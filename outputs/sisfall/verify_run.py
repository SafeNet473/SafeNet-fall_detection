"""Verify saved provenance, class/window policies, and grouped split isolation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pipeline as p
import numpy as np


def verify(run):
    config = json.loads((run / 'config.json').read_text())
    data = Path(config['data_root'])
    with (run / 'trials.csv').open(newline='') as handle:
        trials = list(csv.DictReader(handle))
    with (run / 'windows.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    labels = np.load(run / 'labels_groups_splits.npz')
    split = json.loads((run / 'split_subjects.json').read_text())
    assert len(rows) == config['n_windows'] == len(labels['y'])
    for a in split:
        for b in split:
            if a != b:
                assert set(split[a]).isdisjoint(split[b])
    by_trial = {}
    for i, row in enumerate(rows):
        assert int(row['window_id']) == i
        assert int(row['label']) == labels['y'][i]
        assert row['subject_id'] == labels['subjects'][i]
        assert row['split'] == labels['split'][i]
        assert row['subject_id'] in split[row['split']]
        by_trial.setdefault(int(row['trial_index']), []).append((i, row))
    arrays = {mode: np.load(run / f'X_{mode}.npy', mmap_mode='r') for mode in ('raw', 'filtered')}
    for array in arrays.values():
        assert array.shape == (len(rows), config['window_size'], 3)
    for index, trial in enumerate(trials):
        signal, digest = p.load_trial(data / trial['path'])
        assert digest == trial['sha256'], trial['path']
        expected = p.window_bounds(signal, int(trial['label']), config['window_size'], config['stride'])
        actual = by_trial[index]
        assert len(actual) == len(expected)
        filtered = p.preprocess(signal, 'filtered')
        for (i, row), (start, end, peak) in zip(actual, expected):
            assert (int(row['start_sample']), int(row['end_sample_exclusive']), int(row['peak_sample'])) == (start, end, peak)
            assert row['subject_id'] == trial['subject_id'] and row['path'] == trial['path']
            np.testing.assert_array_equal(arrays['raw'][i], signal[start:end].astype(np.float32))
            np.testing.assert_array_equal(arrays['filtered'][i], filtered[start:end].astype(np.float32))
    for entry in json.loads((run / 'subject_identity_audit.json').read_text()):
        assert hashlib.sha256((data / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    report = dict(verified_trials=len(trials), verified_windows=len(rows), raw_source_hashes_unchanged=True,
                  subject_partitions_disjoint=True, saved_windows_match_recomputed_signals=True)
    (run / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    resolved = args.run.resolve()
    if not any(resolved.is_relative_to((p.PROJECT / folder).resolve()) for folder in ('processed', 'outputs')):
        parser.error('Run must be in processed/ or outputs/')
    verify(resolved)
