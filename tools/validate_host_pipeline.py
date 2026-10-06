"""Classifier-only and raw-g host validation of the unchanged mixed-precision C core.

Run: python tools/validate_host_pipeline.py [--compiler path/to/zig.exe]
Produces individual event tables, paired numeric CSVs, and a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from generate_golden_reference import (
    ROOT, STREAM_NAMES, EVENT_NAMES, c, compare, digest, joblib, np,
    read_csv, reference, write_csv,
)

OUT = ROOT / 'outputs/host_pc_validation'
POINTS = [('balanced', 'balanced_reference'), ('sensitivity', 'sensitivity_oriented_candidate')]


def compile_host(compiler):
    binary = OUT / ('host_validation.exe' if os.name == 'nt' else 'host_validation')
    command = [compiler]
    if Path(compiler).stem.lower() == 'zig':
        command.append('cc')
    command += ['-std=c99', '-O2', '-UNDEBUG', '-Wall', '-Wextra', '-Werror', '-pedantic',
                '-ffp-contract=off', '-Iembedded', 'tests/host_validation.c',
                'embedded/filters.c', 'embedded/features.c', 'embedded/classifier.c',
                'embedded/fall_detector.c', '-o', str(binary)]
    if os.name != 'nt':
        command.append('-lm')
    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(OUT / 'compiler_cache')
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return binary, command


def table(names, python_values, c_values):
    lines = ['| Quantity | Python | C | Absolute error |', '|---|---:|---:|---:|']
    for name, expected, actual in zip(names, python_values, c_values):
        if name == 'label':
            expected_label = 'FALL' if expected else 'ADL'
            actual_label = 'FALL' if actual else 'ADL'
            lines.append(f'| classification | {expected_label} | {actual_label} | '
                         f'{"MATCH" if expected == actual else "MISMATCH"} |')
        else:
            lines.append(f'| {name} | {expected:.17g} | {actual:.17g} | {abs(expected-actual):.8g} |')
    return '\n'.join(lines)


def first_discrepancy(streams, events):
    # Report the first failing sample, then earliest computation at that sample.
    stages = [('Butterworth filter', ['fx', 'fy', 'fz']),
              ('filtered float32 interface', ['ax', 'ay', 'az']),
              ('gravity EMA', ['gx', 'gy', 'gz']),
              ('gravity normalization', ['ux', 'uy', 'uz']),
              ('gravity projection/residual', ['p', 'bx', 'by', 'bz', 'h']),
              ('magnitude', ['m']), ('trigger jerk', ['jerk']),
              ('trigger state', ['high', 'edge', 'trigger'])]
    if 'columns' not in streams:
        return {'stage': 'stream length', 'details': streams}
    failures = []
    for order, (stage, names) in enumerate(stages):
        for name in names:
            column = streams['columns'][name]
            if not column['passed']:
                failures.append((column.get('first_failed_row', 0), order, stage, name, column['max_abs']))
    if failures:
        row, _, stage, name, error = min(failures)
        return {'sample': row, 'stage': stage, 'quantity': name, 'max_abs': error}
    if 'columns' not in events:
        return {'stage': 'event count', 'details': events}
    for name, column in events['columns'].items():
        if not column['passed']:
            return {'stage': 'event bounds/features/classifier', 'quantity': name, **column}
    return None


def main():
    parser = argparse.ArgumentParser()
    local = ROOT / 'outputs/host_toolchain/ziglang/zig.exe'
    parser.add_argument('--compiler', default=str(local) if local.exists()
                        else shutil.which('gcc') or shutil.which('clang'))
    args = parser.parse_args()
    if not args.compiler:
        parser.error('Supply --compiler with a GCC, Clang, or Zig executable')
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [*sorted((ROOT / 'embedded').glob('*.[ch]')),
                 ROOT / 'outputs/mcu_v1_frozen/model_manifest.json',
                 ROOT / 'outputs/causal_trained_models_v1/logistic.joblib',
                 ROOT / 'outputs/sisfall/causal_events.py',
                 ROOT / 'outputs/sisfall/orientation_features.py',
                 ROOT / 'outputs/sisfall/pipeline.py',
                 ROOT / 'processed/baseline_v2/trials.csv',
                 ROOT / 'processed/baseline_v2/config.json']
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}

    # Compile the current core without regenerating headers or touching parameters.
    binary, command = compile_host(args.compiler)
    print('Current mixed-precision C core compiled successfully.', flush=True)
    manifest = json.loads((ROOT / 'outputs/mcu_v1_frozen/model_manifest.json').read_text())
    estimator = joblib.load(ROOT / 'outputs/causal_trained_models_v1/logistic.joblib')
    assert estimator['feature_names'] == manifest['features']['order']
    np.testing.assert_array_equal(estimator['scaler'].mean_, manifest['scaler']['mean'])
    np.testing.assert_array_equal(estimator['scaler'].scale_, manifest['scaler']['scale'])
    np.testing.assert_array_equal(estimator['model'].coef_[0], manifest['logistic']['coefficients'])
    assert estimator['model'].intercept_[0] == manifest['logistic']['intercept']
    np.testing.assert_array_equal(c.SOS, manifest['acceleration_filter']['sos'])
    assert c.of.ALPHA == manifest['gravity']['alpha']

    # A clear previously detected fall and normal ADL come first, then harder cases.
    filenames = ['F01_SA05_R02', 'D01_SA05_R01', 'D03_SA05_R01', 'D05_SA05_R01',
                 'F13_SA05_R01', 'F13_SA05_R02', 'F05_SA05_R01']
    trials = c.read_csv(ROOT / 'processed/baseline_v2/trials.csv')
    config = json.loads((ROOT / 'processed/baseline_v2/config.json').read_text())
    cases = []
    for name in filenames:
        trial = next(row for row in trials if Path(row['path']).stem == name)
        assert trial['split'] == 'validation'
        raw_path = Path(config['data_root']) / trial['path']
        raw, sha = c.p.load_trial(raw_path)
        assert sha == trial['sha256'] and len(raw) == int(trial['n_samples'])
        # Raw SisFall counts/256 are exactly representable in float: no loss at
        # the g-input API, no round-trip from Python g back to integer counts.
        np.testing.assert_array_equal(raw, raw.astype(np.float32).astype(np.float64))
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        input_g = directory / 'input_g.csv'
        np.savetxt(input_g, raw, delimiter=',', fmt='%.17g')
        streams, events = reference(raw, manifest, estimator, 'balanced_reference')
        cases.append(dict(name=name, path=trial['path'], raw=raw, sha=sha,
                          input=input_g, directory=directory, streams=streams, events=events))

    # Classifier only: feed every Python event feature vector from selected raw
    # recordings. This contains both positive and negative reference decisions.
    features = np.concatenate([case['events'][:, 4:7] for case in cases])
    vectors_path = OUT / 'classifier_features.csv'
    np.savetxt(vectors_path, features, delimiter=',', fmt='%.17g')
    standardized = estimator['scaler'].transform(features)
    python_scores = estimator['model'].decision_function(standardized)
    probabilities = estimator['model'].predict_proba(standardized)[:, 1]
    report = {'precision': 'default mixed precision', 'compile_command': command,
              'protected_sha256': before, 'classifier': {}, 'replays': []}
    text = ['# PC validation of the frozen C fall detector',
            'The unchanged C core was compiled before validation. Inputs for full replay are '
            'the existing Python loader\'s chronological XYZ acceleration in g, passed directly '
            'to `fall_detector_push_sample`. No algorithm/model/threshold/header regeneration occurs.',
            '## Classifier-only equivalence',
            'Every Python-reference event feature vector from the selected recordings is tested. '
            'The Python authority is the saved StandardScaler + LogisticRegression, with labels '
            'from `predict_proba >= probability_threshold`. C uses folded weights and a logit threshold.']
    for cli, point in POINTS:
        tau = manifest['documented_operating_points'][point]['score_threshold']
        logit_threshold = math.log(tau / (1 - tau))
        labels = probabilities >= tau
        assert labels.any() and (~labels).any()
        output = OUT / f'classifier_{cli}.csv'
        subprocess.run([str(binary), 'classify', str(vectors_path), str(output), cli], check=True)
        actual = read_csv(output, 7)
        expected = np.column_stack((np.arange(len(features)), features, python_scores,
                                    np.full(len(features), logit_threshold), labels))
        names = ['index', *manifest['features']['order'], 'z', 'threshold', 'label']
        limits = {name: 0 for name in names}
        limits['z'] = 5e-9
        result = compare(expected, actual, names, limits)
        report['classifier'][point] = dict(comparison=result, vectors=len(features),
                                          class_matches=int(np.sum(actual[:, -1] == labels)),
                                          probability_threshold=tau, logit_threshold=logit_threshold)
        write_csv(OUT / f'classifier_python_{cli}.csv', names, expected)
        text += [f'### {point}; probability threshold {tau:.17g}',
                 f'{len(features)} vectors. Score absolute tolerance: 5e-9; features, '
                 'converted thresholds and labels require exact equality.']
        for index, (python_row, c_row) in enumerate(zip(expected, actual)):
            text += [f'Vector {index}', table(names[1:], python_row[1:], c_row[1:])]
        print(f'Classifier {point}: {len(features)} vectors, passed={result["passed"]}, '
              f'max score error={result["columns"]["z"]["max_abs"]:.8g}', flush=True)

    text.append('## Individual raw replay events')
    for case in cases:
        for cli, point in POINTS:
            directory = case['directory'] / cli
            directory.mkdir(exist_ok=True)
            expected_s = case['streams']
            expected_e = reference(case['raw'], manifest, estimator, point)[1]
            streams_path = directory / 'c_streams.csv'
            events_path = directory / 'c_events.csv'
            history_path = directory / 'c_history.csv'
            subprocess.run([str(binary), 'replay', str(case['input']), str(streams_path),
                            str(events_path), str(history_path), cli], check=True)
            actual_s = read_csv(streams_path, 23)
            actual_e = read_csv(events_path, 11)
            history = read_csv(history_path, 5)
            threshold = math.log(manifest['documented_operating_points'][point]['score_threshold'] /
                                 (1 - manifest['documented_operating_points'][point]['score_threshold']))
            expected_e = np.column_stack((expected_e, np.full(len(expected_e), threshold),
                                          np.full(len(expected_e), 300)))
            event_names = [*EVENT_NAMES, 'threshold', 'count']
            limits_s = {name: 2e-10 for name in STREAM_NAMES}
            limits_s.update(jerk=5e-8, high=0, edge=0, trigger=0)
            limits_e = {name: 0 for name in event_names}
            limits_e.update(ga_C2=2e-10, jerk_abs_mean=2e-10, ga_parallel_peak=2e-10, z=5e-9)
            streams = compare(expected_s, actual_s[:, 1:], STREAM_NAMES, limits_s)
            events = compare(expected_e, actual_e, event_names, limits_e)
            np.testing.assert_array_equal(actual_s[:, 0], np.arange(len(case['raw'])))
            # Check actual prehistory scalars against Python's chronological
            # slice for every accepted trigger with enough prehistory.
            triggers = np.flatnonzero(expected_s[:, -1])
            history_expected = []
            for trigger in triggers[triggers >= 100]:
                indices = np.arange(trigger - 100, trigger)
                prior = np.column_stack((expected_s[indices, 15:17], np.abs(expected_s[indices, 17])))
                history_expected.extend(np.column_stack((np.full(100, trigger), indices, prior)))
            history_expected = np.asarray(history_expected).reshape(-1, 5)
            history_result = compare(history_expected, history,
                                     ['trigger', 'index', 'm', 'h', 'abs_p'],
                                     dict(trigger=0, index=0, m=2e-10, h=2e-10, abs_p=2e-10))
            write_csv(directory / 'python_streams.csv', ['index', *STREAM_NAMES],
                      np.column_stack((np.arange(len(expected_s)), expected_s)))
            write_csv(directory / 'python_events.csv', event_names, expected_e)
            write_csv(directory / 'python_history.csv', ['trigger', 'index', 'm', 'h', 'abs_p'], history_expected)
            discrepancy = first_discrepancy(streams, events)
            result = dict(name=case['name'], path=case['path'], raw_sha256=case['sha'],
                          point=point, samples=len(case['raw']), events=len(expected_e),
                          class_matches=int(np.sum(expected_e[:, 8] == actual_e[:, 8]))
                          if expected_e.shape == actual_e.shape else 0,
                          streams=streams, event_comparison=events, history=history_result,
                          first_discrepancy=discrepancy,
                          passed=streams['passed'] and events['passed'] and history_result['passed'])
            report['replays'].append(result)
            text.append(f'### Case: {case["name"]} — {point}')
            for event_index, (python_row, c_row) in enumerate(zip(expected_e, actual_e)):
                event_table = table(event_names, python_row, c_row)
                text += [f'Event {event_index}', event_table]
                # Print every individual event comparison before the summary.
                print(f'Case: {case["name"]} / {cli} / event {event_index}\n{event_table}', flush=True)
            if discrepancy:
                text.append('Earliest discrepancy: ' + json.dumps(discrepancy))
                print('Earliest discrepancy: ' + json.dumps(discrepancy), flush=True)
            text += ['Continuous streams (absolute tolerances: 2e-10, jerk 5e-8):',
                     '| Stream | Maximum absolute error | Mean absolute error |', '|---|---:|---:|']
            if 'columns' in streams:
                text.extend(f'| {name} | {col["max_abs"]:.8g} | {col["mean_abs"]:.8g} |'
                            for name, col in streams['columns'].items())
            text.append(f'Chronological 100-row prehistory comparison passed: {history_result["passed"]}.')

    # A late trigger explicitly exercises >10 full ring wraps. Mutating the
    # decision-time input must change its filtered output but not event features.
    case = cases[0]
    target = case['events'][-1]
    trigger, start, end, decision = map(int, target[:4])
    assert trigger >= 1000 and start == trigger - 100 and end == decision == trigger + 200
    changed = case['raw'][:decision + 1].copy()
    changed[decision] = [12., -12., 12.]
    mutated_path = OUT / 'decision_sample_mutated_g.csv'
    np.savetxt(mutated_path, changed, delimiter=',', fmt='%.17g')
    mutation_streams, mutation_events = OUT / 'mutation_streams.csv', OUT / 'mutation_events.csv'
    subprocess.run([str(binary), 'replay', str(mutated_path), str(mutation_streams),
                    str(mutation_events), str(OUT / 'mutation_history.csv'), 'balanced'], check=True)
    original_events = read_csv(case['directory'] / 'balanced/c_events.csv', 11)
    modified_events = read_csv(mutation_events, 11)
    np.testing.assert_array_equal(original_events, modified_events)
    original_streams = read_csv(case['directory'] / 'balanced/c_streams.csv', 23)
    modified_streams = read_csv(mutation_streams, 23)
    assert np.any(original_streams[decision, 1:4] != modified_streams[decision, 1:4])
    assert len(modified_events) and modified_events[-1, 10] == 300
    # Without arrival of the decision sample the target event cannot be emitted.
    np.savetxt(OUT / 'before_decision_g.csv', case['raw'][:decision], delimiter=',', fmt='%.17g')
    subprocess.run([str(binary), 'replay', str(OUT / 'before_decision_g.csv'),
                    str(OUT / 'before_streams.csv'), str(OUT / 'before_events.csv'),
                    str(OUT / 'before_history.csv'), 'balanced'], check=True)
    np.testing.assert_array_equal(read_csv(OUT / 'before_events.csv', 11), original_events[:-1])
    report['indexing_assertions'] = dict(trigger=trigger, start=start, end=end, decision=decision,
                                        window_samples=300, decision_sample_excluded=True,
                                        no_decision_without_decision_sample=True,
                                        ring_wraps_before_trigger=trigger // 100,
                                        chronological_prehistory_passed=all(x['history']['passed'] for x in report['replays']))
    report['protected_inputs_unchanged'] = all(digest(ROOT / name) == sha for name, sha in before.items())
    classifier_passed = all(x['comparison']['passed'] for x in report['classifier'].values())
    replay_passed = all(x['passed'] for x in report['replays'])
    def maximum(group, names):
        return max(row[group].get('columns', {}).get(name, {'max_abs': 0})['max_abs']
                   for row in report['replays'] for name in names)
    report['summary'] = dict(classifier_only_passed=classifier_passed, raw_replay_passed=replay_passed,
                             recordings=len(cases), samples=sum(len(x['raw']) for x in cases),
                             events_per_operating_point=sum(len(x['events']) for x in cases),
                             exact_class_matches=sum(x['class_matches'] for x in report['replays']),
                             total_decisions=sum(x['events'] for x in report['replays']),
                             maximum_feature_error=maximum('event_comparison', manifest['features']['order']),
                             maximum_replay_score_error=maximum('event_comparison', ['z']),
                             maximum_classifier_only_score_error=max(x['comparison']['columns']['z']['max_abs']
                                                                     for x in report['classifier'].values()),
                             trigger_indices_exact=all(x['streams'].get('columns', {}).get('trigger', {}).get('passed', False)
                                                       for x in report['replays']),
                             event_indices_exact=all(x['event_comparison'].get('columns', {}).get(name, {}).get('passed', False)
                                                     for x in report['replays'] for name in EVENT_NAMES[:4]))
    passed = classifier_passed and replay_passed and report['protected_inputs_unchanged']
    report['passed'] = passed
    text += ['## Explicit window and circular-buffer assertions',
             f'Trigger T={trigger}: start={start}, end={end} exclusive, decision={decision}; '
             f'exactly 300 included samples ({start} through {end-1}). The prehistory buffer has '
             f'wrapped {trigger//100} times. Actual prehistory values were compared in chronological '
             'order against the Python stream for every eligible accepted trigger.',
             'Changing decision-time acceleration to [12, -12, 12] g changed that sample\'s filtered '
             'output while leaving all completed event features, scores, indices and labels bitwise '
             'unchanged in the C output. Omitting that sample withheld the target decision.',
             '## Validation summary', '```json', json.dumps(report['summary'], indent=2), '```',
             f'Core, Python reference and frozen inputs unchanged: {report["protected_inputs_unchanged"]}.',
             'No above-tolerance discrepancy was found.' if passed else 'Validation failed; inspect per-case earliest discrepancies.',
             'The mixed-precision core is ready to proceed to MCU integration and target replay. '
             'This validates PC equivalence, not target execution time, stack/RAM high-water mark, '
             'sensor timing, or single-precision equivalence. Both documented thresholds are tested; '
             'final hardware operating threshold remains explicitly unfrozen. No float build replaces '
             'this reference build. Raw recordings were hash-checked against the existing trial registry; '
             'only validation recordings were replayed, with no fitting or performance selection.',
             'Reproduce: `python tools/validate_host_pipeline.py --compiler PATH_TO_ZIG_GCC_OR_CLANG`. '
             'All actual errors and paired inputs/outputs remain under `outputs/host_pc_validation`. '
             'Score tolerance 5e-9 and feature tolerance 2e-10 allow floating-point summation differences; '
             'indices, counts, thresholds and labels require exact equality.']
    # Table rows need single newlines even when surrounding paragraphs are
    # separated with blank lines. Keep tables valid CommonMark.
    report_text = ('\n\n'.join(text) + '\n').replace('|\n\n|', '|\n|')
    (OUT / 'REPORT.md').write_text(report_text, encoding='utf-8')
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)
    if not passed:
        raise SystemExit('Validation FAILED. See REPORT.md and validation.json for first discrepancies.')


if __name__ == '__main__':
    main()
