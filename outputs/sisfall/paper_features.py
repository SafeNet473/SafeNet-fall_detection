"""Independent SisFall Table 4 features on the existing 200-sample windows.

See FEATURE_EQUATIONS.md for explicit adaptations and numerical conventions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pipeline as p
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from analyze_filtered import markdown_table

FEATURES = ('C2', 'C3', 'C8', 'C9', 'C13')


def extract_features(windows):
    a = np.asarray(windows, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (200, 3) or not np.isfinite(a).all():
        raise ValueError('Expected finite (batch, 200, 3) filtered accelerations in g')
    horizontal = np.sqrt(a[:, :, 0]**2 + a[:, :, 2]**2)
    ranges = np.ptp(a, axis=1)
    variance = np.var(a, axis=1, ddof=1)
    return dict(C2=horizontal.max(axis=1),
                C3=np.sqrt(np.sum(ranges**2, axis=1)),
                C8=np.sqrt(variance[:, 0] + variance[:, 2]),
                C9=np.sqrt(variance.sum(axis=1)),
                C13=np.sum((horizontal[:, :-1] + horizontal[:, 1:]) / 2, axis=1))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def evaluate(y, scores, threshold):
    result = p.metrics(y, scores, threshold)
    precision, recall, _ = precision_recall_curve(y, scores)
    result['pr_auc_trapezoidal'] = float(auc(recall, precision))
    result['average_precision'] = float(average_precision_score(y, scores))
    return result


def run(args):
    source = args.run.resolve()
    config = json.loads((source/'config.json').read_text())
    destination = p.safe_destination(args.out, Path(config['data_root']).resolve())
    assert config['window_size'] == 200 and config['sampling_rate_hz'] == 200
    assert config['filter'] == dict(order=4, cutoff_hz=5, implementation='sosfiltfilt', causal=False)
    input_files = ['X_filtered.npy', 'windows.csv', 'labels_groups_splits.npz',
                   'split_subjects.json', 'config.json', 'baseline_metrics.json', 'scores_filtered.npy']
    hashes = {name: sha(source/name) for name in input_files}
    labels = np.load(source/'labels_groups_splits.npz')
    y, subjects, partitions = labels['y'], labels['subjects'], labels['split']
    split = json.loads((source/'split_subjects.json').read_text())
    with (source/'windows.csv').open(newline='') as handle:
        windows = list(csv.DictReader(handle))
    X = np.load(source/'X_filtered.npy', mmap_mode='r')
    assert X.shape == (len(y), 200, 3) and len(windows) == len(y)
    for i, row in enumerate(windows):
        assert int(row['window_id']) == i and int(row['label']) == y[i]
        assert row['subject_id'] == subjects[i] and row['split'] == partitions[i]
        assert subjects[i] in split[partitions[i]]
    assert len(set(s for group in split.values() for s in group)) == sum(map(len, split.values()))
    scores = {feature: np.empty(len(y), dtype=np.float64) for feature in FEATURES}
    scores['baseline'] = np.load(source/'scores_filtered.npy')
    for start in range(0, len(y), 2048):
        block = X[start:start+2048]
        computed = extract_features(block)
        for feature in FEATURES:
            scores[feature][start:start+len(block)] = computed[feature]
        # Exact same float32 calculation as the saved baseline.
        np.testing.assert_array_equal(np.linalg.norm(block, axis=2).max(axis=1),
                                      scores['baseline'][start:start+len(block)])
    baseline = json.loads((source/'baseline_metrics.json').read_text())['filtered']
    val = partitions == 'validation'
    thresholds = {'baseline': baseline['threshold_g']}
    thresholds.update({f: p.fit_threshold(scores[f][val], y[val]) for f in FEATURES})
    assert p.fit_threshold(scores['baseline'][val], y[val]) == thresholds['baseline']
    destination.mkdir(parents=True)
    np.savez(destination/'feature_scores.npz', window_id=np.arange(len(y)), **scores)
    activities = np.array([r['activity_id'] for r in windows])
    results, grouped, error_rows, curves = [], [], [], []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, partition in zip(axes, ('validation', 'test')):
        mask = partitions == partition
        baseline_metrics = evaluate(y[mask], scores['baseline'][mask], thresholds['baseline'])
        for feature in ('baseline', *FEATURES):
            values, threshold = scores[feature], thresholds[feature]
            metric = evaluate(y[mask], values[mask], threshold)
            if feature == 'baseline':
                for key, value in baseline['metrics'][partition].items():
                    np.testing.assert_allclose(metric[key], value)
            tn, fp, fn, tp = np.array(metric.pop('confusion_matrix')).ravel().tolist()
            results.append(dict(feature=feature, split=partition, threshold=threshold,
                                units='g_sample' if feature == 'C13' else 'g', tn=tn, fp=fp, fn=fn, tp=tp,
                                **metric, **{f'delta_{k}': metric[k]-baseline_metrics[k] for k in
                                ('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal','average_precision')}))
            precision, recall, pr_thresholds = precision_recall_curve(y[mask], values[mask])
            ax.step(recall, precision, where='post', label=f"{feature} AP={metric['average_precision']:.3f}")
            ax.scatter(metric['sensitivity'], metric['precision'], s=15)
            curves.extend(dict(feature=feature, split=partition, precision=float(pr), recall=float(re),
                               threshold=float(pr_thresholds[i]) if i < len(pr_thresholds) else '')
                          for i, (pr,re) in enumerate(zip(precision, recall)))
            prediction = values >= threshold
            for activity in sorted(set(activities[mask])):
                selected = mask & (activities == activity)
                label = int(y[np.flatnonzero(selected)[0]])
                bad = selected & (prediction != y)
                baseline_bad = selected & ((scores['baseline'] >= thresholds['baseline']) != y)
                denominator = int(selected.sum())
                grouped.append(dict(feature=feature, split=partition, activity_code=activity,
                                    error_type='false_negative' if label else 'false_positive',
                                    total=denominator, errors=int(bad.sum()), rate=float(bad.sum()/denominator),
                                    baseline_errors=int(baseline_bad.sum()), baseline_rate=float(baseline_bad.sum()/denominator),
                                    delta_rate=float((bad.sum()-baseline_bad.sum())/denominator)))
                for i in np.flatnonzero(bad):
                    error_rows.append(dict(feature=feature, **windows[i], score=float(values[i]), threshold=threshold))
            assert sum(r['errors'] for r in grouped if r['feature']==feature and r['split']==partition
                       and r['error_type']=='false_positive') == fp
            assert sum(r['errors'] for r in grouped if r['feature']==feature and r['split']==partition
                       and r['error_type']=='false_negative') == fn
        ax.set(title=partition.capitalize(), xlabel='Recall / sensitivity', xlim=(0,1), ylim=(0,1.02))
        ax.axhline(y[mask].mean(), color='gray', ls=':', lw=1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Precision')
    fig.suptitle('Independent filtered SisFall features — unchanged event windows')
    fig.tight_layout()
    fig.savefig(destination/'precision_recall.png', dpi=160)
    fig.savefig(destination/'precision_recall.svg')
    plt.close(fig)
    p.write_csv(destination/'metrics.csv', results)
    p.write_csv(destination/'activity_errors.csv', grouped)
    p.write_csv(destination/'error_windows.csv', error_rows)
    p.write_csv(destination/'precision_recall.csv', curves)
    (destination/'thresholds.json').write_text(json.dumps(dict(selected_on='validation',
        objective='maximum balanced accuracy', prediction_rule='score >= threshold',
        tie_break='lowest threshold', thresholds=thresholds), indent=2))
    # Equivalent conventions change only units, not the ordering of fixed-length windows.
    conventions = dict(C3_literal_RMS='C3 / sqrt(3)', C8_population_std='C8 * sqrt(199/200)',
                       C9_population_std='C9 * sqrt(199/200)', C13_seconds='C13 / 200')
    for f, factor in [('C3', 1/np.sqrt(3)), ('C8', np.sqrt(199/200)),
                      ('C9', np.sqrt(199/200)), ('C13', 1/200)]:
        np.testing.assert_array_equal(scores[f] >= thresholds[f], scores[f]*factor >= thresholds[f]*factor)
    assert all(sha(source/name) == hashes[name] for name in input_files)
    provenance = dict(source_run=str(source), source_sha256=hashes, source_files_unchanged=True,
                      feature_window_samples=200, filter=config['filter'], sample_rate_hz=200,
                      c3_convention='Euclidean norm of axis-wise ranges, consistent with C1 displayed norm',
                      std_ddof=1, c13_integration='trapezoidal, dx=1 sample', equivalent_conventions=conventions,
                      threshold_selection='validation only; max balanced accuracy; lowest threshold on ties',
                      pr_auc='trapezoidal sklearn auc(recall, precision); AP reported separately',
                      source_code_sha256=sha(Path(__file__)),
                      equations_sha256=sha(Path(__file__).with_name('FEATURE_EQUATIONS.md')))
    (destination/'provenance.json').write_text(json.dumps(provenance, indent=2))
    report = ['# Independent SisFall feature comparison',
              'All six scores use the existing filtered 200-sample windows and fixed subjects. No features are combined. Each new threshold maximizes validation balanced accuracy. The baseline uses its saved threshold. Full equations and ambiguities: [FEATURE_EQUATIONS.md](../sisfall/FEATURE_EQUATIONS.md).',
              'PR-AUC below is trapezoidal area; AP is non-interpolated average precision. They are different summaries. Threshold metrics are window-weighted.']
    keys = ('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal','average_precision')
    for partition in ('validation','test'):
        report.append('## '+partition.capitalize())
        table = [dict(feature=r['feature'], threshold=f"{r['threshold']:.9g}", units=r['units'],
                      **{k:f"{r[k]:.2%}" for k in keys}) for r in results if r['split']==partition]
        report.append(markdown_table(table, list(table[0])))
    report += ['## Activity-level errors', 'Each cell is errors / windows (rate). Every fall type has 25 selected events per partition; ADL counts include overlapping windows.']
    for partition in ('validation', 'test'):
        for kind in ('false_positive','false_negative'):
            subset = [r for r in grouped if r['split']==partition and r['error_type']==kind]
            table = []
            for code in sorted({r['activity_code'] for r in subset}):
                row = dict(activity=code)
                for f in ('baseline', *FEATURES):
                    entry = next(r for r in subset if r['activity_code']==code and r['feature']==f)
                    row[f] = f"{entry['errors']}/{entry['total']} ({entry['rate']:.1%})"
                table.append(row)
            report += [f'### {partition}: {kind.replace("_", " ")}', markdown_table(table, list(table[0]))]
    report += ['## Failure-mode summary']
    for f in FEATURES:
        test = next(r for r in results if r['feature']==f and r['split']=='test')
        top = []
        for kind in ('false_positive', 'false_negative'):
            entries = sorted([r for r in grouped if r['feature']==f and r['split']=='test' and r['error_type']==kind],
                             key=lambda r: (-r['errors'],r['activity_code']))[:3]
            top.append(kind.replace('_',' ') + ': ' + ', '.join(f"{r['activity_code']} {r['errors']}/{r['total']}" for r in entries))
        report.append(f"- **{f}**: test balanced accuracy delta {100*test['delta_balanced_accuracy']:+.2f} percentage points; " + '; '.join(top) + '.')
    report += ['## Reproduce',
               'From the project root: `python outputs/sisfall/paper_features.py --out outputs/paper_feature_comparison_rerun`. Existing output directories are refused. Run `python outputs/sisfall/test_paper_features.py` for equation checks.',
               'Input hashes, alignment, baseline-score parity, confusion-count reconciliation and constant-scale equivalence were checked. `provenance.json` records conventions; `feature_scores.npz` aligns with the source window IDs; `metrics.csv` includes baseline deltas; `activity_errors.csv` includes every activity and rate delta; `error_windows.csv` identifies misses and false alarms.',
               'The fixed event extraction is label-aware and the filter is offline. These are comparisons under our protocol, not a reproduction of the paper\'s reported accuracy or a streaming detector evaluation. No test-derived feature or threshold selection is performed.']
    (destination/'REPORT.md').write_text('\n\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps([r for r in results if r['split']=='test'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=p.PROJECT/'processed/baseline_v2')
    parser.add_argument('--out', type=Path, default=p.PROJECT/'outputs/paper_feature_comparison')
    run(parser.parse_args())
