"""Analyze existing filtered predictions without refitting or changing subject splits."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pipeline as p
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def sweep(y, scores, thresholds):
    positives = np.sort(scores[y == 1])
    negatives = np.sort(scores[y == 0])
    if not len(positives) or not len(negatives):
        raise ValueError('Both classes are required')
    fn = np.searchsorted(positives, thresholds, side='left')
    tn = np.searchsorted(negatives, thresholds, side='left')
    tp, fp = len(positives) - fn, len(negatives) - tn
    sensitivity, specificity = tp / len(positives), tn / len(negatives)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) > 0)
    return [dict(threshold_g=float(t), tp=int(a), fp=int(b), tn=int(c), fn=int(d),
                 sensitivity=float(se), specificity=float(sp), precision=float(pr),
                 f1=float(2*a / (2*a+b+d)), balanced_accuracy=float((se+sp)/2))
            for t,a,b,c,d,se,sp,pr in zip(thresholds,tp,fp,tn,fn,sensitivity,specificity,precision)]


def operating_threshold(y, scores, target):
    # Highest observed threshold that meets validation sensitivity, ties included.
    candidates = sweep(y, scores, np.unique(scores))
    return max(r['threshold_g'] for r in candidates if r['sensitivity'] >= target)


def markdown_table(rows, columns):
    result = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for row in rows:
        result.append('| ' + ' | '.join(str(row[k]) for k in columns) + ' |')
    return '\n'.join(result)


def main(args):
    run = args.run.resolve()
    out = p.safe_destination(args.out, Path(json.loads((run/'config.json').read_text())['data_root']).resolve())
    sources = ['windows.csv', 'labels_groups_splits.npz', 'scores_filtered.npy',
               'baseline_metrics.json', 'split_subjects.json', 'config.json']
    hashes = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in sources}
    baseline = json.loads((run/'baseline_metrics.json').read_text())['filtered']
    threshold = baseline['threshold_g']
    split = json.loads((run/'split_subjects.json').read_text())
    labels = np.load(run/'labels_groups_splits.npz')
    y, partitions, subjects = labels['y'], labels['split'], labels['subjects']
    scores = np.load(run/'scores_filtered.npy')
    with (run/'windows.csv').open(newline='') as handle:
        windows = list(csv.DictReader(handle))
    assert len(windows) == len(y) == len(scores) and np.isfinite(scores).all()
    for index, row in enumerate(windows):
        assert int(row['window_id']) == index and int(row['label']) == y[index]
        assert row['subject_id'] == subjects[index] and row['split'] == partitions[index]
        assert subjects[index] in split[partitions[index]]
    for a in split:
        for b in split:
            if a != b:
                assert set(split[a]).isdisjoint(split[b])
    descriptions = {}
    readme = Path(json.loads((run/'config.json').read_text())['data_root'])/'Readme.txt'
    for line in readme.read_text(encoding='cp1252').splitlines():
        match = re.match(r'\|\s*([DF]\d\d)\s*\|\s*([^|]+)\|', line)
        if match:
            descriptions[match[1]] = match[2].strip()
    hashes['dataset_readme'] = hashlib.sha256(readme.read_bytes()).hexdigest()
    activities = np.array([r['activity_id'] for r in windows])
    out.mkdir(parents=True)
    groups, errors, points, pr_rows, sweep_rows = [], [], [], [], []
    validation = partitions == 'validation'
    targets = [.95, .90, .85]
    chosen = [('current', None, threshold)] + [
        (f'validation_sensitivity_{int(target*100)}', target,
         operating_threshold(y[validation], scores[validation], target)) for target in targets]
    thresholds = np.unique(np.r_[scores[np.isin(partitions, ['validation', 'test'])],
                                 np.nextafter(scores.max(), np.inf), [t for _,_,t in chosen]])
    summaries = {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axis, name in zip(axes, ('validation', 'test')):
        mask = partitions == name
        prediction = scores >= threshold
        actual = p.metrics(y[mask], scores[mask], threshold)
        for key, value in baseline['metrics'][name].items():
            np.testing.assert_allclose(actual[key], value)
        for label, kind in ((0, 'false_positive'), (1, 'false_negative')):
            total_errors = int(np.sum(mask & (y == label) & (prediction != y)))
            for activity in sorted(set(activities[mask & (y == label)])):
                selection = mask & (activities == activity) & (y == label)
                error_indices = np.flatnonzero(selection & (prediction != y))
                indices = np.flatnonzero(selection)
                groups.append(dict(split=name, error_type=kind, activity_code=activity,
                                   description=descriptions.get(activity, ''), windows=int(selection.sum()),
                                   errors=len(error_indices), error_rate=len(error_indices)/int(selection.sum()),
                                   share_of_split_errors=len(error_indices)/total_errors if total_errors else 0.,
                                   trials=len({windows[i]['path'] for i in indices}),
                                   trials_with_errors=len({windows[i]['path'] for i in error_indices}),
                                   subjects_with_errors=len(set(subjects[error_indices]))))
                for i in error_indices:
                    errors.append(dict(**windows[i], error_type=kind, score_g=float(scores[i]),
                                       threshold_g=threshold, margin_g=float(scores[i]-threshold)))
        curve_precision, recall, curve_thresholds = precision_recall_curve(y[mask], scores[mask])
        ap = float(average_precision_score(y[mask], scores[mask]))
        prevalence = float(y[mask].mean())
        summaries[name] = dict(average_precision=ap, prevalence=prevalence, current=actual)
        for index in range(len(recall)):
            pr_rows.append(dict(split=name, recall=float(recall[index]), precision=float(curve_precision[index]),
                                threshold_g=float(curve_thresholds[index]) if index < len(curve_thresholds) else ''))
        axis.step(recall, curve_precision, where='post', label=f'PR curve (AP={ap:.3f})')
        axis.axhline(prevalence, color='gray', linestyle=':', label=f'Prevalence={prevalence:.3%}')
        for (point_name, target, t), marker in zip(chosen, ('o', 's', '^', 'D')):
            result = sweep(y[mask], scores[mask], np.array([t]))[0]
            point = dict(operating_point=point_name, selected_on='validation', target_sensitivity=target,
                         split=name, **result)
            points.append(point)
            axis.scatter(result['sensitivity'], result['precision'], marker=marker, s=45,
                         label='Current' if target is None else f'Val target {target:.0%}', zorder=3)
        sweep_rows.extend(dict(split=name, **r) for r in sweep(y[mask], scores[mask], thresholds))
        axis.set(xlabel='Recall / sensitivity', title=name.capitalize(), xlim=(0,1.01), ylim=(0,1.02))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel('Precision')
    fig.suptitle('Filtered magnitude baseline — fixed subject partitions')
    fig.tight_layout()
    fig.savefig(out/'precision_recall.png', dpi=160)
    fig.savefig(out/'precision_recall.svg')
    plt.close(fig)
    for filename, rows in [('activity_errors.csv', groups), ('error_windows.csv', errors),
                           ('operating_points.csv', points), ('threshold_sweep.csv', sweep_rows),
                           ('precision_recall.csv', pr_rows)]:
        p.write_csv(out/filename, rows)
    # Count conservation and identical source hashes protect against alignment/split mistakes.
    for name in ('validation', 'test'):
        for kind, expected in [('false_positive', baseline['metrics'][name]['confusion_matrix'][0][1]),
                               ('false_negative', baseline['metrics'][name]['confusion_matrix'][1][0])]:
            assert sum(r['errors'] for r in groups if r['split'] == name and r['error_type'] == kind) == expected
    assert all(hashlib.sha256((run/name).read_bytes()).hexdigest() == hashes[name] for name in sources)
    summary = dict(source_run=str(run), source_hashes=hashes, current_threshold_g=threshold,
                   splits_unchanged=True, operating_point_selection='highest validation threshold meeting sensitivity target',
                   summaries=summaries)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    report = ['# Filtered baseline failure analysis',
              f'Existing threshold: **{threshold:.9f} g**. Existing subject splits and input artifacts are unchanged.',
              'Errors below use the current threshold. Operating thresholds are selected exclusively on validation: choose the highest threshold meeting the requested sensitivity, which maximizes specificity under that constraint. Test sensitivities are measured outcomes, not guaranteed targets.',
              '## Main failure modes',
              'Jogging (D03/D04) dominates false alarms. A peak-only rule cannot distinguish these high-acceleration ADL windows from fall impacts. Misses are concentrated in seated falls and sitting/standing transitions; their selected filtered peaks fall below the threshold. This is consistent with overlapping peak-magnitude distributions, but does not by itself establish whether filtering or event localization caused a particular miss.',
              '\n'.join(f"- {name}: jogging contributes "
                        f"{sum(r['errors'] for r in groups if r['split']==name and r['activity_code'] in ('D03','D04'))} / "
                        f"{summaries[name]['current']['confusion_matrix'][0][1]} false positives; seated falls F13–F15 contribute "
                        f"{sum(r['errors'] for r in groups if r['split']==name and r['activity_code'] in ('F13','F14','F15'))} / "
                        f"{summaries[name]['current']['confusion_matrix'][1][0]} false negatives."
                        for name in ('validation', 'test')),
              'All three validation sensitivity targets fall short on held-out test subjects. Lowering the threshold to the validation 95% point reduces test misses from 36 to 22 but increases false positives from 2,296 to 3,001. Raising it improves precision at the cost of more missed falls. Peak magnitude alone shows a substantial sensitivity/false-positive tradeoff; temporal or posture features are a reasonable next non-neural experiment, evaluated with the fixed split.',
              '## Operating points']
    display = []
    for r in points:
        display.append(dict(point=r['operating_point'], split=r['split'], threshold_g=f"{r['threshold_g']:.6f}",
                            **{k:f"{r[k]:.2%}" for k in ('sensitivity','specificity','precision','f1','balanced_accuracy')}))
    report.append(markdown_table(display, list(display[0])))
    for kind, title in [('false_positive','ADL false positives'), ('false_negative','Fall false negatives')]:
        report.append('## '+title)
        table = []
        codes = sorted({r['activity_code'] for r in groups if r['error_type'] == kind})
        for code in codes:
            entries = {r['split']:r for r in groups if r['activity_code'] == code}
            table.append(dict(code=code, description=descriptions.get(code,''),
                              validation=f"{entries['validation']['errors']}/{entries['validation']['windows']} ({entries['validation']['error_rate']:.1%})",
                              test=f"{entries['test']['errors']}/{entries['test']['windows']} ({entries['test']['error_rate']:.1%})"))
        report.append(markdown_table(table, list(table[0])))
    report += ['## Interpretation and files',
               '![Precision–recall curves](precision_recall.png)',
               '\n'.join(f"- {name}: average precision {r['average_precision']:.4f}; fall prevalence {r['prevalence']:.2%}." for name,r in summaries.items()),
               '`threshold_sweep.csv` contains every distinct validation/test score boundary plus the selected operating points and a no-positive boundary. Both partitions are evaluated at identical thresholds; the test sweep is descriptive and must not be used to select a deployment threshold. Precision is defined as zero when no windows are predicted positive; the PR curve uses the conventional final (recall=0, precision=1) point without a threshold.',
               '`activity_errors.csv` includes denominators, error rates, error shares, and affected trials/subjects. `error_windows.csv` identifies every erroneous window for inspection. `operating_points.csv` includes confusion counts and all requested metrics. `precision_recall.csv` contains full curve coordinates. `summary.json` records input hashes and checks.',
               'Counts are window-level: overlapping ADL windows can count the same movement more than once; fall trials contribute one selected event each. Compare error rates as well as counts because activity durations and exposure differ. Event-centered labeling and offline filtering remain unchanged. These results do not estimate streaming false alarms/hour. Activity descriptions come from the local raw dataset Readme.txt (read-only).']
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), summary=summaries, operating_points=points), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=p.PROJECT/'processed/baseline_v2')
    parser.add_argument('--out', type=Path, default=p.PROJECT/'outputs/filtered_failure_analysis')
    main(parser.parse_args())
