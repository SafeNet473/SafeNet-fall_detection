"""Fixed-threshold, paired synthetic orientation experiment; no ML fitting."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil

import orientation_features as f
import pipeline as p
import paper_features as paper
from analyze_filtered import markdown_table
import numpy as np


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def run(args):
    source = args.run.resolve()
    paper_run = args.paper_run.resolve()
    config = json.loads((source/'config.json').read_text())
    data = Path(config['data_root'])
    out = p.safe_destination(args.out, data.resolve())
    if args.rotations < 2:
        raise ValueError('At least two random rotation replicates required')
    assert config['window_size'] == 200 and config['sampling_rate_hz'] == 200
    assert config['filter'] == dict(order=4, cutoff_hz=5, implementation='sosfiltfilt', causal=False)
    inputs = [source/name for name in ('X_raw.npy','X_filtered.npy','labels_groups_splits.npz',
              'windows.csv','trials.csv','config.json','split_subjects.json')]
    inputs += [paper_run/name for name in ('metrics.csv','feature_scores.npz','thresholds.json','provenance.json')]
    hashes = {str(path):paper.sha(path) for path in inputs}
    labels = np.load(source/'labels_groups_splits.npz')
    ids = np.flatnonzero(np.isin(labels['split'], ['validation','test']))
    y, subjects, partitions = (labels[key][ids] for key in ('y','subjects','split'))
    split = json.loads((source/'split_subjects.json').read_text())
    all_rows = csv_rows(source/'windows.csv')
    rows = [all_rows[i] for i in ids]
    for i,row in enumerate(rows):
        assert int(row['window_id']) == ids[i] and int(row['label']) == y[i]
        assert row['subject_id'] == subjects[i] and row['split'] == partitions[i]
        assert subjects[i] in split[partitions[i]]
    assert len(set(s for group in split.values() for s in group)) == sum(map(len,split.values()))
    out.mkdir(parents=True)
    # Preserve the original paper result bytes, including baseline comparison.
    shutil.copyfile(paper_run/'metrics.csv', out/'paper_results_unchanged.csv')
    shutil.copyfile(paper_run/'thresholds.json', out/'paper_thresholds_unchanged.json')
    acc = np.load(source/'X_filtered.npy', mmap_mode='r')
    raw = np.load(source/'X_raw.npy', mmap_mode='r')
    gyro = np.lib.format.open_memmap(out/'gyro_filtered_windows.npy', mode='w+', dtype='float32', shape=(len(ids),200,3))
    states = np.empty((len(ids),3))
    trials = csv_rows(source/'trials.csv')
    by_trial = {}
    for i,row in enumerate(rows):
        by_trial.setdefault(int(row['trial_index']), []).append(i)
    raw_hashes = {}
    low_gravity_samples = 0
    for done,(index, selected) in enumerate(by_trial.items(),1):
        trial = trials[index]
        path = data/trial['path']
        digest = paper.sha(path)
        assert digest == trial['sha256'], f'Changed raw source: {path}'
        raw_hashes[trial['path']] = digest
        text = '\n'.join(line.strip() for line in path.read_text(encoding='utf-8-sig').replace(';','\n').splitlines() if line.strip())
        counts = np.loadtxt(io.StringIO(text), delimiter=',', ndmin=2)
        assert counts.shape == (int(trial['n_samples']),9) and np.isfinite(counts).all()
        acceleration = counts[:, :3]*p.SCALE
        gravity = f.causal_gravity(acceleration[None], acceleration[None,0])[0]
        low_gravity_samples += int((np.linalg.norm(gravity,axis=1)<f.GRAVITY_EPS).sum())
        angular = p.preprocess(counts[:,3:6]*(4000/2**16), 'filtered')
        for i in selected:
            start = int(rows[i]['start_sample'])
            end = int(rows[i]['end_sample_exclusive'])
            states[i] = gravity[start-1] if start else acceleration[0]
            gyro[i] = angular[start:end]
            np.testing.assert_array_equal(raw[ids[i]], acceleration[start:end].astype('float32'))
        if done % 300 == 0:
            print(f'Prepared gyro and causal gravity context: {done}/{len(by_trial)} trials', flush=True)
    gyro.flush()
    np.save(out/'gravity_previous_states.npy', states)
    np.save(out/'window_ids.npy', ids)
    # Quantized existing raw windows supply subsequent causal gravity updates.
    # The trial-history EMA state above is the same float64 raw-count history.
    original = {}
    for start in range(0,len(ids),512):
        selected = ids[start:start+512]
        gravity = f.causal_gravity(raw[selected], states[start:start+len(selected)])
        values = f.features(acc[selected], gyro[start:start+len(selected)], gravity)
        for name,value in values.items():
            original.setdefault(name,np.empty(len(ids)))[start:start+len(selected)] = value
    source_scores = np.load(paper_run/'feature_scores.npz')
    for name in paper.FEATURES:
        np.testing.assert_array_equal(original[name], source_scores[name][ids])
    # Preserve baseline float32 scores exactly; mag_peak is the new float64 score.
    np.testing.assert_allclose(original['baseline'],source_scores['baseline'][ids],rtol=2e-7,atol=1e-7)
    original['baseline'] = source_scores['baseline'][ids]
    val = partitions == 'validation'
    thresholds = json.loads((paper_run/'thresholds.json').read_text())['thresholds']
    thresholds.update({name:p.fit_threshold(values[val],y[val]) for name,values in original.items() if name not in thresholds})
    (out/'thresholds.json').write_text(json.dumps(dict(selection='unrotated validation maximum balanced accuracy; lowest threshold ties',
        paper_thresholds_preserved=True, thresholds=thresholds),indent=2))
    np.savez(out/'original_scores.npz', window_id=ids, **original)
    activity = np.array([row['activity_id'] for row in rows])
    metrics, errors, numerical = [], [], []
    scenarios = [('original',None), ('yaw_y_90',np.array([[0,0,1],[0,1,0],[-1,0,0]],float)),
                 ('tilt_x_90',np.array([[1,0,0],[0,0,-1],[0,1,0]],float))]
    scenarios += [(f'random_{r:02}',args.seed+r) for r in range(args.rotations)]
    original_metrics = {}
    original_errors = {}
    for scenario, parameter in scenarios:
        if parameter is None:
            scores = original
        else:
            scores = {name:np.empty(len(ids)) for name in original}
            rng = np.random.default_rng(parameter) if isinstance(parameter,int) else None
            differences = {name:0. for name in original}
            for start in range(0,len(ids),512):
                selected = ids[start:start+512]
                length = len(selected)
                matrix = f.random_rotations(rng,length) if rng is not None else np.broadcast_to(parameter,(length,3,3))
                gravity = f.causal_gravity(raw[selected], states[start:start+length])
                # Linear EMA is equivariant: rotate the complete state trajectory.
                values = f.features(f.rotate(acc[selected],matrix), f.rotate(gyro[start:start+length],matrix), f.rotate(gravity,matrix))
                for name,value in values.items():
                    ref = original['mag_peak' if name=='baseline' else name][start:start+length]
                    differences[name] = max(differences[name],float(np.max(np.abs(value-ref))))
                    if name in f.INVARIANT:
                        np.testing.assert_allclose(value,ref,rtol=1e-10,atol=1e-10)
                        # Algebraically invariant: canonicalize floating roundoff at threshold ties.
                        value = original[name][start:start+length]
                    scores[name][start:start+length] = value
            numerical.extend(dict(scenario=scenario,feature=name,max_absolute_score_change=value,
                                  invariant=name in f.INVARIANT) for name,value in differences.items())
            # Scores suffice to reproduce predictions/PR without storing rotated windows.
            np.savez(out/f'scores_{scenario}.npz', **scores)
        for partition in ('validation','test'):
            mask = partitions == partition
            for name,value in scores.items():
                result = paper.evaluate(y[mask],value[mask],thresholds[name])
                tn,fp,fn,tp = np.array(result.pop('confusion_matrix')).ravel().tolist()
                if scenario == 'original':
                    original_metrics[partition,name] = result
                base = original_metrics[partition,name]
                metrics.append(dict(scenario=scenario,split=partition,feature=name,threshold=thresholds[name],
                    tn=tn,fp=fp,fn=fn,tp=tp,**result,
                    **{f'delta_{key}':result[key]-base[key] for key in ('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal')}))
                prediction = value >= thresholds[name]
                for code in sorted(set(activity[mask])):
                    chosen = mask & (activity == code)
                    count = int(np.sum(prediction[chosen] != y[chosen]))
                    if scenario == 'original':
                        original_errors[partition,name,code] = count
                    errors.append(dict(scenario=scenario,split=partition,feature=name,activity_code=code,
                        error_type='FP' if code.startswith('D') else 'FN', total=int(chosen.sum()),errors=count,
                        rate=count/int(chosen.sum()),delta_errors=count-original_errors[partition,name,code]))
        print(f'Evaluated {scenario}',flush=True)
    p.write_csv(out/'metrics.csv',metrics)
    p.write_csv(out/'activity_errors.csv',errors)
    p.write_csv(out/'numerical_invariance.csv',numerical)
    # Exact preservation of preexisting paper evaluation values.
    for row in csv_rows(paper_run/'metrics.csv'):
        current = next(r for r in metrics if r['scenario']=='original' and r['split']==row['split'] and r['feature']==row['feature'])
        for key in ('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal','average_precision'):
            assert current[key] == float(row[key])
    summary = []
    keys = ('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal')
    for partition in ('validation','test'):
        for name in original:
            selected = [r for r in metrics if r['scenario'].startswith('random') and r['split']==partition and r['feature']==name]
            row = dict(split=partition,feature=name)
            for key in keys:
                values = np.array([r[key] for r in selected])
                row.update({key+'_original':original_metrics[partition,name][key],key+'_mean':float(values.mean()),
                            key+'_min':float(values.min()),key+'_max':float(values.max()),key+'_std':float(values.std(ddof=1))})
            summary.append(row)
    p.write_csv(out/'rotation_summary.csv',summary)
    failure_summary = []
    for partition in ('validation','test'):
        for name in original:
            for code in ('D03','D04','D06','D18','D19','F13','F14','F15'):
                entries = [r for r in errors if r['scenario'].startswith('random') and r['split']==partition and r['feature']==name and r['activity_code']==code]
                rates = np.array([r['rate'] for r in entries])
                failure_summary.append(dict(split=partition,feature=name,activity_code=code,total=entries[0]['total'],
                    original_errors=original_errors[partition,name,code],original_rate=original_errors[partition,name,code]/entries[0]['total'],
                    random_rate_mean=float(rates.mean()),random_rate_min=float(rates.min()),random_rate_max=float(rates.max())))
    p.write_csv(out/'target_activity_summary.csv',failure_summary)
    assert all(paper.sha(Path(path))==digest for path,digest in hashes.items())
    provenance = dict(input_sha256=hashes,inputs_unchanged=True,source_raw_sha256=raw_hashes,
        seed=args.seed,random_replicates=args.rotations,rotation_distribution='Haar SO(3), independent per window and replicate; same rotation for all vectors/state within window',
        gravity_cutoff_hz=.5,gravity_alpha=float(f.ALPHA),gravity_initialization='first raw sample at trial start; state from raw history before each window',
        near_zero_gravity_samples_in_source_trials=low_gravity_samples,filter=config['filter'],gyro_counts_to_deg_per_s=4000/2**16,
        threshold_policy='new features: unrotated validation only; paper and baseline thresholds retained',
        numerical_policy='verify invariant scores at 1e-10 tolerance then reuse original score to avoid threshold-tie roundoff',
        source_code_sha256={name:paper.sha(Path(__file__).with_name(name)) for name in ('orientation_study.py','orientation_features.py')})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    make_report(out,metrics,summary,failure_summary,args.rotations)
    print(json.dumps([r for r in summary if r['split']=='test' and r['feature'] in ('C2','C8','C9','mag_std','ga_C8')],indent=2))


def make_report(out,metrics,summary,failures,repetitions):
    keys = ('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal')
    report = ['# Orientation robustness: fixed waist data, synthetic coordinate rotations',
        'This is not wrist-worn validation. Synthetic constant rotations change coordinates, not sensor location, wrist articulation, impacts, or within-window time-varying orientation. No features are combined or ML models trained. [Equations, causality and MCU costs](../sisfall/ORIENTATION_METHODS.md).',
        '## 1. Unchanged waist setup',
        '`paper_results_unchanged.csv` is a byte-for-byte copy of the previous paper-feature results. Paper thresholds and original metrics were checked for exact equality. All new features use the same windows and subject split; thresholds maximize unrotated validation balanced accuracy.']
    for partition in ('validation','test'):
        rows = [dict(feature=r['feature'],**{key:f"{r[key]:.2%}" for key in keys}) for r in metrics if r['scenario']=='original' and r['split']==partition]
        report += ['### '+partition,markdown_table(rows,list(rows[0]))]
    report += ['## 2. Synthetic orientation stress test',
        f'{repetitions} independent random SO(3) rotations per window, plus 90-degree y-axis yaw and x-axis tilt controls. All thresholds stay fixed. Replicate ranges describe rotation randomness, not subject-level confidence intervals. PR-AUC is trapezoidal, with AP separately in metrics.csv.']
    for partition in ('validation','test'):
        rows = [dict(feature=r['feature'],original_BA=f"{r['balanced_accuracy_original']:.2%}",
                     random_BA_mean=f"{r['balanced_accuracy_mean']:.2%}",
                     random_BA_range=f"{r['balanced_accuracy_min']:.2%}–{r['balanced_accuracy_max']:.2%}",
                     random_PR_AUC=f"{r['pr_auc_trapezoidal_mean']:.4f}") for r in summary if r['split']==partition]
        report += ['### '+partition,markdown_table(rows,list(rows[0]))]
    report += ['## 3. Axis-assumption diagnostic']
    rows = [dict(feature=r['feature'],scenario=r['scenario'],sensitivity=f"{r['sensitivity']:.2%}",specificity=f"{r['specificity']:.2%}",BA=f"{r['balanced_accuracy']:.2%}")
            for r in metrics if r['split']=='test' and r['feature'] in ('C2','C8') and not r['scenario'].startswith('random')]
    report += [markdown_table(rows,list(rows[0])),
        'Y-axis yaw preserves the x/z plane. X-axis tilt mixes vertical and horizontal components. Changes under tilt and arbitrary rotations, with yaw invariance, identify dependence on the original horizontal-plane assumption. C9 is already rotation invariant (trace of the 3-axis covariance); C3 axis-range norm is not generally invariant.',
        '## 4. Requested activity failures']
    for partition in ('validation','test'):
        for name in ('C2','C8','mag_std','mag_peak','ga_C2','ga_C8'):
            rows = [dict(activity=r['activity_code'],original=f"{r['original_errors']}/{r['total']} ({r['original_rate']:.1%})",
                random_mean=f"{r['random_rate_mean']:.1%}",random_range=f"{r['random_rate_min']:.1%}–{r['random_rate_max']:.1%}")
                for r in failures if r['split']==partition and r['feature']==name]
            report += [f'### {partition}: {name}',markdown_table(rows,list(rows[0]))]
    report += ['## Files and reproduction',
        '`metrics.csv`: every feature, split, orientation and metric including deltas. `rotation_summary.csv`: all metric means/ranges/std. `activity_errors.csv`: all ADL/fall rates, denominators and error-count deltas. `target_activity_summary.csv`: requested codes for every feature. `numerical_invariance.csv`: measured score discrepancies before roundoff canonicalization. `thresholds.json`: locked thresholds. Scores align to `window_ids.npy`.',
        'Run `python outputs/sisfall/orientation_study.py --out outputs/orientation_study_rerun` from the project root. Run `python outputs/sisfall/test_orientation.py` for causality, SO(3), decomposition and invariance tests.',
        'Existing zero-phase filtering and event-centered fall selection remain offline. Only gravity estimation is causal. An MCU implementation needs a separate causal-filter/trigger experiment; runtime estimates do not claim bit-exact reproduction of the offline filter. Gravity-aligned invariance assumes the gravity state is expressed in the same frame as the signal. Rapid wrist rotation and dynamic acceleration can invalidate gravity estimates; these tests do not simulate either.']
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=p.PROJECT/'processed/baseline_v2')
    parser.add_argument('--paper-run',type=Path,default=p.PROJECT/'outputs/paper_feature_comparison')
    parser.add_argument('--out',type=Path,default=p.PROJECT/'outputs/orientation_study')
    parser.add_argument('--rotations',type=int,default=10)
    parser.add_argument('--seed',type=int,default=473)
    run(parser.parse_args())
