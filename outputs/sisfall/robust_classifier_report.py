"""Summarize locked classifier outputs and estimate embedded implementation costs."""
import argparse
import csv
import json
from pathlib import Path

import robust_classifiers as r
import numpy as np
from sklearn.metrics import precision_recall_curve


def render(out):
    parameters=json.loads((out/'model_parameters.json').read_text())
    metrics=r.read_csv(out/'metrics.csv')
    activities=r.read_csv(out/'activity_errors.csv')
    checks=r.read_csv(out/'rotation_checks.csv')
    candidate_rows=r.read_csv(out/'validation_candidates.csv')
    lr=parameters['logistic_regression']
    coefficient_rows=[dict(feature=name,training_mean=lr['scaler_mean'][i],training_scale=lr['scaler_scale'][i],
                           standardized_weight=lr['standardized_coefficients'][i],raw_feature_weight=lr['raw_feature_coefficients'][i])
                      for i,name in enumerate(r.FEATURE_NAMES)]
    r.p.write_csv(out/'logistic_coefficients.csv',coefficient_rows)
    names=['baseline','C2','C8','logistic_regression','decision_tree']
    summary=[]
    for partition in ('validation','test'):
        for name in names:
            original=next(v for v in metrics if v['scenario']=='original' and v['split']==partition and v['model']==name)
            rotated=[v for v in metrics if v['scenario'].startswith('random') and v['split']==partition and v['model']==name]
            entry=dict(split=partition,model=name)
            for key in r.METRICS:
                values=np.array([float(v[key]) for v in rotated])
                entry.update({key+'_original':float(original[key]),key+'_mean':float(values.mean()),
                              key+'_min':float(values.min()),key+'_max':float(values.max()),
                              key+'_std':float(values.std(ddof=1)) if len(values)>1 else 0})
            summary.append(entry)
    r.p.write_csv(out/'rotation_summary.csv',summary)
    codes=['D03','D04','D06','D18','D19','F13','F14','F15']
    failure_summary=[]
    for partition in ('validation','test'):
        for name in names:
            for code in codes:
                source=next(v for v in activities if v['scenario']=='original' and v['split']==partition and v['model']==name and v['activity_code']==code)
                randomized=[v for v in activities if v['scenario'].startswith('random') and v['split']==partition and v['model']==name and v['activity_code']==code]
                values=np.array([float(v['rate']) for v in randomized])
                failure_summary.append(dict(split=partition,model=name,activity=code,total=int(source['total']),
                    original_errors=int(source['errors']),original_rate=float(source['rate']),
                    rotation_mean=float(values.mean()),rotation_min=float(values.min()),rotation_max=float(values.max())))
    r.p.write_csv(out/'target_activity_summary.csv',failure_summary)
    findings=['# Non-neural robust classifier findings',
        'Both families are reported; test outcomes were not used to choose between them. Candidate features and grids were declared before fitting. Training-only standardization, validation-only hyperparameter/threshold selection, and the fixed subject partitions were retained.',
        '## Fixed test results']
    table=[]
    for name in names:
        row=next(v for v in metrics if v['scenario']=='original' and v['split']=='test' and v['model']==name)
        table.append(dict(model=name,**{key:f"{float(row[key]):.2%}" for key in r.METRICS},FP=row['fp'],FN=row['fn']))
    findings.append(r.markdown_table(table,list(table[0])))
    findings+=['## Requested failure modes','Original test errors / total; the complete validation breakdown is in REPORT.md.']
    table=[]
    for code in codes:
        row=dict(activity=code)
        for name in names:
            entry=next(v for v in failure_summary if v['split']=='test' and v['model']==name and v['activity']==code)
            row[name]=f"{entry['original_errors']}/{entry['total']} ({entry['original_rate']:.1%})"
        table.append(row)
    findings.append(r.markdown_table(table,list(table[0])))
    findings+=['## Rotations','All model features and predictions were recomputed from rotated vectors and rotated causal gravity history. Unlike the earlier feature study, model scores were not canonicalized to their original values.']
    for name in ('logistic_regression','decision_tree'):
        entries=[v for v in checks if v['model']==name]
        findings.append(f"- {name}: {sum(int(v['changed_predictions']) for v in entries)} changed predictions across validation/test rotation evaluations; "
                        f"maximum score difference {max(float(v['max_score_absolute_error']) for v in entries):.3g}.")
    findings+=['## Interpretation limits',
        'The linear and shallow-tree models combine orientation-robust statistics, but rotation robustness alone does not establish wrist performance. These remain waist recordings with offline filtering and event-centered fall selection. Activity exposure is imbalanced and overlapping ADL windows are correlated; the counts are not false alarms per hour. Validation tuning can overfit its subjects, so its selected score is not an independent estimate.',
        'No further feature, threshold or hyperparameter changes were made after test evaluation. MCU float32 feature extraction and a causal filter/trigger need separate validation; the current export parity check is against the existing offline features.',
        '![Precision–recall comparison](precision_recall.png)',
        'Complete tables: [REPORT.md](REPORT.md), [rotation_summary.csv](rotation_summary.csv), [target_activity_summary.csv](target_activity_summary.csv). Exact fitted features/parameters: [model_parameters.json](model_parameters.json). [MCU estimates](MCU.md).']
    (out/'FINDINGS.md').write_text('\n\n'.join(findings)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred=np.load(out/'original_predictions.npz')
    provenance=json.loads((out/'provenance.json').read_text())
    label_path=next(Path(path) for path in provenance['input_sha256'] if Path(path).name=='labels_groups_splits.npz')
    labels=np.load(label_path)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,partition in zip(axes,('validation','test')):
        mask=labels['split']==partition
        for name in names:
            precision,recall,_=precision_recall_curve(labels['y'][mask],pred[name][mask])
            row=next(v for v in metrics if v['scenario']=='original' and v['split']==partition and v['model']==name)
            ax.step(recall,precision,where='post',label=f"{name} AP={float(row['average_precision']):.3f}")
            ax.scatter(float(row['sensitivity']),float(row['precision']),s=20)
        ax.set(title=partition.capitalize(),xlabel='Recall / sensitivity',xlim=(0,1),ylim=(0,1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower left')
    axes[0].set_ylabel('Precision')
    fig.suptitle('Locked robust models versus unchanged SisFall baselines')
    fig.tight_layout()
    fig.savefig(out/'precision_recall.png',dpi=160)
    fig.savefig(out/'precision_recall.svg')
    plt.close(fig)
    mcu_report(out,parameters)
    # Reconcile every metric row with activity aggregates and fixed thresholds.
    expected_thresholds={name:float(next(v for v in metrics if v['scenario']=='original' and v['model']==name)['threshold']) for name in names}
    for row in metrics:
        assert float(row['threshold'])==expected_thresholds[row['model']]
        matches=[v for v in activities if v['scenario']==row['scenario'] and v['split']==row['split'] and v['model']==row['model']]
        assert sum(int(v['errors']) for v in matches if v['error_type']=='FP')==int(row['fp'])
        assert sum(int(v['errors']) for v in matches if v['error_type']=='FN')==int(row['fn'])
    verified=dict(metric_rows=len(metrics),activity_rows=len(activities),thresholds_fixed=True,activity_counts_reconciled=True,
                  model_rotation_changed_predictions=sum(int(v['changed_predictions']) for v in checks),
                  candidate_count=len(candidate_rows))
    (out/'verification.json').write_text(json.dumps(verified,indent=2))
    print(json.dumps(verified,indent=2))


def mcu_report(out,parameters):
    costs=r.read_csv(r.p.PROJECT/'outputs/orientation_study/mcu_operation_estimates.csv')
    feature_cost={}
    for row in costs:
        for name in row['feature'].split(' / '):
            feature_cost[name]=row
    entries=[]
    for family in ('logistic_regression','decision_tree'):
        data=parameters[family]
        used=list(r.FEATURE_NAMES) if family=='logistic_regression' else data['features_used']
        sums={key:sum(int(feature_cost[name][key]) for name in used) for key in ('additions','multiplications','square_roots')}
        reciprocals=200 if any(name.startswith('ga_') for name in used) else 0
        entries.append(dict(model=family,candidate_features=21,required_features=len(used),window_samples=200,window_seconds=1,
            feature_names=';'.join(used),**{f'independent_feature_upper_bound_{key}':value for key,value in sums.items()},
            shared_gravity_normalization_reciprocals=reciprocals,
            model_multiplications=21 if family=='logistic_regression' else 0,
            model_additions=21 if family=='logistic_regression' else 0,
            model_max_comparisons=1 if family=='logistic_regression' else data['actual_depth']+1,
            model_sqrt=0,model_division=0))
    r.p.write_csv(out/'mcu_complexity.csv',entries)
    tree=parameters['decision_tree']
    n=tree['node_count']
    text=['# MCU-oriented complexity estimates',
        'These are arithmetic/storage estimates for an incremental implementation, not measured MCU cycles or energy. Existing Python evaluation uses float64 features. Feature equations and individual costs remain in [ORIENTATION_METHODS.md](../sisfall/ORIENTATION_METHODS.md). No firmware implementation or float32 accuracy claim is made here.',
        '## Models',
        '**Logistic regression:** 21 features. Fuse the training scaler into exported raw-feature weights: `w_raw = w / scale`, `b_raw = b - dot(w_raw, mean)`. Classify `dot(w_raw,x) + b_raw >= log(tau/(1-tau))`. The final comparison offset can be folded into the intercept. This costs 21 multiplications, roughly 21 additions and one comparison; no sigmoid/exp, sqrt or division is required for a binary decision. Outputting the sigmoid score adds exp and division. Store 21 weights plus fused decision intercept: 22 float32 values = **88 bytes**, preferably flash. `model_parameters.json` retains full-precision coefficients and threshold for conversion.',
        f"**Decision tree:** actual depth **{tree['actual_depth']}**, **{n} nodes**, **{tree['leaf_count']} leaves**, and **{len(tree['features_used'])} distinct used features**: "+', '.join(tree['features_used'])+'. '+
        f"Inference requires at most {tree['actual_depth']} split comparisons plus a final leaf-score threshold comparison, no multiplications, divisions or roots. Precompute leaf decisions to remove the final comparison. "
        f"A simple all-node table with int16 children/feature and float32 split/leaf score takes about 16 bytes/node after alignment: **{16*n} bytes**, preferably flash. Compiler packing and pruning can reduce this. A node index requires only a few runtime bytes.",
        'Only tree-used features are required at inference; exporting this subset follows the fitted training tree and does not use test feature selection. The Python evaluation retained the full 21-column matrix. Logistic regression retains every feature regardless of small or correlated coefficients.',
        'This fitted tree uses only acceleration-derived features: it does not require gyroscope input at inference, even though gyroscope candidates were available during training. Its incremental implementation can therefore omit gyro filtering and magnitude calculation; a causal accelerometer-only SOS filter would require 48 bytes of state rather than the 96-byte six-axis budget below.',
        '## Window and feature extraction',
        '200 samples at 200 Hz, one-second windows, unchanged 100-sample ADL stride and event-centered fall windows. Jerk has 199 differences; pre/post variance uses two 100-sample halves. Gravity context precedes the window, so its state must be continuously carried from past raw samples.',
        '`mcu_complexity.csv` gives conservative sums of independently computing each required feature. These deliberately double-count shared magnitude/EMA work and are upper estimates, not a recommended implementation. Expensive per-sample operations are XYZ magnitude roots and gravity normalization reciprocal/sqrt.',
        '**Shared full-21 implementation:** computing accelerometer magnitude, gyro magnitude and gravity/perpendicular projections once per sample, then sharing moment accumulators, costs approximately **8,000 additions, 7,000 multiplications, 810 square roots and 200 reciprocals per 200-sample feature window**, excluding filtering, sensor scaling and model inference. This is an illustrative budget, not an optimized lower bound. Roots can be avoided for some squared-norm statistics, but gyro/acceleration magnitude means and jerk still require magnitudes; gravity normalization remains costly. Constant reciprocals (1/N, 1/(N−1), fs and alpha) compile to multiplications.',
        'A future causal fourth-order filter with two biquads per axis would add about 60 multiplies + 48 adds per incoming sample for six sensor axes, with 96 bytes of filter state. Six count-to-unit conversions add six multiplies/sample. **This causal filter does not reproduce current forward/backward filtering.** Current offline filtering needs reverse access/full-trial storage and cannot be made causal without changing the evaluation.',
        '## Approximate RAM',
        'For all 21 features, a practical active-window state comprises about 29 float accumulators/extrema plus counters: roughly **132 bytes**. Two staggered accumulator sets for stride 100 use about **264 bytes**; gravity state adds 12 bytes; a prospective causal six-axis SOS filter adds 96 bytes; feature output is 84 bytes; allow ~48 bytes for transient vectors plus a few bytes for inference. Thus approximately **0.5–0.7 KiB of data RAM** can suffice for an optimized incremental full-feature implementation, excluding stack, DMA/sensor buffers, library workspace and model parameters kept in flash. If weights/tree tables reside in RAM, add the model storage above.',
        f"The fitted tree can omit all unused feature streams and use only {4*len(tree['features_used'])} bytes for its feature output; the full-feature **0.5–0.7 KiB** estimate is a conservative shared-buffer budget for either model, not a measured tree-specific minimum.",
        'A straightforward implementation storing a whole acceleration and gyroscope window uses **4,800 bytes** for the two 200×3 float32 arrays, plus states and feature vector. Storing gravity per sample adds another 2,400 bytes, but is unnecessary in a streaming implementation. Overlapping windows can instead use staggered accumulators for sums, moments, extrema, jerk and trapezoid area. Reset jerk/trapezoid previous-sample state at each window start. Pre/post half summaries also update incrementally.',
        'Variance can use Welford updates for numerical stability at a somewhat higher arithmetic cost than sum/sum-of-squares estimates. Model parameters should be stored in flash. Scaler fusion was checked against sklearn in float64; exported tree evaluation was checked exactly, including sklearn float32 input casting. Quantized coefficients, fixed-point features and embedded numerical drift are unverified.',
        '## Causality and wrist limitations',
        'The gravity EMA itself is causal. The **preserved end-to-end protocol is not**: full-trial zero-phase filtering and finding the full-trial largest peak for fall localization use future information. A realistic MCU trigger needs a rolling pre-event buffer and delayed post-event statistics; the current whole-trial peak rule has no causal equivalent without redefining the detector. Constant coordinate rotations also do not model moving a sensor from the waist to the wrist or time-varying wrist orientation. These costs guide the next engineering step, not deployment readiness.']
    (out/'MCU.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    # Human-readable rules that actually use the selected leaf-score threshold.
    def visit(node,depth):
        indent='  '*depth
        left=tree['children_left'][node]
        if left==-1:
            score=tree['positive_score'][node]
            return [f"{indent}leaf score={score:.10g} => {'FALL' if score>=tree['probability_threshold'] else 'ADL'}"]
        feature=r.FEATURE_NAMES[tree['feature'][node]]
        lines=[f"{indent}if {feature} <= {tree['threshold'][node]:.10g}:"]
        lines+=visit(left,depth+1)
        lines.append(indent+'else:')
        lines+=visit(tree['children_right'][node],depth+1)
        return lines
    (out/'decision_tree_tuned_rules.txt').write_text('Inputs cast to float32. Fixed positive-score threshold: '+str(tree['probability_threshold'])+'\n'+'\n'.join(visit(0,0))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    path=args.out.resolve()
    if not path.is_relative_to((r.p.PROJECT/'outputs').resolve()):
        parser.error('Output must be in project outputs/')
    render(path)
