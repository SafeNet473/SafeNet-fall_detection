"""Report the completed six-set diversity study; no fitting or test use."""
import sys,os
sys.dont_write_bytecode=True
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'outputs/causal_feature_diversity_v1'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import causal_feature_diversity as d
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name):
    rows=d.e.m.read(OUT/name)
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            elif v=='':r[k]=None
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows

def main():
    selection=json.loads((OUT/'selection.json').read_text());allmat=json.loads((OUT/'all_candidate_materiality.json').read_text())
    ops=read('operating_points.csv');targets=read('engineering_targets.csv');activity=read('activity_metrics.csv')
    extraction=json.loads((OUT/'extraction_verification.json').read_text());params=json.loads((OUT/'model_parameters.json').read_text())
    lookup={(r['feature_set'],r['operating_point']):r for r in ops}
    selectedmat={r['feature_set']:r for r in selection['materiality']}
    single_names=['plus_gaC8','plus_settling','plus_energy_ratio','plus_orientation_change']
    low_material=[n for n in single_names if selectedmat[n]['material'] and max(selectedmat[n]['sensitivity_gains'].values())>=.01]
    qualifying=[n for n in single_names+['all7'] if selectedmat[n]['material']]
    best=max(single_names,key=lambda n:d.capacity.priority({op:lookup[(n,op)] for op in ['FA_le_1','FA_le_5','FA_le_10','FA_le_20','sens_ge95','sens_ge97','sens_gt98']}))
    answers=[
        '1. **Does any single feature materially improve the low-FA frontier?** '+('Yes: '+', '.join(low_material)+'.' if low_material else 'No selected single-feature model achieves a qualifying >=1-point budget sensitivity gain. Rate-cost improvements, if present, are reported separately.'),
        '2. **Does all7 materially improve the frontier?** '+('Yes under the predefined rule.' if selectedmat['all7']['material'] else 'No under the predefined rule.'),
        '3. **Which adds the most complementary information?** '+best+' has the best selected single-feature low-budget rank (1,5,10,20 priority). This is an empirical ranking, not evidence that it adds material information unless it passes the rule. Fixed-C10 comparisons below help separate regularization from feature effects.',
        '4. **Are gains confined to high-FA regions?** '+('No: qualifying single-feature budget sensitivity gains are present; inspect their exact budgets below.' if low_material else 'There is no qualifying selected single-feature budget sensitivity gain. Any material single-feature result must come from reducing FA cost at a high-sensitivity target; inspect the absolute rates before interpreting it as a low-alarm improvement.'),
        '5. **Enough evidence for an MCU v2 feature set?** '+('The material candidates '+', '.join(qualifying)+' warrant consideration as experimental MCU v2 candidates, with costs and domain limits below. No automatic promotion or independent validation is implied.' if qualifying else 'No. Recommend retaining frozen MCU v1 and stopping feature expansion after these predefined sets.')]
    deltas=[];fixed=[];activity_delta=[]
    for r in ops:
        b=lookup[('frozen_baseline3',r['operating_point'])]
        control=lookup[('baseline3',r['operating_point'])]
        deltas.append(dict(feature_set=r['feature_set'],operating_point=r['operating_point'],
                           sensitivity_delta_pp=100*(r['sensitivity']-b['sensitivity']),FA_delta=r['false_alarms_per_adl_hour']-b['false_alarms_per_adl_hour'],
                           detected_delta=r['detected_falls']-b['detected_falls'],sensitivity_delta_vs_refit_control_pp=100*(r['sensitivity']-control['sensitivity'])))
    for name,p in params.items():
        for op,r in p['fixed_C10_points'].items():
            if r:fixed.append(dict(feature_set=name,C=10,operating_point=op,**r))
    focus=('D06','D18','D19','F13','F14','F15')
    for r in activity:
        if r['activity'] not in focus:continue
        b=next(x for x in activity if x['model']=='frozen_baseline3' and x['activity']==r['activity'] and x['operating_point']==r['operating_point'])
        activity_delta.append(dict(feature_set=r['model'],operating_point=r['operating_point'],activity=r['activity'],
                                   sensitivity=r['sensitivity'],sensitivity_delta_pp=None if r['sensitivity'] is None else 100*(r['sensitivity']-b['sensitivity']),
                                   FA_per_hour=r['false_alarms_per_hour'],FA_per_hour_delta=None if r['false_alarms_per_hour'] is None else r['false_alarms_per_hour']-b['false_alarms_per_hour'],
                                   no_matched_candidate=r['no_matched_candidate'],matched_candidates_rejected=r['matched_candidates_rejected']))
    d.write('frontier_deltas.csv',deltas);d.write('fixed_C10_operating_points.csv',fixed);d.write('focus_activity_changes.csv',activity_delta)
    matrows=[]
    for r in allmat:
        matrows.append(dict(candidate=r['candidate'],material=r['material'],eligible=r['eligible'],
                            both_low_regress=r['both_low_budgets_regress_ge1pp'],
                            **{f'gain_pp_FA{k}':100*v for k,v in r['sensitivity_gains'].items()},
                            **{f'FA_reduction_pct_{k}':100*v for k,v in r['FA_reduction_fractions'].items()}))
    d.write('all_candidate_materiality.csv',matrows)
    fronts=read('pareto_frontiers.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=['frozen_baseline3']+single_names+['all7']
    for ax,limit in zip(axes,(25,200)):
        for name in names:
            r=[x for x in fronts if x['feature_set']==name]
            ax.step([x['false_alarms_per_adl_hour'] for x in r],[100*x['sensitivity'] for x in r],where='post',label=name,lw=2 if name in ('frozen_baseline3','all7') else 1.4)
        ax.axhline(98,color='gray',ls='--',lw=.8);ax.axhline(100*370/375,color='gray',ls=':',lw=.8)
        for v in (1,5):ax.axvline(v,color='gray',ls=':',lw=.5)
        ax.set(xlim=(0,limit),ylim=(75,100),xlabel='ADL false alarms/hour',ylabel='Fall-event sensitivity (%)')
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('Feature diversity: validation-only Pareto frontiers; fixed causal pipeline')
    fig.savefig(OUT/'frontier_comparison.png',dpi=160);fig.savefig(OUT/'frontier_comparison.svg');plt.close(fig)
    table=d.e.m.table
    summary=[]
    for n in selection['models']:
        summary.append(dict(feature_set=n,C=selection['models'][n]['C'],
                            **{f'sens_FA{b}':lookup[n,f'FA_le_{b}']['sensitivity'] for b in (1,5,10,20)},
                            **{f'FA_{s}':lookup[n,s]['false_alarms_per_adl_hour'] for s in ('sens_ge95','sens_ge97','sens_gt98')}))
    d.write('comparison_summary.csv',summary)
    report=['# Controlled physical-feature diversity study for possible MCU v2',
            '## Explicit answers',*answers,
            '## Scope and definitions',
            'Exactly six predefined feature sets and four C values per set were fitted:24 balanced L2 logistic models. No other subset, model family, epsilon, sample interval or C value was searched. MCU v1 and all previous artifacts remain unchanged. Only training/validation raw trials and saved causal candidates are used; no test data or outcomes are used.',
            'The pipeline remains 200 Hz ADXL345 XYZ, causal fourth-order 5 Hz Butterworth, raw-driven 0.5 Hz gravity EMA, the selected magnitude>=1.1g OR magnitude-jerk>=5g/s rising-edge trigger,300-sample refractory, and pre100_post200 window. Its300 samples are [k-100,k+200), available at k+200. No new feature delays the decision.',
            'See [FEATURE_DEFINITIONS.md](FEATURE_DEFINITIONS.md) for equations, units, physical meaning, causality, orientation robustness and MCU costs. ga_C8 retains the existing project sample-variance convention (ddof=1; denominator299). post_mag_variance uses population variance over the late100 samples. The energy ratio uses fixed epsilon1e-6 g². Gravity angle normalizes the mean raw-driven gravity vectors, clamps the dot product, and returns0 radians if either mean norm<1e-8g. Fallback count: '+str(extraction['invalid_gravity_mean_count'])+'.',
            table([dict(feature_set=k,features=', '.join(v['features']),C=v['C']) for k,v in selection['models'].items()],['feature_set','features','C']),
            'frozen_baseline3 is the untouched original C10 reference. baseline3 is a newly fitted experimental C-grid control on the same three features, as requested. Its C10 fit must reproduce the frozen scaler, coefficients and validation scores exactly. All comparisons and materiality judgments use the frozen model, while the refit control exposes regularization-only changes. It is not an additional sensor-feature set.',
            '## Predeclared selection and materiality',
            'Every set has its own training-only StandardScaler and balanced L2/liblinear logistic fit. C grid:0.01,0.1,1,10; max_iter5000, tol1e-8, seed473. No validation refit. Selection prioritizes >98% at<=1FA/h, then >98% at<=5; then detected falls at budgets1,5,10,20 lexicographically; then lower FA cost at>98%,>=95%,>=97%; then smaller C. Each selected model is used across its whole frontier. Feature counts and performance are not traded through an unstated score.',
            'Materiality uses >=1 percentage point gain at one of budgets1/5/10/20 OR >=20% FA reduction at one of targets>=95%,>=97%,>98%, while disqualifying a candidate only if >=1-point regression occurs at BOTH budgets1 AND5. This follows the existing engineering rule (the pasted =1/=20 shorthand is treated as >=). It is a practical gate, not a statistical significance claim.',
            table(matrows,list(matrows[0])),
            '## Frontier comparison',
            '![Saved validation frontiers](frontier_comparison.png)',
            'Plot axes are cropped for detail; all exact values and full nondominated frontiers are in CSVs. Sensitivity columns below are fractions, FA columns are alarms/hour.',
            table(summary,list(summary[0])),
            table(deltas,['feature_set','operating_point','sensitivity_delta_pp','FA_delta','detected_delta','sensitivity_delta_vs_refit_control_pp']),
            '## Operating points and engineering targets',
            table(targets,['feature_set','primary_achieved','secondary_achieved','sensitivity_FA1','sensitivity_FA5','minimum_FA_to_exceed98',
                           'additional_detections_needed_FA1','additional_detections_needed_FA5']),
            'Primary requires sensitivity strictly>98% and ADL FA/h<=1; secondary requires strictly>98% and<=5. With375 fall trials, >98% requires368 detections. The candidate ceiling is370/375=98.6667%, leaving at most two classifier misses. Five no-matched-trigger misses are immutable here. The ceiling prevents100% but is above98%; therefore failure at an alarm budget cannot be attributed solely to the trigger. The closest constraint-wise Pareto points are FA_le_1/FA_le_5 and sens_gt98; no arbitrary joint-distance metric is introduced.',
            table(targets,['feature_set','trigger_misses','classifier_misses_FA1','classifier_misses_FA5']),
            table(ops,['feature_set','operating_point','threshold','sensitivity','false_alarms_per_adl_hour','event_precision',
                       'trial_specificity','trial_balanced_accuracy','detected_falls','missed_falls','total_alarms']),
            'Every unique validation positive score plus a no-alarm boundary is swept. Decisions use score>=threshold. Budget points maximize detections then minimize FA/h and positive decisions. Sensitivity targets minimize FA/h subject to the target, breaking ties by more detections/fewer decisions/higher threshold. At a no-alarm boundary precision is undefined. BA_reference is diagnostic, not a replacement deployment threshold.',
            '## Difficult activities: matched budget comparison',
            'The following directly compares each selected set with frozen MCU v1 at<=5FA/hour. focus_activity_changes.csv covers all requested points; activity_metrics.csv covers every activity. These conditional operating-point comparisons should not be interpreted as causal physical explanations of individual mistakes.',
            table([r for r in activity_delta if r['operating_point']=='FA_le_5'],list(activity_delta[0])),
            'Fall-type sensitivities use25 validation trials per type, so one recovered fall changes a type-specific result by4 percentage points. ADL rates use short, unequal activity exposures. D06/D18/D19 represent quick stairs, stumble and jump; F13/F14/F15 are forward/backward/lateral seated falls. Missing candidates and matched-but-rejected candidates are reported separately.',
            '## Fixed-C10 control',
            table([r for r in fixed if r['operating_point'] in ('FA_le_1','FA_le_5','FA_le_10','FA_le_20','sens_ge95','sens_ge97','sens_gt98')],
                  ['feature_set','C','operating_point','sensitivity','false_alarms_per_adl_hour']),
            'A gain after selecting a different C is a feature-plus-regularization result; these same-C controls help avoid attributing it entirely to additional physical information. No alternative C selection was performed after examining this table.',
            '## Verification, reproducibility and limitations',
            'experiment_plan.json was written before replay/fitting. Raw train/validation source hashes are checked. The existing three feature values and complete candidate times must match the prior experiment exactly, with ring reads restricted to arrived samples. The original baseline predictions and the experimental C10 baseline reproduction are checked exactly. Frozen input hashes, including MCU v1 files, are verified after completion. No new sensor channel is read for model inputs.',
            'The four unit tests cover independent equations and normalization, degenerate means and interval exclusions, constant coordinate rotations, and causal prefix/ring behavior. Saved per-C joblib files, score-boundary sweeps, selected parameters, trial/activity tables and all-candidate materiality records expose the bounded search. Training feature/scaler construction and fitting never use validation labels; validation is used only for the declared selection/evaluation.',
            'Validation has been repeatedly reused in prior studies and is development evidence, not an independent generalization estimate. Candidate labels use the raw-magnitude impact proxy: matched fall candidates are positive and unmatched candidates are negative. Event precision credits at most one detection per fall trial and counts duplicate/unmatched alarms against it; ADL rates divide by3.0679416666666666 recorded hours. Trial BA averages sensitivity and the fraction of ADL trials without an alarm, not an event true-negative rate.',
            'The data are waist-mounted simulated falls, not real wrist-domain validation. Raw gravity may track dynamic acceleration, and a posture-change angle need not equal a true body orientation change. Population/sample moment algebra and constant-rotation invariance do not certify float32 firmware parity. MCU arithmetic/storage estimates exclude shared preprocessing and are not hardware benchmarks. No model or feature set is promoted automatically. Stop after these six sets; any future work requires a separately authorized experiment.',
            'Reproduce in a new output directory using Python -B: causal_feature_diversity.py; test_feature_diversity.py; report_feature_diversity.py; verify_feature_diversity.py. Existing output directories are never overwritten.']
    (OUT/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(answers));print(table(summary,list(summary[0])))

if __name__=='__main__':main()
