"""Summarize the completed predeclared feature study; no fitting or test access."""
import sys
sys.dont_write_bytecode=True
import causal_feature_expansion as e
import json,csv,math
import numpy as np

def numeric(rows):
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            elif v=='':r[k]=None
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows

def main():
    out=e.OUT
    selection=json.loads((out/'selection.json').read_text())
    targets=numeric(e.m.read(out/'engineering_targets.csv'))
    ops=numeric(e.m.read(out/'operating_points.csv'))
    activities=numeric(e.m.read(out/'activity_metrics.csv'))
    pareto=numeric(e.m.read(out/'pareto_frontiers.csv'))
    params=json.loads((out/'model_parameters.json').read_text())
    models=list(selection['models'])
    gate=selection['gate']
    if not gate['five_vs_frozen']['material']:
        recommendation='The 5-feature step fails the predeclared material-improvement screen. Stop here: the 7-feature model was NOT fitted. Recommend freezing the simpler existing 3-feature model; neither the new fits nor operating thresholds replace it.'
    elif 'expanded7' in models and not gate['seven_vs_five']['material']:
        recommendation='The 5-feature step passes the material-improvement screen, but the 7-feature step does not add enough benefit over 5. Stop expansion and retain the simpler 5-feature candidate for this comparison. The original frozen model remains unchanged; no production model or threshold is replaced.'
    else:
        recommendation='Both predeclared expansion steps pass the practical frontier-improvement screen. The 7-feature candidate is retained as an experimental result; stop at the predeclared maximum of seven features. This is not a deployment promotion or authorization for further feature search.'
    # Quantify delta per requested budget/target relative to identical frozen baseline.
    deltas=[]
    for r in ops:
        base=next(b for b in ops if b['feature_set']=='frozen3' and b['operating_point']==r['operating_point'])
        deltas.append(dict(feature_set=r['feature_set'],operating_point=r['operating_point'],
                           sensitivity_delta_pp=100*(r['sensitivity']-base['sensitivity']),
                           false_alarms_per_hour_delta=r['false_alarms_per_adl_hour']-base['false_alarms_per_adl_hour'],
                           detected_falls_delta=r['detected_falls']-base['detected_falls']))
    e.write('frontier_deltas.csv',deltas)
    fixed=[]
    for name,p in params.items():
        for op,row in p['fixed_C10_points'].items():
            if row:fixed.append(dict(feature_set=name,C=10,operating_point=op,**row))
    e.write('fixed_C10_control.csv',fixed)
    table=e.m.table
    cols=['feature_set','operating_point','threshold','sensitivity','false_alarms_per_adl_hour',
          'event_precision','trial_specificity','trial_balanced_accuracy','detected_falls','missed_falls']
    report=['# Controlled causal accelerometer feature expansion',
            'This is a predeclared, training/validation-only logistic study. Raw data, prior experiments, the three-feature frozen baseline, subject partitions, 200 Hz acquisition, causal 5 Hz Butterworth, raw-driven 0.5 Hz gravity EMA, selected trigger, 1.5 s refractory, pre100_post200 timing, matching tolerance and event accounting remain unchanged. No test sensor trials, candidate features or outcomes were used. Only training/validation manifests and raw trial files were replayed.',
            '## Outcome and stopping decision',recommendation,
            'Neither practical materiality nor validation target attainment is a statistical guarantee or hardware/wrist validation. Repeated reuse of the same development subjects creates selection optimism.',
            '## Predeclared progression and selection',
            table([dict(feature_set=name,features=', '.join(v['features']),C=v['C']) for name,v in selection['models'].items()],['feature_set','features','C']),
            'The exact definitions, timing, units and MCU state/arithmetic estimates are in [FEATURE_DEFINITIONS.md](FEATURE_DEFINITIONS.md). The 5-feature set adds late magnitude variance and late-minus-pre magnitude jerk. The prospective 7-feature set additionally adds perpendicular active fraction and late signed-parallel standard deviation. No subset combinations, phase-length search, activity-cutoff tuning, other model family or neural network was tried.',
            'The original train-fitted C=10 three-feature model is reused verbatim, including its scaler; it is not refitted or replaced. Each expanded model is fitted only on training candidates using a training-only StandardScaler and balanced L2/liblinear logistic regression. C is restricted to 0.01, 0.1, 1 and 10, with seed 473, tol 1e-8, max_iter 5000, and no validation refit. Fixed-C=10 controls are exported to help distinguish feature effects from C selection.',
            'Hyperparameter selection was fixed before replay: prioritize achieving >98% sensitivity at <=1 FA/h, then at <=5; next maximize detected falls lexicographically at budgets 1,5,10,20, then minimize FA/h needed for >98%, >=95%, >=97%, then prefer smaller C. This prioritizes the stated low-alarm targets rather than maximizing BA. All reported frontier thresholds come from unique validation positive scores plus the no-alarm boundary. A BA-selected threshold is included as a descriptive reference, not used to select expanded-model C.',
            'Material-improvement gate: >=1 percentage point sensitivity improvement at any requested budget 1/5/10/20 OR >=20% AND >=1 FA/h reduction in the rate required for >=95%, >=97%, or >98%; additionally require no >=1-point sensitivity regression at budgets 1 or 5. The 7-feature fit occurs only if the 5-feature step passes against frozen3. No additional feature search is allowed after this bounded progression. Gate results: '+json.dumps(gate),
            '## Engineering targets and remaining gap',
            '**Primary:** fall-event sensitivity strictly >98% and ADL false alarms/hour <=1. **Secondary:** strictly >98% and <=5. These are engineering targets, not guarantees.',
            table(targets,['feature_set','primary_achieved','secondary_achieved','max_sensitivity_FA_le_1','max_sensitivity_FA_le_5',
                           'minimum_FA_to_exceed98','FA_gap_to_primary','FA_gap_to_secondary',
                           'extra_detected_falls_needed_at_FA1','extra_detected_falls_needed_at_FA5']),
            'Validation has 375 fall trials. Strictly >98% requires at least **368 detected falls (98.1333%)**. Candidate recall is **370/375 = 98.6667%**, leaving room for at most **two classifier misses among matched falls** at a successful operating point. Five misses are unavoidable no-matched-trigger failures. The trigger ceiling is above 98%, so it does not by itself make the target impossible; additional missed matched candidates at a false-alarm budget are classifier/operating-point errors. 100% is impossible under this trigger. Each fitted model can reach the ceiling when accepting every candidate; the question is the false-alarm cost.',
            table(targets,['feature_set','unavoidable_trigger_misses','additional_classifier_misses_at_FA1','additional_classifier_misses_at_FA5']),
            '## Sensitivity–false-alarm operating points',
            '![Saved validation Pareto frontiers at low and high false-alarm budgets](frontier_comparison.png)',
            table([r for r in ops if r['operating_point']!='BA_reference'],cols),
            'For FA budgets, select maximum fall detections and then minimum FA/h, fewer total positive decisions, and higher threshold. For sensitivity targets, select minimum FA/h meeting the target; ties favor more detected falls, fewer total decisions and higher threshold. A sensitivity target may lie between discrete detection counts; >=95% means at least 357/375, >=97% at least 364/375, and >98% at least 368/375.',
            'The closest budget-feasible Pareto points are the FA_le_1 and FA_le_5 rows. The closest sensitivity-feasible point is sens_gt98. These quantify the gap in each constraint without imposing an arbitrary combined distance measure. All nondominated validation points are exported in pareto_frontiers.csv, not just the operating points shown here.',
            table(deltas,['feature_set','operating_point','sensitivity_delta_pp','false_alarms_per_hour_delta','detected_falls_delta']),
            '## Balanced-accuracy diagnostic',table([r for r in ops if r['operating_point']=='BA_reference'],cols),
            'BA is the average of fall-trial sensitivity and ADL-trial specificity (fraction of ADL recordings with no positive decision). It is not event specificity. Event precision credits at most one matched detection per fall trial and divides by all emitted positive decisions, including duplicate and unmatched fall-recording alarms. ADL FA/h uses the full 3.0679416666666666 recorded ADL hours. No additional alarm-merging policy is applied.',
            '## Per-activity failure modes',
            'activity_metrics.csv and trial_results.csv contain all activity/trial outcomes for every listed operating point. At <=5 FA/h, the predeclared difficult activities are:',
            table([r for r in activities if r['operating_point']=='FA_le_5' and r['activity'] in ('D03','D04','D06','D18','D19','F13','F14','F15')],
                  ['model','activity','candidate_recall','sensitivity','false_alarms_per_hour','no_matched_candidate','matched_candidates_rejected']),
            'The per-activity tables separately identify no-matched-candidate failures and falls whose complete matched candidates were all rejected. Activity-specific rates divide by short, unequal exposures and must not be interpreted as expected daily-life alarms/hour. The existing five no-matched-trigger trials remain unchanged; exact trial causes are preserved by the frozen candidate protocol.',
            '## Fixed-C control',
            table([r for r in fixed if r['operating_point'] in ('FA_le_1','FA_le_5','FA_le_10','FA_le_20','sens_ge95','sens_ge97','sens_gt98')],
                  ['feature_set','C','operating_point','sensitivity','false_alarms_per_adl_hour']),
            '## Reproducibility, integrity and limits',
            'experiment_plan.json was written before raw replay and fitting, including all four prospective definitions, gates, targets and C grid. Causal replay checks original manifest hashes; exact candidate times and the original three feature values are verified against the saved causal experiment for every candidate. All four prospective features are cached in one pass, but a skipped 7-feature model is never fitted. Computation at the saved decision time uses only arrived ring samples.',
            'Frozen-model validation scores must match exactly. Source hashes are checked after fitting. feature_tests.json covers independent equations, segment-boundary exclusions, constant signals, rotations and prefix causality; extraction_verification.json records real-stream parity. Model parameters, scalers, scores, sweeps and selected points are saved separately. No deployment threshold or previous artifact is overwritten.',
            'The raw-magnitude peak remains an evaluation-only impact proxy; positive candidate training labels use a +/-1 s trigger match and other candidates are negative. The pipeline is causal but the supervision is weak, and “late settling” is measured relative to a trigger rather than annotated impact. This is waist-mounted, simulated-fall data, not wrist validation. Float64 reference results and algebraic MCU cost estimates do not establish float32 firmware equivalence or measured hardware costs.',
            'Run order in a fresh study output directory: causal_feature_expansion.py, test_causal_expansion.py, report_causal_expansion.py, verify_causal_expansion.py. Use Python -B to preserve existing bytecode/artifacts. No test evaluation is part of this study.']
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(recommendation)

if __name__=='__main__':main()
