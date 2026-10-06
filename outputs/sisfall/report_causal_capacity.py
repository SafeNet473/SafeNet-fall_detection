"""Report existing bounded-capacity results without fitting or test access."""
import sys,os
sys.dont_write_bytecode=True
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'outputs/causal_model_capacity'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import causal_model_capacity as c
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name):
    rows=c.e.m.read(OUT/name)
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            elif v=='':r[k]=None
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows

def main():
    selection=json.loads((OUT/'selection.json').read_text())
    ops=read('operating_points.csv');costs=read('complexity.csv');rank=read('ranking.csv')
    targets=read('engineering_targets.csv');allops=read('candidate_operating_points.csv')
    baseline={r['operating_point']:r for r in allops if r['candidate']=='linear_frozen'}
    audit=[];delta=[]
    for name in dict.fromkeys(r['candidate'] for r in allops):
        subset=[r for r in allops if r['candidate']==name]
        pt={r['operating_point']:r for r in subset}
        item=c.material(baseline,pt)
        audit.append(dict(candidate=name,selected=subset[0]['selected'],eligible=subset[0]['eligible'],
                          material=item['material'],both_low_budgets_regress_ge1pp=item['both_low_budgets_regress_ge1pp'],
                          **{f'gain_pp_FA{b}':100*item['sensitivity_gains'][str(b)] for b in (1,5,10,20)},
                          **{f'FA_reduction_pct_{n}':100*item['FA_reduction_fractions'][n] for n in ('sens_ge95','sens_ge97','sens_gt98')}))
    for r in ops:
        b=baseline[r['operating_point']]
        delta.append(dict(candidate=r['candidate'],operating_point=r['operating_point'],
                          sensitivity_delta_pp=100*(r['sensitivity']-b['sensitivity']),
                          detected_falls_delta=r['detected_falls']-b['detected_falls'],
                          FA_per_hour_delta=r['false_alarms_per_adl_hour']-b['false_alarms_per_adl_hour']))
    c.write('all_candidate_materiality.csv',audit);c.write('baseline_deltas.csv',delta)
    qualifying=[r['candidate'] for r in audit if r['eligible'] and r['material']]
    if not qualifying:
        recommendation='None of the 14 fitted candidates satisfies the predeclared material-improvement rule. Recommend keeping the frozen linear MCU v1 baseline. No tested candidate currently warrants MCU v2 promotion, and no further architectures or feature search were run.'
    else:
        recommendation='Material candidates under the predeclared rule: '+', '.join(qualifying)+'. These remain development candidates only; inspect low-budget regressions and complexity before considering MCU v2. No deployment promotion occurs.'
    # Quantized score ties can make an RF budget infeasible except at no alarms.
    diagnostics=[]
    for family,name in selection['selected'].items():
        rows=read(name+'_validation_sweep.csv')
        positive=[r for r in rows if r['total_alarms']>0]
        highest=max(positive,key=lambda r:r['threshold'])
        diagnostics.append(dict(candidate=name,highest_nonempty_threshold=highest['threshold'],
                                 positive_decisions=highest['total_alarms'],ADL_FA_per_hour=highest['false_alarms_per_adl_hour'],
                                 detected_falls=highest['detected_falls']))
    c.write('top_score_tie_diagnostics.csv',diagnostics)
    fronts=read('pareto_frontiers.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    colors={'linear':'#2367a4','quadratic':'#b75b19','mlp':'#37864c','forest':'#92559e'}
    for ax,limit,title in zip(axes,(25,220),('Low false-alarm budgets','High-sensitivity false-alarm cost')):
        for family,name in selection['selected'].items():
            r=[x for x in fronts if x['family']==family]
            ax.step([x['false_alarms_per_adl_hour'] for x in r],[100*x['sensitivity'] for x in r],where='post',label=family,color=colors[family],lw=1.8)
        for x in (1,5):ax.axvline(x,color='#bbbbbb',ls=':',lw=.8)
        ax.axhline(98,color='#555555',ls='--',lw=.8)
        ax.axhline(100*370/375,color='#555555',ls=':',lw=.8)
        ax.set(xlim=(0,limit),ylim=(80,100),xlabel='ADL false alarms/hour',ylabel='Fall-event sensitivity (%)',title=title)
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower right');fig.suptitle('Validation frontier: same three physical features and causal candidates')
    fig.savefig(OUT/'frontier_comparison.png',dpi=160);fig.savefig(OUT/'frontier_comparison.svg');plt.close(fig)
    table=c.e.m.table
    activities=read('activity_metrics.csv')
    report=['# Controlled causal model-capacity study',
            '## Recommendation',recommendation,
            'This bounded study provides no material evidence that replacing the linear boundary solves the current sensitivity/false-alarm limitation. It does not prove that the three features contain no additional information or that all nonlinear models would fail: only the predeclared small models, one seed, and a narrow regularization grid were tested on repeatedly reused development subjects.',
            '## Fixed system and bounded search',
            'MCU v1, its weights and all prior artifacts are unchanged. The only physical inputs are ga_C2, jerk_abs_mean, ga_parallel_peak, in that exact order. Subject partitions, 200 Hz sampling, causal fourth-order 5 Hz Butterworth, raw-driven 0.5 Hz gravity EMA, magnitude/jerk trigger, 300-sample refractory, pre100_post200 window, matching and event accounting are unchanged. Saved training/validation feature rows are reused; no test sensor data, test candidates or test outcomes are used.',
            '- Linear: reuse the existing frozen train-fitted C=10 logistic model and scaler without refitting.\n'
            '- Quadratic logistic: exactly `[x1,x2,x3,x1²,x2²,x3²,x1*x2,x1*x3,x2*x3]` formed from unstandardized physical features, then training-only StandardScaler; L2/liblinear, C=0.01/0.1/1/10, balanced classes, max_iter=5000, tol=1e-8. Four fits. No new sensor-derived feature is introduced.\n'
            '- Forest: RandomForestClassifier, 5/10/20 trees × depth 2/3, min_samples_leaf=25, max_features=1.0, bootstrap=True, class_weight=balanced, n_jobs=1. Six fits. No boosted or larger forest is tried.\n'
            '- Tiny MLP: 3→4→1 or 3→8→1 only; ReLU hidden layer, logistic output; training-only StandardScaler; L2 alpha=0.001/0.1; LBFGS, max_iter=2000, max_fun=50000, tol=1e-7; no early stopping or validation fitting. Training-only balanced sample weights address candidate class imbalance. Four fits. The optional second hidden layer was not used.\n'
            'All new fits use seed 473. All 14 fits converged within their predeclared bounds. No extra seeds, architectures, retries with enlarged bounds, or features were searched.',
            'Within each family, validation selection prioritizes the primary >98%/<=1 FA/h target, then secondary >98%/<=5, then sensitivity lexicographically at budgets 1/5/10/20, then lower FA at >98%, >=95%, >=97%, then smaller estimated flash and stronger regularization. Every candidate has a saved complete threshold sweep. The selected family representative is one model used across its whole frontier, not a different hyperparameter at each point.',
            '## Stopping rule',
            'A candidate qualifies only with >=1 percentage point higher sensitivity at any budget 1/5/10/20, OR >=20% lower FA/h at a target >=95%, >=97%, or >98%, while avoiding >=1-point regression at BOTH budgets1 AND5. The AND condition follows the request literally; a loss at only one low budget is still shown. There is no extra absolute-rate reduction requirement. All families are evaluated regardless of quadratic results, then the search stops.',
            table(audit,['candidate','selected','material','gain_pp_FA1','gain_pp_FA5','gain_pp_FA10','gain_pp_FA20',
                         'FA_reduction_pct_sens_ge95','FA_reduction_pct_sens_ge97','FA_reduction_pct_sens_gt98']),
            '## Ranked comparison',
            'Ranking is lexicographic: low-FA sensitivities at 1,5,10,20; high-sensitivity FA cost at >98%,95%,97%; then estimated model flash. A first-place ranking does not mean the gain passes materiality.',
            table(rank,['rank','candidate','sensitivity_FA1','sensitivity_FA5','sensitivity_FA10','sensitivity_FA20',
                        'FA_sens_ge95','FA_sens_ge97','FA_sens_gt98','estimated_flash_bytes','multiplications','comparisons']),
            'Sensitivity columns are fractions; rate columns are alarms/hour. Quadratic C=1 improves <=1 FA/h by three falls, +0.8 percentage points, not the required >=1 point (at least four additional falls here). Its approximately 2.56% reduction in FA cost at >=95% is also below 20%, and its >=97% and >98% costs worsen. The selected MLP does not improve the priority budgets. The forest has a coarse/tied highest score block; inspect top_score_tie_diagnostics.csv. A no-alarm threshold is the best <=1 and <=5 point for the selected forest, not a missing result or arbitrary tie-breaking.',
            '## Complete selected operating-point metrics',
            '![Saved selected-family validation Pareto frontiers](frontier_comparison.png)',
            'Plot y-axes are cropped to 80–100% for detail; forest zero-sensitivity budget points are reported in the tables, not visible in this crop.',
            table(ops,['candidate','operating_point','threshold','sensitivity','false_alarms_per_adl_hour','event_precision',
                       'trial_specificity','trial_balanced_accuracy','detected_falls','missed_falls','total_alarms']),
            'All unique validation score boundaries plus a no-alarm boundary are swept with score>=threshold. Budget points maximize sensitivity, then minimize FA/h, total positive decisions, then prefer a higher threshold. Target points minimize FA/h subject to sensitivity, then prefer more detections and fewer decisions. No-alarm precision is undefined (N/A). BA_reference is a diagnostic point on each selected model, not a changed MCU v1 threshold.',
            table(delta,['candidate','operating_point','sensitivity_delta_pp','detected_falls_delta','FA_per_hour_delta']),
            '## Engineering targets and remaining gaps',
            table(targets,['candidate','primary_achieved','secondary_achieved','sensitivity_at_FA1','sensitivity_at_FA5',
                           'minimum_FA_to_exceed98','FA_gap_primary','FA_gap_secondary',
                           'detections_short_of_target_at_FA1','detections_short_of_target_at_FA5']),
            'Strict >98% requires 368 of 375 validation falls. Trigger candidate recall is 370/375=98.6667%, so successful classification can lose at most two matched falls. Five no-matched-trigger misses are unavoidable; the fixed trigger therefore prevents 100%, but does not itself prevent >98%. Additional misses under an alarm budget are classifier/operating-point errors. The FA_le_1/FA_le_5 and sens_gt98 rows are the Pareto points closest along the respective constraints; no arbitrary joint distance metric is introduced.',
            table(targets,['candidate','trigger_misses','classifier_misses_at_FA1','classifier_misses_at_FA5']),
            '## Model-specific MCU complexity',
            table(costs,['candidate','model_numeric_parameters','scaler_statistics','float32_numeric_parameter_bytes','estimated_flash_bytes',
                         'multiplications','additions','comparisons','nonlinear','extra_ram_bytes']),
            'Counts exclude all shared causal preprocessing, sensor-derived feature extraction, and the existing event buffer. They are scalar estimates, not measured firmware cycles, memory high-water marks or power. Model_numeric_parameters counts weights+biases for logistic/MLP. For forests it counts one split threshold per internal node and one positive-class score per leaf (one numeric scalar per node); feature IDs and topology are separate structural parameters included in estimated flash. Scaler_statistics counts additional learned training means/scales, excluded from the model-weight count.',
            'Flash estimates use float32 constants, scaler fusion, and one separate operating threshold. Linear: 4 model constants + threshold =20 bytes; quadratic:10+threshold=44 bytes. Without fusion, scaler statistics add 24 or72 bytes. Quadratic input expansion costs six multiplications in addition to nine dot-product multiplies; extra RAM budgets nine terms plus accumulator (40 bytes). An implementation could stream terms for less RAM, but no such firmware is validated.',
            'For the selected 3→8→1 MLP, weights/biases total 3*8+8+8+1=41, plus an output threshold:168 bytes. The input scaler can be folded into the first layer; the output sigmoid can be replaced by comparison to logit(threshold). It needs32 multiplies,32 adds,8 ReLU comparisons plus the final comparison, and approximately36 bytes of hidden/output working state. The 3→4→1 alternatives have21 model parameters,88 bytes including threshold,16 multiplies/adds,4 ReLUs plus final comparison, and20 bytes activation state. Unfused input scaling adds24 bytes and three subtract/divide operations. Score output, rather than binary inference, adds sigmoid exp/division.',
            'Forest flash assumes an aligned12-byte node storing float32 threshold/leaf score and routing/feature metadata, four bytes per root index, and8 bytes for mean reciprocal and classifier threshold. Worst comparisons sum actual tree depths plus one final threshold comparison. Leaf-score averaging requires T-1 additions and one multiplication by1/T. No nonlinear math library is needed, but branches and routing storage dominate. The floating constants alone are smaller than the full tree table. extra_ram_bytes budgets accumulator plus current-node index, excluding call stack.',
            'These fusion/float32 estimates are algebraic deployment possibilities, not claims of bitwise equivalence. The saved models and exported parameters remain float64 desktop references; no deployment threshold or firmware is created.',
            '## Per-activity failure modes',
            table([r for r in activities if r['operating_point']=='FA_le_5' and r['activity'] in ('D03','D04','D06','D18','D19','F13','F14','F15')],
                  ['model','activity','candidate_recall','sensitivity','false_alarms_per_hour','no_matched_candidate','matched_candidates_rejected']),
            'activity_metrics.csv and trial_results.csv contain every activity and trial at every listed selected operating point. Event sensitivity credits one matched positive candidate per fall trial. All additional positive decisions, including duplicates/unmatched fall-recording alarms, count against event precision; ADL FA/h uses the full3.0679416666666666 ADL hours. ADL trial specificity counts trials without alarms, and trial BA averages it with sensitivity. There is no event true-negative denominator. Model differences cannot repair a missing candidate.',
            '## Evidence and limitations',
            'experiment_plan.json was written before fitting and hashes MCU v1 plus all authoritative inputs and relevant code. verification.json records baseline score identity and unchanged hashes. candidate_operating_points.csv, all_candidate_materiality.csv, per-candidate sweeps and fit_status.csv expose the entire bounded search. model_parameters.json and per-fit joblib files preserve fitted models separately. This run reads only saved train/validation candidates and performs no raw replay or test evaluation.',
            'Training candidate labels remain weak proxy labels: fall trial AND trigger within +/-1 s of the raw-magnitude peak, otherwise negative. Validation was already reused for preprocessing/trigger/window/model development. Small differences may be sampling or selection noise; no significance claim or independent generalization result is made. Waist-mounted simulated falls do not validate a wrist device. No MLP architecture/seed expansion or new feature search follows these results.']
    (OUT/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(recommendation)
    print('Qualifying candidates:',qualifying)

if __name__=='__main__':main()
