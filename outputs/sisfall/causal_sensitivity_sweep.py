"""Read-only fixed-model validation operating-point analysis; standard library only.

Uses saved validation scores. Does not load test files or fit any estimator.
"""
import sys
sys.dont_write_bytecode=True
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MODELS=ROOT/'outputs/causal_trained_models_v1'
CAUSAL=ROOT/'outputs/causal_event_study'
OUT=ROOT/'outputs/causal_sensitivity_analysis'
FEATURES=['ga_C2','jerk_abs_mean','ga_parallel_peak']

def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def write(name,rows,fields=None):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def hashes(paths):return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def above(values,t):return len(values)-bisect.bisect_left(values,t)

def table(rows,keys):
    def fmt(v):return 'N/A' if v is None else f'{v:.6g}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+
                     ['| '+' | '.join(fmt(r.get(k)) for k in keys)+' |' for r in rows])

def main():
    if OUT.exists():raise RuntimeError('Refusing to overwrite an existing analysis.')
    OUT.mkdir(parents=True)
    paths=[MODELS/'model_parameters.json',MODELS/'event_predictions.csv',MODELS/'metrics.csv',
           CAUSAL/'selection.json',CAUSAL/'validation_trial_results.csv',
           ROOT/'processed/baseline_v2/split_subjects.json',Path(__file__)]
    before=hashes(paths)
    params=json.loads((MODELS/'model_parameters.json').read_text())['logistic']
    selected=json.loads((CAUSAL/'selection.json').read_text())
    assert params['feature_names']==FEATURES and selected['design']=='pre100_post200'
    subjects=json.loads((ROOT/'processed/baseline_v2/split_subjects.json').read_text())['validation']
    trials=[r for r in read(CAUSAL/'validation_trial_results.csv')
            if r['key']==selected['key'] and r['design']==selected['design']]
    events=[r for r in read(MODELS/'event_predictions.csv') if r['split']=='validation' and r['model']=='logistic']
    by_path={r['path']:[] for r in trials}
    max_error=0.
    for e in events:
        assert e['path'] in by_path and Path(e['path']).stem.split('_')[1] in subjects
        e['score']=float(e['score']);assert math.isfinite(e['score']) and 0<=e['score']<=1
        z=params['intercept']+sum((float(e[n])-mu)/scale*coef for n,mu,scale,coef in
                                 zip(FEATURES,params['mean'],params['scale'],params['coefficients']))
        probability=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        max_error=max(max_error,abs(probability-e['score']))
        assert math.isclose(probability,e['score'],rel_tol=1e-12,abs_tol=1e-12)
        by_path[e['path']].append(e)
    for t in trials:
        assert t['split']=='validation' and Path(t['path']).stem.split('_')[1] in subjects
        t['label']=int(t['label']);t['duration_s']=float(t['duration_s'])
        for e in by_path[t['path']]:
            match=bool(t['label'] and abs(int(e['trigger'])-int(t['proxy_peak']))<=200)
            assert int(e['proxy_target'])==int(match)
        matches=[e['score'] for e in by_path[t['path']] if int(e['proxy_target'])]
        t['max_matched']=max(matches,default=-math.inf)
        t['max_score']=max((e['score'] for e in by_path[t['path']]),default=-math.inf)
        assert int(t['candidate_hit'])==int(bool(matches))
    falls=[r for r in trials if r['label']];adls=[r for r in trials if not r['label']]
    hours=sum(r['duration_s'] for r in adls)/3600
    ceiling=sum(int(r['candidate_hit']) for r in falls)/len(falls)
    all_scores=sorted(e['score'] for e in events)
    thresholds=sorted(set(all_scores))+[math.nextafter(max(all_scores),math.inf)]
    adl_paths={r['path'] for r in adls}
    adl_scores=sorted(e['score'] for e in events if e['path'] in adl_paths)
    fall_max=sorted(r['max_matched'] for r in falls)
    adl_max=sorted(r['max_score'] for r in adls)
    fall_types=sorted({r['activity'] for r in falls});adl_types=sorted({r['activity'] for r in adls})
    by_fall={a:sorted(r['max_matched'] for r in falls if r['activity']==a) for a in fall_types}
    by_adl={a:sorted(e['score'] for t in adls if t['activity']==a for e in by_path[t['path']]) for a in adl_types}
    adl_hours={a:sum(t['duration_s'] for t in adls if t['activity']==a)/3600 for a in adl_types}
    sweep=[]
    for i,threshold in enumerate(thresholds):
        detected=above(fall_max,threshold);alarms=above(all_scores,threshold)
        adl_alarms=above(adl_scores,threshold);specificity=1-above(adl_max,threshold)/len(adls)
        row=dict(threshold_id=i,threshold=threshold,fall_event_sensitivity=detected/len(falls),
                 event_precision=detected/alarms if alarms else None,adl_false_alarms_per_hour=adl_alarms/hours,
                 adl_trial_specificity=specificity,trial_balanced_accuracy=(detected/len(falls)+specificity)/2,
                 detected_fall_trials=detected,missed_fall_trials=len(falls)-detected,
                 emitted_positive_decisions=alarms,adl_false_alarms=adl_alarms,
                 unmatched_or_duplicate_alarms=alarms-detected)
        row.update({f'{a}_sensitivity':above(by_fall[a],threshold)/len(by_fall[a]) for a in fall_types})
        row.update({f'{a}_false_alarms_per_hour':above(by_adl[a],threshold)/adl_hours[a] for a in adl_types})
        assert row['fall_event_sensitivity']<=ceiling
        sweep.append(row)
    reference=next(r for r in sweep if r['threshold']==params['threshold'])
    existing=next(r for r in read(MODELS/'metrics.csv') if r['model']=='logistic' and r['split']=='validation')
    for a,b in [('fall_event_sensitivity','sensitivity'),('event_precision','event_precision'),
                ('adl_false_alarms_per_hour','false_alarms_per_adl_hour'),('trial_balanced_accuracy','trial_balanced_accuracy')]:
        assert math.isclose(reference[a],float(existing[b]),abs_tol=1e-12)
    assert sweep[-1]['emitted_positive_decisions']==0 and sweep[0]['emitted_positive_decisions']==len(events)
    assert all(a['detected_fall_trials']>=b['detected_fall_trials'] and
               a['emitted_positive_decisions']>=b['emitted_positive_decisions'] for a,b in zip(sweep,sweep[1:]))
    chosen=[('current_BA_reference',reference)]
    for budget in (5,10,20):
        eligible=[r for r in sweep if r['adl_false_alarms_per_hour']<=budget]
        best=max(eligible,key=lambda r:(r['detected_fall_trials'],-r['adl_false_alarms_per_hour'],
                                        -r['emitted_positive_decisions'],r['threshold']))
        chosen.append((f'max_sensitivity_FA_le_{budget}',best))
    for target in (.95,.97):
        eligible=[r for r in sweep if r['fall_event_sensitivity']>=target]
        chosen.append((f'lowest_threshold_sensitivity_ge_{target:.2f}',eligible[0] if eligible else None))
    max_sensitivity=max(r['fall_event_sensitivity'] for r in sweep)
    maximum=[r for r in sweep if r['fall_event_sensitivity']==max_sensitivity]
    chosen.append(('lowest_threshold_max_sensitivity',maximum[0]))
    # Supplemental efficient alternatives: preserve literal requested points above.
    efficient=[]
    for target in (.95,.97,max_sensitivity):
        eligible=[r for r in sweep if r['fall_event_sensitivity']>=target]
        if eligible:efficient.append((f'highest_threshold_sensitivity_ge_{target:.6f}',eligible[-1]))
    selected_rows=[];activity=[]
    for name,row in chosen+efficient:
        if row is None:
            selected_rows.append(dict(operating_point=name,status='unattainable',**{k:None for k in reference},
                                      sensitivity_delta_pp=None,false_alarms_per_hour_delta=None,precision_delta_pp=None,
                                      balanced_accuracy_delta_pp=None,detected_falls_delta=None,positive_decisions_delta=None))
            continue
        selected_rows.append(dict(operating_point=name,status='attainable',**row,
                                  sensitivity_delta_pp=100*(row['fall_event_sensitivity']-reference['fall_event_sensitivity']),
                                  false_alarms_per_hour_delta=row['adl_false_alarms_per_hour']-reference['adl_false_alarms_per_hour'],
                                  precision_delta_pp=100*(row['event_precision']-reference['event_precision']) if row['event_precision'] is not None else None,
                                  balanced_accuracy_delta_pp=100*(row['trial_balanced_accuracy']-reference['trial_balanced_accuracy']),
                                  detected_falls_delta=row['detected_fall_trials']-reference['detected_fall_trials'],
                                  positive_decisions_delta=row['emitted_positive_decisions']-reference['emitted_positive_decisions']))
        for a in fall_types+adl_types:
            is_fall=a in by_fall
            activity.append(dict(operating_point=name,threshold=row['threshold'],activity=a,
                                 sensitivity=row[f'{a}_sensitivity'] if is_fall else None,
                                 detected_falls=above(by_fall[a],row['threshold']) if is_fall else None,
                                 fall_trials=len(by_fall[a]) if is_fall else None,
                                 adl_false_alarms_per_hour=row[f'{a}_false_alarms_per_hour'] if not is_fall else None,
                                 adl_false_alarms=above(by_adl[a],row['threshold']) if not is_fall else None,
                                 adl_hours=adl_hours.get(a)))
    cutoff=maximum[0]['threshold'];misses=[];rejected=[]
    for t in falls:
        if t['max_matched']>=cutoff:continue
        reason=('no_matched_trigger' if not int(t['raw_trigger_hit']) else
                'matched_trigger_incomplete_window' if not int(t['candidate_hit']) else
                'matched_complete_candidate_rejected')
        best=max((e for e in by_path[t['path']] if int(e['proxy_target'])),key=lambda e:e['score'],default=None)
        misses.append(dict(path=t['path'],subject=Path(t['path']).stem.split('_')[1],fall_type=t['activity'],
                           reason=reason,proxy_peak_sample=int(t['proxy_peak']),raw_triggers=int(t['raw_triggers']),
                           complete_candidates=int(t['candidates']),best_matched_score=best['score'] if best else None,
                           **{n:float(best[n]) if best else None for n in FEATURES}))
        if reason=='matched_complete_candidate_rejected':
            for e in by_path[t['path']]:
                if int(e['proxy_target']):
                    rejected.append(dict(path=t['path'],subject=Path(t['path']).stem.split('_')[1],fall_type=t['activity'],
                                         trigger=int(e['trigger']),score=e['score'],**{n:float(e[n]) for n in FEATURES}))
    assert len(misses)==maximum[0]['missed_fall_trials']
    # All finite candidate scores are accepted at the minimum boundary.
    assert not rejected and max_sensitivity==ceiling
    write('threshold_sweep.csv',sweep);write('operating_points.csv',selected_rows)
    write('selected_activity_metrics.csv',activity);write('maximum_sensitivity_misses.csv',misses)
    write('classifier_rejected_candidates.csv',rejected,['path','subject','fall_type','trigger','score']+FEATURES)
    dump('provenance.json',dict(input_sha256=before,feature_order=FEATURES,no_test_access=True,no_training=True,
                               threshold_rule='score >= threshold; unique validation scores plus nextafter(max_score,+infinity)',
                               reference_threshold=params['threshold'],tie_policy='Exact sensitivity ties: fewer ADL alarms/hour, fewer total decisions, highest threshold; no nearly-equal tolerance introduced',
                               lowest_threshold='Literal lowest numerical boundary in the requested sweep; not the highest/most selective threshold meeting a target',
                               validation_fall_trials=len(falls),validation_adl_trials=len(adls),adl_hours=hours,
                               validation_candidates=len(events),thresholds=len(sweep),candidate_recall_ceiling=ceiling,
                               maximum_sensitivity_threshold=cutoff,max_saved_score_equation_error=max_error))
    assert hashes(paths)==before
    dump('verification.json',dict(inputs_unchanged=True,reference_metrics_reproduced=True,monotonicity_verified=True,
                                  no_alarm_boundary_verified=True,all_alarm_boundary_verified=True,
                                  candidate_ceiling_verified=True,all_maximum_sensitivity_misses_accounted_for=True,
                                  no_matched_complete_rejections_at_minimum_boundary=True))
    columns=['operating_point','threshold','fall_event_sensitivity','adl_false_alarms_per_hour','event_precision',
             'adl_trial_specificity','trial_balanced_accuracy','detected_fall_trials','missed_fall_trials','emitted_positive_decisions']
    report=['# Fixed causal logistic model: sensitivity-first validation sweep',
            'Only the classifier decision threshold varies. Model weights, three features, causal filter, gravity EMA, selected trigger, refractory period and pre100_post200 window remain fixed. No test files or outcomes were opened. Previous artifacts are unchanged; these operating points are analysis only and do not replace the deployed/reference threshold.',
            f"The exact reference threshold is **{params['threshold']}**, not rounded 0.8851. Swept **{len(sweep)}** boundaries: every unique saved validation positive score plus a no-alarm boundary immediately above the maximum score. Decisions use score >= threshold. Saved scores were checked against the fixed exported logistic equation.",
            '## Candidate ceiling',
            f"There are {len(falls)} validation fall trials and {len(adls)} ADL trials ({hours:.6f} ADL hours). Complete localized candidate recall is **{ceiling:.6%}**, or {round(ceiling*len(falls))}/{len(falls)} falls. This is the end-to-end sensitivity ceiling under the fixed matching/trigger/window convention. **100% end-to-end sensitivity is impossible by changing only this classifier threshold.**",
            'The impact time is the existing raw-magnitude peak proxy, not an annotated onset. A matched candidate has trigger within +/-1 s of that proxy and a complete emitted window. One detected event per fall trial is credited. Precision = detected fall trials / emitted positive decisions; duplicate/unmatched fall-recording alarms count as false positives. ADL trial specificity is the fraction of ADL trials with no alarm. Trial balanced accuracy averages that specificity and fall-event sensitivity.',
            '## Requested operating points',table(selected_rows[:len(chosen)],columns),
            'For the <=5/10/20 false-alarm budgets, sensitivity is maximized first. Exact sensitivity ties minimize ADL alarms/hour, then total positive decisions, then prefer the highest threshold. No unspecified nearly-equal tolerance is used; one fall changes sensitivity by '+f'{100/len(falls):.6f} percentage points.',
            '**Lowest threshold is interpreted literally within the requested boundary sweep.** If 95%, 97%, and maximum sensitivity are attainable, all three lowest-threshold requests select the minimum observed validation score and accept every candidate. Thresholds below that minimum would produce identical validation decisions, but are outside this finite sweep. These literal points do not minimize false alarms.',
            '## More selective supplemental target points',table(selected_rows[len(chosen):],columns),
            'These supplemental rows use the highest threshold meeting each sensitivity target, which minimizes emitted positives among thresholds meeting that target. They are provided to clarify the tradeoff, not to replace the requested literal lowest-threshold results.',
            '## Changes from the current reference',table(selected_rows,['operating_point','sensitivity_delta_pp','false_alarms_per_hour_delta',
                                                                      'precision_delta_pp','balanced_accuracy_delta_pp','detected_falls_delta','positive_decisions_delta']),
            '## Every remaining miss at the literal maximum-sensitivity threshold',table(misses,list(misses[0])),
            f"Miss counts: no matched trigger = {sum(r['reason']=='no_matched_trigger' for r in misses)}; matched trigger but incomplete window = {sum(r['reason']=='matched_trigger_incomplete_window' for r in misses)}; matched complete candidate rejected by classifier = {len(rejected)}.",
            'At the minimum score boundary every complete candidate is accepted, so there are no classifier-rejected matched candidates to list. The header-only classifier_rejected_candidates.csv records this explicitly. Missing matched candidates have no corresponding logistic score or feature vector; these are left blank rather than invented. The highest threshold attaining the same maximum sensitivity necessarily misses the same no-candidate trials.',
            '## Per-activity results and artifacts',
            'threshold_sweep.csv includes every requested aggregate metric, all 15 per-fall-type sensitivities, and all 19 per-ADL false-alarm rates at **every** threshold. operating_points.csv includes all these columns plus reference deltas. selected_activity_metrics.csv provides long-form counts and exposures for each selected point. maximum_sensitivity_misses.csv lists every missed validation fall with subject/type and cause.',
            'Reproduce with `python -B outputs/sisfall/causal_sensitivity_sweep.py` using a fresh OUT directory; overwriting an existing analysis is refused. This script uses only the Python standard library and never loads/fits an estimator. provenance.json hashes the inputs; verification.json records reference parity, monotonicity, endpoint checks, and unchanged-input checks.']
    (OUT/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(table(selected_rows,columns));print(table(misses,['path','reason']))

if __name__=='__main__':main()
