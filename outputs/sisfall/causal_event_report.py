"""Read completed causal results and report without selection or model fitting."""
import json
import platform
from pathlib import Path
import causal_events as c

def cast(rows):
    integers={'trial_index','label','proxy_peak','raw_triggers','raw_trigger_hit','candidates','candidate_hit',
              'alarms','detected','unmatched_or_duplicate_alarms','prediction','valid'}
    for row in rows:
        for key in integers & row.keys(): row[key]=int(row[key])
        for key in {'duration_s','latency_s'} & row.keys(): row[key]=float(row[key]) if row[key] else None
    return rows

def table(rows,columns):
    def fmt(v):
        if v is None or v=='': return '—'
        if isinstance(v,float): return f'{v:.4f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                     ['| '+' | '.join(fmt(r.get(k)) for k in columns)+' |' for r in rows])

def main():
    out=c.OUT
    assert (out/'evaluation_complete.json').exists()
    selected=json.loads((out/'selection.json').read_text()); key=selected['key']
    metrics=[]; activities=[]; attribution=[]; failures=[]; latency=[]; branches=[]
    for split in ('train','validation','test'):
        rows=[r for r in cast(c.read_csv(out/f'{split}_trial_results.csv')) if r['key']==key]
        oracle=cast(c.read_csv(out/f'{split}_oracle_placement.csv'))
        filter_rows=c.read_csv(out/f'{split}_filter_ablation.csv')
        for design in c.DESIGNS:
            subset=[r for r in rows if r['design']==design]
            metrics.append(dict(split=split,design=design,selected=design==selected['design'],**c.aggregate(subset)))
            for activity in sorted({r['activity'] for r in subset}):
                group=[r for r in subset if r['activity']==activity]
                summary=c.aggregate(group)
                activities.append(dict(split=split,design=design,activity=activity,**summary,
                                       false_negative_rate=None if not group[0]['label'] else 1-summary['sensitivity'],
                                       no_matched_raw_trigger=sum(r['label'] and not r['raw_trigger_hit'] for r in group),
                                       incomplete_window=sum(r['label'] and r['raw_trigger_hit'] and not r['candidate_hit'] for r in group),
                                       matched_candidates_rejected=sum(r['label'] and r['candidate_hit'] and not r['detected'] for r in group)))
            fall=[r for r in subset if r['label']]
            failures.append(dict(split=split,design=design,fall_trials=len(fall),
                                 no_matched_raw_trigger=sum(not r['raw_trigger_hit'] for r in fall),
                                 matched_trigger_but_incomplete_window=sum(r['raw_trigger_hit'] and not r['candidate_hit'] for r in fall),
                                 matched_complete_candidates_all_rejected=sum(r['candidate_hit'] and not r['detected'] for r in fall),
                                 detected=sum(r['detected'] for r in fall)))
            for r in fall:
                if r['detected']: latency.append(dict(split=split,design=design,path=r['path'],latency_s=r['latency_s']))
        original=float(next(r for r in filter_rows if r['mode']=='original')['sensitivity'])
        causal=float(next(r for r in filter_rows if r['mode']=='causal')['sensitivity'])
        base=next(r['sensitivity'] for r in metrics if r['split']==split and r['design']=='pre100_post100')
        attribution.extend([dict(split=split,stage='original_saved_windows',sensitivity=original,delta_from_previous=None),
                            dict(split=split,stage='causal_filter_same_saved_windows',sensitivity=causal,delta_from_previous=causal-original),
                            dict(split=split,stage='causal_trigger_100_100',sensitivity=base,delta_from_previous=base-causal)])
        for design in ('pre50_post150','pre100_post200'):
            sensitivity=next(r['sensitivity'] for r in metrics if r['split']==split and r['design']==design)
            attribution.append(dict(split=split,stage=f'placement_vs_100_100_{design}',sensitivity=sensitivity,delta_from_previous=sensitivity-base))
        for design in c.DESIGNS:
            group=[r for r in oracle if r['design']==design]
            attribution.append(dict(split=split,stage=f'offline_oracle_placement_{design}',
                                    sensitivity=sum(r['prediction'] for r in group)/len(group),delta_from_previous=None))
        events=[e for e in c.read_csv(out/f'{split}_events.csv') if e['key']==key]
        for design in c.DESIGNS:
            for activity in sorted({e['activity'] for e in events}):
                group=[e for e in events if e['design']==design and e['activity']==activity]
                negative=[e for e in group if not int(e['prediction'])]
                low_perp=sum(c.np.float32(float(e['ga_C2']))<=c.LOCKED_TREE['threshold'] for e in negative)
                branches.append(dict(split=split,design=design,activity=activity,candidates=len(group),
                                     accepted=sum(int(e['prediction']) for e in group),
                                     rejected_low_ga_C2=int(low_perp),
                                     rejected_high_jerk_low_parallel=len(negative)-int(low_perp)))
    c.write_csv(out/'summary.csv',metrics); c.write_csv(out/'activity_metrics.csv',activities)
    c.write_csv(out/'attribution.csv',attribution); c.write_csv(out/'failure_decomposition.csv',failures)
    c.write_csv(out/'detected_event_latencies.csv',latency)
    c.write_csv(out/'activity_decision_branches.csv',branches)
    import scipy
    c.dump(out/'report_provenance.json',dict(python=platform.python_version(),numpy=c.np.__version__,scipy=scipy.__version__,
                                           report_code_sha256=c.sha(Path(__file__)),
                                           test_code_sha256=c.sha(Path(__file__).with_name('test_causal_events.py'))))
    cols=['split','design','candidate_recall','candidates_per_adl_hour','sensitivity','false_alarms_per_adl_hour',
          'event_precision','latency_median_s','latency_worst_s']
    chosen=[r for r in metrics if r['selected']]
    filters=[]
    for split in ('validation','test'):
        filters.extend(c.read_csv(out/f'{split}_filter_ablation.csv'))
    text=['# Causal preprocessing and event-window experiment',
          'The classifier, three feature split thresholds, original subject partitions, and raw data are unchanged. '
          'This is a separate preprocessing/trigger experiment, not a replacement of prior baseline artifacts.',
          '## Frozen validation selection',
          f"Trigger: causal filtered magnitude >= {selected['trigger'][0]} g OR absolute magnitude jerk >= {selected['trigger'][1]} g/s "
          '(a null jerk threshold disables that branch). Rising edge of the OR condition, with a fixed 1.5 s refractory interval. '
          'The refractory interval and grid were declared before replay, not tuned on test.',
          f"Selected window: **{selected['design']}**. The 99% localized completed-candidate recall target across all designs "
          f"was **{'met' if selected['target_99_percent_met'] else 'NOT met'}**.",
          'Selection uses validation only: among trigger settings meeting 99% recall in every design, minimize mean ADL candidate rate; '
          'if none qualifies, maximize the minimum recall first, then minimize candidate rate. With that common trigger fixed, '
          'choose the window with highest final validation sensitivity, then lowest ADL false alarms/hour, then shortest post delay. '
          'Training results are descriptive. selection.json was frozen before test replay; all three predeclared window designs '
          'were evaluated together in one test pass, and none was selected using test.',
          table(chosen,cols),
          '## All predeclared windows',table([r for r in metrics if r['split']!='train'],cols),
          'Each window is [trigger-pre, trigger+post), with a decision at trigger+post: exactly 0.5, 0.75, or 1.0 s post-trigger delay. '
          'These contain 200, 200, and 300 samples respectively. The 300-sample variant uses the same feature equations, '
          'including a mean over 299 magnitude differences; the tree is unchanged and was originally developed for 200 samples.',
          '## Definitions and limits',
          '- No annotated onset/impact timestamps are available in the saved pipeline. Each fall trial is assumed to contain one fall; '
          'its raw-magnitude argmax is an **evaluation-only impact proxy**, not ground truth. Neither labels nor this proxy enter trigger generation.',
          '- A candidate is matched when its trigger is within +/-1 s of the proxy. Candidate recall requires a complete, emitted window. '
          'raw_trigger_recall excludes window completeness; any_candidate_trial_recall ignores localization. All are exported in summary.csv.',
          '- A detected fall is a matched candidate accepted by the locked tree. One true positive per fall trial is credited. '
          'Other alarms, including duplicates and unmatched alarms in fall recordings, are false positives for event precision. '
          'No extra alarm-merging policy is applied beyond the trigger refractory interval.',
          '- ADL rates count complete candidate windows or positive decisions divided by full recorded ADL hours, including startup and tails. '
          'Boundary candidates lacking prehistory or the complete post delay are discarded; no shifting, padding, or flushing with future samples.',
          '- Latency is the earliest matched positive decision minus the impact proxy. It is signed and conditional on detection; '
          'negative values mean an alarm precedes the proxy. Worst latency is the largest observed delay among detected events, '
          'not a guarantee for missed falls. Proxy-based recall, precision, and latency require annotated timestamps for definitive interpretation.',
          '- These are waist recordings. Causality and orientation robustness do not validate wrist fall-detection performance.',
          '## Causal implementation',
          'The fourth-order 5 Hz Butterworth at 200 Hz is implemented as two stateful DF-II-transposed SOS sections, one sample per update. '
          'At each trial start, SOS state is sosfilt_zi times the first observed raw sample. Gravity uses the unchanged raw-acceleration '
          '0.5 Hz EMA, initialized from that same first sample. States persist throughout each trial and reset only between trials. '
          'No reverse filtering or delay compensation is used. The one-pass frequency/phase response differs from forward/backward filtering.',
          'Filter and gravity arithmetic are float64; filtered samples retain the previous float32 window interface before float64 feature arithmetic. '
          'The locked export performs its original float32 feature comparisons. This is a causal desktop reference, not a float32 firmware equivalence claim.',
          'The trigger uses only current/past causal magnitude and magnitude differences. The ring stores magnitude, perpendicular magnitude, '
          'and absolute parallel acceleration: 301 x 3 values for the longest window, including the excluded decision-time sample. '
          'That is 3612 bytes in a float32 port or 7224 bytes in this float64 reference, plus SOS, gravity, and queue state. '
          'Reference replay archives derived streams for analysis, but ring aggregation only accesses samples already available at its decision time. '
          'Prefix-invariance tests cover the trigger and emitted features. With 1.5 s refractory and <=1 s post delay, each design has at most one pending event.',
          '## Attribution: filter, localization, and placement',
          'The first comparison uses exactly the existing saved windows and the original cached locked-model features versus causal-filter features. '
          'This isolates preprocessing without changing event positions or ADL sampling. Its specificity/precision are **window-level**, '
          'and must not be compared directly with event-level false alarms/hour or event precision.',
          table(filters,['split','mode','sensitivity','specificity','balanced_accuracy','precision']),
          table([r for r in attribution if r['split']!='train'],['split','stage','sensitivity','delta_from_previous']),
          'Deltas are sensitivity fractions (multiply by 100 for percentage points). The trigger-100/100 delta includes candidate misses, '
          'refractory suppression, boundary losses, and trigger timing versus the previous oracle-centered fall window. '
          'It is not a pure timing-only estimate. Placement deltas keep trigger thresholds and trigger times fixed. '
          'For the 300-sample design they also include the longer feature interval. Offline oracle-placement rows center each design at the '
          'impact proxy on the causal stream to expose placement effects without candidate gating; these are diagnostic only and were not used in selection.',
          table([r for r in failures if r['split']!='train'],list(failures[0])),
          '## Per-activity failures',
          'activity_metrics.csv contains every activity and window design. The following focuses on the predeclared difficult activities for the selected design.',
          table([r for r in activities if r['split'] in ('validation','test') and r['design']==selected['design'] and
                 r['activity'] in ('D03','D04','D06','D18','D19','F13','F14','F15')],
                ['split','activity','candidate_recall','sensitivity','candidates_per_adl_hour','false_alarms_per_adl_hour',
                 'no_matched_raw_trigger','incomplete_window','matched_candidates_rejected']),
          'For ADLs, every alarm is a false alarm. For falls, a missing matched candidate is a trigger/window failure; '
          'a complete matched candidate rejected by the tree is a downstream classification failure. '
          'The per-trial CSVs and event CSVs retain sample bounds, feature values, and decisions for inspection. '
          'activity_decision_branches.csv further separates low-ga_C2 rejections from high-jerk/low-parallel rejections '
          'across all candidates; these candidate counts must not be mistaken for trial-level false negatives.',
          '## Reproduction and integrity',
          'Run test_causal_events.py, then causal_events.py validate, then causal_events.py evaluate, then causal_event_report.py. '
          'Selection/evaluation refuse to overwrite an already frozen/completed run. Use a separate output directory for a new experiment. '
          'experiment_plan.json and selection.json hash the locked tree, its predictor, partitions, source manifests, original features, and replay code. '
          'Each raw trial is checked against its existing SHA-256 manifest during replay. evaluation_complete.json verifies frozen inputs remained unchanged. '
          'No fitting, model hyperparameter tuning, or classifier threshold tuning occurs.']
    (out/'REPORT.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    print(table(chosen,cols))

if __name__=='__main__': main()
