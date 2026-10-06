"""Causal replay experiment. Never fits or modifies the locked classifier.

Run `python outputs/sisfall/causal_events.py validate`, then `... evaluate`.
All artifacts are separate from prior experiments; raw inputs are hash checked.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import pipeline as p
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import orientation_features as of

ROOT = p.PROJECT
OUT = ROOT / 'outputs/causal_event_study'
SOURCE = ROOT / 'processed/baseline_v2'
TREE_FILE = ROOT / 'outputs/simplified_decision_tree/simplified_tree.json'
sys.path.insert(0, str(TREE_FILE.parent))
from predict_simplified import predict as exported_predict
LOCKED_TREE = json.loads(TREE_FILE.read_text())['tree']

def predict(features):
    """Cached JSON traversal with the exact exported scalar comparison semantics."""
    result=[]
    for row in np.asarray(features,dtype=np.float32):
        node=LOCKED_TREE
        while 'prediction' not in node:
            node=node['left' if row[node['feature_index']] <= node['threshold'] else 'right']
        result.append(node['prediction'])
    return np.asarray(result,dtype=np.uint8)

FS = 200
DESIGNS = {'pre100_post100': (100, 100), 'pre50_post150': (50, 150),
           'pre100_post200': (100, 200)}
GRID = [(m, j) for m in (1.1, 1.3, 1.5, 1.8, 2.2)
        for j in (5., 10., 20., 40., None)]
SOS = butter(4, 5, fs=FS, output='sos')
REFRACTORY = 300
MATCH = 200

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def frozen_inputs():
    files = [TREE_FILE, TREE_FILE.with_name('predict_simplified.py'),
             SOURCE/'split_subjects.json', SOURCE/'trials.csv', SOURCE/'windows.csv',
             SOURCE/'config.json', ROOT/'outputs/robust_classifiers/features_all.npy',
             Path(__file__), ROOT/'outputs/sisfall/orientation_features.py']
    return {str(x.relative_to(ROOT)): sha(x) for x in files}

class Frontend:
    """One input sample per call; DF-II-transposed SOS and causal gravity state."""
    def __init__(self, first):
        self.state = sosfilt_zi(SOS)[:, :, None] * first[None, None, :]
        self.gravity = first.copy()

    def push(self, raw):
        value = raw
        for i, (b0, b1, b2, _, a1, a2) in enumerate(SOS):
            y = b0*value + self.state[i, 0]
            z1 = b1*value - a1*y + self.state[i, 1]
            self.state[i, 1] = b2*value - a2*y
            self.state[i, 0] = z1
            value = y
        self.gravity = of.ALPHA*raw + (1-of.ALPHA)*self.gravity
        return value, self.gravity

def derived(acc, gravity):
    # Preserve the previous saved-window float32 acceleration interface.
    acc = acc.astype(np.float32).astype(np.float64)
    unit = gravity / np.maximum(np.linalg.norm(gravity, axis=1)[:, None], of.GRAVITY_EPS)
    parallel = np.sum(acc*unit, axis=1)
    residual = acc-parallel[:, None]*unit
    return np.column_stack((np.linalg.norm(acc, axis=1),
                            np.linalg.norm(residual, axis=1), np.abs(parallel)))

def feature_row(values):
    row = np.zeros(21, dtype=np.float64)
    row[5] = np.abs(np.diff(values[:, 0])*FS).mean()
    row[15] = values[:, 1].max()
    row[18] = values[:, 2].max()
    return row

def triggers(values, magnitude, jerk):
    """Rising edge of magnitude OR absolute magnitude jerk, 1.5 s dead time.

    No labels, gravity, peak annotations, or future samples enter this function.
    At t=0 there is no previous jerk; high initial magnitude counts as an edge.
    Crossings during refractory are discarded, not postponed.
    """
    m = values[:, 0]
    j = np.r_[0., np.abs(np.diff(m)*FS)]
    high = (m >= magnitude) | (j >= (np.inf if jerk is None else jerk))
    crossings = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    result = []
    last = -REFRACTORY
    for t in crossings:
        if t-last >= REFRACTORY:
            result.append(int(t)); last = t
    return result

def ring_windows(values, times_by_key, designs=DESIGNS):
    """Half-open window [trigger-pre, trigger+post), emitted at trigger+post.

    Ring holds only already received scalar samples. One extra sample at decision
    time makes the explicit delay exactly post/200 seconds. Incomplete boundary
    windows are discarded, never shifted/padded. No future signal is read.
    """
    capacity = max(pre+post for pre, post in designs.values())+1
    ring = np.empty((capacity, 3))
    due = {}
    for key, times in times_by_key.items():
        for t in times:
            for design, (pre, post) in designs.items():
                if t >= pre and t+post < len(values):
                    due.setdefault(t+post, []).append((key, design, t, t-pre, t+post))
    events = []
    for now, sample in enumerate(values):
        ring[now % capacity] = sample
        for key, design, trigger, start, end in due.get(now, ()):
            assert end == now and start >= now-capacity+1
            window = ring[np.arange(start, end) % capacity]
            f = feature_row(window)
            events.append(dict(key=key, design=design, trigger=trigger, start=start,
                               end=end, decision=now, ga_C2=float(f[15]),
                               jerk_abs_mean=float(f[5]), ga_parallel_peak=float(f[18]),
                               prediction=int(predict(f[None])[0])))
    return events

def replay(trial, index):
    raw, digest = p.load_trial(Path(json.loads((SOURCE/'config.json').read_text())['data_root']) / trial['path'])
    assert digest == trial['sha256'] and len(raw) == int(trial['n_samples'])
    front = Frontend(raw[0])
    filtered = np.empty_like(raw); gravity = np.empty_like(raw)
    for t, sample in enumerate(raw):
        filtered[t], gravity[t] = front.push(sample)
    # Cached derived values are deterministic results of chronological replay.
    # Triggering/aggregation use these scalar streams, never labels or raw peaks.
    values = derived(filtered, gravity)
    return values, int(np.argmax(np.linalg.norm(raw, axis=1)))

def summarize(trials, events, raw_times):
    grouped = {int(t['trial_index']): [] for t in trials}
    for event in events:
        grouped[event['trial_index']].append(event)
    rows = []
    for trial in trials:
        i = int(trial['trial_index']); label = int(trial['label'])
        ev = grouped[i]; peak = int(trial['proxy_peak'])
        matched = [e for e in ev if abs(e['trigger']-peak) <= MATCH] if label else []
        alarms = [e for e in ev if e['prediction']]
        detections = [e for e in matched if e['prediction']]
        first = min((e['decision'] for e in detections), default=None)
        rows.append(dict(trial_index=i, path=trial['path'], split=trial['split'],
                         activity=trial['activity_id'], label=label,
                         duration_s=int(trial['n_samples'])/FS,
                         proxy_peak=peak, raw_triggers=len(raw_times[i]),
                         raw_trigger_hit=int(label and any(abs(t-peak)<=MATCH for t in raw_times[i])),
                         candidates=len(ev), candidate_hit=int(bool(matched)),
                         alarms=len(alarms), detected=int(bool(detections)),
                         unmatched_or_duplicate_alarms=len(alarms)-int(bool(detections)),
                         latency_s=None if first is None else (first-peak)/FS))
    return rows

def aggregate(rows):
    fall = [r for r in rows if r['label']]; adl = [r for r in rows if not r['label']]
    hours = sum(r['duration_s'] for r in adl)/3600
    tp = sum(r['detected'] for r in fall)
    fp = sum(r['unmatched_or_duplicate_alarms'] for r in rows)
    latency = [r['latency_s'] for r in fall if r['latency_s'] is not None]
    div = lambda a,b: a/b if b else None
    return dict(fall_trials=len(fall), adl_trials=len(adl), adl_hours=hours,
                raw_trigger_recall=div(sum(r['raw_trigger_hit'] for r in fall),len(fall)),
                candidate_recall=div(sum(r['candidate_hit'] for r in fall),len(fall)),
                any_candidate_trial_recall=div(sum(r['candidates']>0 for r in fall),len(fall)),
                candidates_per_adl_hour=div(sum(r['candidates'] for r in adl),hours),
                sensitivity=div(tp,len(fall)),
                false_alarms_per_adl_hour=div(sum(r['alarms'] for r in adl),hours),
                event_precision=div(tp,tp+fp), tp_events=tp, fp_events=fp,
                latency_median_s=float(np.median(latency)) if latency else None,
                latency_worst_s=max(latency) if latency else None)

def ablations(trials, streams, bounds, original):
    rows = []
    for trial in trials:
        i = int(trial['trial_index']); values = streams[i]
        for w in bounds[i]:
            wid = int(w['window_id']); start=int(w['start_sample']); end=int(w['end_sample_exclusive'])
            rows.append(dict(split=trial['split'], activity=trial['activity_id'], label=int(trial['label']),
                             original=int(predict(original[wid:wid+1])[0]),
                             causal=int(predict(feature_row(values[start:end])[None])[0])))
    return rows

def window_metrics(rows, field):
    y=np.array([r['label'] for r in rows]); pred=np.array([r[field] for r in rows])
    tp=int(((y==1)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum())
    return dict(n=len(rows),tp=tp,fn=fn,tn=tn,fp=fp,sensitivity=tp/(tp+fn),
                specificity=tn/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
                precision=tp/(tp+fp) if tp+fp else None)

def run_partition(split, settings, bounds, original):
    trials = [dict(t, trial_index=i) for i,t in enumerate(read_csv(SOURCE/'trials.csv')) if t['split']==split]
    partitions=json.loads((SOURCE/'split_subjects.json').read_text())
    results={key: {'events':[], 'times':{}} for key in settings}
    ablation_rows=[]; oracle=[]
    for done,trial in enumerate(trials,1):
        i=trial['trial_index']
        assert trial['subject_id'] in partitions[split]
        values,peak=replay(trial,i); trial['proxy_peak']=peak
        times={key:triggers(values,*setting) for key,setting in settings.items()}
        events=ring_windows(values,times)
        for key in settings:
            results[key]['times'][i]=times[key]
        for e in events:
            e.update(trial_index=i,path=trial['path'],split=split,activity=trial['activity_id'],label=int(trial['label']))
            results[e['key']]['events'].append(e)
        ablation_rows.extend(ablations([trial],{i:values},bounds,original))
        if int(trial['label']):
            # Offline diagnostic only: same causal stream at the evaluation proxy.
            # This does not enter trigger generation or design selection.
            for design,(pre,post) in DESIGNS.items():
                valid=peak>=pre and peak+post<len(values)
                pred=int(predict(feature_row(values[peak-pre:peak+post])[None])[0]) if valid else 0
                oracle.append(dict(split=split,trial_index=i,activity=trial['activity_id'],design=design,
                                   valid=int(valid),prediction=pred))
        if done%50==0:
            print(f'{split}: replayed {done}/{len(trials)} trials',flush=True)
    all_metrics=[]; all_rows=[]
    for key,result in results.items():
        for design in DESIGNS:
            ev=[e for e in result['events'] if e['design']==design]
            rows=summarize(trials,ev,result['times'])
            for r in rows: r.update(key=key,design=design)
            all_rows.extend(rows)
            all_metrics.append(dict(split=split,key=key,design=design,**aggregate(rows)))
    write_csv(OUT/f'{split}_metrics.csv',all_metrics)
    write_csv(OUT/f'{split}_trial_results.csv',all_rows)
    write_csv(OUT/f'{split}_events.csv',[e for v in results.values() for e in v['events']])
    write_csv(OUT/f'{split}_oracle_placement.csv',oracle)
    write_csv(OUT/f'{split}_filter_ablation_windows.csv',ablation_rows)
    write_csv(OUT/f'{split}_filter_ablation.csv',[dict(split=split,mode=mode,**window_metrics(ablation_rows,mode))
                                              for mode in ('original','causal')])
    return all_metrics,all_rows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['validate','evaluate'])
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    bounds={}
    for w in read_csv(SOURCE/'windows.csv'): bounds.setdefault(int(w['trial_index']),[]).append(w)
    original=np.load(ROOT/'outputs/robust_classifiers/features_all.npy',mmap_mode='r')
    if args.phase=='validate':
        if (OUT/'selection.json').exists(): raise RuntimeError('Selection already frozen; use a new experiment directory to repeat.')
        plan=dict(inputs=frozen_inputs(),designs=DESIGNS,grid=GRID,sos=SOS.tolist(),fs=FS,
                  refractory_samples=REFRACTORY,matching_tolerance_samples=MATCH,
                  selection='Trigger: >=99% completed matched candidate recall in ALL designs, then lowest mean ADL candidate rate; if infeasible maximize minimum recall first. Window: highest validation final sensitivity, then lowest ADL false alarm rate, then shortest post delay.',
                  ground_truth='One assumed event per fall trial; raw magnitude argmax is an evaluation-only proxy. Match trigger within +/-1 s; one-to-one earliest positive decision; all other alarms count as false/duplicates.',
                  boundary='No padding/shifting; require full prehistory and full explicit post delay.',
                  precision='TP matched fall events/(TP + all unmatched or duplicate alarms, including fall-trial alarms)',
                  latency='Earliest matched positive decision minus raw-peak proxy, seconds; signed, detected events only.',
                  filter_initialization='sosfilt_zi * first raw sample; reset only per trial; gravity first raw sample',
                  longer_window='300 samples; same max features; jerk mean over 299 differences, no tree changes',
                  numeric='float64 filter/gravity; filtered sample cast float32 as existing interface; float64 feature calculation; exported float32 tree inputs')
        dump(OUT/'experiment_plan.json',plan)
        settings={f'g{k:02d}':v for k,v in enumerate(GRID)}
        metrics,rows=run_partition('validation',settings,bounds,original)
        def rank(key):
            subset=[m for m in metrics if m['key']==key]
            recall=min(m['candidate_recall'] for m in subset)
            rate=float(np.mean([m['candidates_per_adl_hour'] for m in subset]))
            return (recall>=.99, 0 if recall>=.99 else recall, -rate)
        key=max(settings,key=rank)
        best=max((m for m in metrics if m['key']==key),
                 key=lambda m:(m['sensitivity'],-m['false_alarms_per_adl_hour'],-DESIGNS[m['design']][1]))
        dump(OUT/'selection.json',dict(key=key,trigger=settings[key],design=best['design'],
                                     validation=best,selection_time_unix=time.time(),inputs=frozen_inputs(),
                                     target_99_percent_met=rank(key)[0]))
        print(json.dumps(best,indent=2))
    else:
        if (OUT/'evaluation_complete.json').exists(): raise RuntimeError('Held-out evaluation already completed; no repeated test selection.')
        selected=json.loads((OUT/'selection.json').read_text())
        assert selected['inputs']==frozen_inputs(), 'Frozen inputs/code changed after selection'
        selection_hash=sha(OUT/'selection.json')
        for split in ('train','test'):
            run_partition(split,{selected['key']:selected['trigger']},bounds,original)
        assert sha(OUT/'selection.json')==selection_hash
        assert selected['inputs']==frozen_inputs()
        dump(OUT/'evaluation_complete.json',dict(selection_sha256=selection_hash,completed_time_unix=time.time(),
                                                frozen_inputs_unchanged=True,test_selection=False))

if __name__=='__main__': main()
