"""Train-only fitting and validation-only selection on frozen causal candidates.

No test event/feature/outcome file is opened. Previous outputs are read-only.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import csv
import hashlib
import json
import platform

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/deps'))
import numpy as np
import scipy
import sklearn
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

OUT=ROOT/'outputs/causal_trained_models_v1'
SOURCE=ROOT/'outputs/causal_event_study'
NAMES=['ga_C2','jerk_abs_mean','ga_parallel_peak']

def read(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,
                                   default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def hashes(paths):
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def load(split,selection,subjects):
    assert split in ('train','validation')
    trials=[r for r in read(SOURCE/f'{split}_trial_results.csv')
            if r['key']==selection['key'] and r['design']==selection['design']]
    events=[r for r in read(SOURCE/f'{split}_events.csv')
            if r['key']==selection['key'] and r['design']==selection['design']]
    lookup={int(r['trial_index']):i for i,r in enumerate(trials)}
    for r in trials:
        assert r['split']==split and Path(r['path']).stem.split('_')[1] in subjects[split]
    indices=np.array([lookup[int(r['trial_index'])] for r in events])
    labels=np.array([int(r['label']) for r in trials])
    matched=np.array([bool(labels[i] and abs(int(e['trigger'])-int(trials[i]['proxy_peak']))<=200)
                      for e,i in zip(events,indices)])
    x=np.array([[float(e[n]) for n in NAMES] for e in events])
    assert np.isfinite(x).all() and set(matched)=={False,True}
    return dict(trials=trials,events=events,index=indices,labels=labels,y=matched.astype(int),x=x,
                hours=sum(float(r['duration_s']) for r in trials if not int(r['label']))/3600)

def context(data,scores):
    max_score=np.full(len(data['trials']),-np.inf)
    eligible=(data['labels'][data['index']]==0)|(data['y']==1)
    np.maximum.at(max_score,data['index'][eligible],scores[eligible])
    return max_score

def metrics(data,scores,threshold,max_score=None):
    if max_score is None:max_score=context(data,scores)
    pred=scores>=threshold
    fall=data['labels']==1;adl=~fall
    detected=max_score>=threshold
    tp=int(detected[fall].sum());tn=int((~detected[adl]).sum())
    sens=tp/int(fall.sum());spec=tn/int(adl.sum())
    alarms=int(pred.sum());adl_alarms=int(pred[adl[data['index']]].sum())
    candidate_sens=float(pred[data['y']==1].mean())
    candidate_spec=float((~pred[data['y']==0]).mean())
    return dict(candidate_recall=sum(int(r['candidate_hit']) for r in data['trials'] if int(r['label']))/int(fall.sum()),
                sensitivity=sens,false_alarms_per_adl_hour=adl_alarms/data['hours'],
                event_precision=tp/alarms if alarms else 0.,trial_specificity=spec,
                trial_balanced_accuracy=(sens+spec)/2,candidate_balanced_accuracy=(candidate_sens+candidate_spec)/2,
                detected_falls=tp,missed_falls=int(fall.sum())-tp,adl_alarm_trials=int(adl.sum())-tn,
                adl_false_alarms=adl_alarms,total_alarms=alarms,unmatched_duplicate_alarms=alarms-tp)

def tune(data,scores):
    """All distinct validation prediction boundaries; no train or test refit."""
    mx=context(data,scores)
    best=None
    # Threshold changes at score values; nextafter(max,+inf) permits no alarms.
    for threshold in np.r_[np.unique(scores),np.nextafter(scores.max(),np.inf)]:
        m=metrics(data,scores,float(threshold),mx)
        rank=(m['trial_balanced_accuracy'],m['event_precision'],-m['false_alarms_per_adl_hour'],float(threshold))
        if best is None or rank>best[0]:best=(rank,float(threshold),m)
    return best[1:]

def details(name,data,scores,threshold):
    pred=scores>=threshold;mx=context(data,scores)
    rows=[]
    for i,t in enumerate(data['trials']):
        selected=data['index']==i
        alarm_count=int(pred[selected].sum());label=int(t['label'])
        detected=int(label and mx[i]>=threshold)
        rows.append(dict(model=name,split=t['split'],path=t['path'],activity=t['activity'],label=label,
                         duration_s=float(t['duration_s']),candidate_hit=int(t['candidate_hit']),
                         alarms=alarm_count,detected=detected,
                         no_matched_candidate=int(label and not int(t['candidate_hit'])),
                         matched_candidates_rejected=int(label and int(t['candidate_hit']) and not detected)))
    activities=[]
    for a in sorted({r['activity'] for r in rows}):
        group=[r for r in rows if r['activity']==a];fall=group[0]['label']==1
        hours=sum(r['duration_s'] for r in group)/3600
        activities.append(dict(model=name,split=group[0]['split'],activity=a,trials=len(group),
                               candidate_recall=sum(r['candidate_hit'] for r in group)/len(group) if fall else None,
                               sensitivity=sum(r['detected'] for r in group)/len(group) if fall else None,
                               false_negative_rate=sum(1-r['detected'] for r in group)/len(group) if fall else None,
                               false_alarms_per_hour=sum(r['alarms'] for r in group)/hours if not fall else None,
                               false_positive_trial_rate=sum(r['alarms']>0 for r in group)/len(group) if not fall else None,
                               no_matched_candidate=sum(r['no_matched_candidate'] for r in group),
                               matched_candidates_rejected=sum(r['matched_candidates_rejected'] for r in group)))
    events=[dict(model=name,split=e['split'],path=e['path'],trigger=e['trigger'],start=e['start'],end=e['end'],
                 proxy_target=int(y),score=float(s),prediction=int(v),**{n:float(e[n]) for n in NAMES})
            for e,y,s,v in zip(data['events'],data['y'],scores,pred)]
    return rows,activities,events

def table(rows,columns):
    def fmt(x):return 'N/A' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                     ['| '+' | '.join(fmt(r[k]) for k in columns)+' |' for r in rows])

def main():
    if OUT.exists():raise RuntimeError('Use a fresh output directory; refusing to overwrite any run.')
    OUT.mkdir(parents=True)
    paths=[SOURCE/'selection.json',ROOT/'processed/baseline_v2/split_subjects.json',
           ROOT/'outputs/simplified_decision_tree/simplified_tree.json',Path(__file__)]
    paths += [SOURCE/f'{s}_{kind}.csv' for s in ('train','validation') for kind in ('events','trial_results')]
    before=hashes(paths)
    selection=json.loads((SOURCE/'selection.json').read_text())
    assert selection['design']=='pre100_post200' and selection['trigger']==[1.1,5.0]
    subjects=json.loads((ROOT/'processed/baseline_v2/split_subjects.json').read_text())
    assert not set(subjects['train'])&set(subjects['validation'])
    plan=dict(features=NAMES,preprocessing='Frozen causal_event_study selection; 300-sample pre100_post200',
              training_target='Fall trial AND trigger within +/-200 samples of evaluation-only raw impact proxy; other candidates negative',
              grids=dict(tree_depth=[1,2,3],tree_min_samples_leaf=[10,25,50,100],logistic_C=[.01,.1,1,10]),
              class_weight='balanced',selection='Max validation trial BA; ties event precision, then fewer ADL alarms/hour; model ties simpler depth/node count or smaller C',
              threshold='>= positive probability; all unique validation scores plus no-alarm boundary; threshold ties higher value',
              no_test_access=True,train_only_scaler=True,no_refit_on_validation=True,
              substantial_degradation='Validation trial BA lower than frozen causal reference by >=0.05; also flag sensitivity drop >=0.05 separately',
              input_sha256=before)
    dump('experiment_plan.json',plan)
    data={s:load(s,selection,subjects) for s in ('train','validation')}
    train=data['train'];val=data['validation']
    scaler=StandardScaler().fit(train['x'])
    candidates=[];fitted={}
    for depth in (1,2,3):
        for leaf in (100,50,25,10):
            name=f'tree_d{depth}_leaf{leaf}'
            model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=473)
            model.fit(train['x'],train['y'])
            threshold,m=tune(val,model.predict_proba(val['x'])[:,1])
            candidates.append(dict(candidate=name,family='tree',depth=model.get_depth(),nodes=model.tree_.node_count,
                                   C=None,min_samples_leaf=leaf,threshold=threshold,**m))
            fitted[name]=(model,None)
    for C in (.01,.1,1.,10.):
        name=f'logistic_C{C:g}'
        model=LogisticRegression(C=C,class_weight='balanced',solver='liblinear',max_iter=5000,tol=1e-8,random_state=473)
        model.fit(scaler.transform(train['x']),train['y'])
        assert model.n_iter_[0]<5000
        threshold,m=tune(val,model.predict_proba(scaler.transform(val['x']))[:,1])
        candidates.append(dict(candidate=name,family='logistic',depth=None,nodes=None,C=C,
                               min_samples_leaf=None,threshold=threshold,**m))
        fitted[name]=(model,scaler)
    write('validation_candidates.csv',candidates)
    selected={}
    for family in ('tree','logistic'):
        entries=[r for r in candidates if r['family']==family]
        best=max(entries,key=lambda r:(r['trial_balanced_accuracy'],r['event_precision'],-r['false_alarms_per_adl_hour'],
                                      -(r['depth'] or 0),-(r['nodes'] or 0),-(r['C'] or 0)))
        selected[family]=best
    dump('selection.json',selected)
    params={};summary=[];trial_rows=[];activity_rows=[];event_rows=[]
    for family,best in selected.items():
        model,scale=fitted[best['candidate']]
        joblib.dump(dict(model=model,scaler=scale,feature_names=NAMES,threshold=best['threshold']),OUT/f'{family}.joblib')
        if family=='tree':
            tree=model.tree_
            params[family]=dict(feature_names=NAMES,threshold=best['threshold'],depth=model.get_depth(),
                                nodes=tree.node_count,leaves=model.get_n_leaves(),children_left=tree.children_left.tolist(),
                                children_right=tree.children_right.tolist(),features=tree.feature.tolist(),
                                split_thresholds=tree.threshold.tolist(),class_values=tree.value.tolist(),
                                n_node_samples=tree.n_node_samples.tolist(),classes=model.classes_.tolist())
            (OUT/'tree.txt').write_text(export_text(model,feature_names=NAMES,decimals=12)+
                '\nDisplayed class labels use sklearn argmax; deployment instead thresholds positive probability at '+str(best['threshold'])+'.\n',encoding='utf-8')
        else:
            params[family]=dict(feature_names=NAMES,threshold=best['threshold'],mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                                coefficients=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),C=best['C'])
        for split,d in data.items():
            scores=model.predict_proba(d['x'] if scale is None else scale.transform(d['x']))[:,1]
            summary.append(dict(model=family,split=split,depth=best['depth'],nodes=best['nodes'],threshold=best['threshold'],**metrics(d,scores,best['threshold'])))
            tr,ac,ev=details(family,d,scores,best['threshold']);trial_rows+=tr;activity_rows+=ac;event_rows+=ev
    locked=json.loads((ROOT/'outputs/simplified_decision_tree/simplified_tree.json').read_text())['tree']
    for split,d in data.items():
        predictions=[]
        for event in d['events']:
            node=locked
            while 'prediction' not in node:
                node=node['left' if np.float32(float(event[node['feature']]))<=node['threshold'] else 'right']
            predictions.append(node['prediction'])
        scores=np.array(predictions,dtype=float)
        np.testing.assert_array_equal(scores,[int(e['prediction']) for e in d['events']])
        summary.append(dict(model='offline_frozen',split=split,depth=3,nodes=7,threshold=None,**metrics(d,scores,.5)))
        tr,ac,ev=details('offline_frozen',d,scores,.5);trial_rows+=tr;activity_rows+=ac;event_rows+=ev
    write('metrics.csv',summary);write('trial_results.csv',trial_rows);write('activity_metrics.csv',activity_rows);write('event_predictions.csv',event_rows)
    dump('model_parameters.json',params)
    assert hashes(paths)==before
    dump('verification.json',dict(inputs_unchanged=True,offline_predictions_identical=True,test_data_opened=False,
                                  training_candidates=len(train['y']),training_positive_candidates=int(train['y'].sum()),
                                  validation_candidates=len(val['y']),validation_positive_candidates=int(val['y'].sum()),
                                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    cols=['model','candidate_recall','sensitivity','false_alarms_per_adl_hour','event_precision',
          'trial_specificity','trial_balanced_accuracy','candidate_balanced_accuracy','depth','nodes']
    validation=[r for r in summary if r['split']=='validation']
    reference=next(r for r in validation if r['model']=='offline_frozen')
    comparisons=[]
    for r in validation:
        if r['model']=='offline_frozen':continue
        comparisons.append(dict(model=r['model'],BA_delta=r['trial_balanced_accuracy']-reference['trial_balanced_accuracy'],
                                sensitivity_delta=r['sensitivity']-reference['sensitivity'],
                                substantial_BA_degradation=r['trial_balanced_accuracy']<=reference['trial_balanced_accuracy']-.05,
                                sensitivity_drop_5pp=r['sensitivity']<=reference['sensitivity']-.05))
    write('validation_comparison.csv',comparisons)
    report=['# Three-feature causal-trained models: validation only',
            'New models were fitted from scratch on training subjects only. The frozen offline tree and all prior experiment outputs were preserved. No test outcomes, events, or features were opened. No features were added.',
            '## Fixed pipeline and supervision',
            'Stateful fourth-order 5 Hz Butterworth at 200 Hz; existing raw-acceleration 0.5 Hz gravity EMA; fixed trigger magnitude >=1.1 g OR absolute magnitude jerk >=5 g/s, rising edge with 1.5 s refractory; pre100_post200 window (300 samples, decision after 1 s). Only ga_C2, jerk_abs_mean, ga_parallel_peak in that order. The previous causal replay CSVs provide the exact emitted candidates, including causally available features and preserved window boundaries.',
            'A candidate is a positive training example only in a fall trial with trigger within +/-1 s of its raw-magnitude impact proxy. Other candidates, including unmatched candidates in fall trials, are negatives. This is weak proxy supervision, not timestamp-annotated ground truth. Trials with no localized candidate remain in evaluation denominators. Multiple matching training candidates remain separate examples; evaluation credits at most one true positive per fall trial.',
            '## Fitting and selection',
            'Class weights are balanced from training candidate labels. Tree grid: depths 1/2/3 and minimum leaf samples 10/25/50/100. Logistic grid: C=0.01/0.1/1/10; training-only StandardScaler, liblinear L2. Both use random seed 473. All thresholds are selected from validation positive-score boundaries, including the no-alarm boundary. Neither estimator is refitted on validation.',
            'Selection maximizes validation trial-level balanced accuracy, breaking ties by event precision, fewer ADL alarms/hour, then simpler model. Trial BA averages localized fall sensitivity and ADL-trial specificity (fraction of ADL recordings with no positive decision). Candidate BA uses the candidate proxy labels and excludes missed/non-triggered falls. Neither quantity is an event-level true-negative rate; true-negative events have no natural denominator.',
            'Event precision = detected fall trials / all emitted positive decisions: unmatched fall-trial alarms and duplicates count as false positives. ADL false alarms/hour uses the full recorded ADL duration. The shared validation partition was previously used to select the trigger/window and is reused for model selection, so these are development results, not independent generalization estimates.',
            '## Validation results',table(validation,cols),
            table([dict(model=f,candidate=b['candidate'],threshold=b['threshold']) for f,b in selected.items()],['model','candidate','threshold']),
            '## Comparison with frozen offline tree on identical causal candidates',table(comparisons,list(comparisons[0])),
            'Substantial degradation was predefined as >=5 percentage points lower trial BA; a >=5-point sensitivity drop is also reported separately. No automatic feature expansion is performed, regardless of the result. The comparator above uses identical causal windows; original zero-phase/oracle-window metrics are not a fair direct comparator for this training experiment.',
            '## Per-activity validation failure modes',
            table([r for r in activity_rows if r['split']=='validation'],
                  ['model','activity','trials','candidate_recall','sensitivity','false_negative_rate','false_alarms_per_hour',
                   'false_positive_trial_rate','no_matched_candidate','matched_candidates_rejected']),
            'All three models share candidate misses. Additional fall misses are candidates rejected by that model. ADL false-positive trial rates and alarms/hour answer different questions; both are provided. Event predictions and per-trial outcomes are exported for inspection.',
            '## Artifacts and reproduction',
            'Run `python -B outputs/sisfall/train_causal_models.py` in a fresh run directory (OUT constant); existing output directories are never overwritten. '
            'experiment_plan.json records the grid, objective, feature set and hashes before fitting. validation_candidates.csv records every candidate model and its selected validation threshold. '
            'model_parameters.json exports full tree arrays and logistic coefficients/scaler. tree.txt uses sklearn default labels for display only; deployed decisions use the exported positive-score threshold. '
            'metrics.csv includes descriptive training metrics. verification.json records input preservation and exact frozen-tree prediction agreement. '
            'The new models are validation-selected candidates; no test evaluation or firmware equivalence claim is made.']
    (OUT/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(table(validation,cols))

if __name__=='__main__':main()
