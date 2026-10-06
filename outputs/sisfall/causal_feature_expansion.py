"""Predeclared, gated 3 -> 5 -> 7 causal logistic feature study. No test use."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,hashlib,math
import train_causal_models as m
import causal_events as c
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

ROOT=m.ROOT
OUT=ROOT/'outputs/causal_feature_expansion'
BASE=m.NAMES
ADDED=['late_mag_variance','late_minus_pre_jerk','perp_active_fraction','late_parallel_std']
NAMES=BASE+ADDED
CS=(.01,.1,1.,10.)

def dump(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False,
                                  default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def additions(values,parallel):
    """Window samples 0:100 are pre-trigger; 200:300 are trigger+0.5..1 s."""
    mag=values[:,0]; h=values[:,1]
    return np.array([np.var(mag[200:300],ddof=0),
                     np.abs(np.diff(mag[200:300])*200).mean()-np.abs(np.diff(mag[:100])*200).mean(),
                     np.mean(h>.5),np.std(parallel[200:300],ddof=0)])

def extract(data):
    manifest={r['path']:r for r in m.read(ROOT/'processed/baseline_v2/trials.csv') if r['split'] in ('train','validation')}
    root=Path(json.loads((ROOT/'processed/baseline_v2/config.json').read_text())['data_root'])
    result={};checked=0;maxdiff=0.
    for split,d in data.items():
        array=np.empty((len(d['events']),7));array[:,:3]=d['x']
        by_path={}
        for i,e in enumerate(d['events']):by_path.setdefault(e['path'],[]).append(i)
        for done,t in enumerate(d['trials'],1):
            path=t['path'];assert manifest[path]['split']==split
            raw,digest=c.p.load_trial(root/path);assert digest==manifest[path]['sha256']
            front=c.Frontend(raw[0]);acc=np.empty_like(raw);g=np.empty_like(raw)
            for now,r in enumerate(raw):acc[now],g[now]=front.push(r)
            values=c.derived(acc,g)
            aa=acc.astype(np.float32).astype(np.float64)
            unit=g/np.maximum(np.linalg.norm(g,axis=1)[:,None],c.of.GRAVITY_EPS)
            parallel=np.sum(aa*unit,axis=1)
            times=c.triggers(values,1.1,5.)
            valid=[k for k in times if k>=100 and k+200<len(raw)]
            ids=by_path.get(path,[])
            assert valid==[int(d['events'][i]['trigger']) for i in ids]
            assert len(times)==int(t['raw_triggers'])
            # Only read a candidate's window at its saved availability time.
            ring=np.empty((301,4));due={int(d['events'][i]['end']):i for i in ids}
            for now,sample in enumerate(values):
                ring[now%301,:3]=sample;ring[now%301,3]=parallel[now]
                if now not in due:continue
                i=due[now];e=d['events'][i];start=int(e['start']);end=int(e['end'])
                assert start==int(e['trigger'])-100 and end==int(e['trigger'])+200==now
                window=ring[np.arange(start,end)%301]
                old=c.feature_row(window[:,:3])[[15,5,18]]
                maxdiff=max(maxdiff,float(np.max(np.abs(old-d['x'][i]))))
                np.testing.assert_array_equal(old,d['x'][i])
                array[i,3:]=additions(window[:,:3],window[:,3]);checked+=1
            if done%100==0:print(f'{split}: replay {done}/{len(d["trials"])}',flush=True)
        assert np.isfinite(array).all()
        result[split]=array
        np.save(OUT/f'{split}_features.npy',array)
        write(f'{split}_feature_rows.csv',[dict(path=e['path'],trigger=e['trigger'],start=e['start'],end=e['end'],
                                              label=int(d['y'][i]),**dict(zip(NAMES,map(float,array[i]))))
                                          for i,e in enumerate(d['events'])])
    dump('extraction_verification.json',dict(candidates_checked=checked,base_feature_max_abs_difference=maxdiff,
                                            candidate_times_identical=True,raw_hashes_verified=True,
                                            causal_ring_availability_assertions_passed=True,no_test_replay=True))
    return result

def sweep(d,scores):
    mx=m.context(d,scores);out=[]
    for threshold in np.r_[np.unique(scores),np.nextafter(max(scores),np.inf)]:
        row=dict(threshold=float(threshold),**m.metrics(d,scores,float(threshold),mx))
        if row['total_alarms']==0:row['event_precision']=None
        assert row['sensitivity']<=row['candidate_recall']
        out.append(row)
    return out

def points(rows):
    out={}
    for budget in (1,5,10,20):
        out[f'FA_le_{budget}']=max((r for r in rows if r['false_alarms_per_adl_hour']<=budget),
                                  key=lambda r:(r['detected_falls'],-r['false_alarms_per_adl_hour'],-r['total_alarms'],r['threshold']))
    for name,target,strict in [('sens_ge95',.95,False),('sens_ge97',.97,False),('sens_gt98',.98,True)]:
        eligible=[r for r in rows if (r['sensitivity']>target if strict else r['sensitivity']>=target)]
        out[name]=min(eligible,key=lambda r:(r['false_alarms_per_adl_hour'],-r['detected_falls'],r['total_alarms'],-r['threshold'])) if eligible else None
    out['BA_reference']=max(rows,key=lambda r:(r['trial_balanced_accuracy'],r['event_precision'] or 0.,-r['false_alarms_per_adl_hour'],r['threshold']))
    return out

def rank(pt,C):
    return (pt['FA_le_1']['sensitivity']>.98,pt['FA_le_5']['sensitivity']>.98,
            *(pt[f'FA_le_{b}']['detected_falls'] for b in (1,5,10,20)),
            *(-pt[n]['false_alarms_per_adl_hour'] if pt[n] else -math.inf for n in ('sens_gt98','sens_ge95','sens_ge97')),-C)

def meaningful(base,expanded):
    gains={f'FA_le_{b}':expanded[f'FA_le_{b}']['sensitivity']-base[f'FA_le_{b}']['sensitivity'] for b in (1,5,10,20)}
    improved_rate=[]
    for n in ('sens_ge95','sens_ge97','sens_gt98'):
        if base[n] and expanded[n]:
            a=base[n]['false_alarms_per_adl_hour'];b=expanded[n]['false_alarms_per_adl_hour']
            if a-b>=1 and b<=.8*a:improved_rate.append(n)
    good=(max(gains.values())>=.01 or bool(improved_rate)) and gains['FA_le_1']>-.01 and gains['FA_le_5']>-.01
    return dict(material=good,budget_sensitivity_gains=gains,targets_with_at_least_20pct_and_1FAh_reduction=improved_rate)

def pareto(rows):
    best_by_count={}
    for r in rows:
        k=r['detected_falls']
        if k not in best_by_count or (r['false_alarms_per_adl_hour'],r['total_alarms'],-r['threshold']) < (best_by_count[k]['false_alarms_per_adl_hour'],best_by_count[k]['total_alarms'],-best_by_count[k]['threshold']):best_by_count[k]=r
    result=[];best=math.inf
    for k in sorted(best_by_count,reverse=True):
        r=best_by_count[k]
        if r['false_alarms_per_adl_hour']<best:result.append(r);best=r['false_alarms_per_adl_hour']
    return sorted(result,key=lambda r:r['false_alarms_per_adl_hour'])

def main():
    if OUT.exists():raise RuntimeError('Output exists; prior runs are never overwritten.')
    OUT.mkdir(parents=True)
    selected=json.loads((c.OUT/'selection.json').read_text());assert selected['design']=='pre100_post200' and selected['trigger']==[1.1,5.]
    subjects=json.loads((c.SOURCE/'split_subjects.json').read_text())
    inputs=[Path(__file__),Path(m.__file__),Path(c.__file__),c.OUT/'selection.json',c.SOURCE/'split_subjects.json',c.SOURCE/'trials.csv',
            m.OUT/'logistic.joblib',m.OUT/'model_parameters.json',m.OUT/'event_predictions.csv']
    inputs += [c.OUT/f'{s}_{kind}.csv' for s in ('train','validation') for kind in ('events','trial_results')]
    original_hashes=m.hashes(inputs)
    plan=dict(feature_sets={'frozen3':BASE,'expanded5':NAMES[:5],'expanded7_if_gate_passes':NAMES},
              C_grid=CS,scaler='train-only StandardScaler',estimator='L2 balanced liblinear logistic; max_iter=5000,tol=1e-8,seed=473; no validation refit',
              hyperparameter_rank='Primary success >98% at <=1 FA/h, secondary >98% at <=5; then lexicographic detected falls at budgets 1,5,10,20; then smaller FA/h at >98%,>=95%,>=97%; then smaller C',
              operating_points='All unique validation scores plus no-alarm boundary; >= score comparison. Full frontier plus BA diagnostic and target-specific thresholds; no global deployment threshold changed.',
              material_rule='>=1 percentage point sensitivity gain at any budget 1/5/10/20 OR >=20% AND >=1 FA/h reduction at >=95%,>=97%,>98%; require no >=1-point loss at budgets 1 or 5. Rule is a practical screen, not statistical significance.',
              stop='If 5-feature expansion fails material gate vs frozen3, do not fit 7. At most 7 features; no additional search regardless of outcome.',
              baseline='Reuse already train-fitted frozen C=10 scaler/weights; never refit/replace it. Expanded models also report fixed C=10 control to separate feature gain from C selection.',
              targets=dict(primary='sensitivity >0.98 and ADL FA/h <=1',secondary='sensitivity >0.98 and ADL FA/h <=5'),
              feature_timing='pre indices0:100, late indices200:300; all complete at existing trigger+200 decision',
              added_definitions=dict(late_mag_variance='population variance of ||a|| on late100, g^2',
                                     late_minus_pre_jerk='200*mean(abs(diff(m_late100)))-200*mean(abs(diff(m_pre100))), each99 differences, g/s',
                                     perp_active_fraction='mean(perpendicular_magnitude > 0.5g) over300; fixed untuned threshold, dimensionless',
                                     late_parallel_std='population std of signed dot(a,u) over late100, g'),
              no_test_access=True,input_sha256=original_hashes)
    dump('experiment_plan.json',plan)
    data={s:m.load(s,selected,subjects) for s in ('train','validation')}
    arrays=extract(data)
    frozen=joblib.load(m.OUT/'logistic.joblib')
    validation=data['validation'];train=data['train']
    frozen_scores=frozen['model'].predict_proba(frozen['scaler'].transform(validation['x']))[:,1]
    stored=[float(r['score']) for r in m.read(m.OUT/'event_predictions.csv') if r['model']=='logistic' and r['split']=='validation']
    np.testing.assert_array_equal(frozen_scores,stored)
    baseline_sweep=sweep(validation,frozen_scores);baseline_points=points(baseline_sweep)
    models={'frozen3':dict(scores=frozen_scores,rows=baseline_sweep,points=baseline_points,C=10.,n=3)}
    candidates=[];gate={};exports={}
    for count in (5,7):
        if count==7 and not gate['five_vs_frozen']['material']:break
        name=f'expanded{count}';scale=StandardScaler().fit(arrays['train'][:,:count]);fits=[]
        for C in CS:
            estimator=LogisticRegression(C=C,class_weight='balanced',solver='liblinear',max_iter=5000,tol=1e-8,random_state=473)
            estimator.fit(scale.transform(arrays['train'][:,:count]),train['y']);assert estimator.n_iter_[0]<5000
            scores=estimator.predict_proba(scale.transform(arrays['validation'][:,:count]))[:,1]
            rows=sweep(validation,scores);pt=points(rows)
            fits.append(dict(C=C,model=estimator,scaler=scale,scores=scores,rows=rows,points=pt,n=count))
            for point,r in pt.items():
                if r:candidates.append(dict(feature_set=name,C=C,operating_point=point,**r))
        best=max(fits,key=lambda entry:rank(entry['points'],entry['C']))
        models[name]=best
        gate[f'{"five_vs_frozen" if count==5 else "seven_vs_five"}']=meaningful(models['frozen3' if count==5 else 'expanded5']['points'],best['points'])
        joblib.dump(dict(model=best['model'],scaler=scale,feature_names=NAMES[:count],C=best['C'],operating_points=best['points']),OUT/f'{name}.joblib')
        exports[name]=dict(features=NAMES[:count],C=best['C'],mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                           coefficients=best['model'].coef_[0].tolist(),intercept=float(best['model'].intercept_[0]),
                           fixed_C10_points=next(f['points'] for f in fits if f['C']==10.))
    dump('selection.json',dict(models={k:dict(C=v['C'],features=NAMES[:v['n']],points=v['points']) for k,v in models.items()},gate=gate))
    dump('model_parameters.json',exports);write('hyperparameter_operating_points.csv',candidates)
    point_rows=[];pareto_rows=[];activities=[];trials=[];target_rows=[]
    for name,entry in models.items():
        write(f'{name}_validation_sweep.csv',entry['rows'])
        front=pareto(entry['rows'])
        pareto_rows += [dict(feature_set=name,**r) for r in front]
        np.save(OUT/f'{name}_validation_scores.npy',entry['scores'])
        for point,r in entry['points'].items():
            if r is None:continue
            point_rows.append(dict(feature_set=name,n_features=entry['n'],C=entry['C'],operating_point=point,**r))
            tr,ac,_=m.details(name,validation,entry['scores'],r['threshold'])
            for row in tr:row['operating_point']=point
            for row in ac:row['operating_point']=point
            trials+=tr;activities+=ac
        nfall=int((validation['labels']==1).sum());required=math.floor(.98*nfall)+1
        above98=entry['points']['sens_gt98'];r1=entry['points']['FA_le_1'];r5=entry['points']['FA_le_5']
        target_rows.append(dict(feature_set=name,primary_achieved=r1['sensitivity']>.98,secondary_achieved=r5['sensitivity']>.98,
                                max_sensitivity_FA_le_1=r1['sensitivity'],max_sensitivity_FA_le_5=r5['sensitivity'],
                                extra_detected_falls_needed_at_FA1=max(0,required-r1['detected_falls']),
                                extra_detected_falls_needed_at_FA5=max(0,required-r5['detected_falls']),
                                minimum_FA_to_exceed98=above98['false_alarms_per_adl_hour'] if above98 else None,
                                FA_gap_to_primary=max(0,above98['false_alarms_per_adl_hour']-1) if above98 else None,
                                FA_gap_to_secondary=max(0,above98['false_alarms_per_adl_hour']-5) if above98 else None,
                                candidate_recall=r1['candidate_recall'],unavoidable_trigger_misses=5,
                                additional_classifier_misses_at_FA1=r1['missed_falls']-5,
                                additional_classifier_misses_at_FA5=r5['missed_falls']-5))
    write('operating_points.csv',point_rows);write('pareto_frontiers.csv',pareto_rows)
    write('activity_metrics.csv',activities);write('trial_results.csv',trials);write('engineering_targets.csv',target_rows)
    assert m.hashes(inputs)==original_hashes
    dump('verification.json',dict(frozen_inputs_unchanged=True,baseline_validation_scores_identical=True,
                                  no_test_data_used=True,model_families=['logistic_regression'],max_features_tested=max(v['n'] for v in models.values()),
                                  five_feature_gate=gate['five_vs_frozen']))
    print(json.dumps(target_rows,indent=2));print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
