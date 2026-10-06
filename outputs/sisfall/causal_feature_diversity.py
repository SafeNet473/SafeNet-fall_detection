"""Exactly six predefined feature sets, balanced L2 logistic, train/validation only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,math,warnings
import numpy as np
import causal_feature_expansion as e
import causal_model_capacity as capacity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import joblib

ROOT=e.ROOT
OUT=ROOT/'outputs/causal_feature_diversity_v1'
NEW=['ga_C8','post_mag_variance','pre_post_energy_ratio','gravity_direction_change']
NAMES=e.BASE+NEW
SETS={'baseline3':[0,1,2],'plus_gaC8':[0,1,2,3],'plus_settling':[0,1,2,4],
      'plus_energy_ratio':[0,1,2,5],'plus_orientation_change':[0,1,2,6],'all7':list(range(7))}
ENERGY_EPS=1e-6
GRAVITY_EPS=1e-8
def dump(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False,
                                   default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=e.m.csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def new_features(window):
    """Columns m,h,abs(p),bx,by,bz,Gx,Gy,Gz; exactly 300 arrived samples."""
    assert window.shape==(300,9)
    mag=window[:,0];h=window[:,1];b=window[:,3:6];g=window[:,6:9]
    pre=g[:100].mean(axis=0);post=g[200:300].mean(axis=0)
    npre=np.linalg.norm(pre);npost=np.linalg.norm(post)
    invalid=npre<GRAVITY_EPS or npost<GRAVITY_EPS
    angle=0. if invalid else float(np.arccos(np.clip(np.dot(pre/npre,post/npost),-1.,1.)))
    return np.array([np.sqrt(b.var(axis=0,ddof=1).sum()),mag[200:300].var(ddof=0),
                     np.mean(h[200:300]**2)/(np.mean(h[:100]**2)+ENERGY_EPS),angle]),bool(invalid)

def extract(data):
    manifest={r['path']:r for r in e.m.read(e.c.SOURCE/'trials.csv') if r['split'] in ('train','validation')}
    raw_root=Path(json.loads((e.c.SOURCE/'config.json').read_text())['data_root'])
    arrays={};checked=0;degenerate=[]
    for split,d in data.items():
        x=np.empty((len(d['events']),7));x[:,:3]=d['x'];by_path={}
        for i,r in enumerate(d['events']):by_path.setdefault(r['path'],[]).append(i)
        for done,t in enumerate(d['trials'],1):
            path=t['path'];assert manifest[path]['split']==split
            raw,digest=e.c.p.load_trial(raw_root/path);assert digest==manifest[path]['sha256']
            front=e.c.Frontend(raw[0]);acc=np.empty_like(raw);g=np.empty_like(raw)
            for now,r in enumerate(raw):acc[now],g[now]=front.push(r)
            values=e.c.derived(acc,g);a=acc.astype(np.float32).astype(np.float64)
            u=g/np.maximum(np.linalg.norm(g,axis=1)[:,None],GRAVITY_EPS)
            p=np.sum(a*u,axis=1);b=a-p[:,None]*u
            times=e.c.triggers(values,1.1,5.);ids=by_path.get(path,[])
            assert len(times)==int(t['raw_triggers'])
            assert [k for k in times if k>=100 and k+200<len(raw)]==[int(d['events'][i]['trigger']) for i in ids]
            due={int(d['events'][i]['end']):i for i in ids};ring=np.empty((301,9))
            for now,sample in enumerate(values):
                ring[now%301,:3]=sample;ring[now%301,3:6]=b[now];ring[now%301,6:]=g[now]
                if now not in due:continue
                i=due[now];event=d['events'][i];start=int(event['start'])
                assert start==int(event['trigger'])-100 and now==int(event['trigger'])+200
                window=ring[np.arange(start,now)%301]
                np.testing.assert_array_equal(e.c.feature_row(window[:,:3])[[15,5,18]],d['x'][i])
                x[i,3:],invalid=new_features(window);checked+=1
                if invalid:degenerate.append(dict(split=split,path=path,trigger=event['trigger']))
            if done%150==0:print(f'{split}: replayed {done}/{len(d["trials"])}',flush=True)
        assert np.isfinite(x).all()
        arrays[split]=x;np.save(OUT/f'{split}_features.npy',x)
        write(f'{split}_feature_rows.csv',[dict(path=r['path'],trigger=r['trigger'],start=r['start'],end=r['end'],
                                               proxy_label=int(d['y'][i]),**dict(zip(NAMES,map(float,x[i])))) for i,r in enumerate(d['events'])])
    dump('extraction_verification.json',dict(candidate_count=checked,existing_features_exact=True,trigger_times_exact=True,
                                            raw_hashes_verified=True,invalid_gravity_mean_count=len(degenerate),invalid_gravity_windows=degenerate,
                                            ring_availability_assertions_passed=True,no_test_replay=True))
    return arrays

def main():
    if OUT.exists():raise RuntimeError('Refusing to overwrite versioned study output.')
    OUT.mkdir(parents=True)
    sources=list((ROOT/'outputs/mcu_v1_frozen').iterdir())+[Path(__file__),Path(e.__file__),Path(e.m.__file__),Path(e.c.__file__),Path(capacity.__file__),
       e.m.OUT/'logistic.joblib',e.m.OUT/'event_predictions.csv',e.c.OUT/'selection.json',e.c.SOURCE/'split_subjects.json',e.c.SOURCE/'trials.csv',e.c.SOURCE/'config.json',
       ROOT/'outputs/sisfall/orientation_features.py']
    sources += [e.c.OUT/f'{s}_{kind}.csv' for s in ('train','validation') for kind in ('events','trial_results')]
    hashes=e.m.hashes(sources)
    dump('experiment_plan.json',dict(feature_order=NAMES,sets={k:[NAMES[i] for i in v] for k,v in SETS.items()},
         definitions=dict(ga_C8='sqrt(sum_axis sample variance of b, ddof=1, full300); retain existing ga_C8 convention',
                          post_mag_variance='population variance of magnitude over indices200:300, ddof=0',
                          pre_post_energy_ratio='mean(h[200:300]^2)/(mean(h[0:100]^2)+1e-6)',
                          gravity_direction_change='acos(clamp(dot(normalize(mean(G[0:100])),normalize(mean(G[200:300]))),-1,1)); radians; norm<1e-8 returns0'),
         energy_epsilon_g2=ENERGY_EPS,gravity_epsilon_g=GRAVITY_EPS,variance_convention='existing ga_C8 sample ddof1 retained; new settling population ddof0',
         C_grid=[.01,.1,1.,10.],estimator='balanced L2 liblinear logistic; seed473, tol1e-8,max_iter5000',
         scaler='train-only StandardScaler per feature set, no validation refit',
         selection='Primary target >98% at<=1 FA/h, secondary at<=5; then lexicographic detected falls at1/5/10/20; then smaller FA/h at>98%,>=95%,>=97%; then smaller C',
         material_rule='>=1pp sensitivity gain at any budget1/5/10/20 OR >=20% FA reduction at>=95%,>=97%,>98%; disqualify only if>=1pp regression at BOTH <=1 AND<=5',
         baseline='Keep frozen C10 model untouched as comparison reference. Fit baseline3 C-grid in this experiment as a separate control; C10 reproduction must match saved model exactly.',
         bounds='Exactly six sets x four C values =24 fits. No feature or C additions afterward, regardless of results.',
         no_test_data=True,inputs_sha256=hashes))
    selection=json.loads((e.c.OUT/'selection.json').read_text());assert selection['design']=='pre100_post200' and selection['trigger']==[1.1,5.]
    subjects=json.loads((e.c.SOURCE/'split_subjects.json').read_text())
    data={s:e.m.load(s,selection,subjects) for s in ('train','validation')};arrays=extract(data);val=data['validation'];train=data['train']
    frozen=joblib.load(e.m.OUT/'logistic.joblib');baseline=frozen['model'].predict_proba(frozen['scaler'].transform(val['x']))[:,1]
    old=np.array([float(r['score']) for r in e.m.read(e.m.OUT/'event_predictions.csv') if r['split']=='validation' and r['model']=='logistic'])
    np.testing.assert_array_equal(baseline,old)
    rows=e.sweep(val,baseline);basepoints=e.points(rows)
    models={'frozen_baseline3':dict(scores=baseline,rows=rows,points=basepoints,C=10.,indices=[0,1,2],model=frozen['model'],scaler=frozen['scaler'])}
    candidates=[];status=[];parameters={};all_material=[]
    for name,indices in SETS.items():
        tx=np.ascontiguousarray(arrays['train'][:,indices]);vx=np.ascontiguousarray(arrays['validation'][:,indices])
        scaler=StandardScaler().fit(tx);fits=[]
        for C in (.01,.1,1.,10.):
            model=LogisticRegression(C=C,class_weight='balanced',solver='liblinear',max_iter=5000,tol=1e-8,random_state=473)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);model.fit(scaler.transform(tx),train['y'])
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            scores=model.predict_proba(scaler.transform(vx))[:,1];assert np.isfinite(scores).all()
            sweep=e.sweep(val,scores);pt=e.points(sweep);key=f'{name}_C{C:g}'
            fits.append(dict(C=C,model=model,scaler=scaler,scores=scores,rows=sweep,points=pt,indices=indices,converged=converged))
            status.append(dict(candidate=key,feature_set=name,C=C,converged=converged,warnings='; '.join(str(w.message) for w in caught)))
            all_material.append(dict(candidate=key,feature_set=name,eligible=converged,**capacity.material(basepoints,pt)))
            for op,r in pt.items():
                if r:candidates.append(dict(candidate=key,feature_set=name,C=C,operating_point=op,**r))
            write(f'{key}_validation_sweep.csv',sweep)
            joblib.dump(dict(model=model,scaler=scaler,feature_names=[NAMES[i] for i in indices]),OUT/f'{key}.joblib')
            if name=='baseline3' and C==10.:
                np.testing.assert_array_equal(scaler.mean_,frozen['scaler'].mean_)
                np.testing.assert_array_equal(model.coef_,frozen['model'].coef_)
                np.testing.assert_array_equal(scores,baseline)
            print(f'Fit {key}; converged={converged}',flush=True)
        best=max((r for r in fits if r['converged']),key=lambda r:(*capacity.priority(r['points']),-r['C']))
        models[name]=best
        parameters[name]=dict(features=[NAMES[i] for i in indices],C=best['C'],mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                               coefficients=best['model'].coef_[0].tolist(),intercept=float(best['model'].intercept_[0]),
                               fixed_C10_points=next(r['points'] for r in fits if r['C']==10.))
        joblib.dump(dict(model=best['model'],scaler=scaler,feature_names=parameters[name]['features']),OUT/f'{name}_selected.joblib')
    ops=[];fronts=[];activity=[];trials=[];targets=[];comparison=[]
    for name,obj in models.items():
        write(f'{name}_selected_sweep.csv',obj['rows']);np.save(OUT/f'{name}_validation_scores.npy',obj['scores'])
        fronts += [dict(feature_set=name,**r) for r in e.pareto(obj['rows'])]
        for op,r in obj['points'].items():
            if not r:continue
            ops.append(dict(feature_set=name,C=obj['C'],operating_point=op,**r))
            tr,ac,_=e.m.details(name,val,obj['scores'],r['threshold'])
            for t in tr:t['operating_point']=op
            for t in ac:t['operating_point']=op
            trials+=tr;activity+=ac
        if name!='frozen_baseline3':comparison.append(dict(feature_set=name,**capacity.material(basepoints,obj['points'])))
        pt=obj['points'];h=pt['sens_gt98'];a=pt['FA_le_1'];b=pt['FA_le_5']
        targets.append(dict(feature_set=name,C=obj['C'],primary_achieved=a['sensitivity']>.98,secondary_achieved=b['sensitivity']>.98,
                             sensitivity_FA1=a['sensitivity'],sensitivity_FA5=b['sensitivity'],
                             additional_detections_needed_FA1=max(0,368-a['detected_falls']),additional_detections_needed_FA5=max(0,368-b['detected_falls']),
                             minimum_FA_to_exceed98=h['false_alarms_per_adl_hour'] if h else None,
                             candidate_ceiling=370/375,trigger_misses=5,
                             classifier_misses_FA1=a['missed_falls']-5,classifier_misses_FA5=b['missed_falls']-5))
    write('operating_points.csv',ops);write('pareto_frontiers.csv',fronts);write('activity_metrics.csv',activity);write('trial_results.csv',trials)
    write('engineering_targets.csv',targets);write('candidate_operating_points.csv',candidates);write('fit_status.csv',status)
    dump('selection.json',dict(models={k:dict(C=v['C'],features=[NAMES[i] for i in v['indices']]) for k,v in models.items()},materiality=comparison))
    dump('all_candidate_materiality.json',all_material);dump('model_parameters.json',parameters)
    assert e.m.hashes(sources)==hashes
    dump('verification.json',dict(all_sources_and_MCU_v1_unchanged=True,baseline_scores_exact=True,refitted_baseline_C10_exact=True,
                                  fit_count=24,feature_sets_tested=6,no_test_data_used=True))
    print(json.dumps(targets,indent=2));print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
