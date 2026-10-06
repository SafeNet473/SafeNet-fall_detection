"""Bounded same-physical-feature capacity study; no test access or MCU-v1 edits."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,math,warnings
import causal_feature_expansion as e
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning
import joblib

ROOT=e.ROOT
OUT=ROOT/'outputs/causal_model_capacity'
TERMS=['x1','x2','x3','x1^2','x2^2','x3^2','x1*x2','x1*x3','x2*x3']
def dump(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False,
                                   default=lambda a:a.item() if isinstance(a,np.generic) else str(a)),encoding='utf-8')
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=e.m.csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def quadratic(x):
    return np.column_stack((x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]))

def material(base,other):
    gains={str(b):other[f'FA_le_{b}']['sensitivity']-base[f'FA_le_{b}']['sensitivity'] for b in (1,5,10,20)}
    reductions={n:1-other[n]['false_alarms_per_adl_hour']/base[n]['false_alarms_per_adl_hour']
                for n in ('sens_ge95','sens_ge97','sens_gt98') if base[n] and other[n] and base[n]['false_alarms_per_adl_hour']>0}
    both_low_regress=gains['1']<=-.01 and gains['5']<=-.01
    qualifies=max(gains.values())>=.01 or any(v>=.2 for v in reductions.values())
    return dict(material=qualifies and not both_low_regress,sensitivity_gains=gains,FA_reduction_fractions=reductions,
                both_low_budgets_regress_ge1pp=both_low_regress)

def priority(pt):
    return (pt['FA_le_1']['sensitivity']>.98,pt['FA_le_5']['sensitivity']>.98,
            *(pt[f'FA_le_{b}']['detected_falls'] for b in (1,5,10,20)),
            *(-pt[n]['false_alarms_per_adl_hour'] if pt[n] else -math.inf for n in ('sens_gt98','sens_ge95','sens_ge97')))

def complexity(name,obj):
    model=obj['model'];kind=obj['family']
    common=dict(family=kind,candidate=name,scaler_statistics=0,nodes=None,max_depth=None,trees=None)
    if kind in ('linear','quadratic'):
        n=3 if kind=='linear' else 9
        return dict(**{**common,'scaler_statistics':2*n},model_numeric_parameters=n+1,
                    float32_numeric_parameter_bytes=4*(n+1),estimated_flash_bytes=4*(n+2),
                    multiplications=n+(6 if kind=='quadratic' else 0),additions=n,comparisons=1,
                    nonlinear='none for binary logit comparison; score output adds sigmoid',extra_ram_bytes=4 if n==3 else 40,
                    deployment_form='Scaler fused into coefficients; separate logit threshold; quadratic costs include 6 monomial products')
    if kind=='forest':
        t=len(model.estimators_);nodes=sum(x.tree_.node_count for x in model.estimators_)
        depth=max(x.get_depth() for x in model.estimators_)
        compares=sum(x.get_depth() for x in model.estimators_)+1
        return dict(**{**common,'nodes':nodes,'max_depth':depth,'trees':t},model_numeric_parameters=nodes,
                    float32_numeric_parameter_bytes=nodes*4,estimated_flash_bytes=nodes*12+t*4+8,
                    multiplications=1,additions=t-1,comparisons=compares,nonlinear='tree branches; no exp/sqrt',extra_ram_bytes=8,
                    deployment_form='One float32 threshold OR leaf probability per node; 12-byte aligned node incl routing, 4-byte root index/tree, mean reciprocal and threshold')
    hidden=model.hidden_layer_sizes[0]
    count=sum(a.size for a in model.coefs_)+sum(a.size for a in model.intercepts_)
    return dict(**{**common,'scaler_statistics':6},model_numeric_parameters=count,
                float32_numeric_parameter_bytes=count*4,estimated_flash_bytes=(count+1)*4,
                multiplications=4*hidden,additions=4*hidden,comparisons=hidden+1,
                nonlinear=f'{hidden} ReLU max(0,z); no sigmoid needed for binary logit comparison',extra_ram_bytes=(hidden+1)*4,
                deployment_form='Input scaler fused into first layer; one hidden layer; separate output-logit threshold')

def export(obj):
    model=obj['model'];scale=obj['scaler'];kind=obj['family']
    result=dict(family=kind,input_feature_order=e.BASE,hyperparameters=model.get_params(deep=False),
                classes=model.classes_.tolist(),scaler=None if scale is None else dict(mean=scale.mean_.tolist(),scale=scale.scale_.tolist()),
                terms=TERMS if kind=='quadratic' else e.BASE)
    if kind in ('linear','quadratic'):
        result.update(coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist())
    elif kind=='mlp':
        result.update(weights=[x.tolist() for x in model.coefs_],biases=[x.tolist() for x in model.intercepts_],
                      hidden_activation='relu',output_activation=model.out_activation_,n_iter=int(model.n_iter_))
    else:
        result['trees']=[dict(children_left=t.tree_.children_left.tolist(),children_right=t.tree_.children_right.tolist(),
                               feature=t.tree_.feature.tolist(),threshold=t.tree_.threshold.tolist(),value=t.tree_.value.tolist()) for t in model.estimators_]
    return result

def main():
    if OUT.exists():raise RuntimeError('Refusing to overwrite capacity study.')
    OUT.mkdir(parents=True)
    freeze=ROOT/'outputs/mcu_v1_frozen'
    sources=list(freeze.iterdir())+[Path(__file__),Path(e.__file__),Path(e.m.__file__),
        e.m.OUT/'logistic.joblib',e.m.OUT/'event_predictions.csv',e.c.OUT/'selection.json',e.c.SOURCE/'split_subjects.json']
    sources += [e.c.OUT/f'{s}_{k}.csv' for s in ('train','validation') for k in ('events','trial_results')]
    hashes=e.m.hashes(sources)
    plan=dict(features=e.BASE,quadratic_terms=TERMS,quadratic_C=[.01,.1,1.,10.],
              forest=dict(n_estimators=[5,10,20],max_depth=[2,3],min_samples_leaf=25,max_features=1.,bootstrap=True,class_weight='balanced',seed=473),
              mlp=dict(architectures=[[3,4,1],[3,8,1]],alpha=[.001,.1],activation='relu',solver='lbfgs',max_iter=2000,max_fun=50000,tol=1e-7,seed=473,
                       early_stopping=False,sample_weight='training-only balanced weights; same class imbalance treatment as other fits'),
              preprocessing='Exact existing causal candidate features; training-only StandardScaler for logistic/MLP; no scaling for forest',
              selection='Within family: primary >98% at <=1, secondary >98% at <=5; then lexicographic sensitivity at budgets1,5,10,20; then lower FA at >98%,95%,97%; then lower estimated model flash and stronger regularization.',
              material_rule='>=1pp gain at any budget1/5/10/20 OR >=20% FA reduction at >=95%,>=97%,>98%; disqualify only if >=1pp regression at BOTH budgets1 AND5',
              bounds='14 fits total:4 quadratic+6 forest+4 MLP; all families evaluated; no seed or architecture expansion; no fit of frozen baseline',
              stopping='Stop after these candidates regardless of outcome. Failed-to-converge fits are ineligible, recorded, and not retried with larger bounds.',
              targets=dict(primary='sensitivity >.98 AND FA/hour <=1',secondary='sensitivity >.98 AND FA/hour <=5'),
              no_test_access=True,inputs_sha256=hashes)
    dump('experiment_plan.json',plan)
    selection=e.m.json.loads((e.c.OUT/'selection.json').read_text());subjects=e.m.json.loads((e.c.SOURCE/'split_subjects.json').read_text())
    data={s:e.m.load(s,selection,subjects) for s in ('train','validation')}
    train=data['train'];val=data['validation']
    saved=joblib.load(e.m.OUT/'logistic.joblib')
    baseline=dict(family='linear',model=saved['model'],scaler=saved['scaler'])
    baseline['scores']=baseline['model'].predict_proba(baseline['scaler'].transform(val['x']))[:,1]
    old=np.array([float(r['score']) for r in e.m.read(e.m.OUT/'event_predictions.csv') if r['split']=='validation' and r['model']=='logistic'])
    np.testing.assert_array_equal(baseline['scores'],old)
    baseline['rows']=e.sweep(val,baseline['scores']);baseline['points']=e.points(baseline['rows'])
    fits={'linear_frozen':baseline};attempts=[]
    linear_scale=StandardScaler().fit(train['x']);quad_scale=StandardScaler().fit(quadratic(train['x']))
    weights=compute_sample_weight('balanced',train['y'])
    specs=[]
    for C in (.01,.1,1.,10.):
        specs.append((f'quadratic_C{C:g}','quadratic',LogisticRegression(C=C,class_weight='balanced',solver='liblinear',tol=1e-8,max_iter=5000,random_state=473),quad_scale))
    for trees in (5,10,20):
        for depth in (2,3):
            specs.append((f'forest_t{trees}_d{depth}','forest',RandomForestClassifier(n_estimators=trees,max_depth=depth,min_samples_leaf=25,
                          max_features=1.,bootstrap=True,class_weight='balanced',random_state=473,n_jobs=1),None))
    for hidden in (4,8):
        for alpha in (.001,.1):
            specs.append((f'mlp_h{hidden}_a{alpha:g}','mlp',MLPClassifier(hidden_layer_sizes=(hidden,),activation='relu',solver='lbfgs',alpha=alpha,
                          max_iter=2000,max_fun=50000,tol=1e-7,random_state=473,early_stopping=False),linear_scale))
    for name,family,model,scale in specs:
        X=quadratic(train['x']) if family=='quadratic' else train['x']
        V=quadratic(val['x']) if family=='quadratic' else val['x']
        if scale is not None:X=scale.transform(X);V=scale.transform(V)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            if family=='mlp':model.fit(X,train['y'],sample_weight=weights)
            else:model.fit(X,train['y'])
        convergence=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        scores=model.predict_proba(V)[:,1];assert np.isfinite(scores).all()
        obj=dict(family=family,model=model,scaler=scale,scores=scores)
        obj['rows']=e.sweep(val,scores);obj['points']=e.points(obj['rows'])
        obj['converged']=not convergence
        fits[name]=obj
        joblib.dump(dict(model=model,scaler=scale,feature_order=e.BASE,quadratic_terms=TERMS if family=='quadratic' else None),OUT/f'{name}.joblib')
        np.save(OUT/f'{name}_validation_scores.npy',scores)
        write(f'{name}_validation_sweep.csv',obj['rows'])
        attempts.append(dict(candidate=name,family=family,converged=not convergence,warnings='; '.join(str(w.message) for w in caught),
                             n_iter=int(model.n_iter_) if family=='mlp' else int(model.n_iter_[0]) if family=='quadratic' else None))
        print(f'Completed {name}; converged={not convergence}',flush=True)
    chosen={'linear':'linear_frozen'}
    for family in ('quadratic','forest','mlp'):
        eligible=[name for name,obj in fits.items() if obj['family']==family and obj.get('converged',True)]
        if not eligible:continue
        def key(name):
            obj=fits[name];regularization=-obj['model'].C if family=='quadratic' else obj['model'].alpha if family=='mlp' else 0
            return (*priority(obj['points']),-complexity(name,obj)['estimated_flash_bytes'],regularization)
        chosen[family]=max(eligible,key=key)
    candidate_points=[]
    for name,obj in fits.items():
        for op,r in obj['points'].items():
            if r:candidate_points.append(dict(candidate=name,family=obj['family'],eligible=obj.get('converged',True),selected=name in chosen.values(),operating_point=op,**r))
    write('candidate_operating_points.csv',candidate_points);write('fit_status.csv',attempts)
    write('linear_frozen_validation_sweep.csv',baseline['rows'])
    np.save(OUT/'linear_frozen_validation_scores.npy',baseline['scores'])
    ops=[];fronts=[];activity=[];trials=[];costs=[];comparison=[];targets=[];parameters={}
    for family,name in chosen.items():
        obj=fits[name];parameters[name]=export(obj)
        costs.append(complexity(name,obj))
        for op,r in obj['points'].items():
            if not r:continue
            ops.append(dict(family=family,candidate=name,operating_point=op,**r))
            tr,ac,_=e.m.details(name,val,obj['scores'],r['threshold'])
            for row in tr:row['operating_point']=op
            for row in ac:row['operating_point']=op
            trials+=tr;activity+=ac
        fronts += [dict(family=family,candidate=name,**r) for r in e.pareto(obj['rows'])]
        if family!='linear':comparison.append(dict(candidate=name,**material(baseline['points'],obj['points'])))
        pt=obj['points'];a=pt['FA_le_1'];b=pt['FA_le_5'];h=pt['sens_gt98']
        targets.append(dict(family=family,candidate=name,primary_achieved=a['sensitivity']>.98,secondary_achieved=b['sensitivity']>.98,
                            sensitivity_at_FA1=a['sensitivity'],sensitivity_at_FA5=b['sensitivity'],
                            detections_short_of_target_at_FA1=max(0,368-a['detected_falls']),detections_short_of_target_at_FA5=max(0,368-b['detected_falls']),
                            minimum_FA_to_exceed98=h['false_alarms_per_adl_hour'] if h else None,
                            FA_gap_primary=max(0,h['false_alarms_per_adl_hour']-1) if h else None,
                            FA_gap_secondary=max(0,h['false_alarms_per_adl_hour']-5) if h else None,
                            candidate_ceiling=370/375,trigger_misses=5,
                            classifier_misses_at_FA1=a['missed_falls']-5,classifier_misses_at_FA5=b['missed_falls']-5))
    dump('selection.json',dict(selected=chosen,materiality=comparison))
    dump('model_parameters.json',parameters)
    write('operating_points.csv',ops);write('pareto_frontiers.csv',fronts);write('complexity.csv',costs)
    write('activity_metrics.csv',activity);write('trial_results.csv',trials);write('engineering_targets.csv',targets)
    # Ranking is descriptive, with the requested priority order fixed before data fits.
    order=sorted(chosen.values(),key=lambda name:(*(fits[name]['points'][f'FA_le_{b}']['detected_falls'] for b in (1,5,10,20)),
                   *(-fits[name]['points'][n]['false_alarms_per_adl_hour'] if fits[name]['points'][n] else -math.inf for n in ('sens_gt98','sens_ge95','sens_ge97')),
                   -complexity(name,fits[name])['estimated_flash_bytes']),reverse=True)
    ranking=[]
    for i,name in enumerate(order,1):
        obj=fits[name];cost=complexity(name,obj)
        ranking.append(dict(rank=i,candidate=name,family=obj['family'],
                             **{f'sensitivity_FA{b}':obj['points'][f'FA_le_{b}']['sensitivity'] for b in (1,5,10,20)},
                             **{f'FA_{n}':obj['points'][n]['false_alarms_per_adl_hour'] if obj['points'][n] else None for n in ('sens_ge95','sens_ge97','sens_gt98')},
                             estimated_flash_bytes=cost['estimated_flash_bytes'],multiplications=cost['multiplications'],comparisons=cost['comparisons']))
    write('ranking.csv',ranking)
    assert e.m.hashes(sources)==hashes
    dump('verification.json',dict(all_source_hashes_unchanged=True,mcu_v1_unchanged=True,no_test_data_used=True,
                                  baseline_scores_identical=True,physical_features_exact=e.BASE,fit_count=len(specs),
                                  selected_model_count=len(chosen),max_mlp_hidden_units=8))
    print(json.dumps(ranking,indent=2));print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
