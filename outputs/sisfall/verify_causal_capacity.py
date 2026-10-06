"""Independent capacity-study validation checks; no fitting/test inputs."""
import sys
sys.dont_write_bytecode=True
import json,math
import causal_model_capacity as c
import numpy as np
from scipy.special import expit

def main():
    selection=json.loads((c.OUT/'selection.json').read_text())
    plan=json.loads((c.OUT/'experiment_plan.json').read_text())
    exported=json.loads((c.OUT/'model_parameters.json').read_text())
    trigger=json.loads((c.e.c.OUT/'selection.json').read_text())
    subjects=json.loads((c.e.c.SOURCE/'split_subjects.json').read_text())
    d=c.e.m.load('validation',trigger,subjects);train=c.e.m.load('train',trigger,subjects)
    np.testing.assert_array_equal(c.quadratic(np.array([[2.,3.,5.]])),[[2,3,5,4,9,25,6,10,15]])
    assert len(c.e.m.read(c.OUT/'fit_status.csv'))==14
    points=c.e.m.read(c.OUT/'operating_points.csv');checks=[]
    for family,name in selection['selected'].items():
        obj=c.joblib.load(c.e.m.OUT/'logistic.joblib' if family=='linear' else c.OUT/f'{name}.joblib')
        x=c.quadratic(d['x']) if family=='quadratic' else d['x']
        z=obj['scaler'].transform(x) if obj['scaler'] is not None else x
        actual=obj['model'].predict_proba(z)[:,1]
        np.testing.assert_array_equal(actual,np.load(c.OUT/f'{name}_validation_scores.npy'))
        p=exported[name]
        standard=(x-np.array(p['scaler']['mean']))/np.array(p['scaler']['scale']) if p['scaler'] else x
        if family in ('linear','quadratic'):
            manual=expit((standard@np.array(p['coefficients']).T+np.array(p['intercept'])).ravel())
        elif family=='mlp':
            h=np.maximum(0,standard@np.array(p['weights'][0])+np.array(p['biases'][0]))
            manual=expit((h@np.array(p['weights'][1])+np.array(p['biases'][1])).ravel())
            expected_params=sum(np.array(w).size for w in p['weights'])+sum(np.array(b).size for b in p['biases'])
            assert expected_params==41
        else:
            manual=np.zeros(len(x))
            for tree in p['trees']:
                out=[]
                for row in x.astype(np.float32):
                    node=0
                    while tree['children_left'][node]!=-1:
                        node=tree['children_left'][node] if float(row[tree['feature'][node]])<=tree['threshold'][node] else tree['children_right'][node]
                    out.append(tree['value'][node][0][1])
                manual+=out
            manual/=len(p['trees'])
        np.testing.assert_allclose(manual,actual,rtol=1e-12,atol=1e-12)
        if obj['scaler'] is not None:
            tx=c.quadratic(train['x']) if family=='quadratic' else train['x']
            np.testing.assert_allclose(obj['scaler'].mean_,tx.mean(axis=0),rtol=1e-12,atol=1e-12)
        export_mismatch=0
        subset=[r for r in points if r['candidate']==name]
        for r in subset:
            t=float(r['threshold']);pred=actual>=t
            export_mismatch+=int(((manual>=t)!=pred).sum())
            tp=tn=0
            for i,label in enumerate(d['labels']):
                mask=d['index']==i
                if label:tp+=int(np.any(pred[mask] & (d['y'][mask]==1)))
                else:tn+=int(not np.any(pred[mask]))
            adl=int(pred[d['labels'][d['index']]==0].sum());n=int(pred.sum())
            assert tp==int(r['detected_falls']) and n==int(r['total_alarms'])
            assert math.isclose(adl/d['hours'],float(r['false_alarms_per_adl_hour']),abs_tol=1e-12)
            assert math.isclose(tn/int((d['labels']==0).sum()),float(r['trial_specificity']),abs_tol=1e-12)
            if n:assert math.isclose(tp/n,float(r['event_precision']),abs_tol=1e-12)
            else:assert r['event_precision']==''
        rows=c.e.m.read(c.OUT/f'{name}_validation_sweep.csv')
        for budget in (1,5,10,20):
            r=next(r for r in subset if r['operating_point']==f'FA_le_{budget}')
            eligible=[v for v in rows if float(v['false_alarms_per_adl_hour'])<=budget]
            best=max(int(v['detected_falls']) for v in eligible)
            assert int(r['detected_falls'])==best
            assert float(r['false_alarms_per_adl_hour'])==min(float(v['false_alarms_per_adl_hour']) for v in eligible if int(v['detected_falls'])==best)
        for key,target in [('sens_ge95',.95),('sens_ge97',.97),('sens_gt98',.98)]:
            r=next(r for r in subset if r['operating_point']==key)
            eligible=[v for v in rows if float(v['sensitivity'])>target] if key=='sens_gt98' else [v for v in rows if float(v['sensitivity'])>=target]
            assert float(r['false_alarms_per_adl_hour'])==min(float(v['false_alarms_per_adl_hour']) for v in eligible)
        assert max(float(v['sensitivity']) for v in rows)==370/375
        checks.append(dict(candidate=name,validation_rows=len(x),saved_model_score_mismatches=0,
                           manual_export_max_abs_score_error=float(np.max(np.abs(manual-actual))),
                           manual_export_selected_threshold_prediction_mismatches=export_mismatch))
    # Test the exact requested AND guard independently using synthetic points.
    base={f'FA_le_{b}':{'sensitivity':.9} for b in (1,5,10,20)}
    base.update({k:{'false_alarms_per_adl_hour':10.} for k in ('sens_ge95','sens_ge97','sens_gt98')})
    one=json.loads(json.dumps(base));one['FA_le_1']['sensitivity']=.88;one['FA_le_10']['sensitivity']=.92
    assert c.material(base,one)['material']
    both=json.loads(json.dumps(one));both['FA_le_5']['sensitivity']=.88
    assert not c.material(base,both)['material']
    assert c.e.m.hashes([c.ROOT/p for p in plan['inputs_sha256']])==plan['inputs_sha256']
    c.dump('independent_verification.json',dict(checks=checks,quadratic_order_verified=True,event_accounting_verified=True,
                                               exact_budget_and_target_optimality_verified=True,materiality_AND_guard_verified=True,
                                               train_only_scalers_verified=True,all_source_and_MCU_v1_hashes_unchanged=True,no_test_data_used=True))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
