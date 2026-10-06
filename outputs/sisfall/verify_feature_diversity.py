"""Read-only independent checks of saved train/validation study; never fits."""
import sys
sys.dont_write_bytecode=True
import json, hashlib, platform
import causal_feature_diversity as d
import numpy as np
import sklearn, scipy, joblib
from scipy.special import expit

def main():
    out=d.OUT
    plan=json.loads((out/'experiment_plan.json').read_text())
    for name,digest in plan['inputs_sha256'].items():
        assert hashlib.sha256((d.ROOT/name).read_bytes()).hexdigest()==digest,name
    selection=json.loads((d.e.c.OUT/'selection.json').read_text())
    subjects=json.loads((d.e.c.SOURCE/'split_subjects.json').read_text())
    data={s:d.e.m.load(s,selection,subjects) for s in ('train','validation')}
    x={s:np.load(out/f'{s}_features.npy') for s in data}
    for s in data:np.testing.assert_array_equal(x[s][:,:3],data[s]['x'])
    params=json.loads((out/'model_parameters.json').read_text())
    ops=d.e.m.read(out/'operating_points.csv'); checked=0
    val=data['validation']; fall=val['labels']==1
    for name in ['frozen_baseline3',*d.SETS]:
        indices=[0,1,2] if name=='frozen_baseline3' else d.SETS[name]
        obj=joblib.load(d.e.m.OUT/'logistic.joblib' if name=='frozen_baseline3' else out/f'{name}_selected.joblib')
        xx=np.ascontiguousarray(x['validation'][:,indices])
        scores=obj['model'].predict_proba(obj['scaler'].transform(xx))[:,1]
        np.testing.assert_array_equal(scores,np.load(out/f'{name}_validation_scores.npy'))
        if name!='frozen_baseline3':
            p=params[name]; assert obj['feature_names']==[d.NAMES[i] for i in indices]
            np.testing.assert_array_equal(obj['scaler'].mean_,np.ascontiguousarray(x['train'][:,indices]).mean(axis=0))
            np.testing.assert_array_equal(obj['scaler'].mean_,p['mean'])
            np.testing.assert_array_equal(obj['scaler'].scale_,p['scale'])
            np.testing.assert_array_equal(obj['model'].coef_[0],p['coefficients'])
            assert obj['model'].intercept_[0]==p['intercept']
            manual=expit(((xx-np.array(p['mean']))/np.array(p['scale']))@np.array(p['coefficients'])+p['intercept'])
            np.testing.assert_array_equal(scores,manual)
        sweep=d.e.m.read(out/f'{name}_selected_sweep.csv')
        np.testing.assert_array_equal(np.sort([float(r['threshold']) for r in sweep]),np.r_[np.unique(scores),np.nextafter(scores.max(),np.inf)])
        assert max(float(r['sensitivity']) for r in sweep)==370/375
        for r in [r for r in ops if r['feature_set']==name]:
            pred=scores>=float(r['threshold'])
            hit=np.zeros(len(val['trials']),dtype=bool)
            eligible=(~fall[val['index']])|(val['y']==1)
            for i in val['index'][pred&eligible]:hit[i]=True
            tp=int(hit[fall].sum()); alarms=int(pred.sum())
            fa=int(pred[~fall[val['index']]].sum()); spec=float((~hit[~fall]).mean())
            for key,value in dict(detected_falls=tp,missed_falls=375-tp,total_alarms=alarms,
                                  sensitivity=tp/375,false_alarms_per_adl_hour=fa/val['hours'],
                                  trial_specificity=spec,trial_balanced_accuracy=(tp/375+spec)/2).items():
                assert float(r[key])==value,(name,key)
            if alarms:assert float(r['event_precision'])==tp/alarms
            op=r['operating_point']
            if op.startswith('FA_le_'):
                budget=float(op.split('_')[-1])
                assert tp/375==max(float(a['sensitivity']) for a in sweep if float(a['false_alarms_per_adl_hour'])<=budget)
            elif op.startswith('sens_'):
                target={'sens_ge95':.95,'sens_ge97':.97,'sens_gt98':.98}[op]
                feasible=[a for a in sweep if (float(a['sensitivity'])>target if op=='sens_gt98' else float(a['sensitivity'])>=target)]
                assert fa/val['hours']==min(float(a['false_alarms_per_adl_hour']) for a in feasible)
            checked+=1
    fits=d.e.m.read(out/'fit_status.csv');assert len(fits)==24 and all(r['converged']=='True' for r in fits)
    result=dict(passed=True,operating_points_independently_checked=checked,exact_saved_score_parity=True,
                exact_exported_parameter_score_parity=True,training_only_scaler_means_verified=True,
                boundary_coverage_and_budget_optimality_verified=True,source_hashes_unchanged=True,
                no_test_data_used=True,no_refitting=True,python=platform.python_version(),
                numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    d.dump('independent_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
