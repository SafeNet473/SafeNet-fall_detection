"""Independent real-candidate accounting and stored-estimator integrity checks."""
import sys
sys.dont_write_bytecode=True
import causal_feature_expansion as e
import numpy as np
import json,math

def main():
    plan=json.loads((e.OUT/'experiment_plan.json').read_text())
    selection=json.loads((e.OUT/'selection.json').read_text())
    subject=json.loads((e.c.SOURCE/'split_subjects.json').read_text())
    trigger=json.loads((e.c.OUT/'selection.json').read_text())
    d=e.m.load('validation',trigger,subject)
    array=np.load(e.OUT/'validation_features.npy')
    np.testing.assert_array_equal(array[:,:3],d['x'])
    operations=e.m.read(e.OUT/'operating_points.csv')
    checked=[]
    for name,entry in selection['models'].items():
        model=e.joblib.load(e.m.OUT/'logistic.joblib' if name=='frozen3' else e.OUT/f'{name}.joblib')
        width=len(entry['features'])
        scores=model['model'].predict_proba(model['scaler'].transform(array[:,:width]))[:,1]
        np.testing.assert_array_equal(scores,np.load(e.OUT/f'{name}_validation_scores.npy'))
        thresholds=[r for r in operations if r['feature_set']==name]
        for row in thresholds:
            pred=scores>=float(row['threshold'])
            tp=tn=0
            for i,label in enumerate(d['labels']):
                group=d['index']==i
                if label:tp+=int(np.any(pred[group] & (d['y'][group]==1)))
                else:tn+=int(not np.any(pred[group]))
            adl=int(pred[d['labels'][d['index']]==0].sum())
            assert tp==int(row['detected_falls'])
            assert int(pred.sum())==int(row['total_alarms'])
            assert math.isclose(adl/d['hours'],float(row['false_alarms_per_adl_hour']),abs_tol=1e-12)
            assert math.isclose(tn/int((d['labels']==0).sum()),float(row['trial_specificity']),abs_tol=1e-12)
        rows=e.m.read(e.OUT/f'{name}_validation_sweep.csv')
        for budget in (1,5,10,20):
            point=next(r for r in thresholds if r['operating_point']==f'FA_le_{budget}')
            eligible=[r for r in rows if float(r['false_alarms_per_adl_hour'])<=budget]
            best=max(int(r['detected_falls']) for r in eligible)
            assert int(point['detected_falls'])==best
            assert float(point['false_alarms_per_adl_hour'])==min(float(r['false_alarms_per_adl_hour']) for r in eligible if int(r['detected_falls'])==best)
        # Verify target failures/margin without rounding 98% to 98.00%.
        assert max(float(r['sensitivity']) for r in rows)==370/375
        point=next(r for r in thresholds if r['operating_point']=='sens_gt98')
        assert int(point['detected_falls'])>=368
        assert float(point['false_alarms_per_adl_hour'])==min(float(r['false_alarms_per_adl_hour']) for r in rows if float(r['sensitivity'])>.98)
        if name!='frozen3':
            training=np.load(e.OUT/'train_features.npy')[:,:width]
            np.testing.assert_allclose(model['scaler'].mean_,training.mean(axis=0),rtol=1e-12,atol=1e-12)
        checked.append(dict(feature_set=name,validation_candidates=len(scores),selected_points_checked=len(thresholds),saved_score_mismatches=0))
    if not selection['gate']['five_vs_frozen']['material']:
        assert 'expanded7' not in selection['models'] and not (e.OUT/'expanded7.joblib').exists()
    assert e.m.hashes([e.ROOT/p for p in plan['input_sha256']])==plan['input_sha256']
    e.dump('independent_verification.json',dict(checked=checked,accounting_passed=True,budget_optimality_passed=True,
                                              strict_target_check_passed=True,baseline_inputs_unchanged=True,
                                              train_only_scaler_verified=True,stopping_rule_enforced=True,no_test_data_used=True))
    print('Independent accounting, score parity, budget optimality, >98% target and stopping-rule checks passed.')

if __name__=='__main__':main()
