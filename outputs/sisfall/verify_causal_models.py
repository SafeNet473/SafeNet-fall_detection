"""Verify event accounting and exported models without fitting or test access."""
import sys
sys.dont_write_bytecode=True
import train_causal_models as m
import numpy as np
import json
from scipy.special import expit

def main():
    # One detected fall with two matching alarms and an unmatched alarm, one
    # untriggered fall, one alarmed ADL, one untriggered ADL.
    d=dict(trials=[dict(label=str(y),candidate_hit=str(c)) for y,c in [(1,1),(1,0),(0,0),(0,0)]],
           labels=np.array([1,1,0,0]),index=np.array([0,0,0,2]),y=np.array([1,1,0,0]),hours=2.)
    result=m.metrics(d,np.ones(4),.5)
    assert result['sensitivity']==.5 and result['trial_specificity']==.5
    assert result['event_precision']==.25 and result['unmatched_duplicate_alarms']==3
    assert result['false_alarms_per_adl_hour']==.5 and result['candidate_recall']==.5
    s=json.loads((m.SOURCE/'selection.json').read_text())
    subjects=json.loads((m.ROOT/'processed/baseline_v2/split_subjects.json').read_text())
    params=json.loads((m.OUT/'model_parameters.json').read_text())
    checks=[]
    for split in ('train','validation'):
        data=m.load(split,s,subjects)
        for family in ('tree','logistic'):
            saved=m.joblib.load(m.OUT/f'{family}.joblib')
            model=saved['model'];p=params[family];x=data['x']
            if family=='tree':
                predictions=[]
                for row in x.astype(np.float32):
                    node=0
                    while p['children_left'][node]!=-1:
                        # sklearn widens float32 input to compare double threshold.
                        node=p['children_left'][node] if float(row[p['features'][node]])<=p['split_thresholds'][node] else p['children_right'][node]
                    values=p['class_values'][node][0]
                    # This sklearn version already stores normalized probabilities.
                    # Renormalization can move a leaf score across a tied threshold.
                    predictions.append(values[1]>=p['threshold'])
                actual=model.predict_proba(x)[:,1]>=p['threshold']
            else:
                z=(x-np.array(p['mean']))/np.array(p['scale'])
                score=expit(z@np.array(p['coefficients'])+p['intercept'])
                actual_score=model.predict_proba(saved['scaler'].transform(x))[:,1]
                np.testing.assert_allclose(score,actual_score,rtol=1e-12,atol=1e-12)
                predictions=score>=p['threshold'];actual=actual_score>=p['threshold']
                if split=='train':np.testing.assert_allclose(np.mean(x,axis=0),p['mean'],rtol=1e-12)
            np.testing.assert_array_equal(predictions,actual)
            scores=model.predict_proba(x if saved['scaler'] is None else saved['scaler'].transform(x))[:,1]
            # Independent event accounting from per-trial loops.
            alarms=scores>=p['threshold'];tp=0;tn=0
            for i,label in enumerate(data['labels']):
                indexes=data['index']==i
                if label:tp+=int(np.any(alarms[indexes] & (data['y'][indexes]==1)))
                else:tn+=int(not np.any(alarms[indexes]))
            result=m.metrics(data,scores,p['threshold'])
            assert result['detected_falls']==tp
            assert result['trial_specificity']==tn/int((data['labels']==0).sum())
            assert result['event_precision']==tp/int(alarms.sum())
            checks.append(dict(split=split,model=family,candidates=len(x),export_prediction_mismatches=0))
    plan=json.loads((m.OUT/'experiment_plan.json').read_text())
    assert m.hashes([m.ROOT/p for p in plan['input_sha256']])==plan['input_sha256']
    m.dump('export_verification.json',dict(synthetic_event_accounting_passed=True,
                                          independent_real_event_accounting_passed=True,
                                          training_only_scaler_verified=True,inputs_unchanged=True,
                                          test_data_opened=False,checks=checks))
    tree=params['tree'];lines=['Positive-score threshold: '+repr(tree['threshold'])]
    def walk(i,indent):
        if tree['children_left'][i]==-1:
            score=tree['class_values'][i][0][1]
            lines.append(indent+str(int(score>=tree['threshold']))+' (positive score='+repr(score)+')')
            return
        lines.append(indent+'if '+tree['feature_names'][tree['features'][i]]+' <= '+repr(tree['split_thresholds'][i])+':')
        walk(tree['children_left'][i],indent+'  ')
        lines.append(indent+'else:');walk(tree['children_right'][i],indent+'  ')
    walk(0,'')
    (m.OUT/'tree_thresholded.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified event accounting, train-only scaler, exported predictions and frozen inputs; no test access.')

if __name__=='__main__':main()
