"""Small fixed-split non-neural models using predeclared invariant features only."""
import argparse
import csv
import io
import json
from pathlib import Path
import warnings

import pipeline as p
import numpy as np
import orientation_features as of
import paper_features as pf
from analyze_filtered import markdown_table
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.exceptions import ConvergenceWarning
import joblib

FEATURE_NAMES = (
    'mag_mean','mag_std','mag_rms','mag_peak','mag_ptp',
    'jerk_abs_mean','jerk_std','jerk_rms','jerk_abs_peak','pre_variance','post_variance',
    'gyro_mean','gyro_std','gyro_rms','gyro_peak',
    'ga_C2','ga_C8','ga_C13','ga_parallel_peak','ga_parallel_std','ga_parallel_ptp',
)
assert len(FEATURE_NAMES)==21 and set(FEATURE_NAMES) <= of.INVARIANT
assert not {'C2','C8'} & set(FEATURE_NAMES)
METRICS=('sensitivity','specificity','balanced_accuracy','precision','f1','pr_auc_trapezoidal','average_precision')


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def matrix(features):
    return np.column_stack([features[name] for name in FEATURE_NAMES])


def build_training_features(source, orientation, out, rows, partitions, X, raw):
    cached=np.load(orientation/'original_scores.npz')
    heldout_ids=np.load(orientation/'window_ids.npy')
    expected=np.flatnonzero(partitions!='train')
    np.testing.assert_array_equal(heldout_ids,expected)
    np.testing.assert_array_equal(cached['window_id'],heldout_ids)
    result=np.full((len(rows),len(FEATURE_NAMES)),np.nan)
    result[heldout_ids]=matrix(cached)
    trials=read_csv(source/'trials.csv')
    data=Path(json.loads((source/'config.json').read_text())['data_root'])
    by_trial={}
    for i in np.flatnonzero(partitions=='train'):
        by_trial.setdefault(int(rows[i]['trial_index']),[]).append(i)
    raw_hashes={}
    for done,(index,ids) in enumerate(by_trial.items(),1):
        trial=trials[index]
        path=data/trial['path']
        digest=pf.sha(path)
        assert digest==trial['sha256'],f'Raw source changed: {path}'
        raw_hashes[trial['path']]=digest
        text='\n'.join(line.strip() for line in path.read_text(encoding='utf-8-sig').replace(';','\n').splitlines() if line.strip())
        counts=np.loadtxt(io.StringIO(text),delimiter=',',ndmin=2)
        assert counts.shape==(int(trial['n_samples']),9) and np.isfinite(counts).all()
        acceleration=counts[:,:3]*p.SCALE
        full_g=of.causal_gravity(acceleration[None],acceleration[None,0])[0]
        angular=p.preprocess(counts[:,3:6]*(4000/2**16),'filtered')
        previous=[]
        gyro=[]
        for i in ids:
            start,end=int(rows[i]['start_sample']),int(rows[i]['end_sample_exclusive'])
            previous.append(full_g[start-1] if start else acceleration[0])
            gyro.append(angular[start:end].astype(np.float32))
            np.testing.assert_array_equal(raw[i],acceleration[start:end].astype(np.float32))
        g=of.causal_gravity(raw[ids],np.array(previous))
        result[ids]=matrix(of.features(X[ids],np.stack(gyro),g))
        if done%300==0:
            print(f'Training feature extraction: {done}/{len(by_trial)} trials',flush=True)
    assert np.isfinite(result).all()
    np.save(out/'features_all.npy',result)
    (out/'training_raw_sha256.json').write_text(json.dumps(raw_hashes,indent=2))
    return result


def fit_models(train_x,train_y,val_x,val_y):
    """Only train/validation enter this function; no test data or IDs accepted."""
    scaler=StandardScaler().fit(train_x)
    train_z,val_z=scaler.transform(train_x),scaler.transform(val_x)
    candidates=[]
    fits={}
    for c in (.01,.1,1.,10.):
        model=LogisticRegression(C=c,solver='liblinear',class_weight='balanced',
                                 max_iter=2000,tol=1e-6,random_state=473)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(train_z,train_y)
        values=model.predict_proba(val_z)[:,1]
        threshold=p.fit_threshold(values,val_y)
        key=f'lr_C_{c:g}'
        candidates.append(dict(candidate=key,family='logistic_regression',C=c,max_depth='',min_samples_leaf='',
                               node_count='',threshold=threshold,**pf.evaluate(val_y,values,threshold)))
        fits[key]=(model,scaler)
    for depth in (2,3,4):
        for leaf in (200,50):
            model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,
                    criterion='gini',class_weight='balanced',random_state=473)
            model.fit(train_x,train_y)
            values=model.predict_proba(val_x)[:,1]
            threshold=p.fit_threshold(values,val_y)
            key=f'tree_depth_{depth}_leaf_{leaf}'
            candidates.append(dict(candidate=key,family='decision_tree',C='',max_depth=depth,
                                   min_samples_leaf=leaf,node_count=model.tree_.node_count,threshold=threshold,
                                   **pf.evaluate(val_y,values,threshold)))
            fits[key]=(model,None)
    selected={}
    for family in ('logistic_regression','decision_tree'):
        eligible=[r for r in candidates if r['family']==family]
        # Stable ties: LR smallest C first; trees shallowest, larger leaf size first.
        best=max(eligible,key=lambda r:r['balanced_accuracy'])
        model,normalizer=fits[best['candidate']]
        selected[family]=dict(model=model,scaler=normalizer,threshold=best['threshold'],candidate=best['candidate'])
    return selected,candidates


def scores_for(entry,x):
    values=entry['scaler'].transform(x) if entry['scaler'] is not None else x
    return entry['model'].predict_proba(values)[:,1]


def export_models(models,out):
    description={}
    for family,entry in models.items():
        model=entry['model']
        if family=='logistic_regression':
            scaler=entry['scaler']
            weight=model.coef_[0]/scaler.scale_
            intercept=float(model.intercept_[0]-np.dot(weight,scaler.mean_))
            t=entry['threshold']
            logit_threshold=float(np.log(t/(1-t))) if 0<t<1 else ('-Infinity' if t<=0 else 'Infinity')
            description[family]=dict(candidate=entry['candidate'],features=list(FEATURE_NAMES),
                C=float(model.C),class_weight='balanced',regularization='L2',solver='liblinear',
                standardized_coefficients=model.coef_[0].tolist(),standardized_intercept=float(model.intercept_[0]),
                scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                raw_feature_coefficients=weight.tolist(),raw_feature_intercept=intercept,
                probability_threshold=t,logit_threshold=logit_threshold,classes=model.classes_.tolist())
        else:
            tree=model.tree_
            used=sorted(set(int(i) for i in tree.feature if i>=0))
            values=tree.value[:,0,:]
            # Current sklearn stores normalized proportions and returns them directly.
            # Renormalizing would perturb leaf scores at a tuned threshold boundary.
            np.testing.assert_allclose(values.sum(axis=1),1,rtol=1e-12,atol=1e-12)
            positive=values[:,1]
            description[family]=dict(candidate=entry['candidate'],candidate_features=list(FEATURE_NAMES),
                features_used=[FEATURE_NAMES[i] for i in used],feature_indices_used=used,
                actual_depth=int(model.get_depth()),node_count=int(tree.node_count),leaf_count=int(model.get_n_leaves()),
                max_depth=model.max_depth,min_samples_leaf=model.min_samples_leaf,class_weight='balanced',
                probability_threshold=entry['threshold'],children_left=tree.children_left.tolist(),
                children_right=tree.children_right.tolist(),feature=tree.feature.tolist(),
                threshold=tree.threshold.tolist(),positive_score=positive.tolist(),
                input_cast='float32 before split comparisons (sklearn behavior)',classes=model.classes_.tolist())
            (out/'decision_tree.txt').write_text(export_text(model,feature_names=list(FEATURE_NAMES),decimals=8)+
                '\nIMPORTANT: printed sklearn classes use its default class rule; deployed predictions must threshold the leaf positive score at '+str(entry['threshold'])+' instead. See model_parameters.json.\n')
        joblib.dump(dict(**entry,features=FEATURE_NAMES),out/f'{family}.joblib')
    (out/'model_parameters.json').write_text(json.dumps(description,indent=2))
    return description


def exported_tree_scores(parameters,x):
    x=np.asarray(x,dtype=np.float32)
    result=[]
    for row in x:
        node=0
        while parameters['children_left'][node]!=-1:
            go_left=row[parameters['feature'][node]]<=parameters['threshold'][node]
            node=parameters['children_left' if go_left else 'children_right'][node]
        result.append(parameters['positive_score'][node])
    return np.array(result)


def run(args):
    source,orientation,paper_run=(path.resolve() for path in (args.run,args.orientation,args.paper_run))
    config=json.loads((source/'config.json').read_text())
    out=p.safe_destination(args.out,Path(config['data_root']).resolve())
    assert config['window_size']==200 and config['sampling_rate_hz']==200
    assert config['filter']==dict(order=4,cutoff_hz=5,implementation='sosfiltfilt',causal=False)
    source_files=[source/name for name in ('X_raw.npy','X_filtered.npy','labels_groups_splits.npz','windows.csv','trials.csv','config.json','split_subjects.json')]
    source_files += [orientation/name for name in ('original_scores.npz','window_ids.npy','gravity_previous_states.npy','gyro_filtered_windows.npy','provenance.json')]
    source_files += [paper_run/name for name in ('feature_scores.npz','thresholds.json','metrics.csv')]
    hashes={str(path):pf.sha(path) for path in source_files}
    labels=np.load(source/'labels_groups_splits.npz')
    y,subjects,partitions=(labels[key] for key in ('y','subjects','split'))
    split=json.loads((source/'split_subjects.json').read_text())
    rows=read_csv(source/'windows.csv')
    for i,row in enumerate(rows):
        assert int(row['window_id'])==i and int(row['label'])==y[i]
        assert row['split']==partitions[i] and row['subject_id']==subjects[i]
        assert subjects[i] in split[partitions[i]]
    assert len(set(s for group in split.values() for s in group))==sum(map(len,split.values()))
    X=np.load(source/'X_filtered.npy',mmap_mode='r')
    raw=np.load(source/'X_raw.npy',mmap_mode='r')
    out.mkdir(parents=True)
    # Immutable, predeclared design before fitting or test evaluation.
    plan=dict(features=FEATURE_NAMES,feature_selection='fixed a priori; no outcome-based feature selection',
        logistic_C=[.01,.1,1,10],tree_depth=[2,3,4],tree_min_samples_leaf=[200,50],class_weight='balanced',
        seed=473,threshold='max validation balanced accuracy; lowest threshold ties',
        model_selection='one best per family by validation balanced accuracy; stable grid-order ties',
        refit_on_train_plus_validation=False,rotation_seed=args.seed,rotation_replicates=args.rotations)
    (out/'experiment_plan.json').write_text(json.dumps(plan,indent=2))
    all_features=build_training_features(source,orientation,out,rows,partitions,X,raw)
    train,val=partitions=='train',partitions=='validation'
    models,candidates=fit_models(all_features[train],y[train],all_features[val],y[val])
    p.write_csv(out/'validation_candidates.csv',candidates)
    parameters=export_models(models,out)
    print('Locked both model families and thresholds using training/validation only.',flush=True)
    original_scores={name:scores_for(entry,all_features) for name,entry in models.items()}
    cached_paper=np.load(paper_run/'feature_scores.npz')
    original_scores.update({name:cached_paper[name] for name in ('baseline','C2','C8')})
    thresholds={name:entry['threshold'] for name,entry in models.items()}
    thresholds.update({name:json.loads((paper_run/'thresholds.json').read_text())['thresholds'][name] for name in ('baseline','C2','C8')})
    # Verify numerical export parity on all examples, including no sklearn tree inference.
    lr=parameters['logistic_regression']
    raw_logits=all_features@np.array(lr['raw_feature_coefficients'])+lr['raw_feature_intercept']
    fitted_logits=models['logistic_regression']['model'].decision_function(models['logistic_regression']['scaler'].transform(all_features))
    np.testing.assert_allclose(raw_logits,fitted_logits,rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(exported_tree_scores(parameters['decision_tree'],all_features),original_scores['decision_tree'])
    np.savez(out/'original_predictions.npz',window_id=np.arange(len(y)),**original_scores)
    activities=np.array([row['activity_id'] for row in rows])
    metrics,activity_errors,error_windows,robustness=[],[],[],[]
    original_metrics={}
    def evaluate_scenario(scenario,ids,score_map,include_errors=False):
        for partition in ('train','validation','test'):
            selected=partitions[ids]==partition
            if not selected.any():
                continue
            true_ids=ids[selected]
            for name,values in score_map.items():
                value=values[selected]
                result=pf.evaluate(y[true_ids],value,thresholds[name])
                tn,fp,fn,tp=np.array(result.pop('confusion_matrix')).ravel().tolist()
                if scenario=='original':
                    original_metrics[partition,name]=result
                previous=original_metrics[partition,name]
                metrics.append(dict(scenario=scenario,split=partition,model=name,threshold=thresholds[name],
                    tn=tn,fp=fp,fn=fn,tp=tp,**result,**{f'delta_{k}':result[k]-previous[k] for k in METRICS}))
                prediction=value>=thresholds[name]
                for code in sorted(set(activities[true_ids])):
                    mask=activities[true_ids]==code
                    wrong=mask & (prediction!=y[true_ids])
                    activity_errors.append(dict(scenario=scenario,split=partition,model=name,activity_code=code,
                        error_type='FP' if code.startswith('D') else 'FN',total=int(mask.sum()),errors=int(wrong.sum()),
                        rate=float(wrong.sum()/mask.sum())))
                if include_errors and partition!='train':
                    for local in np.flatnonzero(prediction!=y[true_ids]):
                        i=true_ids[local]
                        error_windows.append(dict(model=name,**rows[i],score=float(value[local]),threshold=thresholds[name]))
    evaluate_scenario('original',np.arange(len(y)),original_scores,True)
    heldout=np.load(orientation/'window_ids.npy')
    gyro=np.load(orientation/'gyro_filtered_windows.npy',mmap_mode='r')
    states=np.load(orientation/'gravity_previous_states.npy')
    # Recompute rotated features from vectors; do NOT reuse canonicalized rotated score caches.
    for repeat in range(args.rotations):
        scenario=f'random_{repeat:02}'
        rng=np.random.default_rng(args.seed+repeat)
        rotated=np.empty((len(heldout),len(FEATURE_NAMES)))
        comparator={name:np.empty(len(heldout)) for name in ('baseline','C2','C8')}
        max_feature_error=0.
        for start in range(0,len(heldout),512):
            ids=heldout[start:start+512]
            length=len(ids)
            R=of.random_rotations(rng,length)
            # Re-run causal estimation with rotated raw samples and rotated initial history state.
            gravity=of.causal_gravity(of.rotate(raw[ids],R),np.einsum('bij,bj->bi',R,states[start:start+length]))
            values=of.features(of.rotate(X[ids],R),of.rotate(gyro[start:start+length],R),gravity)
            rotated[start:start+length]=matrix(values)
            max_feature_error=max(max_feature_error,float(np.max(np.abs(matrix(values)-all_features[ids]))))
            np.testing.assert_allclose(matrix(values),all_features[ids],rtol=1e-10,atol=1e-10)
            for name in comparator:
                comparator[name][start:start+length]=values[name]
        values={name:scores_for(entry,rotated) for name,entry in models.items()}
        for name,value in values.items():
            for partition in ('validation','test'):
                mask=partitions[heldout]==partition
                reference=original_scores[name][heldout][mask]
                predicted=value[mask]
                robustness.append(dict(scenario=scenario,model=name,split=partition,
                    max_feature_absolute_error=max_feature_error,max_score_absolute_error=float(np.max(np.abs(predicted-reference))),
                    changed_predictions=int(np.sum((predicted>=thresholds[name])!=(reference>=thresholds[name]))),
                    score_canonicalization=False))
        # Saved peak baseline uses float32 scores; verify invariance against new float64 mag_peak.
        np.testing.assert_allclose(comparator['baseline'],all_features[heldout,FEATURE_NAMES.index('mag_peak')],rtol=1e-10,atol=1e-10)
        comparator['baseline']=original_scores['baseline'][heldout]
        values.update(comparator)
        np.savez(out/f'predictions_{scenario}.npz',window_id=heldout,**values)
        evaluate_scenario(scenario,heldout,values)
        print(f'Evaluated models on {scenario}',flush=True)
    p.write_csv(out/'metrics.csv',metrics)
    p.write_csv(out/'activity_errors.csv',activity_errors)
    p.write_csv(out/'error_windows.csv',error_windows)
    p.write_csv(out/'rotation_checks.csv',robustness)
    for entry in metrics:
        relevant=[r for r in activity_errors if r['scenario']==entry['scenario'] and r['split']==entry['split'] and r['model']==entry['model']]
        assert sum(r['errors'] for r in relevant if r['error_type']=='FP')==entry['fp']
        assert sum(r['errors'] for r in relevant if r['error_type']=='FN')==entry['fn']
    for row in read_csv(paper_run/'metrics.csv'):
        if row['feature'] in ('baseline','C2','C8'):
            current=original_metrics[row['split'],row['feature']]
            for key in METRICS:
                assert current[key]==float(row[key])
    assert all(pf.sha(Path(path))==digest for path,digest in hashes.items())
    provenance=dict(input_sha256=hashes,inputs_unchanged=True,feature_names=FEATURE_NAMES,
        training_windows=int(train.sum()),validation_windows=int(val.sum()),test_windows=int((partitions=='test').sum()),
        model_selection='validation only; StandardScaler fit on training only; no train+validation refit',
        rotations=dict(replicates=args.rotations,seed=args.seed,per_window_constant=True,models_score_canonicalization=False),
        python_code_sha256=pf.sha(Path(__file__)),
        source_feature_code_sha256=pf.sha(Path(of.__file__)),
        sklearn_version=__import__('sklearn').__version__,numpy_version=np.__version__,joblib_version=joblib.__version__,
        verified_export_parity=True,baseline_metrics_unchanged=True)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    make_report(out,parameters,metrics,activity_errors,robustness)
    print(json.dumps([r for r in metrics if r['scenario']=='original' and r['split']=='test'],indent=2))


def make_report(out,parameters,metrics,activities,robustness):
    names=('baseline','C2','C8','logistic_regression','decision_tree')
    report=['# Small orientation-robust classifiers',
        'Both model families were fit on training subjects only. Hyperparameters and decision thresholds were selected on unrotated validation only, maximizing balanced accuracy. Neither model was refit on train+validation. The fixed test subjects were evaluated only after models/thresholds were locked. No neural network was trained.',
        '## Exact features and fitted models',
        'The predeclared 21 candidates are: '+', '.join('`'+name+'`' for name in FEATURE_NAMES)+'. No raw C2/C8 or other fixed-axis features enter either model. `ga_C2` and `ga_C8` are gravity-aligned new features, not the paper fixed-plane scores.',
        'Logistic regression uses all 21 features with training-only standardization and L2 regularization. Both families use training-derived balanced class weights. Scores are ranking scores, not established calibrated probabilities.',
        f"Selected logistic candidate: **{parameters['logistic_regression']['candidate']}**, threshold {parameters['logistic_regression']['probability_threshold']:.10g}.",
        f"Selected tree: **{parameters['decision_tree']['candidate']}**, actual depth {parameters['decision_tree']['actual_depth']}, "
        f"{parameters['decision_tree']['node_count']} nodes / {parameters['decision_tree']['leaf_count']} leaves, threshold {parameters['decision_tree']['probability_threshold']:.10g}. "
        'Features actually used in splits: '+', '.join(parameters['decision_tree']['features_used'])+'.',
        'See `validation_candidates.csv` for every tried setting. Grid: LR C={0.01,0.1,1,10}; tree depth={2,3,4}, minimum leaf windows={200,50}; fixed random seed 473. Exact BA ties preserve grid order (stronger LR regularization; shallower tree then larger leaf). Threshold ties choose the smallest threshold. Validation results are selection scores, not independent generalization estimates.']
    for partition in ('validation','test'):
        table=[]
        for name in names:
            r=next(r for r in metrics if r['scenario']=='original' and r['split']==partition and r['model']==name)
            table.append(dict(model=name,**{key:f"{r[key]:.2%}" for key in METRICS},FP=r['fp'],FN=r['fn']))
        report+=['## '+partition.capitalize(),markdown_table(table,list(table[0]))]
    report+=['## Random-rotation replication',
        'Uniform SO(3), ten default seeds 473–482, independent rotation per window held constant over its 200 samples; accelerometer, gyroscope and gravity history transformed consistently. Gravity is recomputed from rotated raw samples and prior state. Model feature values/scores are NOT replaced with original values; floating-point changes and changed decisions are reported. Baseline/C2/C8 comparators use their original thresholds.']
    for partition in ('validation','test'):
        table=[]
        for name in names:
            entries=[r for r in metrics if r['scenario'].startswith('random') and r['split']==partition and r['model']==name]
            row=dict(model=name)
            for key in METRICS:
                values=np.array([r[key] for r in entries])
                row[key]=f'{values.mean():.2%} [{values.min():.2%}, {values.max():.2%}]'
            table.append(row)
        report+=['### '+partition,markdown_table(table,list(table[0]))]
    report.append('Changed model predictions across rotation replicates: '+str(sum(r['changed_predictions'] for r in robustness))+
                  '. Tiny threshold-tie changes, if any, are retained rather than hidden; inspect rotation_checks.csv. Replicate ranges reflect rotations, not subject-level confidence intervals.')
    for partition in ('validation','test'):
        report+=['## '+partition+' activity failure rates','Cells show errors/total (rate); D=ADL false positives, F=fall false negatives.']
        codes=sorted({r['activity_code'] for r in activities if r['split']==partition})
        table=[]
        for code in codes:
            row=dict(activity=code)
            for name in names:
                r=next(r for r in activities if r['scenario']=='original' and r['split']==partition and r['model']==name and r['activity_code']==code)
                row[name]=f"{r['errors']}/{r['total']} ({r['rate']:.1%})"
            table.append(row)
        report.append(markdown_table(table,list(table[0])))
    report+=['## Reproduce and inspect',
        '`python outputs/sisfall/robust_classifiers.py --out outputs/robust_classifiers_rerun` from the project root. Existing output directories are refused. Run `python outputs/sisfall/test_robust_classifiers.py` for focused model/export checks.',
        '`model_parameters.json` contains scaler, coefficients, raw-feature fused weights, decision threshold and tree arrays. `decision_tree.txt` aids inspection but its printed default classes are not the tuned leaf-score rule. `.joblib` files preserve sklearn models; only load trusted files. `features_all.npy` aligns with the unchanged source windows.csv.',
        '`metrics.csv` includes all original/rotation metrics and deltas; `activity_errors.csv` covers every activity/scenario; `error_windows.csv` identifies original validation/test failures; `original_predictions.npz` and rotation prediction files retain scores; provenance records input hashes and exact preservation checks.',
        'Feature definitions, gravity estimator and filter conventions remain those in [ORIENTATION_METHODS.md](../sisfall/ORIENTATION_METHODS.md). Pre/post variance means first/last window half, not independently annotated event phases. Existing offline filtering and peak-selected event windows remain unchanged: this is not a causal wrist detector. Synthetic constant frame rotations do not reproduce wrist motion or time-varying orientation. MCU estimates are in MCU.md.']
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=p.PROJECT/'processed/baseline_v2')
    parser.add_argument('--orientation',type=Path,default=p.PROJECT/'outputs/orientation_study')
    parser.add_argument('--paper-run',type=Path,default=p.PROJECT/'outputs/paper_feature_comparison')
    parser.add_argument('--out',type=Path,default=p.PROJECT/'outputs/robust_classifiers')
    parser.add_argument('--rotations',type=int,default=10)
    parser.add_argument('--seed',type=int,default=473)
    args=parser.parse_args()
    if args.rotations<1:
        parser.error('At least one rotation replicate required')
    run(args)
