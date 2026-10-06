"""Prune only by locked thresholded leaf decisions; never fit or tune."""
import csv
import itertools
import json
from functools import lru_cache
from pathlib import Path

import robust_classifiers as r
import numpy as np
import joblib


def prune(parameters, node=0):
    if parameters['children_left'][node] == -1:
        return dict(prediction=int(parameters['positive_score'][node] >= parameters['probability_threshold']),
                    original_leaves=[node])
    left=prune(parameters,parameters['children_left'][node])
    right=prune(parameters,parameters['children_right'][node])
    if 'prediction' in left and 'prediction' in right and left['prediction']==right['prediction']:
        return dict(prediction=left['prediction'],original_leaves=left['original_leaves']+right['original_leaves'])
    return dict(original_node=node,feature_index=parameters['feature'][node],
                feature=parameters['candidate_features'][parameters['feature'][node]],
                threshold=parameters['threshold'][node],left=left,right=right)


def predict(tree, x):
    x=np.asarray(x,dtype=np.float32)
    result=np.empty(len(x),dtype=np.uint8)
    for i,row in enumerate(x):
        node=tree
        while 'prediction' not in node:
            node=node['left' if row[node['feature_index']] <= node['threshold'] else 'right']
        result[i]=node['prediction']
    return result


def size(tree):
    if 'prediction' in tree:
        return 0,1,1,set()
    a,b=size(tree['left']),size(tree['right'])
    return 1+max(a[0],b[0]),1+a[1]+b[1],a[2]+b[2],a[3]|b[3]|{tree['feature']}


def leaves(parameters,node):
    if parameters['children_left'][node]==-1:
        return [node]
    return leaves(parameters,parameters['children_left'][node])+leaves(parameters,parameters['children_right'][node])


def rules(tree,depth=0):
    indent='    '*depth
    if 'prediction' in tree:
        return [indent+f"return {tree['prediction']}  # {'FALL' if tree['prediction'] else 'ADL'}"]
    return ([indent+f"if {tree['feature']} <= {tree['threshold']!r}:"]+rules(tree['left'],depth+1)+
            [indent+'else:']+rules(tree['right'],depth+1))


def main():
    root=r.p.PROJECT
    source=root/'outputs/robust_classifiers'
    out=r.p.safe_destination(root/'outputs/simplified_decision_tree',root/'data')
    inputs=[source/name for name in ('model_parameters.json','decision_tree.joblib','features_all.npy','original_predictions.npz')]
    labels_path=root/'processed/baseline_v2/labels_groups_splits.npz'
    inputs.append(labels_path)
    hashes={str(path):r.pf.sha(path) for path in inputs}
    parameters=json.loads((source/'model_parameters.json').read_text())['decision_tree']
    assert parameters['probability_threshold']==0.7705231308225133
    bundle=joblib.load(source/'decision_tree.joblib')
    model=bundle['model']
    assert bundle['threshold']==parameters['probability_threshold']
    assert list(bundle['features'])==parameters['candidate_features']
    np.testing.assert_array_equal(model.tree_.children_left,parameters['children_left'])
    np.testing.assert_array_equal(model.tree_.children_right,parameters['children_right'])
    np.testing.assert_array_equal(model.tree_.feature,parameters['feature'])
    np.testing.assert_array_equal(model.tree_.threshold,parameters['threshold'])
    np.testing.assert_array_equal(model.tree_.value[:,0,1],parameters['positive_score'])
    X=np.load(source/'features_all.npy',mmap_mode='r')
    labels=np.load(labels_path)
    train=labels['split']=='train'
    assignments=model.apply(X[train])
    y_train=labels['y'][train]
    audit=[]
    for node in leaves(parameters,0):
        mask=assignments==node
        count=int(mask.sum())
        assert count==int(model.tree_.n_node_samples[node])
        positives=int(y_train[mask].sum())
        audit.append(dict(leaf_id=node,positive_score=parameters['positive_score'][node],
            deployed_decision=int(parameters['positive_score'][node]>=parameters['probability_threshold']),
            training_samples=count,training_positive_samples=positives,training_negative_samples=count-positives,
            unweighted_positive_fraction=positives/count,
            weighted_training_samples=float(model.tree_.weighted_n_node_samples[node])))
    internal=[]
    for node in range(model.tree_.node_count):
        if parameters['children_left'][node]==-1:
            continue
        descendants=leaves(parameters,node)
        decisions=sorted({int(parameters['positive_score'][leaf]>=parameters['probability_threshold']) for leaf in descendants})
        internal.append(dict(original_node=node,descendant_leaves=descendants,decisions=decisions,prunable=len(decisions)==1))
    tree=prune(parameters)
    depth,nodes,leaf_count,used=size(tree)
    assert (depth,nodes,leaf_count)==(3,7,4)
    original=(r.exported_tree_scores(parameters,X)>=parameters['probability_threshold']).astype(np.uint8)
    simplified=predict(tree,X)
    sklearn_locked=(model.predict_proba(X)[:,1]>=parameters['probability_threshold']).astype(np.uint8)
    saved=(np.load(source/'original_predictions.npz')['decision_tree']>=parameters['probability_threshold']).astype(np.uint8)
    np.testing.assert_array_equal(original,simplified)
    np.testing.assert_array_equal(original,sklearn_locked)
    np.testing.assert_array_equal(original,saved)
    verification=[]
    for partition in ('train','validation','test'):
        mask=labels['split']==partition
        verification.append(dict(split=partition,rows=int(mask.sum()),mismatches=int(np.sum(original[mask]!=simplified[mask])),
                                 uint8_bytes_identical=original[mask].tobytes()==simplified[mask].tobytes()))
    # Boundary regression against the CURRENT exported predictor's float32 scalar semantics.
    probes=[]
    retained=[0,8,12]
    levels=[]
    for node in retained:
        t=np.float32(parameters['threshold'][node])
        levels.append([np.nextafter(t,np.float32(-np.inf)),t,np.nextafter(t,np.float32(np.inf))])
    for combination in itertools.product(*levels):
        row=np.zeros(21,dtype=np.float32)
        for node,value in zip(retained,combination):
            row[parameters['feature'][node]]=value
        probes.append(row)
    probes=np.array(probes)
    np.testing.assert_array_equal(predict(tree,probes),
        (r.exported_tree_scores(parameters,probes)>=parameters['probability_threshold']).astype(np.uint8))
    # All 8 boolean regions; derive labels from the original model, not any dataset label.
    truth=[]
    for bits in itertools.product((0,1),repeat=3):
        row=np.zeros(21)
        for node,bit in zip(retained,bits):
            row[parameters['feature'][node]]=parameters['threshold'][node]+(-1 if bit==0 else 1)
        value=int(r.exported_tree_scores(parameters,row[None])[0]>=parameters['probability_threshold'])
        assert value==int(bool(bits[0]) and (not bits[1] or bool(bits[2])))
        truth.append((bits,value))
    @lru_cache(None)
    def minimal(indices):
        if len({truth[i][1] for i in indices})==1:
            return 0,0  # internal-node minimum, depth minimum
        choices=[]
        for feature in range(3):
            left=tuple(i for i in indices if truth[i][0][feature]==0)
            right=tuple(i for i in indices if truth[i][0][feature]==1)
            if left and right:
                a,b=minimal(left),minimal(right)
                choices.append((1+a[0]+b[0],1+max(a[1],b[1])))
        return min(v[0] for v in choices),min(v[1] for v in choices)
    assert minimal(tuple(range(8)))==(3,3)
    assert all(r.pf.sha(Path(path))==digest for path,digest in hashes.items())
    out.mkdir(parents=True)
    r.p.write_csv(out/'leaf_audit.csv',audit)
    r.p.write_csv(out/'pruning_audit.csv',internal)
    r.p.write_csv(out/'verification.csv',verification)
    export=dict(kind='binary decision tree; does not preserve probability scores',
        original_positive_score_threshold=parameters['probability_threshold'],input_features=parameters['candidate_features'],
        required_features=sorted(used),input_dtype='float32',
        comparison_semantics='NumPy float32 scalar <= Python float, matching current exported_tree_scores; not a float32 firmware validation',
        depth=depth,node_count=nodes,leaf_count=leaf_count,max_comparisons=depth,tree=tree)
    (out/'simplified_tree.json').write_text(json.dumps(export,indent=2))
    (out/'simplified_tree.txt').write_text('\n'.join(rules(tree))+'\n')
    predictor='''"""Binary-only export. Preserve locked Python comparison semantics."""
import json
from pathlib import Path
import numpy as np

def predict(features):
    """Features must use the original 21-column order; only indices 5/15/18 are read."""
    tree=json.loads(Path(__file__).with_name("simplified_tree.json").read_text())["tree"]
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or features.shape[1]!=21 or not np.isfinite(features).all():
        raise ValueError("Expected finite (N,21) feature matrix")
    result=np.empty(len(features),dtype=np.uint8)
    for i,row in enumerate(features):
        node=tree
        while "prediction" not in node:
            node=node["left" if row[node["feature_index"]] <= node["threshold"] else "right"]
        result[i]=node["prediction"]
    return result
'''
    (out/'predict_simplified.py').write_text(predictor)
    # Execute the exported source directly without import-generated __pycache__ files.
    namespace={'__file__':str(out/'predict_simplified.py')}
    exec(compile(predictor,str(out/'predict_simplified.py'),'exec'),namespace)
    np.testing.assert_array_equal(namespace['predict'](X),original)
    np.savez(out/'prediction_equivalence.npz',window_id=np.arange(len(X)),original=original,simplified=simplified,split=labels['split'])
    summary=dict(source_sha256=hashes,source_files_unchanged=True,locked_threshold=parameters['probability_threshold'],
        leaves= audit,pruning=internal,structure=dict(depth=depth,nodes=nodes,leaves=leaf_count,features=sorted(used),max_comparisons=depth),
        verification=verification,boundary_cases_verified=len(probes),exported_predictor_verified=True,
        minimum_axis_aligned_internal_nodes=3,minimum_depth=3,
        numpy_version=np.__version__,sklearn_version=__import__('sklearn').__version__)
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    report=['# Locked tree binary simplification',
        'No training or tuning. Labels are derived only from the locked positive-score threshold 0.7705231308225133 (>= means fall). Source artifacts remain unchanged.',
        'Leaf scores are class-weighted proportions from class_weight=balanced, not unweighted fractions or calibrated fall probabilities. Training counts are unweighted n_node_samples, independently checked by routing only training rows.',
        r.markdown_table(audit,['leaf_id','positive_score','deployed_decision','training_samples','training_positive_samples','training_negative_samples','unweighted_positive_fraction']),
        'Nodes 1, 2, 5 and 9 have uniform descendant binary decisions. Maximal collapsed subtrees are node 1 (leaves 3/4/6/7 all ADL) and node 9 (leaves 10/11 both FALL). Node 1 subsumes nodes 2 and 5. Node 0, 8 and 12 remain.',
        '```text\n'+'\n'.join(rules(tree))+'\n```',
        'Final depth 3; 7 nodes; 4 leaves; 3 features: ga_C2 (original index 15), jerk_abs_mean (5), ga_parallel_peak (18). At most 3 split comparisons; no runtime leaf probability comparison. jerk_rms and jerk_abs_peak are no longer required for binary decisions.',
        'This is minimal among axis-aligned binary trees for the original finite-input decision function. It depends on three independent predicates A=(ga_C2>t0), B=(jerk_abs_mean>t8), C=(ga_parallel_peak>t12), with output A AND (NOT B OR C). Each variable is essential, requiring at least 3 internal nodes / 7 total nodes; exhaustive recursion over the 8 predicate regions confirms minimum depth 3. No test labels or feature values determine pruning or minimization.',
        r.markdown_table(verification,list(verification[0])),
        'Comparison references: original JSON exported predictor, locked sklearn predict_proba thresholded explicitly (never sklearn predict), and saved original probability predictions. All uint8 prediction arrays match exactly; the serialized new predictor also matches. Boundary probes around each retained float32 cut verify equivalence to the current export.',
        'Only binary decisions are preserved. Leaf scores, ranking and PR-AUC are not preserved by replacing score leaves with labels. Keep the original model for probability-score analysis. Full-precision thresholds are retained; the text export is not a newly quantized C implementation. Embedded float32 comparison/threshold conversion remains a separate task.',
        'Run `python outputs/sisfall/simplify_locked_tree.py` from the project root in a fresh output location (the script refuses to overwrite an existing output). Exported files are under outputs/simplified_decision_tree/.']
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(dict(leaves=audit,structure=summary['structure'],verification=verification),indent=2))


if __name__=='__main__':
    main()
