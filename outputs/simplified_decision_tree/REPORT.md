# Locked tree binary simplification

No training or tuning. Labels are derived only from the locked positive-score threshold 0.7705231308225133 (>= means fall). Source artifacts remain unchanged.

Leaf scores are class-weighted proportions from class_weight=balanced, not unweighted fractions or calibrated fall probabilities. Training counts are unweighted n_node_samples, independently checked by routing only training rows.

| leaf_id | positive_score | deployed_decision | training_samples | training_positive_samples | training_negative_samples | unweighted_positive_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.0029779867376308294 | 0 | 55649 | 3 | 55646 | 5.3909324516163814e-05 |
| 4 | 0.38905640654525 | 0 | 88 | 1 | 87 | 0.011363636363636364 |
| 6 | 0.7168010490882202 | 0 | 206 | 9 | 197 | 0.043689320388349516 |
| 7 | 0.0 | 0 | 1095 | 0 | 1095 | 0.0 |
| 10 | 0.7705231308225133 | 1 | 210 | 12 | 198 | 0.05714285714285714 |
| 11 | 0.9977003328440758 | 1 | 1148 | 1018 | 130 | 0.8867595818815331 |
| 13 | 0.0 | 0 | 664 | 0 | 664 | 0.0 |
| 14 | 0.8602542448217624 | 1 | 50 | 5 | 45 | 0.1 |

Nodes 1, 2, 5 and 9 have uniform descendant binary decisions. Maximal collapsed subtrees are node 1 (leaves 3/4/6/7 all ADL) and node 9 (leaves 10/11 both FALL). Node 1 subsumes nodes 2 and 5. Node 0, 8 and 12 remain.

```text
if ga_C2 <= 0.945275604724884:
    return 0  # ADL
else:
    if jerk_abs_mean <= 14.055817127227783:
        return 1  # FALL
    else:
        if ga_parallel_peak <= 3.3697245121002197:
            return 0  # ADL
        else:
            return 1  # FALL
```

Final depth 3; 7 nodes; 4 leaves; 3 features: ga_C2 (original index 15), jerk_abs_mean (5), ga_parallel_peak (18). At most 3 split comparisons; no runtime leaf probability comparison. jerk_rms and jerk_abs_peak are no longer required for binary decisions.

This is minimal among axis-aligned binary trees for the original finite-input decision function. It depends on three independent predicates A=(ga_C2>t0), B=(jerk_abs_mean>t8), C=(ga_parallel_peak>t12), with output A AND (NOT B OR C). Each variable is essential, requiring at least 3 internal nodes / 7 total nodes; exhaustive recursion over the 8 predicate regions confirms minimum depth 3. No test labels or feature values determine pruning or minimization.

| split | rows | mismatches | uint8_bytes_identical |
| --- | --- | --- | --- |
| train | 59110 | 0 | True |
| validation | 21831 | 0 | True |
| test | 22298 | 0 | True |

Comparison references: original JSON exported predictor, locked sklearn predict_proba thresholded explicitly (never sklearn predict), and saved original probability predictions. All uint8 prediction arrays match exactly; the serialized new predictor also matches. Boundary probes around each retained float32 cut verify equivalence to the current export.

Only binary decisions are preserved. Leaf scores, ranking and PR-AUC are not preserved by replacing score leaves with labels. Keep the original model for probability-score analysis. Full-precision thresholds are retained; the text export is not a newly quantized C implementation. Embedded float32 comparison/threshold conversion remains a separate task.

Run `python outputs/sisfall/simplify_locked_tree.py` from the project root in a fresh output location (the script refuses to overwrite an existing output). Exported files are under outputs/simplified_decision_tree/.
