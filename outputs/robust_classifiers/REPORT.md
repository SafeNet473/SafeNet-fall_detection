# Small orientation-robust classifiers

Both model families were fit on training subjects only. Hyperparameters and decision thresholds were selected on unrotated validation only, maximizing balanced accuracy. Neither model was refit on train+validation. The fixed test subjects were evaluated only after models/thresholds were locked. No neural network was trained.

## Exact features and fitted models

The predeclared 21 candidates are: `mag_mean`, `mag_std`, `mag_rms`, `mag_peak`, `mag_ptp`, `jerk_abs_mean`, `jerk_std`, `jerk_rms`, `jerk_abs_peak`, `pre_variance`, `post_variance`, `gyro_mean`, `gyro_std`, `gyro_rms`, `gyro_peak`, `ga_C2`, `ga_C8`, `ga_C13`, `ga_parallel_peak`, `ga_parallel_std`, `ga_parallel_ptp`. No raw C2/C8 or other fixed-axis features enter either model. `ga_C2` and `ga_C8` are gravity-aligned new features, not the paper fixed-plane scores.

Logistic regression uses all 21 features with training-only standardization and L2 regularization. Both families use training-derived balanced class weights. Scores are ranking scores, not established calibrated probabilities.

Selected logistic candidate: **lr_C_0.1**, threshold 0.8151134927.

Selected tree: **tree_depth_3_leaf_50**, actual depth 3, 15 nodes / 8 leaves, threshold 0.7705231308. Features actually used in splits: jerk_abs_mean, jerk_rms, jerk_abs_peak, ga_C2, ga_parallel_peak.

See `validation_candidates.csv` for every tried setting. Grid: LR C={0.01,0.1,1,10}; tree depth={2,3,4}, minimum leaf windows={200,50}; fixed random seed 473. Exact BA ties preserve grid order (stronger LR regularization; shallower tree then larger leaf). Threshold ties choose the smallest threshold. Validation results are selection scores, not independent generalization estimates.

## Validation

| model | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal | average_precision | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 93.07% | 90.52% | 91.79% | 14.64% | 25.30% | 58.29% | 58.32% | 2035 | 26 |
| C2 | 98.67% | 98.33% | 98.50% | 50.82% | 67.09% | 97.05% | 97.06% | 358 | 5 |
| C8 | 98.67% | 98.42% | 98.54% | 52.19% | 68.27% | 92.72% | 92.72% | 339 | 5 |
| logistic_regression | 99.47% | 99.83% | 99.65% | 90.98% | 95.03% | 99.56% | 99.56% | 37 | 2 |
| decision_tree | 99.73% | 99.53% | 99.63% | 78.74% | 88.00% | 97.79% | 95.78% | 101 | 1 |

## Test

| model | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal | average_precision | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 90.40% | 89.53% | 89.96% | 12.87% | 22.52% | 61.96% | 61.98% | 2296 | 36 |
| C2 | 95.20% | 95.26% | 95.23% | 25.55% | 40.29% | 92.00% | 92.00% | 1040 | 18 |
| C8 | 95.73% | 98.26% | 97.00% | 48.45% | 64.34% | 95.53% | 95.53% | 382 | 16 |
| logistic_regression | 97.07% | 99.95% | 98.51% | 96.81% | 96.94% | 98.44% | 98.45% | 12 | 11 |
| decision_tree | 96.53% | 99.88% | 98.21% | 93.30% | 94.89% | 96.29% | 94.27% | 26 | 13 |

## Random-rotation replication

Uniform SO(3), ten default seeds 473–482, independent rotation per window held constant over its 200 samples; accelerometer, gyroscope and gravity history transformed consistently. Gravity is recomputed from rotated raw samples and prior state. Model feature values/scores are NOT replaced with original values; floating-point changes and changed decisions are reported. Baseline/C2/C8 comparators use their original thresholds.

### validation

| model | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 93.07% [93.07%, 93.07%] | 90.52% [90.52%, 90.52%] | 91.79% [91.79%, 91.79%] | 14.64% [14.64%, 14.64%] | 25.30% [25.30%, 25.30%] | 58.29% [58.29%, 58.29%] | 58.32% [58.32%, 58.32%] |
| C2 | 96.24% [95.47%, 97.07%] | 72.60% [72.38%, 72.87%] | 84.42% [84.10%, 84.75%] | 5.78% [5.74%, 5.82%] | 10.91% [10.83%, 10.98%] | 42.48% [38.46%, 44.68%] | 42.53% [38.51%, 44.73%] |
| C8 | 98.21% [97.60%, 98.93%] | 84.36% [84.24%, 84.47%] | 91.29% [90.94%, 91.64%] | 9.89% [9.79%, 9.95%] | 17.97% [17.79%, 18.08%] | 20.41% [19.50%, 21.05%] | 20.57% [19.66%, 21.24%] |
| logistic_regression | 99.47% [99.47%, 99.47%] | 99.83% [99.83%, 99.83%] | 99.65% [99.65%, 99.65%] | 90.98% [90.98%, 90.98%] | 95.03% [95.03%, 95.03%] | 99.56% [99.56%, 99.56%] | 99.56% [99.56%, 99.56%] |
| decision_tree | 99.73% [99.73%, 99.73%] | 99.53% [99.53%, 99.53%] | 99.63% [99.63%, 99.63%] | 78.74% [78.74%, 78.74%] | 88.00% [88.00%, 88.00%] | 97.79% [97.79%, 97.79%] | 95.78% [95.78%, 95.78%] |

### test

| model | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 90.40% [90.40%, 90.40%] | 89.53% [89.53%, 89.53%] | 89.96% [89.96%, 89.96%] | 12.87% [12.87%, 12.87%] | 22.52% [22.52%, 22.52%] | 61.96% [61.96%, 61.96%] | 61.98% [61.98%, 61.98%] |
| C2 | 94.85% [93.33%, 96.53%] | 71.99% [71.76%, 72.29%] | 83.42% [82.74%, 84.28%] | 5.48% [5.41%, 5.57%] | 10.35% [10.24%, 10.54%] | 44.83% [38.03%, 48.02%] | 44.86% [38.07%, 48.06%] |
| C8 | 97.04% [95.47%, 98.93%] | 84.68% [84.50%, 84.85%] | 90.86% [90.05%, 91.79%] | 9.77% [9.60%, 9.93%] | 17.76% [17.45%, 18.05%] | 26.70% [23.74%, 30.67%] | 26.76% [23.81%, 30.72%] |
| logistic_regression | 97.07% [97.07%, 97.07%] | 99.95% [99.95%, 99.95%] | 98.51% [98.51%, 98.51%] | 96.81% [96.81%, 96.81%] | 96.94% [96.94%, 96.94%] | 98.44% [98.44%, 98.44%] | 98.45% [98.45%, 98.45%] |
| decision_tree | 96.53% [96.53%, 96.53%] | 99.88% [99.88%, 99.88%] | 98.21% [98.21%, 98.21%] | 93.30% [93.30%, 93.30%] | 94.89% [94.89%, 94.89%] | 96.29% [96.29%, 96.29%] | 94.27% [94.27%, 94.27%] |

Changed model predictions across rotation replicates: 0. Tiny threshold-tie changes, if any, are retained rather than hidden; inspect rotation_checks.csv. Replicate ranges reflect rotations, not subject-level confidence intervals.

## validation activity failure rates

Cells show errors/total (rate); D=ADL false positives, F=fall false negatives.

| activity | baseline | C2 | C8 | logistic_regression | decision_tree |
| --- | --- | --- | --- | --- | --- |
| D01 | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) |
| D02 | 0/1589 (0.0%) | 19/1589 (1.2%) | 0/1589 (0.0%) | 0/1589 (0.0%) | 0/1589 (0.0%) |
| D03 | 345/1592 (21.7%) | 13/1592 (0.8%) | 1/1592 (0.1%) | 1/1592 (0.1%) | 2/1592 (0.1%) |
| D04 | 1203/1592 (75.6%) | 56/1592 (3.5%) | 226/1592 (14.2%) | 0/1592 (0.0%) | 47/1592 (3.0%) |
| D05 | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) |
| D06 | 269/1219 (22.1%) | 0/1219 (0.0%) | 1/1219 (0.1%) | 1/1219 (0.1%) | 5/1219 (0.4%) |
| D07 | 0/915 (0.0%) | 0/915 (0.0%) | 0/915 (0.0%) | 0/915 (0.0%) | 0/915 (0.0%) |
| D08 | 0/917 (0.0%) | 20/917 (2.2%) | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) |
| D09 | 0/918 (0.0%) | 20/918 (2.2%) | 0/918 (0.0%) | 0/918 (0.0%) | 0/918 (0.0%) |
| D10 | 0/919 (0.0%) | 30/919 (3.3%) | 2/919 (0.2%) | 0/919 (0.0%) | 0/919 (0.0%) |
| D11 | 13/913 (1.4%) | 2/913 (0.2%) | 0/913 (0.0%) | 1/913 (0.1%) | 5/913 (0.5%) |
| D12 | 0/920 (0.0%) | 4/920 (0.4%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) |
| D13 | 0/574 (0.0%) | 43/574 (7.5%) | 28/574 (4.9%) | 13/574 (2.3%) | 15/574 (2.6%) |
| D14 | 0/920 (0.0%) | 90/920 (9.8%) | 54/920 (5.9%) | 12/920 (1.3%) | 12/920 (1.3%) |
| D15 | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) |
| D16 | 0/912 (0.0%) | 0/912 (0.0%) | 0/912 (0.0%) | 0/912 (0.0%) | 0/912 (0.0%) |
| D17 | 0/1948 (0.0%) | 0/1948 (0.0%) | 0/1948 (0.0%) | 7/1948 (0.4%) | 0/1948 (0.0%) |
| D18 | 52/573 (9.1%) | 44/573 (7.7%) | 25/573 (4.4%) | 0/573 (0.0%) | 11/573 (1.9%) |
| D19 | 153/575 (26.6%) | 17/575 (3.0%) | 2/575 (0.3%) | 2/575 (0.3%) | 4/575 (0.7%) |
| F01 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F02 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F03 | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) |
| F04 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F05 | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |
| F06 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F07 | 4/25 (16.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F08 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F09 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F10 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F11 | 2/25 (8.0%) | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F12 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F13 | 10/25 (40.0%) | 1/25 (4.0%) | 4/25 (16.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F14 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F15 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |

## test activity failure rates

Cells show errors/total (rate); D=ADL false positives, F=fall false negatives.

| activity | baseline | C2 | C8 | logistic_regression | decision_tree |
| --- | --- | --- | --- | --- | --- |
| D01 | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) |
| D02 | 0/1752 (0.0%) | 0/1752 (0.0%) | 0/1752 (0.0%) | 0/1752 (0.0%) | 0/1752 (0.0%) |
| D03 | 427/1751 (24.4%) | 304/1751 (17.4%) | 8/1751 (0.5%) | 0/1751 (0.0%) | 0/1751 (0.0%) |
| D04 | 1476/1751 (84.3%) | 605/1751 (34.6%) | 305/1751 (17.4%) | 0/1751 (0.0%) | 2/1751 (0.1%) |
| D05 | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) |
| D06 | 193/1174 (16.4%) | 13/1174 (1.1%) | 18/1174 (1.5%) | 0/1174 (0.0%) | 0/1174 (0.0%) |
| D07 | 0/916 (0.0%) | 0/916 (0.0%) | 0/916 (0.0%) | 0/916 (0.0%) | 0/916 (0.0%) |
| D08 | 4/920 (0.4%) | 2/920 (0.2%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) |
| D09 | 0/920 (0.0%) | 2/920 (0.2%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) |
| D10 | 0/919 (0.0%) | 10/919 (1.1%) | 4/919 (0.4%) | 0/919 (0.0%) | 0/919 (0.0%) |
| D11 | 8/900 (0.9%) | 0/900 (0.0%) | 0/900 (0.0%) | 2/900 (0.2%) | 2/900 (0.2%) |
| D12 | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) |
| D13 | 0/574 (0.0%) | 25/574 (4.4%) | 16/574 (2.8%) | 8/574 (1.4%) | 12/574 (2.1%) |
| D14 | 0/917 (0.0%) | 47/917 (5.1%) | 13/917 (1.4%) | 1/917 (0.1%) | 0/917 (0.0%) |
| D15 | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) |
| D16 | 0/894 (0.0%) | 0/894 (0.0%) | 0/894 (0.0%) | 0/894 (0.0%) | 0/894 (0.0%) |
| D17 | 0/1870 (0.0%) | 0/1870 (0.0%) | 0/1870 (0.0%) | 0/1870 (0.0%) | 0/1870 (0.0%) |
| D18 | 37/574 (6.4%) | 32/574 (5.6%) | 18/574 (3.1%) | 0/574 (0.0%) | 10/574 (1.7%) |
| D19 | 151/574 (26.3%) | 0/574 (0.0%) | 0/574 (0.0%) | 1/574 (0.2%) | 0/574 (0.0%) |
| F01 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F02 | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 1/25 (4.0%) |
| F03 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F04 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F05 | 0/25 (0.0%) | 5/25 (20.0%) | 5/25 (20.0%) | 5/25 (20.0%) | 5/25 (20.0%) |
| F06 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F07 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F08 | 4/25 (16.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) |
| F09 | 4/25 (16.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F10 | 4/25 (16.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |
| F11 | 2/25 (8.0%) | 11/25 (44.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 2/25 (8.0%) |
| F12 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F13 | 6/25 (24.0%) | 1/25 (4.0%) | 1/25 (4.0%) | 2/25 (8.0%) | 2/25 (8.0%) |
| F14 | 5/25 (20.0%) | 0/25 (0.0%) | 6/25 (24.0%) | 1/25 (4.0%) | 2/25 (8.0%) |
| F15 | 6/25 (24.0%) | 0/25 (0.0%) | 2/25 (8.0%) | 1/25 (4.0%) | 0/25 (0.0%) |

## Reproduce and inspect

`python outputs/sisfall/robust_classifiers.py --out outputs/robust_classifiers_rerun` from the project root. Existing output directories are refused. Run `python outputs/sisfall/test_robust_classifiers.py` for focused model/export checks.

`model_parameters.json` contains scaler, coefficients, raw-feature fused weights, decision threshold and tree arrays. `decision_tree.txt` aids inspection but its printed default classes are not the tuned leaf-score rule. `.joblib` files preserve sklearn models; only load trusted files. `features_all.npy` aligns with the unchanged source windows.csv.

`metrics.csv` includes all original/rotation metrics and deltas; `activity_errors.csv` covers every activity/scenario; `error_windows.csv` identifies original validation/test failures; `original_predictions.npz` and rotation prediction files retain scores; provenance records input hashes and exact preservation checks.

Feature definitions, gravity estimator and filter conventions remain those in [ORIENTATION_METHODS.md](../sisfall/ORIENTATION_METHODS.md). Pre/post variance means first/last window half, not independently annotated event phases. Existing offline filtering and peak-selected event windows remain unchanged: this is not a causal wrist detector. Synthetic constant frame rotations do not reproduce wrist motion or time-varying orientation. MCU estimates are in MCU.md.
