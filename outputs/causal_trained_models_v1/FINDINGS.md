# Three-feature causal training findings

Both models were trained from scratch using training candidates only. Hyperparameters and score thresholds were selected on validation only. No test feature, event, or outcome file was opened. The offline reference and all previous experiment outputs remain unchanged.

The fixed feature order is `ga_C2`, `jerk_abs_mean`, `ga_parallel_peak`. The fixed preprocessing, trigger, and 300-sample event window are inherited from the completed causal-transfer diagnostic. Candidate recall is 98.67% for every model on validation: 370 of 375 fall trials had localized complete candidates.

| Validation model | Final sensitivity | ADL false alarms/hour | Event precision | Trial balanced accuracy |
|---|---:|---:|---:|---:|
| Frozen offline-trained tree | 84.80% | 12.39 | 87.85% | 90.56% |
| Causal-trained tree | 94.13% | 47.91 | 67.75% | 92.43% |
| Causal-trained logistic regression | 93.33% | 4.89 | 93.09% | 95.53% |

The new tree improves sensitivity by 9.33 percentage points but produces 147 ADL alarms versus 38 for the frozen tree, over 3.06794 ADL hours. The selection objective uses ADL **trial** specificity, so multiple alarms within one ADL trial are penalized only once by balanced accuracy; false alarms/hour exposes the resulting cost. This objective was fixed before fitting and was not changed after observing the results.

The logistic reference has the more favorable validation tradeoff: 350 detected falls versus 318 for the frozen reference, and 15 ADL alarms versus 38. These are development results from a reused validation partition, not independent test estimates. No additional features were introduced or selected.

## Selected models

- Tree: maximum/actual depth 2, 7 nodes, 4 leaves, minimum leaf samples 100, training-balanced class weights; positive-probability threshold **0.971060451555281**. It was offered all three features, but its splits use only `ga_C2` and `jerk_abs_mean`. No pruning or further feature-selection experiment was performed.
- Logistic regression: training-only standardization, L2 regularization, **C=10**, training-balanced class weights; positive-probability threshold **0.8850929456953477**. It uses all three features.

## Failure modes

All models miss the same five falls without a localized candidate. Beyond those, the causal tree rejects all matched candidates in 17 fall trials; logistic regression does so in 20, versus 52 for the frozen tree.

F13/F14/F15 validation sensitivities are:

| Model | F13 | F14 | F15 |
|---|---:|---:|---:|
| Frozen offline tree | 64% | 88% | 84% |
| Causal tree | 96% | 96% | 88% |
| Causal logistic | 80% | 100% | 92% |

The causal tree introduces substantial jogging alarms: D03 and D04 rates are 108 and 297 alarms/hour. Logistic regression has zero alarms for both, and reduces D06/D18/D19 rates to approximately 5.76/24.00/84.00 per hour, compared with the frozen tree's 57.60/48.00/108.00. Activity-specific exposures are short, so these rates are descriptive rather than long-term deployment estimates.

## Verification and limits

The exported tree and logistic parameters reproduce every saved-model prediction on 12,253 training and 4,387 validation candidates, with zero mismatches. Synthetic duplicate-alarm/no-candidate cases and independently computed real event counts pass. The scaler mean was verified against training features only. Input hashes and frozen offline predictions are unchanged.

Tree leaf probabilities are already normalized in this sklearn version. Re-normalizing them can perturb a score tied to the selected threshold; the verified exporter uses the saved probabilities directly. No model or threshold was changed during verification.

Matching and training targets use the existing raw-magnitude impact proxy with a +/-1-second trigger tolerance, not annotated impact timestamps. Unmatched candidates in fall trials are treated as negative training examples under this convention. Event precision counts duplicate and unmatched fall-recording alarms as false positives. Trial BA and candidate BA are explicitly separated in [REPORT.md](REPORT.md); neither is an event true-negative rate.
