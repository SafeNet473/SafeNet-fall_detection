# Three-feature causal-trained models: validation only

New models were fitted from scratch on training subjects only. The frozen offline tree and all prior experiment outputs were preserved. No test outcomes, events, or features were opened. No features were added.

## Fixed pipeline and supervision

Stateful fourth-order 5 Hz Butterworth at 200 Hz; existing raw-acceleration 0.5 Hz gravity EMA; fixed trigger magnitude >=1.1 g OR absolute magnitude jerk >=5 g/s, rising edge with 1.5 s refractory; pre100_post200 window (300 samples, decision after 1 s). Only ga_C2, jerk_abs_mean, ga_parallel_peak in that order. The previous causal replay CSVs provide the exact emitted candidates, including causally available features and preserved window boundaries.

A candidate is a positive training example only in a fall trial with trigger within +/-1 s of its raw-magnitude impact proxy. Other candidates, including unmatched candidates in fall trials, are negatives. This is weak proxy supervision, not timestamp-annotated ground truth. Trials with no localized candidate remain in evaluation denominators. Multiple matching training candidates remain separate examples; evaluation credits at most one true positive per fall trial.

## Fitting and selection

Class weights are balanced from training candidate labels. Tree grid: depths 1/2/3 and minimum leaf samples 10/25/50/100. Logistic grid: C=0.01/0.1/1/10; training-only StandardScaler, liblinear L2. Both use random seed 473. All thresholds are selected from validation positive-score boundaries, including the no-alarm boundary. Neither estimator is refitted on validation.

Selection maximizes validation trial-level balanced accuracy, breaking ties by event precision, fewer ADL alarms/hour, then simpler model. Trial BA averages localized fall sensitivity and ADL-trial specificity (fraction of ADL recordings with no positive decision). Candidate BA uses the candidate proxy labels and excludes missed/non-triggered falls. Neither quantity is an event-level true-negative rate; true-negative events have no natural denominator.

Event precision = detected fall trials / all emitted positive decisions: unmatched fall-trial alarms and duplicates count as false positives. ADL false alarms/hour uses the full recorded ADL duration. The shared validation partition was previously used to select the trigger/window and is reused for model selection, so these are development results, not independent generalization estimates.

## Validation results

| model | candidate_recall | sensitivity | false_alarms_per_adl_hour | event_precision | trial_specificity | trial_balanced_accuracy | candidate_balanced_accuracy | depth | nodes |
|---|---|---|---|---|---|---|---|---|---|
| tree | 0.9867 | 0.9413 | 47.9149 | 0.6775 | 0.9073 | 0.9243 | 0.9476 | 2 | 7 |
| logistic | 0.9867 | 0.9333 | 4.8893 | 0.9309 | 0.9773 | 0.9553 | 0.9615 | N/A | N/A |
| offline_frozen | 0.9867 | 0.8480 | 12.3862 | 0.8785 | 0.9633 | 0.9056 | 0.9093 | 3 | 7 |

| model | candidate | threshold |
|---|---|---|
| tree | tree_d2_leaf100 | 0.9711 |
| logistic | logistic_C10 | 0.8851 |

## Comparison with frozen offline tree on identical causal candidates

| model | BA_delta | sensitivity_delta | substantial_BA_degradation | sensitivity_drop_5pp |
|---|---|---|---|---|
| tree | 0.0187 | 0.0933 | False | False |
| logistic | 0.0497 | 0.0853 | False | False |

Substantial degradation was predefined as >=5 percentage points lower trial BA; a >=5-point sensitivity drop is also reported separately. No automatic feature expansion is performed, regardless of the result. The comparator above uses identical causal windows; original zero-phase/oracle-window metrics are not a fair direct comparator for this training experiment.

## Per-activity validation failure modes

| model | activity | trials | candidate_recall | sensitivity | false_negative_rate | false_alarms_per_hour | false_positive_trial_rate | no_matched_candidate | matched_candidates_rejected |
|---|---|---|---|---|---|---|---|---|---|
| tree | D01 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D02 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D03 | 8 | N/A | N/A | N/A | 108.0000 | 0.2500 | 0 | 0 |
| tree | D04 | 8 | N/A | N/A | N/A | 297.0000 | 0.6250 | 0 | 0 |
| tree | D05 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D06 | 25 | N/A | N/A | N/A | 86.4041 | 0.5200 | 0 | 0 |
| tree | D07 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D08 | 40 | N/A | N/A | N/A | 7.5002 | 0.0250 | 0 | 0 |
| tree | D09 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D10 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D11 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D12 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D13 | 25 | N/A | N/A | N/A | 84.0014 | 0.2400 | 0 | 0 |
| tree | D14 | 40 | N/A | N/A | N/A | 60.0000 | 0.1500 | 0 | 0 |
| tree | D15 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D16 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D17 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| tree | D18 | 25 | N/A | N/A | N/A | 168.0084 | 0.5200 | 0 | 0 |
| tree | D19 | 25 | N/A | N/A | N/A | 144.0000 | 0.2800 | 0 | 0 |
| tree | F01 | 25 | 0.9600 | 0.8800 | 0.1200 | N/A | N/A | 1 | 2 |
| tree | F02 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| tree | F03 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| tree | F04 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| tree | F05 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| tree | F06 | 25 | 0.9600 | 0.9600 | 0.0400 | N/A | N/A | 1 | 0 |
| tree | F07 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| tree | F08 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| tree | F09 | 25 | 1.0000 | 0.9600 | 0.0400 | N/A | N/A | 0 | 1 |
| tree | F10 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| tree | F11 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| tree | F12 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| tree | F13 | 25 | 1.0000 | 0.9600 | 0.0400 | N/A | N/A | 0 | 1 |
| tree | F14 | 25 | 1.0000 | 0.9600 | 0.0400 | N/A | N/A | 0 | 1 |
| tree | F15 | 25 | 1.0000 | 0.8800 | 0.1200 | N/A | N/A | 0 | 3 |
| logistic | D01 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D02 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D03 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D04 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D05 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D06 | 25 | N/A | N/A | N/A | 5.7603 | 0.0400 | 0 | 0 |
| logistic | D07 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D08 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D09 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D10 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D11 | 40 | N/A | N/A | N/A | 7.5005 | 0.0250 | 0 | 0 |
| logistic | D12 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D13 | 25 | N/A | N/A | N/A | 12.0002 | 0.0400 | 0 | 0 |
| logistic | D14 | 40 | N/A | N/A | N/A | 22.5000 | 0.0750 | 0 | 0 |
| logistic | D15 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D16 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D17 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| logistic | D18 | 25 | N/A | N/A | N/A | 24.0012 | 0.0800 | 0 | 0 |
| logistic | D19 | 25 | N/A | N/A | N/A | 84.0000 | 0.2000 | 0 | 0 |
| logistic | F01 | 25 | 0.9600 | 0.8800 | 0.1200 | N/A | N/A | 1 | 2 |
| logistic | F02 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| logistic | F03 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| logistic | F04 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| logistic | F05 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| logistic | F06 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| logistic | F07 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| logistic | F08 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| logistic | F09 | 25 | 1.0000 | 0.9600 | 0.0400 | N/A | N/A | 0 | 1 |
| logistic | F10 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| logistic | F11 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| logistic | F12 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| logistic | F13 | 25 | 1.0000 | 0.8000 | 0.2000 | N/A | N/A | 0 | 5 |
| logistic | F14 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| logistic | F15 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| offline_frozen | D01 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D02 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D03 | 8 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D04 | 8 | N/A | N/A | N/A | 63.0000 | 0.2500 | 0 | 0 |
| offline_frozen | D05 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D06 | 25 | N/A | N/A | N/A | 57.6028 | 0.3600 | 0 | 0 |
| offline_frozen | D07 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D08 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D09 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D10 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D11 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D12 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D13 | 25 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D14 | 40 | N/A | N/A | N/A | 7.5000 | 0.0250 | 0 | 0 |
| offline_frozen | D15 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D16 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D17 | 40 | N/A | N/A | N/A | 0.0000 | 0.0000 | 0 | 0 |
| offline_frozen | D18 | 25 | N/A | N/A | N/A | 48.0024 | 0.1600 | 0 | 0 |
| offline_frozen | D19 | 25 | N/A | N/A | N/A | 108.0000 | 0.2000 | 0 | 0 |
| offline_frozen | F01 | 25 | 0.9600 | 0.6400 | 0.3600 | N/A | N/A | 1 | 8 |
| offline_frozen | F02 | 25 | 1.0000 | 0.8800 | 0.1200 | N/A | N/A | 0 | 3 |
| offline_frozen | F03 | 25 | 1.0000 | 0.8800 | 0.1200 | N/A | N/A | 0 | 3 |
| offline_frozen | F04 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| offline_frozen | F05 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| offline_frozen | F06 | 25 | 0.9600 | 0.8000 | 0.2000 | N/A | N/A | 1 | 4 |
| offline_frozen | F07 | 25 | 1.0000 | 0.8000 | 0.2000 | N/A | N/A | 0 | 5 |
| offline_frozen | F08 | 25 | 0.9600 | 0.7600 | 0.2400 | N/A | N/A | 1 | 5 |
| offline_frozen | F09 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| offline_frozen | F10 | 25 | 1.0000 | 0.9200 | 0.0800 | N/A | N/A | 0 | 2 |
| offline_frozen | F11 | 25 | 0.9600 | 0.9200 | 0.0800 | N/A | N/A | 1 | 1 |
| offline_frozen | F12 | 25 | 1.0000 | 1.0000 | 0.0000 | N/A | N/A | 0 | 0 |
| offline_frozen | F13 | 25 | 1.0000 | 0.6400 | 0.3600 | N/A | N/A | 0 | 9 |
| offline_frozen | F14 | 25 | 1.0000 | 0.8800 | 0.1200 | N/A | N/A | 0 | 3 |
| offline_frozen | F15 | 25 | 1.0000 | 0.8400 | 0.1600 | N/A | N/A | 0 | 4 |

All three models share candidate misses. Additional fall misses are candidates rejected by that model. ADL false-positive trial rates and alarms/hour answer different questions; both are provided. Event predictions and per-trial outcomes are exported for inspection.

## Artifacts and reproduction

Run `python -B outputs/sisfall/train_causal_models.py` in a fresh run directory (OUT constant); existing output directories are never overwritten. experiment_plan.json records the grid, objective, feature set and hashes before fitting. validation_candidates.csv records every candidate model and its selected validation threshold. model_parameters.json exports full tree arrays and logistic coefficients/scaler. tree.txt uses sklearn default labels for display only; deployed decisions use the exported positive-score threshold. metrics.csv includes descriptive training metrics. verification.json records input preservation and exact frozen-tree prediction agreement. The new models are validation-selected candidates; no test evaluation or firmware equivalence claim is made.
