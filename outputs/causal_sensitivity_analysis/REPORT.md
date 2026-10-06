# Fixed causal logistic model: sensitivity-first validation sweep

Only the classifier decision threshold varies. Model weights, three features, causal filter, gravity EMA, selected trigger, refractory period and pre100_post200 window remain fixed. No test files or outcomes were opened. Previous artifacts are unchanged; these operating points are analysis only and do not replace the deployed/reference threshold.

The exact reference threshold is **0.8850929456953477**, not rounded 0.8851. Swept **4388** boundaries: every unique saved validation positive score plus a no-alarm boundary immediately above the maximum score. Decisions use score >= threshold. Saved scores were checked against the fixed exported logistic equation.

## Candidate ceiling

There are 375 validation fall trials and 572 ADL trials (3.067942 ADL hours). Complete localized candidate recall is **98.666667%**, or 370/375 falls. This is the end-to-end sensitivity ceiling under the fixed matching/trigger/window convention. **100% end-to-end sensitivity is impossible by changing only this classifier threshold.**

The impact time is the existing raw-magnitude peak proxy, not an annotated onset. A matched candidate has trigger within +/-1 s of that proxy and a complete emitted window. One detected event per fall trial is credited. Precision = detected fall trials / emitted positive decisions; duplicate/unmatched fall-recording alarms count as false positives. ADL trial specificity is the fraction of ADL trials with no alarm. Trial balanced accuracy averages that specificity and fall-event sensitivity.

## Requested operating points

| operating_point | threshold | fall_event_sensitivity | adl_false_alarms_per_hour | event_precision | adl_trial_specificity | trial_balanced_accuracy | detected_fall_trials | missed_fall_trials | emitted_positive_decisions |
|---|---|---|---|---|---|---|---|---|---|
| current_BA_reference | 0.885093 | 0.933333 | 4.88927 | 0.930851 | 0.977273 | 0.955303 | 350 | 25 | 376 |
| max_sensitivity_FA_le_5 | 0.885093 | 0.933333 | 4.88927 | 0.930851 | 0.977273 | 0.955303 | 350 | 25 | 376 |
| max_sensitivity_FA_le_10 | 0.8622 | 0.938667 | 6.84498 | 0.916667 | 0.97028 | 0.954473 | 352 | 23 | 384 |
| max_sensitivity_FA_le_20 | 0.735288 | 0.968 | 17.2754 | 0.840278 | 0.924825 | 0.946413 | 363 | 12 | 432 |
| lowest_threshold_sensitivity_ge_0.95 | 1.47663e-09 | 0.986667 | 1072.05 | 0.0843401 | 0.0874126 | 0.53704 | 370 | 5 | 4387 |
| lowest_threshold_sensitivity_ge_0.97 | 1.47663e-09 | 0.986667 | 1072.05 | 0.0843401 | 0.0874126 | 0.53704 | 370 | 5 | 4387 |
| lowest_threshold_max_sensitivity | 1.47663e-09 | 0.986667 | 1072.05 | 0.0843401 | 0.0874126 | 0.53704 | 370 | 5 | 4387 |

For the <=5/10/20 false-alarm budgets, sensitivity is maximized first. Exact sensitivity ties minimize ADL alarms/hour, then total positive decisions, then prefer the highest threshold. No unspecified nearly-equal tolerance is used; one fall changes sensitivity by 0.266667 percentage points.

**Lowest threshold is interpreted literally within the requested boundary sweep.** If 95%, 97%, and maximum sensitivity are attainable, all three lowest-threshold requests select the minimum observed validation score and accept every candidate. Thresholds below that minimum would produce identical validation decisions, but are outside this finite sweep. These literal points do not minimize false alarms.

## More selective supplemental target points

| operating_point | threshold | fall_event_sensitivity | adl_false_alarms_per_hour | event_precision | adl_trial_specificity | trial_balanced_accuracy | detected_fall_trials | missed_fall_trials | emitted_positive_decisions |
|---|---|---|---|---|---|---|---|---|---|
| highest_threshold_sensitivity_ge_0.950000 | 0.774203 | 0.952 | 12.7121 | 0.870732 | 0.942308 | 0.947154 | 357 | 18 | 410 |
| highest_threshold_sensitivity_ge_0.970000 | 0.610153 | 0.970667 | 24.7723 | 0.791304 | 0.898601 | 0.934634 | 364 | 11 | 460 |
| highest_threshold_sensitivity_ge_0.986667 | 0.0250379 | 0.986667 | 340.619 | 0.207982 | 0.274476 | 0.630571 | 370 | 5 | 1779 |

These supplemental rows use the highest threshold meeting each sensitivity target, which minimizes emitted positives among thresholds meeting that target. They are provided to clarify the tradeoff, not to replace the requested literal lowest-threshold results.

## Changes from the current reference

| operating_point | sensitivity_delta_pp | false_alarms_per_hour_delta | precision_delta_pp | balanced_accuracy_delta_pp | detected_falls_delta | positive_decisions_delta |
|---|---|---|---|---|---|---|
| current_BA_reference | 0 | 0 | 0 | 0 | 0 | 0 |
| max_sensitivity_FA_le_5 | 0 | 0 | 0 | 0 | 0 | 0 |
| max_sensitivity_FA_le_10 | 0.533333 | 1.95571 | -1.41844 | -0.0829837 | 2 | 8 |
| max_sensitivity_FA_le_20 | 3.46667 | 12.3862 | -9.05733 | -0.889044 | 13 | 56 |
| lowest_threshold_sensitivity_ge_0.95 | 5.33333 | 1067.17 | -84.6511 | -41.8263 | 20 | 4011 |
| lowest_threshold_sensitivity_ge_0.97 | 5.33333 | 1067.17 | -84.6511 | -41.8263 | 20 | 4011 |
| lowest_threshold_max_sensitivity | 5.33333 | 1067.17 | -84.6511 | -41.8263 | 20 | 4011 |
| highest_threshold_sensitivity_ge_0.950000 | 1.86667 | 7.82283 | -6.01194 | -0.814918 | 7 | 34 |
| highest_threshold_sensitivity_ge_0.970000 | 3.73333 | 19.883 | -13.9547 | -2.0669 | 14 | 84 |
| highest_threshold_sensitivity_ge_0.986667 | 5.33333 | 335.73 | -72.2869 | -32.4732 | 20 | 1403 |

## Every remaining miss at the literal maximum-sensitivity threshold

| path | subject | fall_type | reason | proxy_peak_sample | raw_triggers | complete_candidates | best_matched_score | ga_C2 | jerk_abs_mean | ga_parallel_peak |
|---|---|---|---|---|---|---|---|---|---|---|
| SA05/F01_SA05_R01.txt | SA05 | F01 | no_matched_trigger | 1822 | 5 | 5 | N/A | N/A | N/A | N/A |
| SA05/F04_SA05_R05.txt | SA05 | F04 | no_matched_trigger | 1823 | 5 | 5 | N/A | N/A | N/A | N/A |
| SA05/F06_SA05_R01.txt | SA05 | F06 | no_matched_trigger | 1682 | 5 | 5 | N/A | N/A | N/A | N/A |
| SA05/F11_SA05_R04.txt | SA05 | F11 | no_matched_trigger | 1400 | 2 | 2 | N/A | N/A | N/A | N/A |
| SA19/F08_SA19_R02.txt | SA19 | F08 | no_matched_trigger | 1234 | 1 | 1 | N/A | N/A | N/A | N/A |

Miss counts: no matched trigger = 5; matched trigger but incomplete window = 0; matched complete candidate rejected by classifier = 0.

At the minimum score boundary every complete candidate is accepted, so there are no classifier-rejected matched candidates to list. The header-only classifier_rejected_candidates.csv records this explicitly. Missing matched candidates have no corresponding logistic score or feature vector; these are left blank rather than invented. The highest threshold attaining the same maximum sensitivity necessarily misses the same no-candidate trials.

## Per-activity results and artifacts

threshold_sweep.csv includes every requested aggregate metric, all 15 per-fall-type sensitivities, and all 19 per-ADL false-alarm rates at **every** threshold. operating_points.csv includes all these columns plus reference deltas. selected_activity_metrics.csv provides long-form counts and exposures for each selected point. maximum_sensitivity_misses.csv lists every missed validation fall with subject/type and cause.

Reproduce with `python -B outputs/sisfall/causal_sensitivity_sweep.py` using a fresh OUT directory; overwriting an existing analysis is refused. This script uses only the Python standard library and never loads/fits an estimator. provenance.json hashes the inputs; verification.json records reference parity, monotonicity, endpoint checks, and unchanged-input checks.
