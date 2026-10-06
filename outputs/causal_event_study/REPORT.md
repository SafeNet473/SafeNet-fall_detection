# Causal preprocessing and event-window experiment

The classifier, three feature split thresholds, original subject partitions, and raw data are unchanged. This is a separate preprocessing/trigger experiment, not a replacement of prior baseline artifacts.

## Frozen validation selection

Trigger: causal filtered magnitude >= 1.1 g OR absolute magnitude jerk >= 5.0 g/s (a null jerk threshold disables that branch). Rising edge of the OR condition, with a fixed 1.5 s refractory interval. The refractory interval and grid were declared before replay, not tuned on test.

Selected window: **pre100_post200**. The 99% localized completed-candidate recall target across all designs was **NOT met**.

Selection uses validation only: among trigger settings meeting 99% recall in every design, minimize mean ADL candidate rate; if none qualifies, maximize the minimum recall first, then minimize candidate rate. With that common trigger fixed, choose the window with highest final validation sensitivity, then lowest ADL false alarms/hour, then shortest post delay. Training results are descriptive. selection.json was frozen before test replay; all three predeclared window designs were evaluated together in one test pass, and none was selected using test.

| split | design | candidate_recall | candidates_per_adl_hour | sensitivity | false_alarms_per_adl_hour | event_precision | latency_median_s | latency_worst_s |
|---|---|---|---|---|---|---|---|---|
| train | pre100_post200 | 0.9819 | 1112.1307 | 0.8340 | 26.7403 | 0.7769 | 0.9425 | 1.7900 |
| validation | pre100_post200 | 0.9867 | 1072.0543 | 0.8480 | 12.3862 | 0.8785 | 0.8825 | 1.8800 |
| test | pre100_post200 | 0.9760 | 1145.9057 | 0.7760 | 2.2362 | 0.9479 | 0.9150 | 1.8200 |

## All predeclared windows

| split | design | candidate_recall | candidates_per_adl_hour | sensitivity | false_alarms_per_adl_hour | event_precision | latency_median_s | latency_worst_s |
|---|---|---|---|---|---|---|---|---|
| validation | pre100_post100 | 0.9867 | 1078.5733 | 0.7147 | 9.1266 | 0.8963 | 0.4400 | 1.3850 |
| validation | pre50_post150 | 0.9867 | 1078.2474 | 0.7707 | 9.4526 | 0.9003 | 0.6350 | 1.5200 |
| validation | pre100_post200 | 0.9867 | 1072.0543 | 0.8480 | 12.3862 | 0.8785 | 0.8825 | 1.8800 |
| test | pre100_post100 | 0.9760 | 1155.8090 | 0.6160 | 1.2778 | 0.9706 | 0.4800 | 1.3200 |
| test | pre50_post150 | 0.9760 | 1153.2533 | 0.6667 | 1.9168 | 0.9579 | 0.6700 | 1.5700 |
| test | pre100_post200 | 0.9760 | 1145.9057 | 0.7760 | 2.2362 | 0.9479 | 0.9150 | 1.8200 |

Each window is [trigger-pre, trigger+post), with a decision at trigger+post: exactly 0.5, 0.75, or 1.0 s post-trigger delay. These contain 200, 200, and 300 samples respectively. The 300-sample variant uses the same feature equations, including a mean over 299 magnitude differences; the tree is unchanged and was originally developed for 200 samples.

## Definitions and limits

- No annotated onset/impact timestamps are available in the saved pipeline. Each fall trial is assumed to contain one fall; its raw-magnitude argmax is an **evaluation-only impact proxy**, not ground truth. Neither labels nor this proxy enter trigger generation.

- A candidate is matched when its trigger is within +/-1 s of the proxy. Candidate recall requires a complete, emitted window. raw_trigger_recall excludes window completeness; any_candidate_trial_recall ignores localization. All are exported in summary.csv.

- A detected fall is a matched candidate accepted by the locked tree. One true positive per fall trial is credited. Other alarms, including duplicates and unmatched alarms in fall recordings, are false positives for event precision. No extra alarm-merging policy is applied beyond the trigger refractory interval.

- ADL rates count complete candidate windows or positive decisions divided by full recorded ADL hours, including startup and tails. Boundary candidates lacking prehistory or the complete post delay are discarded; no shifting, padding, or flushing with future samples.

- Latency is the earliest matched positive decision minus the impact proxy. It is signed and conditional on detection; negative values mean an alarm precedes the proxy. Worst latency is the largest observed delay among detected events, not a guarantee for missed falls. Proxy-based recall, precision, and latency require annotated timestamps for definitive interpretation.

- These are waist recordings. Causality and orientation robustness do not validate wrist fall-detection performance.

## Causal implementation

The fourth-order 5 Hz Butterworth at 200 Hz is implemented as two stateful DF-II-transposed SOS sections, one sample per update. At each trial start, SOS state is sosfilt_zi times the first observed raw sample. Gravity uses the unchanged raw-acceleration 0.5 Hz EMA, initialized from that same first sample. States persist throughout each trial and reset only between trials. No reverse filtering or delay compensation is used. The one-pass frequency/phase response differs from forward/backward filtering.

Filter and gravity arithmetic are float64; filtered samples retain the previous float32 window interface before float64 feature arithmetic. The locked export performs its original float32 feature comparisons. This is a causal desktop reference, not a float32 firmware equivalence claim.

The trigger uses only current/past causal magnitude and magnitude differences. The ring stores magnitude, perpendicular magnitude, and absolute parallel acceleration: 301 x 3 values for the longest window, including the excluded decision-time sample. That is 3612 bytes in a float32 port or 7224 bytes in this float64 reference, plus SOS, gravity, and queue state. Reference replay archives derived streams for analysis, but ring aggregation only accesses samples already available at its decision time. Prefix-invariance tests cover the trigger and emitted features. With 1.5 s refractory and <=1 s post delay, each design has at most one pending event.

## Attribution: filter, localization, and placement

The first comparison uses exactly the existing saved windows and the original cached locked-model features versus causal-filter features. This isolates preprocessing without changing event positions or ADL sampling. Its specificity/precision are **window-level**, and must not be compared directly with event-level false alarms/hour or event precision.

| split | mode | sensitivity | specificity | balanced_accuracy | precision |
|---|---|---|---|---|---|
| validation | original | 0.9973333333333333 | 0.9952926920208799 | 0.9963130126771066 | 0.7873684210526316 |
| validation | causal | 0.8746666666666667 | 0.9964112602535421 | 0.9355389634601043 | 0.8098765432098766 |
| test | original | 0.9653333333333334 | 0.9988140309264243 | 0.9820736821298788 | 0.9329896907216495 |
| test | causal | 0.8106666666666666 | 0.9994526296583497 | 0.9050596481625082 | 0.9620253164556962 |

| split | stage | sensitivity | delta_from_previous |
|---|---|---|---|
| validation | original_saved_windows | 0.9973 | — |
| validation | causal_filter_same_saved_windows | 0.8747 | -0.1227 |
| validation | causal_trigger_100_100 | 0.7147 | -0.1600 |
| validation | placement_vs_100_100_pre50_post150 | 0.7707 | 0.0560 |
| validation | placement_vs_100_100_pre100_post200 | 0.8480 | 0.1333 |
| validation | offline_oracle_placement_pre100_post100 | 0.8747 | — |
| validation | offline_oracle_placement_pre50_post150 | 0.8880 | — |
| validation | offline_oracle_placement_pre100_post200 | 0.8880 | — |
| test | original_saved_windows | 0.9653 | — |
| test | causal_filter_same_saved_windows | 0.8107 | -0.1547 |
| test | causal_trigger_100_100 | 0.6160 | -0.1947 |
| test | placement_vs_100_100_pre50_post150 | 0.6667 | 0.0507 |
| test | placement_vs_100_100_pre100_post200 | 0.7760 | 0.1600 |
| test | offline_oracle_placement_pre100_post100 | 0.8107 | — |
| test | offline_oracle_placement_pre50_post150 | 0.8213 | — |
| test | offline_oracle_placement_pre100_post200 | 0.8240 | — |

Deltas are sensitivity fractions (multiply by 100 for percentage points). The trigger-100/100 delta includes candidate misses, refractory suppression, boundary losses, and trigger timing versus the previous oracle-centered fall window. It is not a pure timing-only estimate. Placement deltas keep trigger thresholds and trigger times fixed. For the 300-sample design they also include the longer feature interval. Offline oracle-placement rows center each design at the impact proxy on the causal stream to expose placement effects without candidate gating; these are diagnostic only and were not used in selection.

| split | design | fall_trials | no_matched_raw_trigger | matched_trigger_but_incomplete_window | matched_complete_candidates_all_rejected | detected |
|---|---|---|---|---|---|---|
| validation | pre100_post100 | 375 | 5 | 0 | 102 | 268 |
| validation | pre50_post150 | 375 | 5 | 0 | 81 | 289 |
| validation | pre100_post200 | 375 | 5 | 0 | 52 | 318 |
| test | pre100_post100 | 375 | 9 | 0 | 135 | 231 |
| test | pre50_post150 | 375 | 9 | 0 | 116 | 250 |
| test | pre100_post200 | 375 | 9 | 0 | 75 | 291 |

## Per-activity failures

activity_metrics.csv contains every activity and window design. The following focuses on the predeclared difficult activities for the selected design.

| split | activity | candidate_recall | sensitivity | candidates_per_adl_hour | false_alarms_per_adl_hour | no_matched_raw_trigger | incomplete_window | matched_candidates_rejected |
|---|---|---|---|---|---|---|---|---|
| validation | D03 | — | — | 2074.5000 | 0.0000 | 0 | 0 | 0 |
| validation | D04 | — | — | 2146.5000 | 63.0000 | 0 | 0 | 0 |
| validation | D06 | — | — | 731.5551 | 57.6028 | 0 | 0 | 0 |
| validation | D18 | — | — | 1728.0864 | 48.0024 | 0 | 0 | 0 |
| validation | D19 | — | — | 660.0000 | 108.0000 | 0 | 0 | 0 |
| validation | F13 | 1.0000 | 0.6400 | — | — | 0 | 0 | 9 |
| validation | F14 | 1.0000 | 0.8800 | — | — | 0 | 0 | 3 |
| validation | F15 | 1.0000 | 0.8400 | — | — | 0 | 0 | 4 |
| test | D03 | — | — | 2164.0909 | 0.0000 | 0 | 0 | 0 |
| test | D04 | — | — | 2151.8304 | 8.1819 | 0 | 0 | 0 |
| test | D06 | — | — | 816.0068 | 6.0001 | 0 | 0 | 0 |
| test | D18 | — | — | 1704.0284 | 24.0004 | 0 | 0 | 0 |
| test | D19 | — | — | 612.0102 | 12.0002 | 0 | 0 | 0 |
| test | F13 | 1.0000 | 0.5200 | — | — | 0 | 0 | 12 |
| test | F14 | 1.0000 | 0.7200 | — | — | 0 | 0 | 7 |
| test | F15 | 1.0000 | 0.6800 | — | — | 0 | 0 | 8 |

For ADLs, every alarm is a false alarm. For falls, a missing matched candidate is a trigger/window failure; a complete matched candidate rejected by the tree is a downstream classification failure. The per-trial CSVs and event CSVs retain sample bounds, feature values, and decisions for inspection. activity_decision_branches.csv further separates low-ga_C2 rejections from high-jerk/low-parallel rejections across all candidates; these candidate counts must not be mistaken for trial-level false negatives.

## Reproduction and integrity

Run test_causal_events.py, then causal_events.py validate, then causal_events.py evaluate, then causal_event_report.py. Selection/evaluation refuse to overwrite an already frozen/completed run. Use a separate output directory for a new experiment. experiment_plan.json and selection.json hash the locked tree, its predictor, partitions, source manifests, original features, and replay code. Each raw trial is checked against its existing SHA-256 manifest during replay. evaluation_complete.json verifies frozen inputs remained unchanged. No fitting, model hyperparameter tuning, or classifier threshold tuning occurs.
