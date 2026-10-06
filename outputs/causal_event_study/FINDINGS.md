# Findings from the frozen causal experiment

The locked classifier loses substantial sensitivity when moved to causal preprocessing. A high-recall trigger alone does not resolve the loss. No classifier, threshold, or partition was changed.

The validation-selected design uses filtered magnitude >= 1.1 g OR absolute magnitude jerk >= 5 g/s, rising-edge triggering with 1.5 s refractory, and a window with 0.5 s pre-trigger / 1.0 s post-trigger. The 99% localized candidate-recall target was not met: validation recall was 98.67%. The selection was frozen before the single fixed-test replay.

| Test window: pre / post | Candidate recall | Candidates / ADL hour | Final sensitivity | False alarms / ADL hour | Event precision | Median latency | Worst observed latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 / 0.5 s | 97.60% | 1155.81 | 61.60% | 1.28 | 97.06% | 0.480 s | 1.320 s |
| 0.25 / 0.75 s | 97.60% | 1153.25 | 66.67% | 1.92 | 95.79% | 0.670 s | 1.570 s |
| **0.5 / 1.0 s (selected on validation)** | **97.60%** | **1145.91** | **77.60%** | **2.24** | **94.79%** | **0.915 s** | **1.820 s** |

There were 375 test fall trials and 3.130275 recorded ADL hours. Latency and localized recall use the original raw-magnitude peak as an evaluation-only impact proxy, not an annotated fall onset. The trigger never sees that proxy or labels. Event precision counts unmatched/duplicate alarms in fall trials as false positives too. Results do not establish wrist performance or long-term false-alarm rates.

## What caused the sensitivity change?

- **Causal filtering alone:** on identical saved windows, sensitivity changed from 96.53% to 81.07%, a loss of **15.47 percentage points**.
- **Trigger localization and candidate gating:** replacing oracle-centered fall windows with trigger-centered 0.5/0.5 s windows reduced sensitivity to 61.60%, another **19.47 points**. This includes missing candidates and timing differences, not just timing alone.
- **Window placement:** keeping trigger times fixed, moving to 0.25/0.75 s recovered **5.07 points**. Extending to 0.5/1.0 s recovered **16.00 points**. The latter changes both placement and feature interval length to 300 samples; the locked tree was originally developed with 200 samples.
- With offline oracle localization, changing 0.5/0.5 s to 0.5/1.0 s increased causal sensitivity only from 81.07% to 82.40%. The larger improvement under trigger localization is consistent with the longer post-trigger interval compensating for early trigger placement. It does not restore the original filter's behavior.

These are sequential, protocol-specific ablations, not independent causal-effect estimates. Original fixed-window specificity and precision cannot be compared directly with event false alarms/hour and event precision.

## Failure modes of the selected design

Of 375 test falls, **291 were detected**, **9 had no localized trigger**, and **75 had complete localized candidates but all were rejected by the tree**. No localized candidates were lost to incomplete boundary windows.

All F13/F14/F15 test trials had localized candidates, but their final sensitivities were **52%, 72%, and 68%**, respectively: 12/25, 7/25, and 8/25 misses. These are downstream tree/window failures rather than absent candidates.

The seven ADL false alarms came from D04 (2), D06 (1), D13 (1), D18 (2), and D19 (1). D03 produced no false alarms despite frequent candidates. Candidate traffic remains high: about **19.1 candidates per ADL minute** for the selected design.

See [REPORT.md](REPORT.md) for definitions, all validation results, implementation details, and reproducibility. [activity_metrics.csv](activity_metrics.csv) and [activity_decision_branches.csv](activity_decision_branches.csv) separate per-activity trial outcomes from candidate-level rejection branches.

All five regression tests passed, including causal prefix invariance, exact ring ordering/delay, feature-equation agreement, and locked-tree comparison parity. Frozen-input hashes were unchanged after evaluation; every raw trial was checked against its original manifest hash.
