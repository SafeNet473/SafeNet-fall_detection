# Independent SisFall feature comparison

All six scores use the existing filtered 200-sample windows and fixed subjects. No features are combined. Each new threshold maximizes validation balanced accuracy. The baseline uses its saved threshold. Full equations and ambiguities: [FEATURE_EQUATIONS.md](../sisfall/FEATURE_EQUATIONS.md).

PR-AUC below is trapezoidal area; AP is non-interpolated average precision. They are different summaries. Threshold metrics are window-weighted.

## Validation

| feature | threshold | units | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2.20946598 | g | 93.07% | 90.52% | 91.79% | 14.64% | 25.30% | 58.29% | 58.32% |
| C2 | 1.23052612 | g | 98.67% | 98.33% | 98.50% | 50.82% | 67.09% | 97.05% | 97.06% |
| C3 | 2.63532167 | g | 97.07% | 92.24% | 94.65% | 17.93% | 30.27% | 63.29% | 63.33% |
| C8 | 0.447458296 | g | 98.67% | 98.42% | 98.54% | 52.19% | 68.27% | 92.72% | 92.72% |
| C9 | 0.76186194 | g | 97.07% | 89.58% | 93.32% | 14.00% | 24.47% | 26.68% | 26.90% |
| C13 | 124.625752 | g_sample | 97.07% | 86.74% | 91.90% | 11.34% | 20.31% | 17.86% | 17.94% |

## Test

| feature | threshold | units | sensitivity | specificity | balanced_accuracy | precision | f1 | pr_auc_trapezoidal | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2.20946598 | g | 90.40% | 89.53% | 89.96% | 12.87% | 22.52% | 61.96% | 61.98% |
| C2 | 1.23052612 | g | 95.20% | 95.26% | 95.23% | 25.55% | 40.29% | 92.00% | 92.00% |
| C3 | 2.63532167 | g | 91.47% | 90.90% | 91.18% | 14.66% | 25.28% | 71.05% | 71.07% |
| C8 | 0.447458296 | g | 95.73% | 98.26% | 97.00% | 48.45% | 64.34% | 95.53% | 95.53% |
| C9 | 0.76186194 | g | 86.13% | 86.88% | 86.51% | 10.10% | 18.07% | 44.26% | 44.30% |
| C13 | 124.625752 | g_sample | 93.07% | 85.37% | 89.22% | 9.81% | 17.75% | 25.21% | 25.29% |

## Activity-level errors

Each cell is errors / windows (rate). Every fall type has 25 selected events per partition; ADL counts include overlapping windows.

### validation: false positive

| activity | baseline | C2 | C3 | C8 | C9 | C13 |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) | 0/1584 (0.0%) |
| D02 | 0/1589 (0.0%) | 19/1589 (1.2%) | 0/1589 (0.0%) | 0/1589 (0.0%) | 0/1589 (0.0%) | 4/1589 (0.3%) |
| D03 | 345/1592 (21.7%) | 13/1592 (0.8%) | 154/1592 (9.7%) | 1/1592 (0.1%) | 532/1592 (33.4%) | 31/1592 (1.9%) |
| D04 | 1203/1592 (75.6%) | 56/1592 (3.5%) | 1102/1592 (69.2%) | 226/1592 (14.2%) | 1385/1592 (87.0%) | 198/1592 (12.4%) |
| D05 | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 1/1956 (0.1%) |
| D06 | 269/1219 (22.1%) | 0/1219 (0.0%) | 248/1219 (20.3%) | 1/1219 (0.1%) | 210/1219 (17.2%) | 0/1219 (0.0%) |
| D07 | 0/915 (0.0%) | 0/915 (0.0%) | 0/915 (0.0%) | 0/915 (0.0%) | 0/915 (0.0%) | 86/915 (9.4%) |
| D08 | 0/917 (0.0%) | 20/917 (2.2%) | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) | 66/917 (7.2%) |
| D09 | 0/918 (0.0%) | 20/918 (2.2%) | 0/918 (0.0%) | 0/918 (0.0%) | 0/918 (0.0%) | 107/918 (11.7%) |
| D10 | 0/919 (0.0%) | 30/919 (3.3%) | 0/919 (0.0%) | 2/919 (0.2%) | 0/919 (0.0%) | 110/919 (12.0%) |
| D11 | 13/913 (1.4%) | 2/913 (0.2%) | 0/913 (0.0%) | 0/913 (0.0%) | 0/913 (0.0%) | 134/913 (14.7%) |
| D12 | 0/920 (0.0%) | 4/920 (0.4%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 571/920 (62.1%) |
| D13 | 0/574 (0.0%) | 43/574 (7.5%) | 0/574 (0.0%) | 28/574 (4.9%) | 2/574 (0.3%) | 211/574 (36.8%) |
| D14 | 0/920 (0.0%) | 90/920 (9.8%) | 0/920 (0.0%) | 54/920 (5.9%) | 0/920 (0.0%) | 920/920 (100.0%) |
| D15 | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 64/920 (7.0%) |
| D16 | 0/912 (0.0%) | 0/912 (0.0%) | 0/912 (0.0%) | 0/912 (0.0%) | 0/912 (0.0%) | 215/912 (23.6%) |
| D17 | 0/1948 (0.0%) | 0/1948 (0.0%) | 0/1948 (0.0%) | 0/1948 (0.0%) | 0/1948 (0.0%) | 105/1948 (5.4%) |
| D18 | 52/573 (9.1%) | 44/573 (7.7%) | 40/573 (7.0%) | 25/573 (4.4%) | 25/573 (4.4%) | 16/573 (2.8%) |
| D19 | 153/575 (26.6%) | 17/575 (3.0%) | 122/575 (21.2%) | 2/575 (0.3%) | 82/575 (14.3%) | 6/575 (1.0%) |

### validation: false negative

| activity | baseline | C2 | C3 | C8 | C9 | C13 |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F02 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F03 | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) | 1/25 (4.0%) |
| F04 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 1/25 (4.0%) |
| F05 | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 1/25 (4.0%) |
| F06 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F07 | 4/25 (16.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 1/25 (4.0%) |
| F08 | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 2/25 (8.0%) | 0/25 (0.0%) |
| F09 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F10 | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |
| F11 | 2/25 (8.0%) | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 7/25 (28.0%) |
| F12 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F13 | 10/25 (40.0%) | 1/25 (4.0%) | 6/25 (24.0%) | 4/25 (16.0%) | 5/25 (20.0%) | 0/25 (0.0%) |
| F14 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F15 | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |

### test: false positive

| activity | baseline | C2 | C3 | C8 | C9 | C13 |
| --- | --- | --- | --- | --- | --- | --- |
| D01 | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) | 0/1744 (0.0%) |
| D02 | 0/1752 (0.0%) | 0/1752 (0.0%) | 0/1752 (0.0%) | 0/1752 (0.0%) | 0/1752 (0.0%) | 24/1752 (1.4%) |
| D03 | 427/1751 (24.4%) | 304/1751 (17.4%) | 222/1751 (12.7%) | 8/1751 (0.5%) | 932/1751 (53.2%) | 278/1751 (15.9%) |
| D04 | 1476/1751 (84.3%) | 605/1751 (34.6%) | 1481/1751 (84.6%) | 305/1751 (17.4%) | 1703/1751 (97.3%) | 353/1751 (20.2%) |
| D05 | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) | 0/1956 (0.0%) |
| D06 | 193/1174 (16.4%) | 13/1174 (1.1%) | 162/1174 (13.8%) | 18/1174 (1.5%) | 159/1174 (13.5%) | 2/1174 (0.2%) |
| D07 | 0/916 (0.0%) | 0/916 (0.0%) | 0/916 (0.0%) | 0/916 (0.0%) | 0/916 (0.0%) | 61/916 (6.7%) |
| D08 | 4/920 (0.4%) | 2/920 (0.2%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 17/920 (1.8%) |
| D09 | 0/920 (0.0%) | 2/920 (0.2%) | 0/920 (0.0%) | 0/920 (0.0%) | 0/920 (0.0%) | 142/920 (15.4%) |
| D10 | 0/919 (0.0%) | 10/919 (1.1%) | 0/919 (0.0%) | 4/919 (0.4%) | 0/919 (0.0%) | 103/919 (11.2%) |
| D11 | 8/900 (0.9%) | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 27/900 (3.0%) |
| D12 | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) | 0/917 (0.0%) | 452/917 (49.3%) |
| D13 | 0/574 (0.0%) | 25/574 (4.4%) | 0/574 (0.0%) | 16/574 (2.8%) | 0/574 (0.0%) | 233/574 (40.6%) |
| D14 | 0/917 (0.0%) | 47/917 (5.1%) | 0/917 (0.0%) | 13/917 (1.4%) | 0/917 (0.0%) | 917/917 (100.0%) |
| D15 | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 0/900 (0.0%) | 127/900 (14.1%) |
| D16 | 0/894 (0.0%) | 0/894 (0.0%) | 0/894 (0.0%) | 0/894 (0.0%) | 0/894 (0.0%) | 252/894 (28.2%) |
| D17 | 0/1870 (0.0%) | 0/1870 (0.0%) | 0/1870 (0.0%) | 0/1870 (0.0%) | 0/1870 (0.0%) | 214/1870 (11.4%) |
| D18 | 37/574 (6.4%) | 32/574 (5.6%) | 26/574 (4.5%) | 18/574 (3.1%) | 11/574 (1.9%) | 6/574 (1.0%) |
| D19 | 151/574 (26.3%) | 0/574 (0.0%) | 105/574 (18.3%) | 0/574 (0.0%) | 71/574 (12.4%) | 0/574 (0.0%) |

### test: false negative

| activity | baseline | C2 | C3 | C8 | C9 | C13 |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F02 | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 2/25 (8.0%) |
| F03 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F04 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F05 | 0/25 (0.0%) | 5/25 (20.0%) | 0/25 (0.0%) | 5/25 (20.0%) | 0/25 (0.0%) | 5/25 (20.0%) |
| F06 | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 1/25 (4.0%) |
| F07 | 2/25 (8.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) |
| F08 | 4/25 (16.0%) | 0/25 (0.0%) | 4/25 (16.0%) | 1/25 (4.0%) | 8/25 (32.0%) | 0/25 (0.0%) |
| F09 | 4/25 (16.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 1/25 (4.0%) | 0/25 (0.0%) |
| F10 | 4/25 (16.0%) | 0/25 (0.0%) | 2/25 (8.0%) | 0/25 (0.0%) | 4/25 (16.0%) | 0/25 (0.0%) |
| F11 | 2/25 (8.0%) | 11/25 (44.0%) | 2/25 (8.0%) | 1/25 (4.0%) | 4/25 (16.0%) | 14/25 (56.0%) |
| F12 | 1/25 (4.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 0/25 (0.0%) | 4/25 (16.0%) |
| F13 | 6/25 (24.0%) | 1/25 (4.0%) | 6/25 (24.0%) | 1/25 (4.0%) | 5/25 (20.0%) | 0/25 (0.0%) |
| F14 | 5/25 (20.0%) | 0/25 (0.0%) | 8/25 (32.0%) | 6/25 (24.0%) | 17/25 (68.0%) | 0/25 (0.0%) |
| F15 | 6/25 (24.0%) | 0/25 (0.0%) | 10/25 (40.0%) | 2/25 (8.0%) | 13/25 (52.0%) | 0/25 (0.0%) |

## Failure-mode summary

C8 has the highest balanced accuracy on both validation (98.54%, narrowly ahead of C2 at 98.50%) and test (97.00%). Relative to the baseline, its test false positives fall from 2,296 to 382 and misses from 36 to 16. Test D04 still causes 305 of its 382 false positives; F14 remains its largest missed-fall group. C2 substantially reduces false alarms, but misses 11/25 F11 events. These different weaknesses matter even where aggregate scores are close.

C3 and C9 retain large jogging errors, with C9 flagging 1,703/1,751 fast-jogging windows. Including vertical-axis variation is consistent with these errors, though the aggregate comparison alone does not isolate a causal mechanism. C9 also misses 17/25 F14 and 13/25 F15 events.

C13 flags every test D14 window (917/917) and 452/917 D12 windows. Its uncentered horizontal-magnitude integral includes sustained acceleration and gravity projected onto sensor x/z during lying postures; unlike C8, it does not remove a constant axis offset. That algebraic property is consistent with these observed posture-related false positives. We have not added gravity removal because it would change the requested equation. C13's poor ranking cannot be repaired by changing sample-based integration to seconds: that only rescales every score and threshold.

Validation threshold tuning does not optimize precision or F1. C8 still has test precision 48.45% at the selected balanced-accuracy threshold, despite high PR-AUC. Subject generalization and threshold tradeoffs remain visible; no feature is promoted or combined on the basis of test outcomes.

- **C2**: test balanced accuracy delta +5.26 percentage points; false positive: D04 605/1751, D03 304/1751, D14 47/917; false negative: F11 11/25, F05 5/25, F02 1/25.

- **C3**: test balanced accuracy delta +1.22 percentage points; false positive: D04 1481/1751, D03 222/1751, D06 162/1174; false negative: F15 10/25, F14 8/25, F13 6/25.

- **C8**: test balanced accuracy delta +7.03 percentage points; false positive: D04 305/1751, D06 18/1174, D18 18/574; false negative: F14 6/25, F05 5/25, F15 2/25.

- **C9**: test balanced accuracy delta -3.46 percentage points; false positive: D04 1703/1751, D03 932/1751, D06 159/1174; false negative: F14 17/25, F15 13/25, F08 8/25.

- **C13**: test balanced accuracy delta -0.75 percentage points; false positive: D14 917/917, D12 452/917, D04 353/1751; false negative: F11 14/25, F05 5/25, F12 4/25.

## Reproduce

From the project root: `python outputs/sisfall/paper_features.py --out outputs/paper_feature_comparison_rerun`. Existing output directories are refused. Run `python outputs/sisfall/test_paper_features.py` for equation checks.

Input hashes, alignment, baseline-score parity, confusion-count reconciliation and constant-scale equivalence were checked. `provenance.json` records conventions; `feature_scores.npz` aligns with the source window IDs; `metrics.csv` includes baseline deltas; `activity_errors.csv` includes every activity and rate delta; `error_windows.csv` identifies misses and false alarms.

The fixed event extraction is label-aware and the filter is offline. These are comparisons under our protocol, not a reproduction of the paper's reported accuracy or a streaming detector evaluation. No test-derived feature or threshold selection is performed.
