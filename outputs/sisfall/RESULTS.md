# Initial baseline results

Completed run: `processed/baseline_v2`. Seed 42; 22/8/8 train/validation/test subjects; 4,500 trials; 103,239 windows (59,110/21,831/22,298). Five ambiguous subject files were excluded and recorded in the run audit. Raw sources were not edited.

| Signal | Validation-selected threshold (g) | Test sensitivity | Test specificity | Test balanced accuracy | Test precision | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 3.047143 | 88.53% | 87.10% | 87.82% | 10.51% | 18.78% |
| 5 Hz filtered | 2.209466 | 90.40% | 89.53% | 89.96% | 12.87% | 22.52% |

The filtered mode also has higher validation balanced accuracy (91.79% versus 90.51%). The held-out test set contains 21,923 ADL windows and 375 selected fall events. Confusion matrices, AUC and all partitions are in `baseline_metrics.json`.

These are window-level results on a label-aware event extraction protocol. Low precision reflects the large number of ADL false positives. They do not measure continuous fall detection, event latency or false alarms per hour. Zero-phase filtering is offline; a causal MCU pipeline needs separate evaluation. No neural network was trained.

Seven unit tests cover parsing/conversion, ambiguous-subject handling, windows and edge cases, filter behavior, reproducible subject splits and grouped folds, exhaustive threshold selection, and output protection. See `verification.json` in the completed run for the full artifact verification result.
