# SisFall preprocessing and threshold baseline

All code, dependencies, tests and plots live in `outputs/` or `processed/`. Raw `data/` files are opened for reading only. Run commands from the project root. Requires Python 3.10+.

The completed initial run is `processed/baseline_v2/`. `processed/baseline_v1/` is an incomplete diagnostic run that stopped on a whitespace-only trailing line; do not use it for evaluation. The parser now handles those lines, with a regression test. Choose a fresh directory such as `processed/baseline_v3/` when rerunning.

## 1. Install and test

```powershell
python -m pip install --target outputs/deps numpy scipy scikit-learn matplotlib
python outputs/sisfall/test_pipeline.py
```

## 2. Build both modes and evaluate

```powershell
python outputs/sisfall/pipeline.py --data data --out processed/baseline_v3
```

The default seed is 42, with approximately 60/20/20 percent train/validation/test subjects. Splitting is stratified by presence of fall recordings (not by windows); each partition must have both classes. Integer rounding changes exact fractions. Subject age cohorts are not explicitly stratified. Existing output directories are refused, preventing accidental replacement. Use a new run directory for each experiment.

For identical subjects on subsequent runs:

```powershell
python outputs/sisfall/pipeline.py --out processed/reproduce_v2 --split-json processed/baseline_v2/split_subjects.json
```

Use the saved `requirements-lock.txt` to install the same direct dependency versions. The run records Python/library versions, parameters, source-code hash and per-trial SHA-256 hashes. Do not move code out of `outputs/sisfall` without updating the project-root calculation.

## 3. Processing decisions

1. Discover filenames recursively; extract subject, activity, numeric trial, and class (`D` = ADL/0, `F` = fall/1). Documentation text files are ignored and listed in the run configuration. Malformed trial-like filenames and duplicate trial identifiers fail explicitly. If a subject folder conflicts with a filename subject, default to exclusion and record the source hash and reason in `subject_identity_audit.json`. This dataset has five such files under SA15 named D17_SE15_R01–R05.txt. Only use `--subject-mismatch folder` when you explicitly intend to override those subject labels from folder names; raw files remain unchanged.
2. Validate finite nine-column input; convert only columns 1–3 using `32 / 8192 = 0.00390625 g/count`. All axes retain their original orientation; there is no normalization or gravity subtraction.
3. Save raw and filtered XYZ separately. Filter the entire trial with a fourth-order 5 Hz Butterworth SOS at 200 Hz, using forward/backward `sosfiltfilt`. This is offline zero-phase filtering: the forward/backward response has doubled effective order and is not a causal MCU filter. Filtering each independent trial does not mix subjects.
4. ADL uses 200-sample windows every 100 samples and discards incomplete tails. Fall trials produce exactly one 200-sample window around the largest **raw** XYZ magnitude. Ties use the earliest peak; boundary windows shift to fit the recording, without padding. Trials shorter than the window fail. `peak_event` is the replaceable event locator accepted by `window_bounds`.
5. The same window boundaries are used in both signal modes. No non-event segments of fall trials are included. This label-aware heuristic may select an unrelated peak and evaluates selected events, not continuous fall detection. It does not establish false alarms per hour or event detection latency.
6. Plot three distinct training activities per class by default. Each plot overlays raw/filtered axes and magnitude, with the selected fall event shaded. These are deterministic examples, not a comprehensive quality audit.

## 4. Artifacts and baseline

- `trials.csv`: parsed metadata, sample counts, source hashes and split.
- `windows.csv`: array row ID, source trial, subject/activity/trial, label, partition, start/end sample indices and raw peak index (`-1` for ADL). End indices are exclusive.
- `X_raw.npy`, `X_filtered.npy`: float32 arrays shaped `(windows, 200, 3)`, in g, memory-mappable for future CNN work.
- `labels_groups_splits.npz`: aligned `y`, `subjects`, and `split` arrays; no pickle required.
- `scores_*.npy`: maximum magnitude in each window.
- `baseline_metrics.json`: threshold and train/validation/test metrics for each mode. Confusion matrix order is `[[TN, FP], [FN, TP]]`.
- `split_subjects.json`, `config.json`, `requirements-lock.txt`, and `plots/`.

The simple non-neural baseline predicts fall when maximum magnitude is at least a threshold. The threshold maximizes validation balanced accuracy over all distinct score boundaries (ties choose the lowest threshold). There are no fitted training parameters for this scalar baseline. Test subjects are untouched during tuning. Both predefined modes are reported; select a mode on validation results, not test results. Metrics are window-weighted and class imbalance is substantial, so inspect sensitivity, specificity, precision, F1 and ROC AUC alongside balanced accuracy. This is an initial heuristic baseline, not a claimed reproduction of a specific publication.

## 5. Grouped cross-validation and reuse

```python
# In a script under outputs/sisfall/:
from pathlib import Path
from pipeline import grouped_folds
import numpy as np

run = Path('processed/baseline_v2')
X = np.load(run / 'X_raw.npy', mmap_mode='r')
info = np.load(run / 'labels_groups_splits.npz')
# Preserve the held-out test subjects when developing with CV.
development = np.flatnonzero(info['split'] != 'test')
for train_local, val_local in grouped_folds(
        info['y'][development], info['subjects'][development], method='groupkfold', n_splits=5):
    train_indices, val_indices = development[train_local], development[val_local]
    # Fit scaling/features/models on train_indices only; validate on val_indices.
    assert not set(info['subjects'][train_indices]) & set(info['subjects'][val_indices])
```

Use `method='loso'` for leave-one-subject-out. For an outer evaluation fold, threshold tuning requires a separate inner subject split; never tune on the outer held-out subject. Some held-out subjects have only ADL, so sensitivity/AUC for those folds are undefined. `metrics` returns null for undefined class-dependent metrics. Future learned normalization must fit training subjects only; future MCU work should explicitly validate a causal filter and streaming event trigger.

## 6. Verify a completed run

```powershell
python outputs/sisfall/verify_run.py processed/baseline_v2
```

This recomputes windows in both modes, checks aligned labels and subject partitions, and compares current raw-file hashes against ingestion hashes, including excluded ambiguous files. It writes `verification.json` only after every check passes.
