"""Read-only SisFall ingestion, subject splits, event extraction, and baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
# Optional isolated dependency installation; no global environment changes.
sys.path.insert(0, str(PROJECT / "outputs" / "deps"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib"))
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, roc_auc_score

PATTERN = re.compile(r"^(?P<activity>[DF]\d{2})_(?P<subject>S[AE]\d{2})_R(?P<trial>\d{2})\.txt$", re.I)
SCALE = 32 / 2**13
FS = 200


def metadata(data_root: Path, mismatch_policy="exclude", audit=None) -> list[dict]:
    if mismatch_policy not in {"exclude", "folder"}:
        raise ValueError("Unknown subject mismatch policy")
    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        match = PATTERN.fullmatch(path.name)
        if not match:
            if re.match(r"^[DF]\d", path.name, re.I):
                raise ValueError(f"Unrecognized trial filename: {path}")
            continue  # Dataset documentation is not a sensor trial.
        activity, subject = match["activity"].upper(), match["subject"].upper()
        folder = path.parent.name.upper()
        if re.fullmatch(r"S[AE]\d{2}", folder) and folder != subject:
            if audit is not None:
                audit.append(dict(path=path.relative_to(data_root).as_posix(), filename_subject=subject,
                                  folder_subject=folder, action=mismatch_policy,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if mismatch_policy == "exclude":
                continue
            subject = folder
        rows.append(dict(path=path.relative_to(data_root).as_posix(),
                         subject_id=subject, activity_id=activity,
                         trial_number=int(match["trial"]),
                         class_name="fall" if activity.startswith("F") else "ADL",
                         label=int(activity.startswith("F"))))
    if not rows:
        raise ValueError(f"No trials in {data_root}")
    keys = [(r["subject_id"], r["activity_id"], r["trial_number"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate subject/activity/trial identifiers")
    return rows


def load_trial(path: Path) -> tuple[np.ndarray, str]:
    blob = path.read_bytes()
    # SisFall has comma-delimited rows terminated by semicolons.
    text = "\n".join(line.strip() for line in blob.decode("utf-8-sig").replace(";", "\n").splitlines()
                     if line.strip())
    try:
        counts = np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2)
    except ValueError as error:
        raise ValueError(f"Malformed sensor data: {path}: {error}") from error
    if counts.shape[0] == 0 or counts.shape[1] != 9 or not np.isfinite(counts).all():
        raise ValueError(f"Expected finite N x 9 sensor data: {path}, {counts.shape}")
    return counts[:, :3] * SCALE, hashlib.sha256(blob).hexdigest()


def preprocess(signal: np.ndarray, mode: str, cutoff: float = 5, fs: int = FS) -> np.ndarray:
    if mode == "raw":
        return signal.copy()
    if mode != "filtered":
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    if not 0 < cutoff < fs / 2:
        raise ValueError("Cutoff must be between zero and Nyquist")
    # Fourth-order design, forward/backward offline filtering of the full trial.
    return sosfiltfilt(butter(4, cutoff, fs=fs, output="sos"), signal, axis=0)


def peak_event(signal: np.ndarray, size: int = 200) -> tuple[int, int, int]:
    """Replaceable event locator; earliest maximum wins ties, edges are shifted."""
    if size < 1 or len(signal) < size:
        raise ValueError("Trial is shorter than the requested window")
    peak = int(np.argmax(np.linalg.norm(signal, axis=1)))
    start = int(np.clip(peak - size // 2, 0, len(signal) - size))
    return start, start + size, peak


def window_bounds(signal: np.ndarray, label: int, size=200, stride=100,
                  event_locator=peak_event) -> list[tuple[int, int, int]]:
    if size < 1 or stride < 1 or len(signal) < size:
        raise ValueError("Invalid size/stride or trial shorter than window")
    if label == 1:
        return [event_locator(signal, size)]
    if label != 0:
        raise ValueError("Label must be 0 or 1")
    return [(start, start + size, -1) for start in range(0, len(signal) - size + 1, stride)]


def split_subjects(rows, seed=42, validation_fraction=.2, test_fraction=.2):
    """Seeded subject split stratified by whether the subject has fall trials."""
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one")
    subjects = sorted({r["subject_id"] for r in rows})
    fall_subjects = {r["subject_id"] for r in rows if r["label"] == 1}
    rng = np.random.default_rng(seed)
    result = dict(train=[], validation=[], test=[])
    for has_falls in (False, True):
        cohort = [s for s in subjects if (s in fall_subjects) == has_falls]
        if not cohort:
            continue
        rng.shuffle(cohort)
        if len(cohort) < 3:
            raise ValueError("Each nonempty subject stratum needs >=3 subjects; supply a split JSON")
        nv = max(1, round(len(cohort) * validation_fraction))
        nt = max(1, round(len(cohort) * test_fraction))
        if nv + nt >= len(cohort):
            raise ValueError("Fractions leave no training subjects in a stratum")
        result["validation"].extend(cohort[:nv])
        result["test"].extend(cohort[nv:nv + nt])
        result["train"].extend(cohort[nv + nt:])
    return validate_split(rows, result)


def validate_split(rows, split):
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Split JSON needs train, validation, test lists")
    flattened = [s for values in split.values() for s in values]
    if any(not values for values in split.values()) or len(flattened) != len(set(flattened)):
        raise ValueError("Partitions must be nonempty and subjects must not overlap")
    if set(flattened) != {r["subject_id"] for r in rows}:
        raise ValueError("Split must cover exactly the dataset subjects")
    for name, subjects in split.items():
        if {r["label"] for r in rows if r["subject_id"] in subjects} != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
    return {k: sorted(v) for k, v in split.items()}


def grouped_folds(y, subjects, method="groupkfold", n_splits=5):
    """Indices usable with raw windows or features. Tune only inside each train fold."""
    if method not in {"groupkfold", "loso"}:
        raise ValueError("Method must be groupkfold or loso")
    splitter = LeaveOneGroupOut() if method == "loso" else GroupKFold(n_splits=n_splits)
    return splitter.split(np.zeros(len(y)), y, groups=subjects)


def fit_threshold(scores, y):
    """Maximize balanced accuracy, considering every distinct prediction boundary."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold tuning requires both classes")
    values, inverse = np.unique(scores, return_inverse=True)
    pos = np.bincount(inverse, weights=y, minlength=len(values))
    neg = np.bincount(inverse, weights=1 - y, minlength=len(values))
    tp = np.r_[pos.sum(), pos.sum() - np.cumsum(pos)]
    tn = np.r_[0, np.cumsum(neg)]
    quality = (tp / pos.sum() + tn / neg.sum()) / 2
    thresholds = np.r_[values, np.nextafter(values[-1], np.inf)]
    return float(thresholds[int(np.argmax(quality))])


def metrics(y, scores, threshold):
    tn, fp, fn, tp = map(int, confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(n=len(y), confusion_matrix=[[tn, fp], [fn, tp]],
                sensitivity=recall, specificity=specificity,
                balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                precision=tp / (tp + fp) if tp + fp else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                roc_auc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None)


def plot_trial(raw, filtered, row, destination, bounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = np.arange(len(raw)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for axis, index in zip(axes[:3], range(3)):
        axis.plot(time, raw[:, index], alpha=.6, lw=.7, label="Raw")
        axis.plot(time, filtered[:, index], lw=1, label="5 Hz filtered")
        axis.set_ylabel(f"a{'xyz'[index]} (g)")
    axes[3].plot(time, np.linalg.norm(raw, axis=1), alpha=.6, lw=.7)
    axes[3].plot(time, np.linalg.norm(filtered, axis=1), lw=1)
    axes[3].set_ylabel("Magnitude (g)")
    if row["label"]:
        start, end, peak = bounds[0]
        for axis in axes:
            axis.axvspan(start / FS, end / FS, color="orange", alpha=.2)
            axis.axvline(peak / FS, color="red", ls=":", lw=.7)
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{Path(row['path']).stem} — {row['class_name']}")
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_destination(path, data_root):
    path = path.resolve()
    allowed = [PROJECT / "processed", PROJECT / "outputs"]
    if not any(path.is_relative_to(root.resolve()) for root in allowed):
        raise ValueError("Generated files must be under project processed/ or outputs/")
    if path == data_root or path.is_relative_to(data_root) or data_root.is_relative_to(path):
        raise ValueError("Output and raw dataset directories must not overlap")
    if path.exists():
        raise ValueError(f"Refusing to overwrite an existing run: {path}")
    return path


def run(args):
    data_root = args.data.resolve()
    destination = safe_destination(args.out, data_root)
    audit = []
    rows = metadata(data_root, args.subject_mismatch, audit)
    split = (validate_split(rows, json.loads(args.split_json.read_text())) if args.split_json
             else split_subjects(rows, args.seed, args.validation_fraction, args.test_fraction))
    partition = {s: name for name, subjects in split.items() for s in subjects}
    destination.mkdir(parents=True)
    (destination / "plots").mkdir()
    (destination / "subject_identity_audit.json").write_text(json.dumps(audit, indent=2))
    (destination / "split_subjects.json").write_text(json.dumps(split, indent=2))
    # Fixed train-only examples: first available trial of distinct activities.
    examples = set()
    for label in (0, 1):
        activities = sorted({r["activity_id"] for r in rows if r["label"] == label
                             and partition[r["subject_id"]] == "train"})[:args.plots_per_class]
        for activity in activities:
            examples.add(next(r["path"] for r in rows if r["activity_id"] == activity
                             and partition[r["subject_id"]] == "train"))
    window_rows = []
    # Stage arrays per trial to keep memory bounded, then consolidate into .npy memmaps.
    stage = destination / "staging"
    stage.mkdir()
    for trial_index, row in enumerate(rows):
        signal, digest = load_trial(data_root / row["path"])
        bounds = window_bounds(signal, row["label"], args.window_size, args.stride)
        row.update(n_samples=len(signal), sha256=digest, split=partition[row["subject_id"]])
        filtered = preprocess(signal, "filtered")
        for mode, values in (("raw", signal), ("filtered", filtered)):
            np.save(stage / f"{trial_index}_{mode}.npy",
                    np.stack([values[start:end] for start, end, _ in bounds]).astype(np.float32))
        for start, end, peak in bounds:
            window_rows.append(dict(window_id=len(window_rows), trial_index=trial_index,
                                    path=row["path"], subject_id=row["subject_id"], activity_id=row["activity_id"],
                                    trial_number=row["trial_number"], label=row["label"], split=row["split"],
                                    start_sample=start, end_sample_exclusive=end, peak_sample=peak))
        if row["path"] in examples:
            plot_trial(signal, filtered, row, destination / "plots" / f"{Path(row['path']).stem}.png", bounds)
        if (trial_index + 1) % 250 == 0:
            print(f"Loaded {trial_index + 1}/{len(rows)} trials", flush=True)
    write_csv(destination / "trials.csv", rows)
    write_csv(destination / "windows.csv", window_rows)
    y = np.array([r["label"] for r in window_rows], dtype=np.int8)
    groups = np.array([r["subject_id"] for r in window_rows])
    partitions = np.array([r["split"] for r in window_rows])
    np.savez(destination / "labels_groups_splits.npz", y=y, subjects=groups, split=partitions)
    reports = {}
    for mode in ("raw", "filtered"):
        windows = np.lib.format.open_memmap(destination / f"X_{mode}.npy", mode="w+", dtype="float32",
                                           shape=(len(y), args.window_size, 3))
        offset = 0
        for index in range(len(rows)):
            part_path = stage / f"{index}_{mode}.npy"
            values = np.load(part_path)
            windows[offset:offset + len(values)] = values
            offset += len(values)
            part_path.unlink()  # Only this run's generated staging file.
        windows.flush()
        scores = np.empty(len(y), dtype=np.float64)
        for start in range(0, len(y), 2048):
            scores[start:start + 2048] = np.linalg.norm(windows[start:start + 2048], axis=2).max(axis=1)
        del windows
        threshold = fit_threshold(scores[partitions == "validation"], y[partitions == "validation"])
        reports[mode] = dict(threshold_g=threshold, tuned_on="validation", objective="balanced_accuracy",
                             metrics={name: metrics(y[partitions == name], scores[partitions == name], threshold)
                                      for name in split})
        np.save(destination / f"scores_{mode}.npy", scores)
    stage.rmdir()
    (destination / "baseline_metrics.json").write_text(json.dumps(reports, indent=2))
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")}
    config = dict(seed=args.seed, data_root=str(data_root), sampling_rate_hz=FS, channels="ADXL345 xyz",
                  counts_to_g=SCALE, filter=dict(order=4, cutoff_hz=5, implementation="sosfiltfilt", causal=False),
                  window_size=args.window_size, stride=args.stride, event_locator="raw_magnitude_argmax",
                  endpoint_policy="shift window inside recording; no padding; ADL incomplete tails dropped",
                  n_trials=len(rows), n_windows=len(y), python=platform.python_version(), versions=versions,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
                  baseline="max magnitude >= threshold; validation-tuned, no fitted training parameters")
    config["subject_mismatch_policy"] = args.subject_mismatch
    config["ignored_text_files"] = [p.relative_to(data_root).as_posix() for p in sorted(data_root.rglob("*.txt"))
                                    if not PATTERN.fullmatch(p.name)]
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    (destination / "requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n")
    print(json.dumps(dict(output=str(destination), trials=len(rows), windows=len(y), baseline=reports), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT / "processed" / "baseline_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--plots-per-class", type=int, default=3)
    parser.add_argument("--subject-mismatch", choices=["exclude", "folder"], default="exclude",
                        help="Exclude ambiguous subjects (default), or explicitly trust containing subject folder")
    args = parser.parse_args()
    if args.window_size <= 0 or args.stride <= 0 or args.plots_per_class < 0:
        parser.error("Window/stride must be positive; plot count must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
