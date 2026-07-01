from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.features import (  # noqa: E402
    discover_subjects,
    load_training_features,
    read_mat_v5,
    standardize_from_train,
)


SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MER-PS iteration experiments.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--output", default="experiments/results/iteration_001.json")
    parser.add_argument("--skip-signal", action="store_true")
    parser.add_argument("--ridge-alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--smooth-windows", default="3,5,9")
    parser.add_argument(
        "--no-baseline-correction",
        action="store_true",
        help="Build signal features without subtracting the 5-second pre-video baseline.",
    )
    parser.add_argument(
        "--fnirs-types",
        default="0,1,2",
        help="Comma-separated fNIRS signal type indices to use, e.g. 0,1,2 or 0,1,2,3,4,5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    ridge_alphas = [float(item) for item in args.ridge_alphas.split(",") if item.strip()]
    smooth_windows = [int(item) for item in args.smooth_windows.split(",") if item.strip()]
    fnirs_types = tuple(int(item) for item in args.fnirs_types.split(",") if item.strip())

    labels = load_labels(data_root, train_subjects + val_subjects)
    train_ids = [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects]
    val_ids = [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in val_subjects]
    y_train = np.stack([labels[sample_id] for sample_id in train_ids]).astype(np.float32)
    y_val = np.stack([labels[sample_id] for sample_id in val_ids]).astype(np.float32)

    results: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    center = np.full_like(y_val, 128.0)
    predictions["Center128"] = center
    results.append(score("Center128", y_val, center, "No signal; constant center of VA plane."))

    global_mean = np.tile(y_train.mean(axis=0, keepdims=True), (len(val_ids), 1))
    predictions["TrainMean"] = global_mean
    results.append(score("TrainMean", y_val, global_mean, "No signal; train-subject global mean."))

    video_mean = predict_video_time_mean(train_ids, y_train, val_ids)
    predictions["VideoTimeMean"] = video_mean
    results.append(
        score(
            "VideoTimeMean",
            y_val,
            video_mean,
            "No signal; mean trajectory by video and second from train subjects.",
        )
    )

    feature_summary = None
    if not args.skip_signal:
        features = load_or_build_feature_cache(
            data_root=data_root,
            cache_path=Path(args.feature_cache),
            subjects=train_subjects + val_subjects,
            train_subjects=train_subjects,
            baseline_correction=not args.no_baseline_correction,
            fnirs_types=fnirs_types,
        )
        feature_summary = features["summary"]
        sample_subjects = features["sample_subjects"]
        sample_ids = features["sample_ids"]
        x = features["x"]
        y_raw = features["y_raw"]
        train_idx = np.flatnonzero(np.isin(sample_subjects, train_subjects))
        val_idx = np.flatnonzero(np.isin(sample_subjects, val_subjects))
        x_train = x[train_idx]
        x_val = x[val_idx]
        y_train_signal = y_raw[train_idx]
        y_val_signal = y_raw[val_idx]
        val_signal_ids = sample_ids[val_idx].astype(str).tolist()

        if val_signal_ids != val_ids:
            id_to_prediction_order = {sample_id: index for index, sample_id in enumerate(val_signal_ids)}
            reorder = np.asarray([id_to_prediction_order[sample_id] for sample_id in val_ids])
        else:
            reorder = None

        ridge_predictions: dict[str, np.ndarray] = {}
        signal_train_ids = sample_ids[train_idx].astype(str).tolist()
        train_prior_signal = predict_video_time_mean(train_ids, y_train, signal_train_ids)
        val_prior_signal_order = predict_video_time_mean(train_ids, y_train, val_signal_ids)
        if reorder is not None:
            val_prior_signal = val_prior_signal_order[reorder]
        else:
            val_prior_signal = val_prior_signal_order

        for alpha in ridge_alphas:
            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train_signal)
            pred = np.clip(model.predict(x_val), 1.0, 255.0)
            if reorder is not None:
                pred = pred[reorder]
            name = f"RidgeSignal_a{format_alpha(alpha)}"
            ridge_predictions[name] = pred
            predictions[name] = pred
            results.append(
                score(
                    name,
                    y_val,
                    pred,
                    "EEG bandpower + fNIRS statistics flattened into multi-output Ridge.",
                )
            )

            for window in smooth_windows:
                smooth = smooth_predictions(val_ids, pred, window=window)
                smooth_name = f"{name}_smooth{window}"
                predictions[smooth_name] = smooth
                results.append(
                    score(
                        smooth_name,
                        y_val,
                        smooth,
                        f"RidgeSignal with per-trial moving-average smoothing window={window}.",
                )
            )

            residual_model = Ridge(alpha=alpha)
            residual_model.fit(x_train, y_train_signal - train_prior_signal)
            residual = residual_model.predict(x_val)
            residual_pred_signal_order = np.clip(val_prior_signal_order + residual, 1.0, 255.0)
            if reorder is not None:
                residual_pred = residual_pred_signal_order[reorder]
            else:
                residual_pred = residual_pred_signal_order
            residual_name = f"ResidualRidge_a{format_alpha(alpha)}"
            predictions[residual_name] = residual_pred
            results.append(
                score(
                    residual_name,
                    y_val,
                    residual_pred,
                    "VideoTimeMean prior plus EEG/fNIRS Ridge residual prediction.",
                )
            )
            for window in smooth_windows:
                smooth = smooth_predictions(val_ids, residual_pred, window=window)
                smooth_name = f"{residual_name}_smooth{window}"
                predictions[smooth_name] = smooth
                results.append(
                    score(
                        smooth_name,
                        y_val,
                        smooth,
                        f"ResidualRidge with per-trial moving-average smoothing window={window}.",
                    )
                )

        best_ridge_name = min(
            ridge_predictions,
            key=lambda item: float(score(item, y_val, ridge_predictions[item], "")["overall_mae"]),
        )
        best_ridge = ridge_predictions[best_ridge_name]
        for alpha in (0.25, 0.50, 0.75):
            blended = np.clip(alpha * best_ridge + (1.0 - alpha) * video_mean, 1.0, 255.0)
            name = f"Blend_{best_ridge_name}_VideoTimeMean_{alpha:.2f}"
            predictions[name] = blended
            results.append(
                score(
                    name,
                    y_val,
                    blended,
                    "Validation sweep blend of signal Ridge and video-time trajectory prior.",
                )
            )

    results = sorted(results, key=lambda item: float(item["overall_mae"]))
    output = {
        "data_root": str(data_root),
        "metric": "overall_mae = mean absolute error over valence and arousal, raw [1,255] scale",
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
        },
        "feature_summary": feature_summary,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def expand_subjects(value: str) -> list[str]:
    subjects: list[str] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            prefix, start_idx = start.rsplit("_", 1)
            _, end_idx = end.rsplit("_", 1)
            subjects.extend(f"{prefix}_{idx}" for idx in range(int(start_idx), int(end_idx) + 1))
        else:
            subjects.append(chunk)
    return subjects


def load_labels(data_root: Path, subjects: list[str]) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for subject in subjects:
        mat = read_mat_v5(data_root / "annotations" / f"{subject}_label.mat")
        for key in sorted((key for key in mat if key.startswith("video_")), key=video_sort_key):
            video = int(key.split("_", 1)[1])
            value = np.asarray(mat[key], dtype=np.float32)
            if value.shape[0] != 2 and value.shape[1] == 2:
                value = value.T
            for timestamp in range(value.shape[1]):
                sample_id = f"{subject}_V{video:02d}_T{timestamp:03d}"
                labels[sample_id] = value[:, timestamp]
    return labels


def predict_video_time_mean(train_ids: list[str], y_train: np.ndarray, val_ids: list[str]) -> np.ndarray:
    global_mean = y_train.mean(axis=0)
    video_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    video_count: dict[int, int] = defaultdict(int)
    vt_sum: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    vt_count: dict[tuple[int, int], int] = defaultdict(int)

    for sample_id, target in zip(train_ids, y_train):
        _, video, timestamp = parse_sample_id(sample_id)
        video_sum[video] += target
        video_count[video] += 1
        vt_sum[(video, timestamp)] += target
        vt_count[(video, timestamp)] += 1

    pred = np.empty((len(val_ids), 2), dtype=np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, timestamp = parse_sample_id(sample_id)
        key = (video, timestamp)
        if vt_count[key]:
            pred[index] = vt_sum[key] / vt_count[key]
        elif video_count[video]:
            pred[index] = video_sum[video] / video_count[video]
        else:
            pred[index] = global_mean
    return pred


def load_or_build_feature_cache(
    data_root: Path,
    cache_path: Path,
    subjects: list[str],
    train_subjects: list[str],
    baseline_correction: bool = True,
    fnirs_types: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "x": data["x"],
                "y_raw": data["y_raw"],
                "sample_subjects": data["sample_subjects"].astype(str),
                "sample_ids": data["sample_ids"].astype(str),
                "summary": json.loads(str(data["summary"].item())),
            }

    eeg, fnirs, y_unit, sample_subjects, subject_names, sample_ids = load_training_features(
        data_root,
        subjects=subjects,
        fnirs_types=fnirs_types,
        baseline_correction=baseline_correction,
        include_sample_ids=True,
        verbose=True,
    )
    train_idx = np.flatnonzero(np.isin(sample_subjects, train_subjects))
    eeg, fnirs, stats = standardize_from_train(eeg, fnirs, train_idx)
    x = np.concatenate([eeg.reshape(eeg.shape[0], -1), fnirs.reshape(fnirs.shape[0], -1)], axis=1)
    y_raw = y_unit * 254.0 + 1.0
    summary = {
        "subjects": subject_names,
        "samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape),
        "standardization_keys": sorted(stats),
        "baseline_correction": bool(baseline_correction),
        "fnirs_types": list(fnirs_types),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x.astype(np.float32),
        y_raw=y_raw.astype(np.float32),
        sample_subjects=sample_subjects.astype(str),
        sample_ids=sample_ids.astype(str),
        summary=json.dumps(summary, ensure_ascii=False),
    )
    return {
        "x": x.astype(np.float32),
        "y_raw": y_raw.astype(np.float32),
        "sample_subjects": sample_subjects.astype(str),
        "sample_ids": sample_ids.astype(str),
        "summary": summary,
    }


def score(name: str, y_true: np.ndarray, y_pred: np.ndarray, notes: str) -> dict[str, object]:
    errors = np.abs(y_pred - y_true)
    return {
        "method": name,
        "overall_mae": round(float(errors.mean()), 4),
        "valence_mae": round(float(errors[:, 0].mean()), 4),
        "arousal_mae": round(float(errors[:, 1].mean()), 4),
        "overall_mse": round(float(((y_pred - y_true) ** 2).mean()), 4),
        "notes": notes,
    }


def smooth_predictions(sample_ids: list[str], pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    radius = window // 2
    out = pred.copy()
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        items = sorted(items)
        indices = [index for _, index in items]
        values = pred[indices]
        for local_index, global_index in enumerate(indices):
            start = max(0, local_index - radius)
            stop = min(len(indices), local_index + radius + 1)
            out[global_index] = values[start:stop].mean(axis=0)
    return out


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def video_sort_key(key: str) -> int:
    return int(key.split("_", 1)[1])


def format_alpha(alpha: float) -> str:
    return str(alpha).replace(".", "p")


if __name__ == "__main__":
    main()
