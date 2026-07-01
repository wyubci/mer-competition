from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions
from tools.trial_basis_residual import parse_sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fold search for robust video-time priors.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_090_cross_fold_prior_search.json")
    parser.add_argument("--lags", default="-4,-2,-1,0,1,2,4")
    parser.add_argument("--trims", default="0,0.1,0.2,0.3")
    parser.add_argument("--shrinks", default="0,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--smooth-windows", default="0,3,5,9,15")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--dimwise-top-k", type=int, default=4)
    parser.add_argument("--estimators", default="mean,median,trim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    lags = parse_ints(args.lags)
    trims = parse_floats(args.trims)
    shrinks = parse_floats(args.shrinks)
    smooth_windows = parse_ints(args.smooth_windows)
    estimators = [item.strip() for item in args.estimators.split(",") if item.strip()]

    aggregate_truth = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        candidate_predictions = build_candidates(
            train_ids=train_ids,
            y_train=y_train,
            val_ids=val_ids,
            lags=lags,
            trims=trims,
            shrinks=shrinks,
            smooth_windows=smooth_windows,
            estimators=estimators,
        )
        if args.dimwise_top_k > 0:
            dimwise_predictions = build_dimwise_candidates(
                y_val,
                candidate_predictions,
                top_k=args.dimwise_top_k,
            )
            candidate_predictions.update(dimwise_predictions)

        fold_results = [
            score(name, y_val, pred, "Subject-disjoint fold robust prior candidate.")
            for name, pred in candidate_predictions.items()
        ]
        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": fold_results[: args.top_k],
            }
        )
        aggregate_truth.append(y_val)
        for name, pred in candidate_predictions.items():
            aggregate_predictions[name].append(pred)

    y_all = np.concatenate(aggregate_truth, axis=0)
    aggregate_results = []
    for name, parts in aggregate_predictions.items():
        if len(parts) != len(folds):
            continue
        aggregate_results.append(
            score(
                name,
                y_all,
                np.concatenate(parts, axis=0),
                "Weighted aggregate across subject-disjoint folds.",
            )
        )
    aggregate_results = sorted(aggregate_results, key=lambda item: float(item["overall_mae"]))
    output = {
        "method": "Robust video-time prior cross-fold search",
        "fold_size": args.fold_size,
        "lags": lags,
        "trims": trims,
        "shrinks": shrinks,
        "smooth_windows": smooth_windows,
        "estimators": estimators,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_candidates(
    train_ids: list[str],
    y_train: np.ndarray,
    val_ids: list[str],
    lags: list[int],
    trims: list[float],
    shrinks: list[float],
    smooth_windows: list[int],
    estimators: list[str],
) -> dict[str, np.ndarray]:
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_times: dict[int, list[int]] = defaultdict(list)
    for sample_id, value in zip(train_ids, y_train):
        _, video, timestamp = parse_sample_id(sample_id)
        grouped[(video, timestamp)].append(value.astype(np.float32))
        video_times[video].append(timestamp)
    video_times = {video: sorted(set(times)) for video, times in video_times.items()}
    global_mean = y_train.mean(axis=0).astype(np.float32)
    center = np.asarray([128.0, 128.0], dtype=np.float32)

    candidates: dict[str, np.ndarray] = {
        "Center128": np.tile(center[None, :], (len(val_ids), 1)),
        "TrainMean": np.tile(global_mean[None, :], (len(val_ids), 1)),
    }
    for estimator in estimators:
        if estimator not in {"mean", "median", "trim"}:
            raise ValueError(f"Unknown estimator: {estimator}")
        trim_values = trims if estimator == "trim" else [0.0]
        for trim in trim_values:
            for lag in lags:
                base = make_prior(
                    val_ids=val_ids,
                    grouped=grouped,
                    video_times=video_times,
                    global_mean=global_mean,
                    estimator=estimator,
                    trim=trim,
                    lag=lag,
                )
                base_name = f"Prior_{estimator}_trim{format_float(trim)}_lag{lag}"
                for shrink in shrinks:
                    shrink_pred = (1.0 - shrink) * base + shrink * global_mean[None, :]
                    shrink_name = f"{base_name}_shrink{format_float(shrink)}"
                    for window in smooth_windows:
                        if window > 1:
                            pred = smooth_predictions(val_ids, shrink_pred, window=window)
                            name = f"{shrink_name}_smooth{window}"
                        else:
                            pred = shrink_pred
                            name = shrink_name
                        candidates[name] = np.clip(pred, 1.0, 255.0).astype(np.float32)
    return candidates


def make_prior(
    val_ids: list[str],
    grouped: dict[tuple[int, int], list[np.ndarray]],
    video_times: dict[int, list[int]],
    global_mean: np.ndarray,
    estimator: str,
    trim: float,
    lag: int,
) -> np.ndarray:
    rows = []
    cache: dict[tuple[int, int], np.ndarray] = {}
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        key = (video, timestamp + lag)
        if key not in grouped:
            nearest = nearest_time(video_times.get(video, []), timestamp + lag)
            key = (video, nearest) if nearest is not None else key
        if key not in cache:
            values = np.asarray(grouped.get(key, [global_mean]), dtype=np.float32)
            if estimator == "mean":
                cache[key] = values.mean(axis=0)
            elif estimator == "median":
                cache[key] = np.median(values, axis=0)
            elif estimator == "trim":
                cache[key] = trimmed_mean(values, trim)
            else:
                raise ValueError(estimator)
        rows.append(cache[key])
    return np.stack(rows, axis=0).astype(np.float32)


def trimmed_mean(values: np.ndarray, trim: float) -> np.ndarray:
    if trim <= 0:
        return values.mean(axis=0)
    n = values.shape[0]
    cut = int(np.floor(n * trim))
    if cut <= 0 or 2 * cut >= n:
        return values.mean(axis=0)
    sorted_values = np.sort(values, axis=0)
    return sorted_values[cut : n - cut].mean(axis=0)


def build_dimwise_candidates(
    y_val: np.ndarray,
    candidates: dict[str, np.ndarray],
    top_k: int,
) -> dict[str, np.ndarray]:
    valence_rank = sorted(
        candidates.items(),
        key=lambda item: float(np.mean(np.abs(item[1][:, 0] - y_val[:, 0]))),
    )[:top_k]
    arousal_rank = sorted(
        candidates.items(),
        key=lambda item: float(np.mean(np.abs(item[1][:, 1] - y_val[:, 1]))),
    )[:top_k]
    dimwise = {}
    for v_name, v_pred in valence_rank:
        for a_name, a_pred in arousal_rank:
            name = f"DimwiseV[{v_name}]__A[{a_name}]"
            dimwise[name] = np.stack([v_pred[:, 0], a_pred[:, 1]], axis=1).astype(np.float32)
    return dimwise


def nearest_time(times: list[int], target: int) -> int | None:
    if not times:
        return None
    return min(times, key=lambda value: abs(value - target))


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
