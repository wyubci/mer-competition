from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import (  # noqa: E402
    build_candidates as build_reference_candidates,
    build_prior_stats,
    ids_for_subjects,
    labels_to_array,
    parse_floats,
    parse_ints,
)
from tools.cross_fold_oof_prior_stacking import make_pattern_098  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


@dataclass
class RobustStats:
    grouped: dict[tuple[int, int], np.ndarray]
    video_times: dict[int, list[int]]
    global_mean: np.ndarray
    global_median: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fold robust video-time aggregation: trimmed mean, winsor mean, drop extremes."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_540_trimmed_prior.json")
    parser.add_argument("--lags", default="-4,-3,-2,-1,0,1,2")
    parser.add_argument("--smooths", default="5,9,11,15,21,43,51,61")
    parser.add_argument(
        "--estimators",
        default=(
            "mean,median,drop1,drop2,trim5,trim10,trim15,trim20,"
            "winsor5,winsor10,winsor15,winsor20,huber15,huber20,q45,q50,q55"
        ),
    )
    parser.add_argument("--blend-weights", default="0.25,0.5,0.75")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    estimators = parse_strings(args.estimators)
    lags = parse_ints(args.lags)
    smooths = parse_ints(args.smooths)
    blend_weights = parse_floats(args.blend_weights)

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] trimmed/winsor prior search", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        stats = build_robust_stats(train_ids, y_train)

        reference = build_reference_predictions(train_ids, y_train, val_ids)
        fold_predictions: dict[str, np.ndarray] = dict(reference)
        robust_priors = build_robust_prior_grid(stats, val_ids, estimators, lags, smooths)
        fold_predictions.update(robust_priors)

        pattern_ref = reference["PatternPrior_098_reference"]
        for name, pred in robust_priors.items():
            mixed = pattern_ref.copy()
            mixed[:, 0] = pred[:, 0]
            fold_predictions[f"V[{name}]_A[Pattern098]"] = clip(mixed)
            mixed = pattern_ref.copy()
            mixed[:, 1] = pred[:, 1]
            fold_predictions[f"V[Pattern098]_A[{name}]"] = clip(mixed)
            for weight in blend_weights:
                fold_predictions[f"Blend_{name}_Pattern098_w{fmt(weight)}"] = clip(
                    weight * pred + (1.0 - weight) * pattern_ref
                )

        robust_pattern = build_pattern098_with_robust_mean_refs(stats, val_ids, estimators)
        fold_predictions.update(robust_pattern)

        fold_results = [
            evaluate_candidate(metric_acc, name, y_val, pred, "Robust video-time aggregation prior.")
            for name, pred in fold_predictions.items()
        ]
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[: args.top_k],
            }
        )

    aggregate_results = sorted(
        [finalize_metric(name, payload) for name, payload in metric_acc.items()],
        key=lambda item: float(item["overall_mae"]),
    )
    output = {
        "method": "Robust video-time aggregation search",
        "note": (
            "Tests replacing ordinary video-time means with drop-extreme, trimmed, winsorized, "
            "Huber, and quantile centers. Pattern variants replace only the meanPrior references "
            "inside PatternPrior_098 while keeping the same subject-disjoint folds."
        ),
        "fold_size": args.fold_size,
        "estimators": estimators,
        "lags": lags,
        "smooths": smooths,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_reference_predictions(train_ids: list[str], y_train: np.ndarray, val_ids: list[str]) -> dict[str, np.ndarray]:
    ref_stats = build_prior_stats(train_ids, y_train)
    candidates = build_reference_candidates(
        stats=ref_stats,
        val_ids=val_ids,
        base_lag=-2,
        base_smooth=11,
        alt_lag=-1,
        alt_smooth=9,
        q_lows=[15.0, 20.0],
        q_highs=[45.0, 50.0, 55.0, 60.0, 70.0],
        max_gates=[0.25, 0.35, 0.45, 0.5, 0.55],
        long_smooths=[43, 51, 61],
        ensemble_weights=[0.5],
    )
    return {
        "RobustMedian_lag-2_smooth11_reference": candidates["RobustMedian_lag-2_smooth11"],
        "PatternPrior_098_reference": make_pattern_098(val_ids, candidates),
    }


def build_robust_stats(train_ids: list[str], y_train: np.ndarray) -> RobustStats:
    grouped_lists: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_times: dict[int, set[int]] = defaultdict(set)
    for sample_id, value in zip(train_ids, y_train):
        _, video, timestamp = parse_sample_id(sample_id)
        grouped_lists[(video, timestamp)].append(value.astype(np.float32))
        video_times[video].add(timestamp)
    return RobustStats(
        grouped={key: np.asarray(values, dtype=np.float32) for key, values in grouped_lists.items()},
        video_times={video: sorted(times) for video, times in video_times.items()},
        global_mean=y_train.mean(axis=0).astype(np.float32),
        global_median=np.median(y_train, axis=0).astype(np.float32),
    )


def build_robust_prior_grid(
    stats: RobustStats,
    val_ids: list[str],
    estimators: list[str],
    lags: list[int],
    smooths: list[int],
) -> dict[str, np.ndarray]:
    out = {}
    for estimator in estimators:
        for lag in lags:
            raw = make_prior(stats, val_ids, estimator=estimator, lag=lag, smooth=0)
            for smooth in smooths:
                pred = smooth_predictions(val_ids, raw, smooth).astype(np.float32) if smooth > 1 else raw
                out[f"{estimator}_lag{lag}_smooth{smooth}"] = clip(pred)
    return out


def build_pattern098_with_robust_mean_refs(
    stats: RobustStats,
    val_ids: list[str],
    estimators: list[str],
) -> dict[str, np.ndarray]:
    base_raw = make_prior(stats, val_ids, estimator="median", lag=-2, smooth=0)
    base = smooth_predictions(val_ids, base_raw, 11).astype(np.float32)
    dispersion = make_dispersion(stats, val_ids, lag=-2)
    dispersion = smooth_predictions(val_ids, dispersion, 11).astype(np.float32)

    smooth51 = smooth_predictions(val_ids, base_raw, 51).astype(np.float32)
    smooth61 = smooth_predictions(val_ids, base_raw, 61).astype(np.float32)
    current_v_dynamic = uncertainty_blend(base, smooth61, dispersion, q_low=15.0, q_high=45.0, max_gate=0.45)
    current_a_stable = uncertainty_blend(base, smooth51, dispersion, q_low=20.0, q_high=45.0, max_gate=0.50)

    slopes = slope_by_trial(val_ids, base)
    v_stable = np.abs(slopes[:, 0]) <= np.percentile(np.abs(slopes[:, 0]), 65.0)
    a_stable = np.abs(slopes[:, 1]) <= np.percentile(np.abs(slopes[:, 1]), 55.0)

    out = {}
    for estimator in estimators:
        robust_ref = make_prior(stats, val_ids, estimator=estimator, lag=-2, smooth=11)
        v_stable_ref = uncertainty_blend(base, robust_ref, dispersion, q_low=15.0, q_high=45.0, max_gate=0.25)
        a_dynamic_ref = uncertainty_blend(base, robust_ref, dispersion, q_low=20.0, q_high=55.0, max_gate=0.55)

        pred = base.copy()
        pred[:, 0] = np.where(v_stable, v_stable_ref[:, 0], current_v_dynamic[:, 0])
        pred[:, 1] = np.where(a_stable, current_a_stable[:, 1], a_dynamic_ref[:, 1])
        out[f"Pattern098_RobustMeanRefs[{estimator}]"] = clip(pred)

        pred_v = base.copy()
        pred_v[:, 0] = np.where(v_stable, v_stable_ref[:, 0], current_v_dynamic[:, 0])
        pred_v[:, 1] = np.where(a_stable, current_a_stable[:, 1], base[:, 1])
        out[f"Pattern098_VmeanRefOnly[{estimator}]"] = clip(pred_v)

        pred_a = base.copy()
        pred_a[:, 0] = base[:, 0]
        pred_a[:, 1] = np.where(a_stable, current_a_stable[:, 1], a_dynamic_ref[:, 1])
        out[f"Pattern098_AmeanRefOnly[{estimator}]"] = clip(pred_a)
    return out


def make_prior(stats: RobustStats, val_ids: list[str], estimator: str, lag: int, smooth: int) -> np.ndarray:
    rows = []
    cache: dict[tuple[int, int, str], np.ndarray] = {}
    fallback = stats.global_median if estimator in {"median", "q50"} else stats.global_mean
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        query_time = timestamp + lag
        if (video, query_time) not in stats.grouped:
            nearest = nearest_time(stats.video_times.get(video, []), query_time)
            query_time = nearest if nearest is not None else query_time
        key = (video, query_time, estimator)
        if key not in cache:
            arr = stats.grouped.get((video, query_time))
            cache[key] = robust_center(arr, estimator, fallback).astype(np.float32)
        rows.append(cache[key])
    pred = np.stack(rows, axis=0).astype(np.float32)
    if smooth > 1:
        pred = smooth_predictions(val_ids, pred, smooth).astype(np.float32)
    return clip(pred)


def make_dispersion(stats: RobustStats, val_ids: list[str], lag: int) -> np.ndarray:
    rows = []
    cache: dict[tuple[int, int], np.ndarray] = {}
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        query_time = timestamp + lag
        if (video, query_time) not in stats.grouped:
            nearest = nearest_time(stats.video_times.get(video, []), query_time)
            query_time = nearest if nearest is not None else query_time
        key = (video, query_time)
        if key not in cache:
            arr = stats.grouped.get(key)
            if arr is None:
                cache[key] = np.zeros(2, dtype=np.float32)
            else:
                med = np.median(arr, axis=0)
                cache[key] = np.median(np.abs(arr - med[None, :]), axis=0).astype(np.float32)
        rows.append(cache[key])
    return np.stack(rows, axis=0).astype(np.float32)


def robust_center(arr: np.ndarray | None, estimator: str, fallback: np.ndarray) -> np.ndarray:
    if arr is None or arr.size == 0:
        return fallback.astype(np.float32)
    x = np.asarray(arr, dtype=np.float32)
    if estimator == "mean":
        return x.mean(axis=0)
    if estimator in {"median", "q50"}:
        return np.median(x, axis=0)
    if estimator.startswith("q"):
        return np.percentile(x, float(estimator[1:]), axis=0)
    if estimator.startswith("drop"):
        drop = int(estimator[4:])
        return trimmed_mean_by_count(x, drop)
    if estimator.startswith("trim"):
        percent = float(estimator[4:])
        drop = int(np.floor(x.shape[0] * percent / 100.0))
        if drop == 0 and percent > 0.0 and x.shape[0] >= 10:
            drop = 1
        return trimmed_mean_by_count(x, drop)
    if estimator.startswith("winsor"):
        percent = float(estimator[6:])
        return winsorized_mean(x, percent)
    if estimator.startswith("huber"):
        scale = float(estimator[5:]) / 10.0
        return huber_center(x, scale)
    raise ValueError(f"Unknown estimator: {estimator}")


def trimmed_mean_by_count(x: np.ndarray, drop: int) -> np.ndarray:
    if drop <= 0 or x.shape[0] <= 2 * drop:
        return x.mean(axis=0)
    sorted_x = np.sort(x, axis=0)
    return sorted_x[drop : x.shape[0] - drop].mean(axis=0)


def winsorized_mean(x: np.ndarray, percent: float) -> np.ndarray:
    if percent <= 0.0:
        return x.mean(axis=0)
    low = np.percentile(x, percent, axis=0)
    high = np.percentile(x, 100.0 - percent, axis=0)
    return np.clip(x, low[None, :], high[None, :]).mean(axis=0)


def huber_center(x: np.ndarray, scale: float) -> np.ndarray:
    center = np.median(x, axis=0)
    mad = np.median(np.abs(x - center[None, :]), axis=0)
    sigma = np.maximum(1.4826 * mad, 1.0)
    threshold = scale * sigma
    for _ in range(6):
        residual = x - center[None, :]
        weights = np.minimum(1.0, threshold[None, :] / np.maximum(np.abs(residual), 1e-6))
        center = (weights * x).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-6)
    return center


def uncertainty_blend(
    base: np.ndarray,
    reference: np.ndarray,
    dispersion: np.ndarray,
    q_low: float,
    q_high: float,
    max_gate: float,
) -> np.ndarray:
    low = np.percentile(dispersion, q_low, axis=0)
    high = np.percentile(dispersion, q_high, axis=0)
    gate = np.clip((dispersion - low[None, :]) / np.maximum(high - low, 1e-6)[None, :], 0.0, 1.0)
    gate = np.clip(max_gate * gate, 0.0, 1.0).astype(np.float32)
    return clip((1.0 - gate) * base + gate * reference)


def slope_by_trial(sample_ids: list[str], values: np.ndarray) -> np.ndarray:
    slopes = np.zeros_like(values, dtype=np.float32)
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        seq = values[indices].astype(np.float32)
        if len(indices) >= 3:
            grad = np.gradient(seq, axis=0)
        else:
            grad = np.zeros_like(seq)
        slopes[indices] = grad
    return slopes


def evaluate_candidate(
    metric_acc: dict[str, dict[str, object]],
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    notes: str,
) -> dict[str, object]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    payload = metric_acc.setdefault(
        name,
        {"sum_abs": np.zeros(2, dtype=np.float64), "sum_sq": 0.0, "count": 0, "notes": notes},
    )
    payload["sum_abs"] = np.asarray(payload["sum_abs"]) + abs_err.sum(axis=0)
    payload["sum_sq"] = float(payload["sum_sq"]) + float((err**2).sum())
    payload["count"] = int(payload["count"]) + int(y_true.shape[0])
    return {
        "method": name,
        "overall_mae": round(float(abs_err.mean()), 4),
        "valence_mae": round(float(abs_err[:, 0].mean()), 4),
        "arousal_mae": round(float(abs_err[:, 1].mean()), 4),
        "overall_mse": round(float((err**2).mean()), 4),
        "notes": notes,
    }


def finalize_metric(name: str, payload: dict[str, object]) -> dict[str, object]:
    count = int(payload["count"])
    sum_abs = np.asarray(payload["sum_abs"], dtype=np.float64)
    return {
        "method": name,
        "overall_mae": round(float(sum_abs.sum() / (2 * count)), 4),
        "valence_mae": round(float(sum_abs[0] / count), 4),
        "arousal_mae": round(float(sum_abs[1] / count), 4),
        "overall_mse": round(float(payload["sum_sq"]) / (2 * count), 4),
        "notes": str(payload["notes"]),
    }


def nearest_time(times: list[int], target: int) -> int | None:
    if not times:
        return None
    return min(times, key=lambda value: abs(value - target))


def clip(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 255.0).astype(np.float32)


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def fmt(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
