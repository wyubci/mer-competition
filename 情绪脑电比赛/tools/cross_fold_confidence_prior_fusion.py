from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions
from tools.trial_basis_residual import parse_sample_id


@dataclass
class PriorStats:
    grouped: dict[tuple[int, int], list[np.ndarray]]
    video_times: dict[int, list[int]]
    global_mean: np.ndarray
    global_median: np.ndarray
    video_mean: dict[int, np.ndarray]
    video_median: dict[int, np.ndarray]
    dispersion_reference: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fold confidence-aware fusion of robust video-time priors."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_094_confidence_prior_fusion.json")
    parser.add_argument("--base-lag", type=int, default=-2)
    parser.add_argument("--base-smooth", type=int, default=11)
    parser.add_argument("--alt-lag", type=int, default=-1)
    parser.add_argument("--alt-smooth", type=int, default=9)
    parser.add_argument("--quantile-lows", default="40,50,60,70")
    parser.add_argument("--quantile-highs", default="75,85,90,95")
    parser.add_argument("--max-gates", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--long-smooths", default="15,21,31")
    parser.add_argument("--ensemble-weights", default="0.25,0.4,0.5,0.6,0.75")
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    q_lows = parse_floats(args.quantile_lows)
    q_highs = parse_floats(args.quantile_highs)
    max_gates = parse_floats(args.max_gates)
    long_smooths = parse_ints(args.long_smooths)
    ensemble_weights = parse_floats(args.ensemble_weights)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        stats = build_prior_stats(train_ids, y_train)
        fold_predictions = build_candidates(
            stats=stats,
            val_ids=val_ids,
            base_lag=args.base_lag,
            base_smooth=args.base_smooth,
            alt_lag=args.alt_lag,
            alt_smooth=args.alt_smooth,
            q_lows=q_lows,
            q_highs=q_highs,
            max_gates=max_gates,
            long_smooths=long_smooths,
            ensemble_weights=ensemble_weights,
        )
        fold_results = [
            score(name, y_val, pred, "Confidence-aware robust prior fusion.")
            for name, pred in fold_predictions.items()
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
        for name, pred in fold_predictions.items():
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
        "method": "Confidence-aware robust prior fusion",
        "note": (
            "Only training labels and target sample_ids are used. Dispersion is the median "
            "absolute deviation across train subjects at each shifted video-time key."
        ),
        "fold_size": args.fold_size,
        "base_prior": {"estimator": "median", "lag": args.base_lag, "smooth_window": args.base_smooth},
        "alt_prior": {"estimator": "median", "lag": args.alt_lag, "smooth_window": args.alt_smooth},
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_prior_stats(train_ids: list[str], y_train: np.ndarray) -> PriorStats:
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_values: dict[int, list[np.ndarray]] = defaultdict(list)
    video_times: dict[int, list[int]] = defaultdict(list)
    for sample_id, value in zip(train_ids, y_train):
        _, video, timestamp = parse_sample_id(sample_id)
        value = value.astype(np.float32)
        grouped[(video, timestamp)].append(value)
        video_values[video].append(value)
        video_times[video].append(timestamp)

    video_times = {video: sorted(set(times)) for video, times in video_times.items()}
    video_mean = {
        video: np.asarray(values, dtype=np.float32).mean(axis=0) for video, values in video_values.items()
    }
    video_median = {
        video: np.median(np.asarray(values, dtype=np.float32), axis=0) for video, values in video_values.items()
    }
    dispersion = []
    for values in grouped.values():
        arr = np.asarray(values, dtype=np.float32)
        med = np.median(arr, axis=0)
        dispersion.append(np.median(np.abs(arr - med[None, :]), axis=0))
    return PriorStats(
        grouped=grouped,
        video_times=video_times,
        global_mean=y_train.mean(axis=0).astype(np.float32),
        global_median=np.median(y_train, axis=0).astype(np.float32),
        video_mean=video_mean,
        video_median=video_median,
        dispersion_reference=np.asarray(dispersion, dtype=np.float32),
    )


def build_candidates(
    stats: PriorStats,
    val_ids: list[str],
    base_lag: int,
    base_smooth: int,
    alt_lag: int,
    alt_smooth: int,
    q_lows: list[float],
    q_highs: list[float],
    max_gates: list[float],
    long_smooths: list[int],
    ensemble_weights: list[float],
) -> dict[str, np.ndarray]:
    base_raw, dispersion = make_prior_and_dispersion(stats, val_ids, base_lag, smooth_window=0)
    base = smooth_predictions(val_ids, base_raw, base_smooth).astype(np.float32)
    dispersion = smooth_predictions(val_ids, dispersion, base_smooth).astype(np.float32)
    alt = make_prior(stats, val_ids, alt_lag, alt_smooth, estimator="median")
    mean_prior = make_prior(stats, val_ids, base_lag, base_smooth, estimator="mean")
    global_mean = np.tile(stats.global_mean[None, :], (len(val_ids), 1))
    global_median = np.tile(stats.global_median[None, :], (len(val_ids), 1))
    video_mean = make_video_level(stats.video_mean, stats.global_mean, val_ids)
    video_median = make_video_level(stats.video_median, stats.global_median, val_ids)

    candidates: dict[str, np.ndarray] = {
        f"RobustMedian_lag{base_lag}_smooth{base_smooth}": base,
        f"AltRobustMedian_lag{alt_lag}_smooth{alt_smooth}": alt,
    }
    for weight in ensemble_weights:
        pred = weight * base + (1.0 - weight) * alt
        candidates[f"StablePriorEnsemble_baseW{format_float(weight)}"] = clip(pred)

    references = {
        "globalMean": global_mean,
        "globalMedian": global_median,
        "videoMean": video_mean,
        "videoMedian": video_median,
        "meanPrior": mean_prior,
    }
    for window in long_smooths:
        references[f"smooth{window}"] = smooth_predictions(val_ids, base_raw, window).astype(np.float32)

    for q_low in q_lows:
        for q_high in q_highs:
            if q_high <= q_low:
                continue
            low = np.percentile(stats.dispersion_reference, q_low, axis=0)
            high = np.percentile(stats.dispersion_reference, q_high, axis=0)
            gate = uncertainty_gate(dispersion, low, high)
            for max_gate in max_gates:
                scaled_gate = np.clip(max_gate * gate, 0.0, 1.0).astype(np.float32)
                for ref_name, ref in references.items():
                    pred = (1.0 - scaled_gate) * base + scaled_gate * ref
                    name = (
                        f"UncertaintyBlend_{ref_name}_q{format_float(q_low)}-"
                        f"{format_float(q_high)}_g{format_float(max_gate)}"
                    )
                    candidates[name] = clip(pred)

    return candidates


def make_prior(
    stats: PriorStats,
    val_ids: list[str],
    lag: int,
    smooth_window: int,
    estimator: str,
) -> np.ndarray:
    prior, _ = make_prior_and_dispersion(stats, val_ids, lag, smooth_window=0, estimator=estimator)
    if smooth_window > 1:
        prior = smooth_predictions(val_ids, prior, smooth_window).astype(np.float32)
    return clip(prior)


def make_prior_and_dispersion(
    stats: PriorStats,
    val_ids: list[str],
    lag: int,
    smooth_window: int,
    estimator: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    dispersions = []
    value_cache: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    fallback = stats.global_median if estimator == "median" else stats.global_mean
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        key = (video, timestamp + lag)
        if key not in stats.grouped:
            nearest = nearest_time(stats.video_times.get(video, []), timestamp + lag)
            key = (video, nearest) if nearest is not None else key
        cache_key = (key[0], key[1], estimator)
        if cache_key not in value_cache:
            arr = np.asarray(stats.grouped.get(key, [fallback]), dtype=np.float32)
            if estimator == "mean":
                center = arr.mean(axis=0)
            elif estimator == "median":
                center = np.median(arr, axis=0)
            else:
                raise ValueError(estimator)
            med = np.median(arr, axis=0)
            disp = np.median(np.abs(arr - med[None, :]), axis=0)
            value_cache[cache_key] = (center.astype(np.float32), disp.astype(np.float32))
        center, disp = value_cache[cache_key]
        rows.append(center)
        dispersions.append(disp)
    prior = np.stack(rows, axis=0).astype(np.float32)
    dispersion = np.stack(dispersions, axis=0).astype(np.float32)
    if smooth_window > 1:
        prior = smooth_predictions(val_ids, prior, smooth_window).astype(np.float32)
        dispersion = smooth_predictions(val_ids, dispersion, smooth_window).astype(np.float32)
    return prior, dispersion


def make_video_level(video_values: dict[int, np.ndarray], fallback: np.ndarray, val_ids: list[str]) -> np.ndarray:
    rows = []
    for sample_id in val_ids:
        _, video, _ = parse_sample_id(sample_id)
        rows.append(video_values.get(video, fallback))
    return np.stack(rows, axis=0).astype(np.float32)


def uncertainty_gate(dispersion: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    denom = np.maximum(high - low, 1e-6)
    return np.clip((dispersion - low[None, :]) / denom[None, :], 0.0, 1.0).astype(np.float32)


def nearest_time(times: list[int], target: int) -> int | None:
    if not times:
        return None
    return min(times, key=lambda value: abs(value - target))


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def clip(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 255.0).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
