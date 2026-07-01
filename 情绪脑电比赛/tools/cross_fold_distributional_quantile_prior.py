from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_signal_residual_over_pattern_prior import make_pattern_prior
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions
from tools.trial_basis_residual import parse_sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributional quantile video-time priors for MAE-oriented MER-PS decoding."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_101_distributional_quantile_prior.json")
    parser.add_argument("--quantiles", default="0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument("--lags", default="-4,-3,-2,-1,0,1")
    parser.add_argument("--smooth-windows", default="0,5,9,11,15,21,31,45,61")
    parser.add_argument("--dimwise-top-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        candidates = build_quantile_candidates(
            train_ids=train_ids,
            y_train=y_train,
            val_ids=val_ids,
            quantiles=parse_floats(args.quantiles),
            lags=parse_ints(args.lags),
            smooth_windows=parse_ints(args.smooth_windows),
        )
        candidates["PatternPrior_098"] = make_pattern_prior(train_ids, y_train, val_ids)

        fold_results = [
            score(name, y_val, pred, "Distributional quantile video-time prior candidate.")
            for name, pred in candidates.items()
        ]
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "candidate_count": len(candidates),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[
                    : args.top_k
                ],
            }
        )
        aggregate_truth.append(y_val)
        for name, pred in candidates.items():
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
    dimwise_results = build_metric_dimwise(aggregate_predictions, aggregate_truth, args.dimwise_top_n)

    output = {
        "method": "Distributional quantile video-time prior",
        "note": (
            "Instead of assuming the conditional median is always best after smoothing, this "
            "searches conditional quantile trajectories under subject-disjoint validation."
        ),
        "fold_size": args.fold_size,
        "grid": {
            "quantiles": parse_floats(args.quantiles),
            "lags": parse_ints(args.lags),
            "smooth_windows": parse_ints(args.smooth_windows),
        },
        "aggregate_results": aggregate_results[: args.top_k],
        "dimwise_results": dimwise_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_quantile_candidates(
    train_ids: list[str],
    y_train: np.ndarray,
    val_ids: list[str],
    quantiles: list[float],
    lags: list[int],
    smooth_windows: list[int],
) -> dict[str, np.ndarray]:
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_times: dict[int, list[int]] = defaultdict(list)
    for sample_id, value in zip(train_ids, y_train):
        _, video, timestamp = parse_sample_id(sample_id)
        grouped[(video, timestamp)].append(value.astype(np.float32))
        video_times[video].append(timestamp)
    video_times = {video: sorted(set(times)) for video, times in video_times.items()}
    global_quantiles = {
        q: np.quantile(y_train, q, axis=0).astype(np.float32) for q in quantiles
    }
    candidates: dict[str, np.ndarray] = {}
    for q_v in quantiles:
        for q_a in quantiles:
            for lag in lags:
                pred = quantile_prior(
                    val_ids=val_ids,
                    grouped=grouped,
                    video_times=video_times,
                    q_v=q_v,
                    q_a=q_a,
                    global_v=global_quantiles[q_v][0],
                    global_a=global_quantiles[q_a][1],
                    lag=lag,
                )
                base_name = f"QuantilePrior_qv{format_float(q_v)}_qa{format_float(q_a)}_lag{lag}"
                for window in smooth_windows:
                    if window > 1:
                        candidates[f"{base_name}_smooth{window}"] = smooth_predictions(
                            val_ids,
                            pred,
                            window,
                        )
                    else:
                        candidates[base_name] = pred
    return {name: np.clip(pred, 1.0, 255.0).astype(np.float32) for name, pred in candidates.items()}


def quantile_prior(
    val_ids: list[str],
    grouped: dict[tuple[int, int], list[np.ndarray]],
    video_times: dict[int, list[int]],
    q_v: float,
    q_a: float,
    global_v: float,
    global_a: float,
    lag: int,
) -> np.ndarray:
    cache: dict[tuple[int, int, float, float], np.ndarray] = {}
    rows = []
    fallback = np.asarray([global_v, global_a], dtype=np.float32)
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        key = (video, timestamp + lag)
        if key not in grouped:
            nearest = nearest_time(video_times.get(video, []), timestamp + lag)
            key = (video, nearest) if nearest is not None else key
        cache_key = (key[0], key[1], q_v, q_a)
        if cache_key not in cache:
            values = np.asarray(grouped.get(key, [fallback]), dtype=np.float32)
            cache[cache_key] = np.asarray(
                [
                    np.quantile(values[:, 0], q_v),
                    np.quantile(values[:, 1], q_a),
                ],
                dtype=np.float32,
            )
        rows.append(cache[cache_key])
    return np.stack(rows, axis=0).astype(np.float32)


def nearest_time(times: list[int], target: int) -> int | None:
    if not times:
        return None
    return min(times, key=lambda value: abs(value - target))


def build_metric_dimwise(
    aggregate_predictions: dict[str, list[np.ndarray]],
    aggregate_truth: list[np.ndarray],
    top_n: int,
) -> list[dict[str, object]]:
    y = np.concatenate(aggregate_truth, axis=0)
    preds = {
        name: np.concatenate(parts, axis=0)
        for name, parts in aggregate_predictions.items()
        if len(parts) == len(aggregate_truth)
    }
    summaries = []
    for name, pred in preds.items():
        diff = pred - y
        summaries.append(
            {
                "method": name,
                "valence_mae": float(np.abs(diff[:, 0]).mean()),
                "arousal_mae": float(np.abs(diff[:, 1]).mean()),
                "valence_mse": float((diff[:, 0] ** 2).mean()),
                "arousal_mse": float((diff[:, 1] ** 2).mean()),
            }
        )
    top_v = sorted(summaries, key=lambda item: item["valence_mae"])[:top_n]
    top_a = sorted(summaries, key=lambda item: item["arousal_mae"])[:top_n]
    rows = []
    for v_item in top_v:
        for a_item in top_a:
            rows.append(
                {
                    "method": f"QuantileDimwise_V[{v_item['method']}]__A[{a_item['method']}]",
                    "overall_mae": round((v_item["valence_mae"] + a_item["arousal_mae"]) / 2.0, 4),
                    "valence_mae": round(v_item["valence_mae"], 4),
                    "arousal_mae": round(a_item["arousal_mae"], 4),
                    "overall_mse": round((v_item["valence_mse"] + a_item["arousal_mse"]) / 2.0, 4),
                    "notes": "Metric-composed quantile dimwise candidate.",
                }
            )
    return sorted(rows, key=lambda item: float(item["overall_mae"]))


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
