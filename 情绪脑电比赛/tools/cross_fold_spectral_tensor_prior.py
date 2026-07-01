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
        description="Cross-fold spectral/tensor trajectory priors for MER-PS."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_100_spectral_tensor_prior.json")
    parser.add_argument("--lags", default="-3,-2,-1,0")
    parser.add_argument("--fourier-keeps", default="3,5,7,9,13,17")
    parser.add_argument("--ssa-windows", default="9,15,21,31")
    parser.add_argument("--ssa-ranks", default="1,2,3,4,6")
    parser.add_argument("--svd-ranks", default="0,1,2,3,4,6,8")
    parser.add_argument("--smooth-windows", default="0,5,9,15")
    parser.add_argument("--dimwise-top-n", type=int, default=24)
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
        candidates = build_spectral_candidates(
            train_subjects=train_subjects,
            train_ids=train_ids,
            y_train=y_train,
            val_ids=val_ids,
            lags=parse_ints(args.lags),
            fourier_keeps=parse_ints(args.fourier_keeps),
            ssa_windows=parse_ints(args.ssa_windows),
            ssa_ranks=parse_ints(args.ssa_ranks),
            svd_ranks=parse_ints(args.svd_ranks),
            smooth_windows=parse_ints(args.smooth_windows),
        )
        candidates["PatternPrior_098"] = make_pattern_prior(train_ids, y_train, val_ids)

        fold_results = [
            score(name, y_val, pred, "Spectral/tensor trajectory prior candidate.")
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
        "method": "Spectral/Tensor trajectory priors",
        "note": (
            "A non-local trajectory framework: Fourier low-pass, Hankel SSA denoising, and "
            "subject-by-time SVD denoising are used instead of local moving-average-only priors."
        ),
        "fold_size": args.fold_size,
        "grid": {
            "lags": parse_ints(args.lags),
            "fourier_keeps": parse_ints(args.fourier_keeps),
            "ssa_windows": parse_ints(args.ssa_windows),
            "ssa_ranks": parse_ints(args.ssa_ranks),
            "svd_ranks": parse_ints(args.svd_ranks),
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


def build_spectral_candidates(
    train_subjects: list[str],
    train_ids: list[str],
    y_train: np.ndarray,
    val_ids: list[str],
    lags: list[int],
    fourier_keeps: list[int],
    ssa_windows: list[int],
    ssa_ranks: list[int],
    svd_ranks: list[int],
    smooth_windows: list[int],
) -> dict[str, np.ndarray]:
    trajectories = build_video_trajectories(train_subjects, train_ids, y_train)
    candidates: dict[str, np.ndarray] = {}

    for lag in lags:
        median_prior = trajectory_prior(trajectories, val_ids, lag, source="median")
        mean_prior = trajectory_prior(trajectories, val_ids, lag, source="mean")
        add_smoothed(candidates, f"MedianTrajectory_lag{lag}", val_ids, median_prior, smooth_windows)
        add_smoothed(candidates, f"MeanTrajectory_lag{lag}", val_ids, mean_prior, smooth_windows)

        for keep in fourier_keeps:
            pred = np.stack(
                [
                    lowpass_by_trial(val_ids, median_prior[:, 0], keep),
                    lowpass_by_trial(val_ids, median_prior[:, 1], keep),
                ],
                axis=1,
            )
            add_smoothed(candidates, f"FourierMedian_lag{lag}_keep{keep}", val_ids, pred, [0])

        for window in ssa_windows:
            for rank in ssa_ranks:
                pred = np.stack(
                    [
                        ssa_by_trial(val_ids, median_prior[:, 0], window, rank),
                        ssa_by_trial(val_ids, median_prior[:, 1], window, rank),
                    ],
                    axis=1,
                )
                add_smoothed(candidates, f"SSAMedian_lag{lag}_w{window}_r{rank}", val_ids, pred, [0])

        for rank in svd_ranks:
            for agg in ("mean", "median"):
                pred = svd_tensor_prior(trajectories, val_ids, lag, rank, agg)
                add_smoothed(candidates, f"SubjectTimeSVD_lag{lag}_r{rank}_{agg}", val_ids, pred, smooth_windows)

    return {name: clip(pred) for name, pred in candidates.items()}


def build_video_trajectories(
    train_subjects: list[str],
    train_ids: list[str],
    y_train: np.ndarray,
) -> dict[int, dict[str, object]]:
    by_video_subject: dict[tuple[int, str], dict[int, np.ndarray]] = defaultdict(dict)
    for sample_id, value in zip(train_ids, y_train):
        subject, video, timestamp = parse_sample_id(sample_id)
        by_video_subject[(video, subject)][timestamp] = value.astype(np.float32)

    out: dict[int, dict[str, object]] = {}
    videos = sorted({video for video, _ in by_video_subject})
    for video in videos:
        times = sorted(
            {
                timestamp
                for (item_video, _), values in by_video_subject.items()
                if item_video == video
                for timestamp in values
            }
        )
        time_to_col = {timestamp: index for index, timestamp in enumerate(times)}
        matrix = np.full((len(train_subjects), len(times), 2), np.nan, dtype=np.float32)
        for row, subject in enumerate(train_subjects):
            values = by_video_subject.get((video, subject), {})
            for timestamp, value in values.items():
                matrix[row, time_to_col[timestamp]] = value
        median = np.nanmedian(matrix, axis=0).astype(np.float32)
        mean = np.nanmean(matrix, axis=0).astype(np.float32)
        out[video] = {
            "times": times,
            "matrix": matrix,
            "median": median,
            "mean": mean,
        }
    return out


def trajectory_prior(
    trajectories: dict[int, dict[str, object]],
    val_ids: list[str],
    lag: int,
    source: str,
) -> np.ndarray:
    rows = []
    global_center = global_video_center(trajectories, source)
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        item = trajectories.get(video)
        if item is None:
            rows.append(global_center)
            continue
        rows.append(sample_from_trajectory(item["times"], item[source], timestamp + lag))
    return np.stack(rows, axis=0).astype(np.float32)


def svd_tensor_prior(
    trajectories: dict[int, dict[str, object]],
    val_ids: list[str],
    lag: int,
    rank: int,
    agg: str,
) -> np.ndarray:
    denoised_cache: dict[tuple[int, int, str], np.ndarray] = {}
    rows = []
    global_center = global_video_center(trajectories, "median")
    for sample_id in val_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        item = trajectories.get(video)
        if item is None:
            rows.append(global_center)
            continue
        key = (video, rank, agg)
        if key not in denoised_cache:
            denoised_cache[key] = denoise_video_matrix(item["matrix"], rank, agg)
        rows.append(sample_from_trajectory(item["times"], denoised_cache[key], timestamp + lag))
    return np.stack(rows, axis=0).astype(np.float32)


def denoise_video_matrix(matrix: np.ndarray, rank: int, agg: str) -> np.ndarray:
    outputs = []
    for dim in range(2):
        x = matrix[:, :, dim].astype(np.float32)
        col_median = np.nanmedian(x, axis=0).astype(np.float32)
        filled = np.where(np.isnan(x), col_median[None, :], x)
        if rank > 0:
            centered = filled - col_median[None, :]
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            keep = min(rank, s.size)
            reconstructed = col_median[None, :] + (u[:, :keep] * s[:keep]) @ vt[:keep]
        else:
            reconstructed = np.tile(col_median[None, :], (filled.shape[0], 1))
        if agg == "mean":
            outputs.append(np.mean(reconstructed, axis=0))
        elif agg == "median":
            outputs.append(np.median(reconstructed, axis=0))
        else:
            raise ValueError(agg)
    return np.stack(outputs, axis=1).astype(np.float32)


def add_smoothed(
    candidates: dict[str, np.ndarray],
    name: str,
    val_ids: list[str],
    pred: np.ndarray,
    smooth_windows: list[int],
) -> None:
    for window in smooth_windows:
        if window > 1:
            candidates[f"{name}_smooth{window}"] = smooth_predictions(val_ids, pred, window)
        else:
            candidates[name] = pred


def lowpass_by_trial(sample_ids: list[str], values: np.ndarray, keep: int) -> np.ndarray:
    out = values.astype(np.float32).copy()
    for indices in grouped_indices(sample_ids).values():
        trial = values[indices].astype(np.float32)
        freq = np.fft.rfft(trial)
        filtered = np.zeros_like(freq)
        filtered[: min(keep, freq.size)] = freq[: min(keep, freq.size)]
        out[indices] = np.fft.irfft(filtered, n=trial.size).astype(np.float32)
    return out


def ssa_by_trial(sample_ids: list[str], values: np.ndarray, window: int, rank: int) -> np.ndarray:
    out = values.astype(np.float32).copy()
    for indices in grouped_indices(sample_ids).values():
        out[indices] = ssa_denoise(values[indices].astype(np.float32), window, rank)
    return out


def ssa_denoise(values: np.ndarray, window: int, rank: int) -> np.ndarray:
    n = int(values.size)
    if n < 3:
        return values.copy()
    window = max(2, min(window, n - 1))
    cols = n - window + 1
    hankel = np.stack([values[start : start + window] for start in range(cols)], axis=1)
    u, s, vt = np.linalg.svd(hankel, full_matrices=False)
    keep = max(1, min(rank, s.size))
    reconstructed = (u[:, :keep] * s[:keep]) @ vt[:keep]
    out = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    for col in range(cols):
        out[col : col + window] += reconstructed[:, col]
        count[col : col + window] += 1.0
    return out / np.maximum(count, 1.0)


def grouped_indices(sample_ids: list[str]) -> dict[tuple[str, int], np.ndarray]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    return {
        key: np.asarray([index for _, index in sorted(items)], dtype=np.int64)
        for key, items in groups.items()
    }


def sample_from_trajectory(times: list[int], values: np.ndarray, target: int) -> np.ndarray:
    if not times:
        return values.mean(axis=0)
    if target in times:
        index = times.index(target)
    else:
        index = min(range(len(times)), key=lambda item: abs(times[item] - target))
    return values[index].astype(np.float32)


def global_video_center(trajectories: dict[int, dict[str, object]], source: str) -> np.ndarray:
    parts = [np.asarray(item[source], dtype=np.float32).reshape(-1, 2) for item in trajectories.values()]
    if not parts:
        return np.asarray([128.0, 128.0], dtype=np.float32)
    return np.median(np.concatenate(parts, axis=0), axis=0).astype(np.float32)


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
                    "method": f"SpectralDimwise_V[{v_item['method']}]__A[{a_item['method']}]",
                    "overall_mae": round((v_item["valence_mae"] + a_item["arousal_mae"]) / 2.0, 4),
                    "valence_mae": round(v_item["valence_mae"], 4),
                    "arousal_mae": round(a_item["arousal_mae"], 4),
                    "overall_mse": round((v_item["valence_mse"] + a_item["arousal_mse"]) / 2.0, 4),
                    "notes": "Metric-composed spectral/tensor dimwise candidate.",
                }
            )
    return sorted(rows, key=lambda item: float(item["overall_mae"]))


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def clip(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 255.0).astype(np.float32)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
