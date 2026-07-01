from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125  # noqa: E402
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    build_oof_training_set,
    make_candidates as make_existing_candidates,
    make_feature_matrix,
    make_pattern_098,
    parse_strings,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.cross_fold_residual_field_module import make_manual_200  # noqa: E402
from tools.cross_fold_to200_architectures import make_previous_167  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


DIM_NAMES = ("valence", "arousal")


@dataclass(frozen=True)
class CandidateSpec:
    estimator: str
    lag: int
    smooth: int

    @property
    def name(self) -> str:
        return f"{self.estimator}_lag{self.lag}_s{self.smooth}"


@dataclass
class PriorCache:
    grouped: dict[tuple[int, int], np.ndarray]
    video_times: dict[int, list[int]]
    global_mean: np.ndarray
    global_median: np.ndarray
    video_mean: dict[int, np.ndarray]
    video_median: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive stimulus-response prior. Candidate lag/smoothing/estimator choices are "
            "selected with inner leave-one-subject-out CV inside each outer train fold."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_536_adaptive_stimulus_prior.json")
    parser.add_argument("--lags", default="-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8")
    parser.add_argument("--smooths", default="0,3,5,7,9,11,15,21,31,43,61")
    parser.add_argument("--estimators", default="median,trimmed,mean")
    parser.add_argument("--temperatures", default="0.15,0.3,0.6,1.0,2.0,4.0")
    parser.add_argument("--top-soft", default="3,5,9,15")
    parser.add_argument("--blend-weights", default="0.05,0.1,0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--include-p200", action="store_true")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    specs = make_specs(parse_ints(args.lags), parse_ints(args.smooths), parse_strings(args.estimators))
    temperatures = parse_floats(args.temperatures)
    top_soft = parse_ints(args.top_soft)
    blend_weights = parse_floats(args.blend_weights)
    candidate_pool = parse_strings(args.candidate_pool)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []
    selector_summaries = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] adaptive prior inner-CV", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)

        loss_sum, loss_count = inner_cv_candidate_losses(labels, train_subjects, specs)
        cache = build_prior_cache(train_ids, y_train)
        pred_stack = np.stack([predict_candidate(cache, val_ids, spec) for spec in specs], axis=0)

        existing_candidates = make_existing_candidates(train_ids, y_train, val_ids, args)
        pattern_098 = make_pattern_098(val_ids, existing_candidates)

        fold_predictions: dict[str, np.ndarray] = {
            "098_PatternPrior_reference": pattern_098,
            "AdaptiveGlobalHard": compose_global_hard(pred_stack, loss_sum),
            "AdaptiveVideoDimHard": compose_video_dim_hard(pred_stack, val_ids, loss_sum, loss_count),
        }

        for top_k in top_soft:
            for temp in temperatures:
                fold_predictions[f"AdaptiveGlobalSoft_top{top_k}_t{fmt(temp)}"] = compose_global_soft(
                    pred_stack, loss_sum, top_k=top_k, temperature=temp
                )
                fold_predictions[f"AdaptiveVideoDimSoft_top{top_k}_t{fmt(temp)}"] = compose_video_dim_soft(
                    pred_stack, val_ids, loss_sum, loss_count, top_k=top_k, temperature=temp
                )

        fold_predictions.update(make_dimwise_mixes(fold_predictions, pattern_098, "098"))

        p200 = None
        if args.include_p200:
            p200 = make_p200_fold(
                labels=labels,
                train_subjects=train_subjects,
                train_ids=train_ids,
                y_train=y_train,
                val_ids=val_ids,
                pattern_098=pattern_098,
                existing_candidates=existing_candidates,
                candidate_pool=candidate_pool,
                args=args,
                seed=args.seed + fold_index * 197,
            )
            fold_predictions["200_CurrentManualFusion_reference"] = p200
            adaptive_names = [
                name
                for name in fold_predictions
                if name.startswith("Adaptive") and not name.startswith("AdaptiveMix")
            ]
            for name in adaptive_names:
                adaptive = fold_predictions[name]
                for weight in blend_weights:
                    fused = (1.0 - weight) * p200 + weight * adaptive
                    fold_predictions[f"P200_plus_{name}_w{fmt(weight)}"] = clip(fused)
                dimmix = p200.copy()
                dimmix[:, 0] = adaptive[:, 0]
                fold_predictions[f"V[{name}]_A[P200]"] = clip(dimmix)
                dimmix = p200.copy()
                dimmix[:, 1] = adaptive[:, 1]
                fold_predictions[f"V[P200]_A[{name}]"] = clip(dimmix)

        fold_results = [
            score(name, y_val, pred, "Adaptive stimulus-response prior with inner train-fold CV.")
            for name, pred in fold_predictions.items()
        ]
        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))

        selector_summaries.append(selector_summary(specs, loss_sum, loss_count))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "candidate_count": len(specs),
                "val_samples": len(val_ids),
                "best_results": fold_results[: args.top_k],
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
        "method": "Adaptive Stimulus-Response Prior",
        "note": (
            "The model treats sample_id video/time as a stimulus-response curve problem. "
            "For each outer fold, each candidate prior is scored only by inner "
            "leave-one-subject-out training subjects, then hard-selected or soft-ensembled "
            "per dimension/video before evaluating held-out subjects."
        ),
        "fold_size": args.fold_size,
        "subjects": subjects,
        "candidate_count": len(specs),
        "spec_grid": {
            "lags": parse_ints(args.lags),
            "smooths": parse_ints(args.smooths),
            "estimators": parse_strings(args.estimators),
        },
        "include_p200": bool(args.include_p200),
        "aggregate_results": aggregate_results[: args.top_k],
        "selector_summaries": selector_summaries,
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_specs(lags: list[int], smooths: list[int], estimators: list[str]) -> list[CandidateSpec]:
    specs = []
    allowed = {"median", "mean", "trimmed"}
    for estimator in estimators:
        if estimator not in allowed:
            raise ValueError(f"Unsupported estimator: {estimator}")
        for lag in lags:
            for smooth in smooths:
                specs.append(CandidateSpec(estimator=estimator, lag=lag, smooth=smooth))
    return specs


def build_prior_cache(sample_ids: list[str], y: np.ndarray) -> PriorCache:
    grouped_lists: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_lists: dict[int, list[np.ndarray]] = defaultdict(list)
    video_times_raw: dict[int, set[int]] = defaultdict(set)
    for sample_id, value in zip(sample_ids, y):
        _, video, timestamp = parse_sample_id(sample_id)
        grouped_lists[(video, timestamp)].append(value.astype(np.float32))
        video_lists[video].append(value.astype(np.float32))
        video_times_raw[video].add(timestamp)

    grouped = {
        key: np.asarray(values, dtype=np.float32) for key, values in grouped_lists.items()
    }
    video_mean = {
        video: np.asarray(values, dtype=np.float32).mean(axis=0) for video, values in video_lists.items()
    }
    video_median = {
        video: np.median(np.asarray(values, dtype=np.float32), axis=0).astype(np.float32)
        for video, values in video_lists.items()
    }
    return PriorCache(
        grouped=grouped,
        video_times={video: sorted(times) for video, times in video_times_raw.items()},
        global_mean=y.mean(axis=0).astype(np.float32),
        global_median=np.median(y, axis=0).astype(np.float32),
        video_mean=video_mean,
        video_median=video_median,
    )


def predict_candidate(cache: PriorCache, target_ids: list[str], spec: CandidateSpec) -> np.ndarray:
    rows = []
    value_cache: dict[tuple[int, int, str], np.ndarray] = {}
    for sample_id in target_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        query_time = timestamp + spec.lag
        if (video, query_time) not in cache.grouped:
            query_time = nearest_time(cache.video_times.get(video, []), query_time)
        key = (video, query_time if query_time is not None else -1, spec.estimator)
        if key not in value_cache:
            if query_time is None:
                value = cache.global_median if spec.estimator != "mean" else cache.global_mean
            else:
                arr = cache.grouped.get((video, query_time))
                if arr is None:
                    value = cache.video_median.get(video, cache.global_median)
                else:
                    value = estimate_center(arr, spec.estimator)
            value_cache[key] = value.astype(np.float32)
        rows.append(value_cache[key])
    pred = np.stack(rows, axis=0).astype(np.float32)
    if spec.smooth > 1:
        pred = smooth_predictions(target_ids, pred, spec.smooth).astype(np.float32)
    return clip(pred)


def estimate_center(arr: np.ndarray, estimator: str) -> np.ndarray:
    if estimator == "mean":
        return arr.mean(axis=0)
    if estimator == "median":
        return np.median(arr, axis=0)
    if estimator == "trimmed":
        if arr.shape[0] < 6:
            return np.median(arr, axis=0)
        lower = np.percentile(arr, 15.0, axis=0)
        upper = np.percentile(arr, 85.0, axis=0)
        clipped = np.clip(arr, lower[None, :], upper[None, :])
        return clipped.mean(axis=0)
    raise ValueError(estimator)


def inner_cv_candidate_losses(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    specs: list[CandidateSpec],
) -> tuple[np.ndarray, np.ndarray]:
    loss_sum = np.zeros((len(specs), 16, 2), dtype=np.float64)
    loss_count = np.zeros((len(specs), 16, 2), dtype=np.float64)
    for held_subject in train_subjects:
        fit_subjects = [subject for subject in train_subjects if subject != held_subject]
        fit_ids = ids_for_subjects(labels, fit_subjects)
        held_ids = ids_for_subjects(labels, [held_subject])
        y_fit = labels_to_array(labels, fit_ids)
        y_held = labels_to_array(labels, held_ids)
        cache = build_prior_cache(fit_ids, y_fit)
        videos = np.asarray([parse_sample_id(sample_id)[1] for sample_id in held_ids], dtype=np.int16)
        for spec_index, spec in enumerate(specs):
            pred = predict_candidate(cache, held_ids, spec)
            err = np.abs(pred - y_held)
            for video in range(1, 16):
                mask = videos == video
                if not np.any(mask):
                    continue
                loss_sum[spec_index, video] += err[mask].sum(axis=0)
                loss_count[spec_index, video] += mask.sum()
    return loss_sum, loss_count


def compose_global_hard(pred_stack: np.ndarray, loss_sum: np.ndarray) -> np.ndarray:
    loss = loss_sum[:, 1:16, :].sum(axis=1)
    best = np.argmin(loss, axis=0)
    out = np.empty(pred_stack.shape[1:], dtype=np.float32)
    for dim in range(2):
        out[:, dim] = pred_stack[best[dim], :, dim]
    return clip(out)


def compose_global_soft(
    pred_stack: np.ndarray,
    loss_sum: np.ndarray,
    top_k: int,
    temperature: float,
) -> np.ndarray:
    loss = loss_sum[:, 1:16, :].sum(axis=1)
    out = np.empty(pred_stack.shape[1:], dtype=np.float32)
    for dim in range(2):
        weights = soft_weights(loss[:, dim], top_k=top_k, temperature=temperature)
        out[:, dim] = np.tensordot(weights, pred_stack[:, :, dim], axes=(0, 0))
    return clip(out)


def compose_video_dim_hard(
    pred_stack: np.ndarray,
    sample_ids: list[str],
    loss_sum: np.ndarray,
    loss_count: np.ndarray,
) -> np.ndarray:
    avg = average_loss(loss_sum, loss_count)
    out = np.empty(pred_stack.shape[1:], dtype=np.float32)
    videos = [parse_sample_id(sample_id)[1] for sample_id in sample_ids]
    for row, video in enumerate(videos):
        for dim in range(2):
            best = int(np.argmin(avg[:, video, dim]))
            out[row, dim] = pred_stack[best, row, dim]
    return clip(out)


def compose_video_dim_soft(
    pred_stack: np.ndarray,
    sample_ids: list[str],
    loss_sum: np.ndarray,
    loss_count: np.ndarray,
    top_k: int,
    temperature: float,
) -> np.ndarray:
    avg = average_loss(loss_sum, loss_count)
    out = np.empty(pred_stack.shape[1:], dtype=np.float32)
    videos = [parse_sample_id(sample_id)[1] for sample_id in sample_ids]
    weight_cache: dict[tuple[int, int], np.ndarray] = {}
    for row, video in enumerate(videos):
        for dim in range(2):
            key = (video, dim)
            if key not in weight_cache:
                weight_cache[key] = soft_weights(avg[:, video, dim], top_k=top_k, temperature=temperature)
            out[row, dim] = np.dot(weight_cache[key], pred_stack[:, row, dim])
    return clip(out)


def make_dimwise_mixes(
    predictions: dict[str, np.ndarray],
    reference: np.ndarray,
    reference_name: str,
) -> dict[str, np.ndarray]:
    out = {}
    adaptive_names = [name for name in predictions if name.startswith("Adaptive")]
    for name in adaptive_names:
        pred = predictions[name]
        mixed = reference.copy()
        mixed[:, 0] = pred[:, 0]
        out[f"V[{name}]_A[{reference_name}]"] = clip(mixed)
        mixed = reference.copy()
        mixed[:, 1] = pred[:, 1]
        out[f"V[{reference_name}]_A[{name}]"] = clip(mixed)
    return out


def make_p200_fold(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    train_ids: list[str],
    y_train: np.ndarray,
    val_ids: list[str],
    pattern_098: np.ndarray,
    existing_candidates: dict[str, np.ndarray],
    candidate_pool: list[str],
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    x_train, y_oof, prior_train, _ = build_oof_training_set(
        labels=labels,
        train_subjects=train_subjects,
        candidate_pool=candidate_pool,
        args=args,
    )
    oof_train_ids = []
    for subject in train_subjects:
        oof_train_ids.extend(ids_for_subjects(labels, [subject]))
    residual_target = (y_oof - prior_train).astype(np.float32)
    x_val = make_feature_matrix(val_ids, existing_candidates, candidate_pool, pattern_098)
    candidate_stack = np.stack([existing_candidates[name] for name in candidate_pool], axis=0).astype(np.float32)
    candidate_std = candidate_stack.std(axis=0).astype(np.float32)
    ref104, _, _ = make_reference_104(
        x_train=x_train,
        residual_target=residual_target,
        x_val=x_val,
        prior_val=pattern_098,
        val_ids=val_ids,
        seed=seed,
    )
    previous_125 = make_previous_125(ref104, val_ids)
    previous_167 = make_previous_167(
        previous_125=previous_125,
        oof_train_ids=oof_train_ids,
        y_train=y_oof,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=val_ids,
    )
    p200, _ = make_manual_200(
        previous_167=previous_167,
        oof_train_ids=oof_train_ids,
        y_train=y_oof,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=val_ids,
        candidate_std=candidate_std,
    )
    return clip(p200)


def average_loss(loss_sum: np.ndarray, loss_count: np.ndarray) -> np.ndarray:
    global_loss = loss_sum[:, 1:16, :].sum(axis=1) / np.maximum(
        loss_count[:, 1:16, :].sum(axis=1), 1.0
    )
    avg = loss_sum / np.maximum(loss_count, 1.0)
    for video in range(1, 16):
        missing = loss_count[:, video, :] <= 0
        for dim in range(2):
            avg[missing[:, dim], video, dim] = global_loss[missing[:, dim], dim]
    return avg


def soft_weights(loss: np.ndarray, top_k: int, temperature: float) -> np.ndarray:
    top_k = max(1, min(int(top_k), loss.shape[0]))
    order = np.argsort(loss)[:top_k]
    shifted = loss[order] - loss[order[0]]
    weights_top = np.exp(-shifted / max(float(temperature), 1e-6))
    weights_top /= np.maximum(weights_top.sum(), 1e-12)
    weights = np.zeros_like(loss, dtype=np.float32)
    weights[order] = weights_top.astype(np.float32)
    return weights


def selector_summary(
    specs: list[CandidateSpec],
    loss_sum: np.ndarray,
    loss_count: np.ndarray,
) -> dict[str, object]:
    avg = average_loss(loss_sum, loss_count)
    summary: dict[str, object] = {"global_best": {}, "video_dim_best": {}}
    global_loss = loss_sum[:, 1:16, :].sum(axis=1) / np.maximum(
        loss_count[:, 1:16, :].sum(axis=1), 1.0
    )
    for dim, dim_name in enumerate(DIM_NAMES):
        best = int(np.argmin(global_loss[:, dim]))
        summary["global_best"][dim_name] = {
            "spec": specs[best].name,
            "inner_mae": round(float(global_loss[best, dim]), 4),
        }
        video_items = []
        for video in range(1, 16):
            best = int(np.argmin(avg[:, video, dim]))
            video_items.append(
                {
                    "video": video,
                    "spec": specs[best].name,
                    "inner_mae": round(float(avg[best, video, dim]), 4),
                }
            )
        summary["video_dim_best"][dim_name] = video_items
    return summary


def nearest_time(times: list[int], target: int) -> int | None:
    if not times:
        return None
    return min(times, key=lambda value: abs(value - target))


def clip(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 255.0).astype(np.float32)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def fmt(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
