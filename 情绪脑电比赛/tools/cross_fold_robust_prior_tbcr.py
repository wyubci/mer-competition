from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions
from tools.trial_basis_residual import (
    fit_basis_coefficients,
    load_feature_cache,
    order_predictions,
    parse_sample_id,
    reconstruct_predictions,
    trial_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subject-disjoint validation of TBCR residuals over RobustMedianPrior."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_092_robust_prior_tbcr.json")
    parser.add_argument("--prior-lag", type=int, default=-2)
    parser.add_argument("--prior-smooth", type=int, default=11)
    parser.add_argument("--basis-count", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--valence-scales", default="0,0.05,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--arousal-scales", default="0,0.025,0.05,0.075,0.1,0.15,0.25")
    parser.add_argument("--valence-clips", default="2,3,4.5,6")
    parser.add_argument("--arousal-clips", default="2,4,6,8")
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    cache = load_feature_cache(Path(args.feature_cache))
    valence_scales = parse_floats(args.valence_scales)
    arousal_scales = parse_floats(args.arousal_scales)
    valence_clips = parse_floats(args.valence_clips)
    arousal_clips = parse_floats(args.arousal_clips)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        prior_val = robust_median_prior(
            reference_ids=train_ids,
            reference_y=y_train,
            target_ids=val_ids,
            lag=args.prior_lag,
            smooth_window=args.prior_smooth,
        )
        train_trials, val_trials = build_robust_prior_trials(
            cache=cache,
            labels=labels,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            train_ids=train_ids,
            y_train=y_train,
            val_ids=val_ids,
            prior_val=prior_val,
            lag=args.prior_lag,
            smooth_window=args.prior_smooth,
        )
        residual_mean = predict_tbcr_residual(
            train_trials=train_trials,
            val_trials=val_trials,
            val_ids=val_ids,
            basis_count=args.basis_count,
            alpha=args.alpha,
            feature_mode="mean_std",
        )
        residual_slope = predict_tbcr_residual(
            train_trials=train_trials,
            val_trials=val_trials,
            val_ids=val_ids,
            basis_count=args.basis_count,
            alpha=args.alpha,
            feature_mode="mean_std_slope",
        )

        fold_predictions = {"RobustMedianPrior": prior_val}
        for scale_v in valence_scales:
            for scale_a in arousal_scales:
                for clip_v in valence_clips:
                    cv = np.clip(scale_v * residual_mean[:, 0], -clip_v, clip_v)
                    for clip_a in arousal_clips:
                        ca = np.clip(scale_a * residual_slope[:, 1], -clip_a, clip_a)
                        pred = np.clip(prior_val + np.stack([cv, ca], axis=1), 1.0, 255.0)
                        name = (
                            f"RobustPrior_TBCR_sv{format_float(scale_v)}_sa{format_float(scale_a)}"
                            f"_cv{format_float(clip_v)}_ca{format_float(clip_a)}"
                        )
                        fold_predictions[name] = pred.astype(np.float32)

        fold_results = [
            score(name, y_val, pred, "RobustMedianPrior plus TBCR correction.")
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
        "method": "RobustMedianPrior + TBCR residual validation",
        "note": (
            "Training residual targets use leave-subject-out RobustMedianPrior inside each outer "
            "training fold to reduce train-prior leakage."
        ),
        "fold_size": args.fold_size,
        "prior": {
            "estimator": "median",
            "lag": args.prior_lag,
            "smooth_window": args.prior_smooth,
        },
        "basis_count": args.basis_count,
        "alpha": args.alpha,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_robust_prior_trials(
    cache: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    val_subjects: list[str],
    train_ids: list[str],
    y_train: np.ndarray,
    val_ids: list[str],
    prior_val: np.ndarray,
    lag: int,
    smooth_window: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject_train_refs = {}
    for subject in train_subjects:
        ref_ids = [sample_id for sample_id in train_ids if not sample_id.startswith(subject + "_V")]
        subject_train_refs[subject] = (ref_ids, labels_to_array(labels, ref_ids))
    val_prior_by_id = {sample_id: prior for sample_id, prior in zip(val_ids, prior_val)}

    groups: dict[tuple[str, int], list[tuple[int, int, str]]] = defaultdict(list)
    wanted_subjects = set(train_subjects + val_subjects)
    for index, sample_id in enumerate(cache["sample_ids"].astype(str).tolist()):
        subject, video, timestamp = parse_sample_id(sample_id)
        if subject in wanted_subjects:
            groups[(subject, video)].append((timestamp, index, sample_id))

    train_trials = []
    val_trials = []
    for (subject, video), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        ordered = sorted(items)
        sample_ids = [sample_id for _, _, sample_id in ordered]
        indices = np.asarray([index for _, index, _ in ordered], dtype=np.int64)
        y = np.stack([labels[sample_id] for sample_id in sample_ids], axis=0).astype(np.float32)
        if subject in train_subjects:
            ref_ids, ref_y = subject_train_refs[subject]
            prior = robust_median_prior(ref_ids, ref_y, sample_ids, lag, smooth_window)
            target = train_trials
        else:
            prior = np.stack([val_prior_by_id[sample_id] for sample_id in sample_ids], axis=0)
            target = val_trials
        target.append(
            {
                "subject": subject,
                "video": video,
                "sample_ids": sample_ids,
                "x": cache["x"][indices],
                "y": y,
                "prior": prior.astype(np.float32),
                "residual": (y - prior).astype(np.float32),
            }
        )
    return train_trials, val_trials


def robust_median_prior(
    reference_ids: list[str],
    reference_y: np.ndarray,
    target_ids: list[str],
    lag: int,
    smooth_window: int,
) -> np.ndarray:
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_times: dict[int, list[int]] = defaultdict(list)
    for sample_id, value in zip(reference_ids, reference_y):
        _, video, timestamp = parse_sample_id(sample_id)
        grouped[(video, timestamp)].append(value.astype(np.float32))
        video_times[video].append(timestamp)
    video_times = {video: sorted(set(times)) for video, times in video_times.items()}
    global_median = np.median(reference_y, axis=0).astype(np.float32)
    cache: dict[tuple[int, int], np.ndarray] = {}
    rows = []
    for sample_id in target_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        key = (video, timestamp + lag)
        if key not in grouped:
            nearest = nearest_time(video_times.get(video, []), timestamp + lag)
            key = (video, nearest) if nearest is not None else key
        if key not in cache:
            cache[key] = np.median(np.asarray(grouped.get(key, [global_median])), axis=0)
        rows.append(cache[key])
    prior = np.stack(rows, axis=0).astype(np.float32)
    if smooth_window > 1:
        prior = smooth_predictions(target_ids, prior, smooth_window).astype(np.float32)
    return prior


def predict_tbcr_residual(
    train_trials: list[dict[str, object]],
    val_trials: list[dict[str, object]],
    val_ids: list[str],
    basis_count: int,
    alpha: float,
    feature_mode: str,
) -> np.ndarray:
    x_train = np.stack([trial_features(trial["x"], feature_mode) for trial in train_trials], axis=0)
    x_val = np.stack([trial_features(trial["x"], feature_mode) for trial in val_trials], axis=0)
    y_coeff_train = np.stack(
        [fit_basis_coefficients(trial["residual"], basis_count) for trial in train_trials],
        axis=0,
    ).reshape(len(train_trials), -1)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_coeff_train)
    coeff_pred = model.predict(x_val).reshape(len(val_trials), basis_count, 2)
    pred_by_id = reconstruct_predictions(val_trials, coeff_pred, basis_count)
    tbcr_pred = order_predictions(val_ids, pred_by_id)
    prior_by_id: dict[str, np.ndarray] = {}
    for trial in val_trials:
        for sample_id, prior in zip(trial["sample_ids"], trial["prior"]):
            prior_by_id[str(sample_id)] = np.asarray(prior, dtype=np.float32)
    prior_val = np.stack([prior_by_id[sample_id] for sample_id in val_ids], axis=0).astype(np.float32)
    return (tbcr_pred - prior_val).astype(np.float32)


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


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
