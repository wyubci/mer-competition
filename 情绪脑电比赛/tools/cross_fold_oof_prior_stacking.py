from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import (  # noqa: E402
    build_candidates,
    build_prior_stats,
    ids_for_subjects,
    labels_to_array,
    parse_floats,
    parse_ints,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


PATTERN_098 = {
    "valence": {
        "split_q": 65.0,
        "stable": "UncertaintyBlend_meanPrior_q15p0-45p0_g0p25",
        "dynamic": "UncertaintyBlend_smooth61_q15p0-45p0_g0p45",
    },
    "arousal": {
        "split_q": 55.0,
        "stable": "UncertaintyBlend_smooth51_q20p0-45p0_g0p5",
        "dynamic": "UncertaintyBlend_meanPrior_q20p0-55p0_g0p55",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOF prior stacking/meta-calibration over robust MER-PS trajectory experts."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_102_oof_prior_stacking.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--models", default="ridge_residual,hgb_residual,extra_residual,ridge_direct")
    parser.add_argument("--ridge-alphas", default="1,10,50,100,300,1000")
    parser.add_argument("--residual-scales", default="0,0.05,0.1,0.15,0.2,0.35,0.5")
    parser.add_argument("--direct-blends", default="0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--smooth-windows", default="0,5,9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    candidate_pool = parse_strings(args.candidate_pool)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(
            f"[fold {fold_index}] building OOF train features for {len(train_subjects)} subjects",
            flush=True,
        )
        x_train, y_train, prior_train, train_rows = build_oof_training_set(
            labels=labels,
            train_subjects=train_subjects,
            candidate_pool=candidate_pool,
            args=args,
        )

        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        val_candidates = make_candidates(train_ids, y_outer_train, val_ids, args)
        prior_val = make_pattern_098(val_ids, val_candidates)
        x_val = make_feature_matrix(val_ids, val_candidates, candidate_pool, prior_val)

        fold_predictions: dict[str, np.ndarray] = {"PatternPrior_098": prior_val}
        fit_and_predict_models(
            fold_predictions=fold_predictions,
            x_train=x_train,
            y_train=y_train,
            prior_train=prior_train,
            x_val=x_val,
            prior_val=prior_val,
            val_ids=val_ids,
            args=args,
        )

        fold_results = [
            score(name, y_val, pred, "OOF prior stacking/meta-calibration.")
            for name, pred in fold_predictions.items()
        ]
        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "train_oof_rows": train_rows,
                "val_samples": len(val_ids),
                "feature_dim": int(x_train.shape[1]),
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
        "method": "OOF prior stacking/meta-calibration",
        "note": (
            "For each outer fold, training meta-features are generated by inner "
            "leave-one-subject-out priors. The meta-model never sees a subject's own labels "
            "inside its prior features."
        ),
        "fold_size": args.fold_size,
        "candidate_pool_size": len(candidate_pool),
        "candidate_pool": candidate_pool,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_oof_training_set(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    candidate_pool: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x_parts = []
    y_parts = []
    prior_parts = []
    for subject in train_subjects:
        fit_subjects = [item for item in train_subjects if item != subject]
        fit_ids = ids_for_subjects(labels, fit_subjects)
        target_ids = ids_for_subjects(labels, [subject])
        y_fit = labels_to_array(labels, fit_ids)
        y_target = labels_to_array(labels, target_ids)
        candidates = make_candidates(fit_ids, y_fit, target_ids, args)
        prior = make_pattern_098(target_ids, candidates)
        x_parts.append(make_feature_matrix(target_ids, candidates, candidate_pool, prior))
        y_parts.append(y_target)
        prior_parts.append(prior)
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    prior = np.concatenate(prior_parts, axis=0).astype(np.float32)
    return x, y, prior, int(y.shape[0])


def make_candidates(
    train_ids: list[str],
    y_train: np.ndarray,
    target_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    stats = build_prior_stats(train_ids, y_train)
    candidates = build_candidates(
        stats=stats,
        val_ids=target_ids,
        base_lag=-2,
        base_smooth=11,
        alt_lag=-1,
        alt_smooth=9,
        q_lows=parse_floats(args.quantile_lows),
        q_highs=parse_floats(args.quantile_highs),
        max_gates=parse_floats(args.max_gates),
        long_smooths=parse_ints(args.long_smooths),
        ensemble_weights=parse_floats(args.ensemble_weights),
    )
    required = set(parse_strings(args.candidate_pool))
    required.update(
        [
            "RobustMedian_lag-2_smooth11",
            PATTERN_098["valence"]["stable"],
            PATTERN_098["valence"]["dynamic"],
            PATTERN_098["arousal"]["stable"],
            PATTERN_098["arousal"]["dynamic"],
        ]
    )
    missing = sorted(required.difference(candidates))
    if missing:
        raise KeyError(f"Missing candidates: {missing}")
    return candidates


def make_pattern_098(sample_ids: list[str], candidates: dict[str, np.ndarray]) -> np.ndarray:
    base = candidates["RobustMedian_lag-2_smooth11"]
    slopes = prior_slope_by_trial(sample_ids, base)
    pred = np.zeros((len(sample_ids), 2), dtype=np.float32)

    v_abs = np.abs(slopes[:, 0])
    v_threshold = float(np.percentile(v_abs, PATTERN_098["valence"]["split_q"]))
    v_stable = v_abs <= v_threshold
    pred[:, 0] = np.where(
        v_stable,
        candidates[PATTERN_098["valence"]["stable"]][:, 0],
        candidates[PATTERN_098["valence"]["dynamic"]][:, 0],
    )

    a_abs = np.abs(slopes[:, 1])
    a_threshold = float(np.percentile(a_abs, PATTERN_098["arousal"]["split_q"]))
    a_stable = a_abs <= a_threshold
    pred[:, 1] = np.where(
        a_stable,
        candidates[PATTERN_098["arousal"]["stable"]][:, 1],
        candidates[PATTERN_098["arousal"]["dynamic"]][:, 1],
    )
    return np.clip(pred, 1.0, 255.0).astype(np.float32)


def prior_slope_by_trial(sample_ids: list[str], values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    slopes = np.zeros_like(arr, dtype=np.float32)
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        if len(indices) <= 1:
            continue
        trial_values = arr[indices]
        if arr.ndim == 1:
            local = np.gradient(trial_values).astype(np.float32)
        else:
            local = np.gradient(trial_values, axis=0).astype(np.float32)
        slopes[indices] = local
    return slopes


def make_feature_matrix(
    sample_ids: list[str],
    candidates: dict[str, np.ndarray],
    candidate_pool: list[str],
    pattern_prior: np.ndarray,
) -> np.ndarray:
    candidate_stack = np.stack([candidates[name] for name in candidate_pool], axis=0).astype(np.float32)
    candidate_flat = np.transpose(candidate_stack, (1, 0, 2)).reshape(len(sample_ids), -1)
    candidate_mean = candidate_stack.mean(axis=0)
    candidate_median = np.median(candidate_stack, axis=0)
    candidate_std = candidate_stack.std(axis=0)
    candidate_min = candidate_stack.min(axis=0)
    candidate_max = candidate_stack.max(axis=0)
    candidate_range = candidate_max - candidate_min

    prior_slope = prior_slope_by_trial(sample_ids, pattern_prior)
    prior_accel = prior_slope_by_trial(sample_ids, prior_slope)
    base_slope = prior_slope_by_trial(sample_ids, candidates["RobustMedian_lag-2_smooth11"])
    time_features = build_time_features(sample_ids)

    features = [
        candidate_flat,
        pattern_prior,
        prior_slope,
        np.abs(prior_slope),
        prior_accel,
        np.abs(prior_accel),
        base_slope,
        np.abs(base_slope),
        candidate_mean,
        candidate_median,
        candidate_std,
        candidate_range,
        time_features,
    ]
    return np.concatenate(features, axis=1).astype(np.float32)


def build_time_features(sample_ids: list[str]) -> np.ndarray:
    videos = []
    timestamps = []
    for sample_id in sample_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        videos.append(video)
        timestamps.append(timestamp)
    max_by_video: dict[int, int] = defaultdict(int)
    for video, timestamp in zip(videos, timestamps):
        max_by_video[int(video)] = max(max_by_video[int(video)], int(timestamp))
    time_norm = np.asarray(
        [timestamp / max(max_by_video[int(video)], 1) for video, timestamp in zip(videos, timestamps)],
        dtype=np.float32,
    )
    video_onehot = np.zeros((len(sample_ids), 15), dtype=np.float32)
    for index, video in enumerate(videos):
        if 1 <= int(video) <= 15:
            video_onehot[index, int(video) - 1] = 1.0
    return np.concatenate(
        [
            time_norm[:, None],
            (time_norm**2)[:, None],
            np.sin(2 * np.pi * time_norm)[:, None],
            np.cos(2 * np.pi * time_norm)[:, None],
            np.sin(4 * np.pi * time_norm)[:, None],
            np.cos(4 * np.pi * time_norm)[:, None],
            ((np.asarray(videos, dtype=np.float32) - 8.0) / 7.0)[:, None],
            video_onehot,
        ],
        axis=1,
    ).astype(np.float32)


def fit_and_predict_models(
    fold_predictions: dict[str, np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    prior_train: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    val_ids: list[str],
    args: argparse.Namespace,
) -> None:
    models = parse_strings(args.models)
    residual_scales = parse_floats(args.residual_scales)
    direct_blends = parse_floats(args.direct_blends)
    smooth_windows = parse_ints(args.smooth_windows)
    residual_target = (y_train - prior_train).astype(np.float32)

    if "ridge_residual" in models:
        for alpha in parse_floats(args.ridge_alphas):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, residual_target)
            residual = model.predict(x_val).astype(np.float32)
            add_residual_family(
                fold_predictions,
                f"OOFStack_RidgeResidual_a{format_float(alpha)}",
                prior_val,
                residual,
                val_ids,
                residual_scales,
                smooth_windows,
            )

    if "hgb_residual" in models:
        hgb = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.035,
                max_iter=120,
                max_leaf_nodes=11,
                min_samples_leaf=90,
                l2_regularization=0.15,
                random_state=args.seed,
            )
        )
        hgb.fit(x_train, residual_target)
        residual = hgb.predict(x_val).astype(np.float32)
        add_residual_family(
            fold_predictions,
            "OOFStack_HGBResidual_l1_leaf11",
            prior_val,
            residual,
            val_ids,
            residual_scales,
            smooth_windows,
        )

    if "extra_residual" in models:
        extra = ExtraTreesRegressor(
            n_estimators=260,
            max_depth=6,
            min_samples_leaf=70,
            max_features=0.35,
            random_state=args.seed + 7,
            n_jobs=-1,
        )
        extra.fit(x_train, residual_target)
        residual = extra.predict(x_val).astype(np.float32)
        add_residual_family(
            fold_predictions,
            "OOFStack_ExtraResidual_d6_leaf70",
            prior_val,
            residual,
            val_ids,
            residual_scales,
            smooth_windows,
        )

    if "ridge_direct" in models:
        for alpha in parse_floats(args.ridge_alphas):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, y_train)
            direct = np.clip(model.predict(x_val), 1.0, 255.0).astype(np.float32)
            for blend in direct_blends:
                pred = np.clip((1.0 - blend) * prior_val + blend * direct, 1.0, 255.0)
                base_name = f"OOFStack_RidgeDirect_a{format_float(alpha)}_blend{format_float(blend)}"
                add_smooth_family(fold_predictions, base_name, pred, val_ids, smooth_windows)


def add_residual_family(
    fold_predictions: dict[str, np.ndarray],
    base_name: str,
    prior: np.ndarray,
    residual: np.ndarray,
    sample_ids: list[str],
    scales: list[float],
    smooth_windows: list[int],
) -> None:
    for scale in scales:
        pred = np.clip(prior + scale * residual, 1.0, 255.0).astype(np.float32)
        name = f"{base_name}_scale{format_float(scale)}"
        add_smooth_family(fold_predictions, name, pred, sample_ids, smooth_windows)


def add_smooth_family(
    fold_predictions: dict[str, np.ndarray],
    base_name: str,
    pred: np.ndarray,
    sample_ids: list[str],
    smooth_windows: list[int],
) -> None:
    fold_predictions[base_name] = pred
    for window in smooth_windows:
        if window <= 1:
            continue
        fold_predictions[f"{base_name}_smooth{window}"] = smooth_predictions(sample_ids, pred, window)


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
