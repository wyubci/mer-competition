from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_tbcr_validation import predict_tbcr_residual
from tools.run_iteration_experiments import expand_subjects, load_labels, predict_video_time_mean, score
from tools.trial_basis_residual import build_trials, load_feature_cache, parse_sample_id, trial_features


SCALE_GRID = np.asarray([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only adaptive gate for TBCR corrections under subject-disjoint folds."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--output", default="experiments/results/iteration_089_adaptive_tbcr_gate.json")
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--inner-fold-size", type=int, default=4)
    parser.add_argument("--basis-count", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_subjects = expand_subjects(args.subjects)
    outer_folds = [
        all_subjects[start : start + args.fold_size]
        for start in range(0, len(all_subjects), args.fold_size)
    ]
    labels = load_labels(Path(args.data_root), all_subjects)
    cache = load_feature_cache(Path(args.feature_cache))

    fold_outputs = []
    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = {}

    for fold_index, val_subjects in enumerate(outer_folds, start=1):
        train_subjects = [subject for subject in all_subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        prior_val = predict_video_time_mean(train_ids, y_train, val_ids)

        outer_trials = build_trials(cache, labels, train_ids, y_train)
        outer_train_trials = [trial for trial in outer_trials if trial["subject"] in train_subjects]
        outer_val_trials = [trial for trial in outer_trials if trial["subject"] in val_subjects]

        fixed_correction = predict_dual_correction(
            train_trials=outer_train_trials,
            eval_trials=outer_val_trials,
            eval_ids=val_ids,
            prior_eval=prior_val,
            basis_count=args.basis_count,
            alpha=args.alpha,
        )
        fixed_pred = np.clip(prior_val + fixed_correction, 1.0, 255.0)

        meta_x, meta_y_scale, meta_summary = build_outer_meta_training(
            labels=labels,
            cache=cache,
            train_subjects=train_subjects,
            inner_fold_size=args.inner_fold_size,
            basis_count=args.basis_count,
            alpha=args.alpha,
        )
        gate_model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        gate_model.fit(meta_x, meta_y_scale)

        val_meta_x, val_trial_indices = build_meta_features(
            train_trials=outer_train_trials,
            eval_trials=outer_val_trials,
            eval_ids=val_ids,
            prior=prior_val,
            correction=fixed_correction,
        )
        pred_scales = np.clip(gate_model.predict(val_meta_x), 0.0, 1.0).astype(np.float32)
        mean_scale = np.clip(meta_y_scale.mean(axis=0), 0.0, 1.0).astype(np.float32)
        positive_rate = (meta_y_scale > 0).mean(axis=0).astype(np.float32)

        adaptive_correction = apply_trial_scales(fixed_correction, val_trial_indices, pred_scales)
        mean_scale_correction = apply_trial_scales(
            fixed_correction,
            val_trial_indices,
            np.tile(mean_scale[None, :], (len(val_trial_indices), 1)),
        )
        half_adaptive_correction = apply_trial_scales(
            fixed_correction,
            val_trial_indices,
            0.5 * pred_scales,
        )
        binary_gate_scales = (pred_scales >= 0.5).astype(np.float32)
        binary_gate_correction = apply_trial_scales(
            fixed_correction,
            val_trial_indices,
            binary_gate_scales,
        )

        predictions = {
            "VideoTimeMean": prior_val,
            "FixedDualTBCR": fixed_pred,
            "AdaptiveTBCR_mean_oof_scale": np.clip(prior_val + mean_scale_correction, 1.0, 255.0),
            "AdaptiveTBCR_ridge_scale": np.clip(prior_val + adaptive_correction, 1.0, 255.0),
            "AdaptiveTBCR_half_ridge_scale": np.clip(
                prior_val + half_adaptive_correction,
                1.0,
                255.0,
            ),
            "AdaptiveTBCR_binary_gate": np.clip(prior_val + binary_gate_correction, 1.0, 255.0),
        }

        fold_results = [
            score(name, y_val, pred, "Outer-fold subject-disjoint validation.")
            for name, pred in predictions.items()
        ]
        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "meta_summary": meta_summary
                | {
                    "mean_best_scale_valence": round(float(mean_scale[0]), 6),
                    "mean_best_scale_arousal": round(float(mean_scale[1]), 6),
                    "positive_rate_valence": round(float(positive_rate[0]), 6),
                    "positive_rate_arousal": round(float(positive_rate[1]), 6),
                    "pred_scale_mean_valence": round(float(pred_scales[:, 0].mean()), 6),
                    "pred_scale_mean_arousal": round(float(pred_scales[:, 1].mean()), 6),
                },
                "results": fold_results,
            }
        )
        aggregate_truth.append(y_val)
        for name, pred in predictions.items():
            aggregate_predictions.setdefault(name, []).append(pred)

    y_all = np.concatenate(aggregate_truth, axis=0)
    aggregate_results = []
    for name, parts in aggregate_predictions.items():
        aggregate_results.append(
            score(
                name,
                y_all,
                np.concatenate(parts, axis=0),
                "Weighted aggregate across outer subject-disjoint folds.",
            )
        )
    aggregate_results = sorted(aggregate_results, key=lambda item: float(item["overall_mae"]))

    output = {
        "method": "AdaptiveTBCR train-only meta gate",
        "note": (
            "For each outer fold, inner subject-disjoint OOF predictions on the outer train subjects "
            "are used to learn trial-level correction scales. Outer validation labels are never used "
            "for gate fitting."
        ),
        "fold_size": args.fold_size,
        "inner_fold_size": args.inner_fold_size,
        "basis_count": args.basis_count,
        "alpha": args.alpha,
        "ridge_alpha": args.ridge_alpha,
        "scale_grid": SCALE_GRID.tolist(),
        "aggregate_results": aggregate_results,
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_outer_meta_training(
    labels: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    train_subjects: list[str],
    inner_fold_size: int,
    basis_count: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    inner_folds = [
        train_subjects[start : start + inner_fold_size]
        for start in range(0, len(train_subjects), inner_fold_size)
    ]
    x_parts = []
    y_parts = []
    improve_parts = []
    for holdout_subjects in inner_folds:
        inner_train_subjects = [
            subject for subject in train_subjects if subject not in holdout_subjects
        ]
        inner_train_ids = ids_for_subjects(labels, inner_train_subjects)
        holdout_ids = ids_for_subjects(labels, holdout_subjects)
        y_inner_train = labels_to_array(labels, inner_train_ids)
        y_holdout = labels_to_array(labels, holdout_ids)
        prior_holdout = predict_video_time_mean(inner_train_ids, y_inner_train, holdout_ids)
        trials = build_trials(cache, labels, inner_train_ids, y_inner_train)
        inner_train_trials = [
            trial for trial in trials if trial["subject"] in inner_train_subjects
        ]
        holdout_trials = [trial for trial in trials if trial["subject"] in holdout_subjects]
        correction = predict_dual_correction(
            train_trials=inner_train_trials,
            eval_trials=holdout_trials,
            eval_ids=holdout_ids,
            prior_eval=prior_holdout,
            basis_count=basis_count,
            alpha=alpha,
        )
        meta_x, trial_indices = build_meta_features(
            train_trials=inner_train_trials,
            eval_trials=holdout_trials,
            eval_ids=holdout_ids,
            prior=prior_holdout,
            correction=correction,
        )
        best_scales, improvements = best_trial_scales(
            y_true=y_holdout,
            prior=prior_holdout,
            correction=correction,
            trial_indices=trial_indices,
        )
        x_parts.append(meta_x)
        y_parts.append(best_scales)
        improve_parts.append(improvements)
    meta_x_all = np.concatenate(x_parts, axis=0).astype(np.float32)
    meta_y_all = np.concatenate(y_parts, axis=0).astype(np.float32)
    improve_all = np.concatenate(improve_parts, axis=0).astype(np.float32)
    summary = {
        "inner_folds": len(inner_folds),
        "meta_trials": int(meta_x_all.shape[0]),
        "meta_features": int(meta_x_all.shape[1]),
        "mean_improvement_valence": round(float(improve_all[:, 0].mean()), 6),
        "mean_improvement_arousal": round(float(improve_all[:, 1].mean()), 6),
    }
    return meta_x_all, meta_y_all, summary


def predict_dual_correction(
    train_trials: list[dict[str, object]],
    eval_trials: list[dict[str, object]],
    eval_ids: list[str],
    prior_eval: np.ndarray,
    basis_count: int,
    alpha: float,
) -> np.ndarray:
    residual_mean = predict_tbcr_residual(
        train_trials=train_trials,
        val_trials=eval_trials,
        val_ids=eval_ids,
        prior_val=prior_eval,
        basis_count=basis_count,
        alpha=alpha,
        feature_mode="mean_std",
    )
    residual_slope = predict_tbcr_residual(
        train_trials=train_trials,
        val_trials=eval_trials,
        val_ids=eval_ids,
        prior_val=prior_eval,
        basis_count=basis_count,
        alpha=alpha,
        feature_mode="mean_std_slope",
    )
    return np.stack(
        [
            np.clip(1.0 * residual_mean[:, 0], -4.5, 4.5),
            np.clip(0.10 * residual_slope[:, 1], -6.0, 6.0),
        ],
        axis=1,
    ).astype(np.float32)


def build_meta_features(
    train_trials: list[dict[str, object]],
    eval_trials: list[dict[str, object]],
    eval_ids: list[str],
    prior: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    id_to_index = {sample_id: index for index, sample_id in enumerate(eval_ids)}
    distance_models = {
        mode: fit_distance_reference(train_trials, mode)
        for mode in ("mean_std", "mean_std_slope")
    }
    rows = []
    trial_indices = []
    for trial in eval_trials:
        indices = np.asarray([id_to_index[sample_id] for sample_id in trial["sample_ids"]])
        trial_indices.append(indices)
        subject = str(trial["subject"])
        video = int(trial["video"])
        prior_trial = prior[indices]
        correction_trial = correction[indices]
        length = len(indices)
        features: list[float] = [
            video / 15.0,
            length / 180.0,
        ]
        for dim in range(2):
            p = prior_trial[:, dim]
            c = correction_trial[:, dim]
            features.extend(
                [
                    float(p.mean()) / 255.0,
                    float(p.std()) / 64.0,
                    float(p[-1] - p[0]) / 128.0,
                    float(np.mean(np.abs(np.gradient(p)))) / 32.0,
                    float(c.mean()) / 8.0,
                    float(c.std()) / 8.0,
                    float(np.mean(np.abs(c))) / 8.0,
                    float(np.max(np.abs(c))) / 8.0,
                    float(c[-1] - c[0]) / 8.0,
                ]
            )
        for mode, reference in distance_models.items():
            stats = distance_stats(trial, mode, reference)
            features.extend(stats)
        rows.append(features)
    return np.asarray(rows, dtype=np.float32), trial_indices


def fit_distance_reference(
    train_trials: list[dict[str, object]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([trial_features(trial["x"], mode) for trial in train_trials], axis=0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def distance_stats(
    trial: dict[str, object],
    mode: str,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[float]:
    train_z, mean, std = reference
    x = trial_features(trial["x"], mode)
    z = (x - mean) / std
    distances = np.sqrt(np.mean((train_z - z[None, :]) ** 2, axis=1))
    distances = np.sort(distances)
    k = min(5, len(distances))
    return [
        float(distances[0]),
        float(distances[:k].mean()),
        float(np.median(distances)),
    ]


def best_trial_scales(
    y_true: np.ndarray,
    prior: np.ndarray,
    correction: np.ndarray,
    trial_indices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    best_scales = np.zeros((len(trial_indices), 2), dtype=np.float32)
    improvements = np.zeros((len(trial_indices), 2), dtype=np.float32)
    for trial_index, indices in enumerate(trial_indices):
        for dim in range(2):
            y = y_true[indices, dim]
            base = prior[indices, dim]
            corr = correction[indices, dim]
            baseline_mae = float(np.mean(np.abs(base - y)))
            losses = [
                float(np.mean(np.abs(np.clip(base + scale * corr, 1.0, 255.0) - y)))
                for scale in SCALE_GRID
            ]
            best_idx = int(np.argmin(losses))
            best_scales[trial_index, dim] = float(SCALE_GRID[best_idx])
            improvements[trial_index, dim] = baseline_mae - losses[best_idx]
    return best_scales, improvements


def apply_trial_scales(
    correction: np.ndarray,
    trial_indices: list[np.ndarray],
    scales: np.ndarray,
) -> np.ndarray:
    adjusted = np.zeros_like(correction, dtype=np.float32)
    for trial_index, indices in enumerate(trial_indices):
        adjusted[indices] = correction[indices] * scales[trial_index][None, :]
    return adjusted


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


if __name__ == "__main__":
    main()
