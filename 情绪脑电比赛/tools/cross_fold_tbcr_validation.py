from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.residual_attention_tbcr import residual_attention_by_trial
from tools.run_iteration_experiments import expand_subjects, load_labels, predict_video_time_mean, score
from tools.trial_basis_residual import (
    build_trials,
    fit_basis_coefficients,
    load_feature_cache,
    order_predictions,
    reconstruct_predictions,
    trial_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subject-disjoint multi-fold validation for VideoTimeMean + TBCR/LRAG."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--output", default="experiments/results/iteration_088_cross_fold_tbcr.json")
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--basis-count", type=int, default=4)
    parser.add_argument("--alpha-stable", type=float, default=0.25)
    parser.add_argument("--alpha-aggressive", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_subjects = expand_subjects(args.subjects)
    folds = [
        all_subjects[start : start + args.fold_size]
        for start in range(0, len(all_subjects), args.fold_size)
    ]
    labels = load_labels(Path(args.data_root), all_subjects)
    cache = load_feature_cache(Path(args.feature_cache))

    fold_outputs = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    aggregate_truth: list[np.ndarray] = []
    aggregate_counts: dict[str, int] = defaultdict(int)

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in all_subjects if subject not in val_subjects]
        train_ids = [
            sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
        ]
        val_ids = [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in val_subjects]
        y_train = np.stack([labels[sample_id] for sample_id in train_ids]).astype(np.float32)
        y_val = np.stack([labels[sample_id] for sample_id in val_ids]).astype(np.float32)
        prior_val = predict_video_time_mean(train_ids, y_train, val_ids)
        trials = build_trials(cache, labels, train_ids, y_train)
        train_trials = [trial for trial in trials if trial["subject"] in train_subjects]
        val_trials = [trial for trial in trials if trial["subject"] in val_subjects]

        fold_results: list[dict[str, object]] = [
            score("VideoTimeMean", y_val, prior_val, "Mean trajectory by video/time from train subjects.")
        ]
        predictions = {"VideoTimeMean": prior_val}

        for alpha_label, alpha in (("stable", args.alpha_stable), ("aggressive", args.alpha_aggressive)):
            tbcr_mean = predict_tbcr_residual(
                train_trials=train_trials,
                val_trials=val_trials,
                val_ids=val_ids,
                prior_val=prior_val,
                basis_count=args.basis_count,
                alpha=alpha,
                feature_mode="mean_std",
            )
            tbcr_slope = predict_tbcr_residual(
                train_trials=train_trials,
                val_trials=val_trials,
                val_ids=val_ids,
                prior_val=prior_val,
                basis_count=args.basis_count,
                alpha=alpha,
                feature_mode="mean_std_slope",
            )

            stable_correction = np.stack(
                [
                    np.clip(1.0 * tbcr_mean[:, 0], -4.5, 4.5),
                    np.clip(0.10 * tbcr_slope[:, 1], -6.0, 6.0),
                ],
                axis=1,
            )
            stable_pred = np.clip(prior_val + stable_correction, 1.0, 255.0)
            stable_name = f"Prior_DualTBCR_{alpha_label}_084params"
            predictions[stable_name] = stable_pred
            fold_results.append(
                score(
                    stable_name,
                    y_val,
                    stable_pred,
                    "VideoTimeMean plus fixed Dual-TBCR correction using 084 parameters.",
                )
            )

            aggressive_correction = np.stack(
                [
                    np.clip(0.75 * tbcr_mean[:, 0], -4.2, 4.2),
                    np.clip(0.10 * tbcr_slope[:, 1], -5.5, 5.5),
                ],
                axis=1,
            )
            aggressive_pred = np.clip(prior_val + aggressive_correction, 1.0, 255.0)
            aggressive_name = f"Prior_DualTBCR_{alpha_label}_085params"
            predictions[aggressive_name] = aggressive_pred
            fold_results.append(
                score(
                    aggressive_name,
                    y_val,
                    aggressive_pred,
                    "VideoTimeMean plus fixed Dual-TBCR correction using 085 parameters.",
                )
            )

            lrag_correction = np.stack(
                [
                    residual_attention_by_trial(
                        val_ids,
                        prior_val[:, 0],
                        aggressive_correction[:, 0],
                        gamma=0.65,
                        temperature=3.0,
                        distance_sigma=0.3,
                        mode="interp",
                    ),
                    residual_attention_by_trial(
                        val_ids,
                        prior_val[:, 1],
                        aggressive_correction[:, 1],
                        gamma=0.65,
                        temperature=3.0,
                        distance_sigma=0.3,
                        mode="interp",
                    ),
                ],
                axis=1,
            )
            lrag_correction[:, 0] = np.clip(lrag_correction[:, 0], -4.2, 4.2)
            lrag_correction[:, 1] = np.clip(lrag_correction[:, 1], -5.5, 5.5)
            lrag_pred = np.clip(prior_val + lrag_correction, 1.0, 255.0)
            lrag_name = f"Prior_LRAG_{alpha_label}_087params"
            predictions[lrag_name] = lrag_pred
            fold_results.append(
                score(
                    lrag_name,
                    y_val,
                    lrag_pred,
                    "VideoTimeMean plus fixed Dual-TBCR correction and LRAG attention.",
                )
            )

        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": fold_results,
            }
        )
        aggregate_truth.append(y_val)
        for name, pred in predictions.items():
            aggregate_predictions[name].append(pred)
            aggregate_counts[name] += len(val_ids)

    y_all = np.concatenate(aggregate_truth, axis=0)
    aggregate_results = []
    for name, parts in aggregate_predictions.items():
        pred_all = np.concatenate(parts, axis=0)
        aggregate_results.append(
            score(
                name,
                y_all,
                pred_all,
                "Weighted aggregate across subject-disjoint folds.",
            )
        )
    aggregate_results = sorted(aggregate_results, key=lambda item: float(item["overall_mae"]))

    output = {
        "method": "Subject-disjoint cross-fold validation for prior-only TBCR/LRAG",
        "important_note": (
            "This validates TBCR/LRAG residual correction over VideoTimeMean only. "
            "It does not validate SCEW deep checkpoints, which require retraining each fold."
        ),
        "fold_size": args.fold_size,
        "basis_count": args.basis_count,
        "alpha_stable": args.alpha_stable,
        "alpha_aggressive": args.alpha_aggressive,
        "aggregate_results": aggregate_results,
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def predict_tbcr_residual(
    train_trials: list[dict[str, object]],
    val_trials: list[dict[str, object]],
    val_ids: list[str],
    prior_val: np.ndarray,
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
    return tbcr_pred - prior_val


if __name__ == "__main__":
    main()
