from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import (  # noqa: E402
    evaluate_candidate,
    evaluate_residual_grid,
    finalize_metric,
    load_or_build_precomputed,
    sanitize,
)
from tools.cross_fold_signal_residual_over_pattern_prior import (  # noqa: E402
    make_leave_subject_out_train_prior,
    make_pattern_prior,
)
from tools.run_iteration_experiments import expand_subjects, load_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested OOF EEG-fNIRS reliability fusion gates.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_baseline.npz")
    parser.add_argument("--output", default="experiments/results/iteration_240_246_neurovascular_oof_gate.json")
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--meta-alpha", type=float, default=1000.0)
    parser.add_argument("--scales", default="0.03,0.05,0.08,0.12,0.20")
    parser.add_argument("--clips", default="1,2,4")
    parser.add_argument("--smooth-windows", default="0,5")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(data_root, subjects, Path(args.precompute_cache))
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}
    scales = parse_floats(args.scales)
    clips = parse_floats(args.clips)
    smooth_windows = parse_ints(args.smooth_windows)

    views = {
        "eeg": pre["eeg_lag"],
        "fnirs": pre["fnirs_slow"],
        "nv": pre["neurovascular"],
        "early": pre["early_concat"],
    }

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] nested OOF fusion train={len(train_subjects)} val={val_subjects}", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        prior_train = make_leave_subject_out_train_prior(labels, train_subjects, train_ids)
        prior_val = make_pattern_prior(train_ids, y_train, val_ids)
        residual_train = (y_train - prior_train).astype(np.float32)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)

        fold_results = [
            evaluate_candidate(metric_acc, "098_PatternPrior_reference", y_val, prior_val, "Strong non-signal prior.")
        ]
        expert_oof, expert_val = nested_expert_predictions(
            views=views,
            train_idx=train_idx,
            val_idx=val_idx,
            train_ids=train_ids,
            train_subjects=train_subjects,
            residual_train=residual_train,
            alpha=args.alpha,
        )
        module_residuals = build_oof_gate_modules(
            expert_oof=expert_oof,
            expert_val=expert_val,
            residual_train=residual_train,
            meta_alpha=args.meta_alpha,
        )
        for name, residual_val in module_residuals.items():
            fold_results.extend(
                evaluate_residual_grid(
                    metric_acc=metric_acc,
                    base_name=name,
                    y_val=y_val,
                    prior_val=prior_val,
                    val_ids=val_ids,
                    residual_val=residual_val,
                    scales=scales,
                    clips=clips,
                    smooth_windows=smooth_windows,
                )
            )
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
        "method": "Nested OOF neurovascular reliability gates",
        "note": (
            "For each outer fold, EEG/fNIRS/neurovascular experts are predicted for training rows "
            "by inner leave-one-subject-out models. Fusion gates are trained only on these OOF expert "
            "predictions, then applied to validation expert predictions."
        ),
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def nested_expert_predictions(
    views: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_ids: list[str],
    train_subjects: list[str],
    residual_train: np.ndarray,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    expert_oof = {name: np.zeros_like(residual_train, dtype=np.float32) for name in views}
    expert_val = {}
    subject_by_row = np.asarray([sample_id.split("_V", 1)[0] for sample_id in train_ids])

    for view_name, matrix in views.items():
        x_train_all = matrix[train_idx]
        x_val = matrix[val_idx]
        for subject in train_subjects:
            inner_val = subject_by_row == subject
            inner_train = ~inner_val
            expert_oof[view_name][inner_val] = fit_predict_ridge(
                x_train_all[inner_train],
                residual_train[inner_train],
                x_train_all[inner_val],
                alpha,
            )
        expert_val[view_name] = fit_predict_ridge(x_train_all, residual_train, x_val, alpha)
    return expert_oof, expert_val


def build_oof_gate_modules(
    expert_oof: dict[str, np.ndarray],
    expert_val: dict[str, np.ndarray],
    residual_train: np.ndarray,
    meta_alpha: float,
) -> dict[str, np.ndarray]:
    eeg_oof = expert_oof["eeg"]
    fnirs_oof = expert_oof["fnirs"]
    nv_oof = expert_oof["nv"]
    early_oof = expert_oof["early"]
    eeg_val = expert_val["eeg"]
    fnirs_val = expert_val["fnirs"]
    nv_val = expert_val["nv"]
    early_val = expert_val["early"]

    modules = {}
    modules["240_OOFLinearStack_EEG_FNIRS"] = meta_ridge(
        stack_features(eeg_oof, fnirs_oof),
        residual_train,
        stack_features(eeg_val, fnirs_val),
        meta_alpha,
    )
    modules["241_OOFLinearStack_EEG_FNIRS_NV"] = meta_ridge(
        stack_features(eeg_oof, fnirs_oof, nv_oof),
        residual_train,
        stack_features(eeg_val, fnirs_val, nv_val),
        meta_alpha,
    )
    modules["242_OOFAgreementWeighted"] = agreement_weighted(eeg_oof, fnirs_oof, eeg_val, fnirs_val, residual_train)
    modules["243_OOFDisagreementAwareStack"] = meta_ridge(
        disagreement_features(eeg_oof, fnirs_oof, nv_oof, early_oof),
        residual_train,
        disagreement_features(eeg_val, fnirs_val, nv_val, early_val),
        meta_alpha,
    )
    consensus_oof = agreement_weighted(eeg_oof, fnirs_oof, eeg_oof, fnirs_oof, residual_train)
    consensus_val = agreement_weighted(eeg_oof, fnirs_oof, eeg_val, fnirs_val, residual_train)
    modules["244_OOFHelpfulGateConsensus"] = helpful_gate(
        train_features=disagreement_features(eeg_oof, fnirs_oof, nv_oof, early_oof),
        val_features=disagreement_features(eeg_val, fnirs_val, nv_val, early_val),
        residual_train=residual_train,
        consensus_train=consensus_oof,
        consensus_val=consensus_val,
    )
    modules["245_NeurovascularHelpfulGate"] = helpful_gate(
        train_features=disagreement_features(nv_oof, early_oof),
        val_features=disagreement_features(nv_val, early_val),
        residual_train=residual_train,
        consensus_train=nv_oof,
        consensus_val=nv_val,
    )
    modules["246_ConsensusConfidenceShrink"] = confidence_shrink(eeg_val, fnirs_val)
    return modules


def stack_features(*preds: np.ndarray) -> np.ndarray:
    parts = list(preds)
    if len(preds) >= 2:
        parts.extend([np.abs(preds[0] - preds[1]), preds[0] * preds[1]])
    return sanitize(np.concatenate(parts, axis=1))


def disagreement_features(*preds: np.ndarray) -> np.ndarray:
    parts = list(preds)
    for left, right in zip(preds[:-1], preds[1:]):
        parts.append(np.abs(left - right))
        parts.append(left * right)
        parts.append((np.sign(left) == np.sign(right)).astype(np.float32))
    return sanitize(np.concatenate(parts, axis=1))


def meta_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(sanitize(x_train), y_train)
    return model.predict(sanitize(x_val)).astype(np.float32)


def agreement_weighted(
    eeg_oof: np.ndarray,
    fnirs_oof: np.ndarray,
    eeg_val: np.ndarray,
    fnirs_val: np.ndarray,
    residual_train: np.ndarray,
) -> np.ndarray:
    eeg_mse = ((eeg_oof - residual_train) ** 2).mean(axis=0) + 1e-6
    fnirs_mse = ((fnirs_oof - residual_train) ** 2).mean(axis=0) + 1e-6
    eeg_weight = (1.0 / eeg_mse) / (1.0 / eeg_mse + 1.0 / fnirs_mse)
    consensus = eeg_weight[None, :] * eeg_val + (1.0 - eeg_weight[None, :]) * fnirs_val
    agreement = (np.sign(eeg_val) == np.sign(fnirs_val)).astype(np.float32)
    magnitude_ratio = np.minimum(np.abs(eeg_val), np.abs(fnirs_val)) / (
        np.maximum(np.abs(eeg_val), np.abs(fnirs_val)) + 1e-3
    )
    confidence = agreement * (0.35 + 0.65 * magnitude_ratio)
    return (confidence * consensus).astype(np.float32)


def helpful_gate(
    train_features: np.ndarray,
    val_features: np.ndarray,
    residual_train: np.ndarray,
    consensus_train: np.ndarray,
    consensus_val: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(consensus_val, dtype=np.float32)
    for dim in range(2):
        helpful = (np.abs(residual_train[:, dim] - consensus_train[:, dim]) < np.abs(residual_train[:, dim])).astype(int)
        if helpful.min() == helpful.max():
            probability = np.full(consensus_val.shape[0], float(helpful.mean()), dtype=np.float32)
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.2, max_iter=1000, class_weight="balanced", solver="lbfgs"),
            )
            model.fit(sanitize(train_features), helpful)
            probability = model.predict_proba(sanitize(val_features))[:, 1].astype(np.float32)
        out[:, dim] = probability * consensus_val[:, dim]
    return out


def confidence_shrink(eeg_val: np.ndarray, fnirs_val: np.ndarray) -> np.ndarray:
    consensus = 0.5 * (eeg_val + fnirs_val)
    ratio = np.minimum(np.abs(eeg_val), np.abs(fnirs_val)) / (
        np.maximum(np.abs(eeg_val), np.abs(fnirs_val)) + 1e-3
    )
    agreement = (np.sign(eeg_val) == np.sign(fnirs_val)).astype(np.float32)
    confidence = agreement * ratio
    return (confidence * consensus).astype(np.float32)


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(sanitize(x_train), y_train)
    return model.predict(sanitize(x_val)).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
