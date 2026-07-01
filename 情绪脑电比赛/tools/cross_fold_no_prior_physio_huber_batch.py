from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.cross_fold_no_prior_physio_adaptive_batch import (  # noqa: E402
    add_metric,
    agreement_average,
    finalize_metric,
    group_center,
    group_zscore,
    pca_head_predict,
    pca_ridge_predict,
    temporal_delta,
)
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-video-prior Huber physiological batch 376-395.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_376_395_no_prior_physio_huber.json")
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root=data_root,
        subjects=subjects,
        cache_path=Path(args.precompute_cache),
        fnirs_types=(0, 1, 2, 3, 4, 5),
        feature_normalization="none",
        baseline_correction=True,
    )
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] train={len(train_subjects)} val={val_subjects}", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)
        center = np.full_like(y_val, 128.0)

        x_raw_train = pre["early_concat"][train_idx]
        x_raw_val = pre["early_concat"][val_idx]
        x_eeg_train = pre["eeg_lag"][train_idx]
        x_eeg_val = pre["eeg_lag"][val_idx]
        x_fnirs_train = pre["fnirs_slow"][train_idx]
        x_fnirs_val = pre["fnirs_slow"][val_idx]
        x_neuro_train = pre["neurovascular"][train_idx]
        x_neuro_val = pre["neurovascular"][val_idx]

        ridge16_raw = smooth_v(
            val_ids,
            pca_ridge_predict(x_raw_train, y_train, x_raw_val, components=16, alpha=10000.0)[:, 0],
        )
        huber16_raw = smooth_v(val_ids, huber_predict(x_raw_train, y_train[:, 0], x_raw_val, components=16))
        reference_354 = agreement_gate_from_values(center, smooth_v(
            val_ids,
            pca_ridge_predict(x_raw_train, y_train, x_raw_val, components=8, alpha=10000.0)[:, 0],
        ), ridge16_raw, threshold=8.0)

        candidates = build_candidates(
            center=center,
            train_ids=train_ids,
            val_ids=val_ids,
            y_train=y_train,
            x_raw_train=x_raw_train,
            x_raw_val=x_raw_val,
            x_eeg_train=x_eeg_train,
            x_eeg_val=x_eeg_val,
            x_fnirs_train=x_fnirs_train,
            x_fnirs_val=x_fnirs_val,
            x_neuro_train=x_neuro_train,
            x_neuro_val=x_neuro_val,
            ridge16_raw=ridge16_raw,
            huber16_raw=huber16_raw,
        )
        references = {
            "Reference_321_Center128_noPrior": center,
            "Reference_354_AgreementGatedPCA16Valence_CenterArousal": reference_354,
            "Reference_364_HuberPCA16Valence_CenterArousal": from_valence(center, huber16_raw),
        }
        fold_predictions = {**references, **candidates}

        fold_results = []
        for name, pred in fold_predictions.items():
            pred = np.clip(pred.astype(np.float32), 1.0, 255.0)
            note = candidate_note(name)
            fold_results.append(score(name, y_val, pred, note))
            add_metric(metric_acc, name, y_val, pred, note)
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
        "method": "No-video-prior Huber physiological module batch",
        "iteration_range": "376-395",
        "note": (
            "No candidate uses video/time label priors. This batch studies the Huber valence head: "
            "PCA dimension, modality source, subject/trial alignment, agreement gating, residual "
            "caps/shrinkage, and tree/boosted references."
        ),
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_candidates(
    center: np.ndarray,
    train_ids: list[str],
    val_ids: list[str],
    y_train: np.ndarray,
    x_raw_train: np.ndarray,
    x_raw_val: np.ndarray,
    x_eeg_train: np.ndarray,
    x_eeg_val: np.ndarray,
    x_fnirs_train: np.ndarray,
    x_fnirs_val: np.ndarray,
    x_neuro_train: np.ndarray,
    x_neuro_val: np.ndarray,
    ridge16_raw: np.ndarray,
    huber16_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}

    train_subject_center, val_subject_center = group_center(x_raw_train, train_ids, x_raw_val, val_ids, "subject")
    train_subject_z, val_subject_z = group_zscore(x_raw_train, train_ids, x_raw_val, val_ids, "subject")
    train_trial_center, val_trial_center = group_center(x_raw_train, train_ids, x_raw_val, val_ids, "trial")
    train_delta = temporal_delta(x_raw_train, train_ids)
    val_delta = temporal_delta(x_raw_val, val_ids)
    train_level_delta = np.concatenate([x_raw_train, train_delta], axis=1)
    val_level_delta = np.concatenate([x_raw_val, val_delta], axis=1)

    candidates["376_HuberPCA8Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_raw_train, y_train[:, 0], x_raw_val, 8))
    )
    candidates["377_HuberPCA24Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_raw_train, y_train[:, 0], x_raw_val, 24))
    )
    candidates["378_HuberPCA32Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_raw_train, y_train[:, 0], x_raw_val, 32))
    )
    candidates["379_HuberSubjectZPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(train_subject_z, y_train[:, 0], val_subject_z, 16))
    )
    candidates["380_HuberSubjectCenterPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(train_subject_center, y_train[:, 0], val_subject_center, 16))
    )
    candidates["381_HuberTrialCenterPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(train_trial_center, y_train[:, 0], val_trial_center, 16))
    )
    candidates["382_HuberEEGPCA8Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_eeg_train, y_train[:, 0], x_eeg_val, 8))
    )
    candidates["383_HuberFNIRSPCA8Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_fnirs_train, y_train[:, 0], x_fnirs_val, 8))
    )
    candidates["384_HuberNeuroPCA8Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_neuro_train, y_train[:, 0], x_neuro_val, min(8, x_neuro_train.shape[1])))
    )
    candidates["385_HuberDeltaPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(train_delta, y_train[:, 0], val_delta, 16))
    )
    candidates["386_HuberLevelDeltaPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(train_level_delta, y_train[:, 0], val_level_delta, 16))
    )

    elastic16_raw = smooth_v(val_ids, pca_head_predict(x_raw_train, y_train[:, 0], x_raw_val, 16, "elastic"))
    subject_huber_raw = smooth_v(val_ids, huber_predict(train_subject_z, y_train[:, 0], val_subject_z, 16))
    candidates["387_HuberRidgeAgreementGateValence_CenterArousal"] = agreement_average(
        center, huber16_raw, ridge16_raw, threshold=8.0
    )
    candidates["388_HuberElasticAgreementGateValence_CenterArousal"] = agreement_average(
        center, huber16_raw, elastic16_raw, threshold=8.0
    )
    candidates["389_HuberSubjectAgreementGateValence_CenterArousal"] = agreement_average(
        center, huber16_raw, subject_huber_raw, threshold=8.0
    )
    candidates["390_HuberCap8Valence_CenterArousal"] = from_valence(center, cap_residual(huber16_raw, 8.0))
    candidates["391_HuberCap12Valence_CenterArousal"] = from_valence(center, cap_residual(huber16_raw, 12.0))
    candidates["392_HuberShrink80Valence_CenterArousal"] = from_valence(center, 128.0 + 0.80 * (huber16_raw - 128.0))

    y_winsor = winsorize(y_train[:, 0], lower=0.10, upper=0.90)
    candidates["393_WinsorTargetHuberPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, huber_predict(x_raw_train, y_winsor, x_raw_val, 16))
    )
    candidates["394_ExtraTreesPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, tree_predict(x_raw_train, y_train[:, 0], x_raw_val, "extra"))
    )
    candidates["395_GBRHuberPCA16Valence_CenterArousal"] = from_valence(
        center, smooth_v(val_ids, tree_predict(x_raw_train, y_train[:, 0], x_raw_val, "gbr_huber"))
    )

    return candidates


def huber_predict(x_train: np.ndarray, y_train_v: np.ndarray, x_val: np.ndarray, components: int) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026),
        HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=300),
    )
    model.fit(sanitize(x_train), y_train_v)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def tree_predict(x_train: np.ndarray, y_train_v: np.ndarray, x_val: np.ndarray, kind: str) -> np.ndarray:
    if kind == "extra":
        reg = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=16,
            random_state=2026,
            n_jobs=-1,
        )
    elif kind == "gbr_huber":
        reg = GradientBoostingRegressor(
            loss="huber",
            n_estimators=80,
            learning_rate=0.04,
            max_depth=2,
            random_state=2026,
        )
    else:
        raise ValueError(kind)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=16, svd_solver="randomized", random_state=2026),
        reg,
    )
    model.fit(sanitize(x_train), y_train_v)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def smooth_v(val_ids: list[str], values: np.ndarray) -> np.ndarray:
    return smooth_predictions(val_ids, np.asarray(values, dtype=np.float32).reshape(-1, 1), 5)[:, 0]


def from_valence(center: np.ndarray, valence: np.ndarray) -> np.ndarray:
    out = center.copy()
    out[:, 0] = np.asarray(valence, dtype=np.float32)
    return out


def cap_residual(valence: np.ndarray, cap: float) -> np.ndarray:
    return 128.0 + np.clip(np.asarray(valence, dtype=np.float32) - 128.0, -cap, cap)


def winsorize(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    low = float(np.quantile(values, lower))
    high = float(np.quantile(values, upper))
    return np.clip(values, low, high).astype(np.float32)


def agreement_gate_from_values(center: np.ndarray, pred_a_v: np.ndarray, pred_b_v: np.ndarray, threshold: float) -> np.ndarray:
    out = center.copy()
    disagreement = np.abs(pred_a_v - pred_b_v)
    out[:, 0] = np.where(disagreement < threshold, pred_b_v, center[:, 0])
    return out


def candidate_note(name: str) -> str:
    if name.startswith("Reference"):
        return "Reference from previous physiological-only batches."
    if "PCA8" in name or "PCA24" in name or "PCA32" in name:
        return "Huber valence head with changed low-rank dimensionality."
    if "Subject" in name or "Trial" in name:
        return "Huber valence head under unlabeled subject/trial feature alignment."
    if "EEG" in name or "FNIRS" in name or "Neuro" in name:
        return "Single-view Huber expert to test modality-specific contribution."
    if "Delta" in name:
        return "Huber valence head on temporal-difference physiological representation."
    if "Agreement" in name:
        return "Use Huber only when another physiological expert agrees."
    if "Cap" in name or "Shrink" in name:
        return "Constrain Huber residual magnitude around the center."
    if "Winsor" in name:
        return "Train Huber on winsorized valence targets to reduce label-tail influence."
    if "ExtraTrees" in name or "GBR" in name:
        return "Tree-based nonlinear reference over low-rank physiological features."
    return "No-video-prior Huber physiological candidate."


if __name__ == "__main__":
    main()
