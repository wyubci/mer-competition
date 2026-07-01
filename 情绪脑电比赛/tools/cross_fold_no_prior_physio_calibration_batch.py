from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.cross_fold_no_prior_physio_adaptive_batch import add_metric, finalize_metric, pca_ridge_predict  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-video-prior physiological calibration batch 396-415.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_396_415_no_prior_physio_calibration.json")
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

        x_train = pre["early_concat"][train_idx]
        x_val = pre["early_concat"][val_idx]
        huber_train_raw, huber_val_raw = fit_huber_train_val(x_train, y_train[:, 0], x_val, components=16)
        ridge_val_s5 = smooth_1d(
            val_ids,
            pca_ridge_predict(x_train, y_train, x_val, components=16, alpha=10000.0)[:, 0],
            window=5,
        )
        candidates = build_candidates(
            center=center,
            train_ids=train_ids,
            val_ids=val_ids,
            y_train_v=y_train[:, 0],
            huber_train_raw=huber_train_raw,
            huber_val_raw=huber_val_raw,
            ridge_val_s5=ridge_val_s5,
        )
        references = {
            "Reference_321_Center128_noPrior": center,
            "Reference_364_HuberPCA16Valence_CenterArousal": from_valence(
                center, smooth_1d(val_ids, huber_val_raw, window=5)
            ),
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
        "method": "No-video-prior physiological output calibration batch",
        "iteration_range": "396-415",
        "note": (
            "No candidate uses video/time label priors. This batch keeps HuberPCA16 fixed and searches "
            "sequence smoothing, residual soft clipping, asymmetric scaling, train-prediction calibration, "
            "and unlabeled subject-level prediction centering."
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
    y_train_v: np.ndarray,
    huber_train_raw: np.ndarray,
    huber_val_raw: np.ndarray,
    ridge_val_s5: np.ndarray,
) -> dict[str, np.ndarray]:
    train_s5 = smooth_1d(train_ids, huber_train_raw, window=5)
    val_s5 = smooth_1d(val_ids, huber_val_raw, window=5)
    candidates: dict[str, np.ndarray] = {}
    candidates["396_HuberSmooth0Valence_CenterArousal"] = from_valence(center, huber_val_raw)
    candidates["397_HuberSmooth3Valence_CenterArousal"] = from_valence(center, smooth_1d(val_ids, huber_val_raw, 3))
    candidates["398_HuberSmooth7Valence_CenterArousal"] = from_valence(center, smooth_1d(val_ids, huber_val_raw, 7))
    candidates["399_HuberSmooth9Valence_CenterArousal"] = from_valence(center, smooth_1d(val_ids, huber_val_raw, 9))
    candidates["400_HuberMedian5Valence_CenterArousal"] = from_valence(center, median_smooth(val_ids, huber_val_raw, 5))
    candidates["401_HuberExpA03Valence_CenterArousal"] = from_valence(center, exp_smooth(val_ids, huber_val_raw, alpha=0.30))
    candidates["402_HuberExpA06Valence_CenterArousal"] = from_valence(center, exp_smooth(val_ids, huber_val_raw, alpha=0.60))
    candidates["403_HuberTanhResidual12Valence_CenterArousal"] = from_valence(center, soft_tanh(val_s5, scale=12.0))
    candidates["404_HuberTanhResidual20Valence_CenterArousal"] = from_valence(center, soft_tanh(val_s5, scale=20.0))
    candidates["405_HuberClip20Valence_CenterArousal"] = from_valence(center, hard_clip(val_s5, cap=20.0))
    candidates["406_HuberAsymP08N10Valence_CenterArousal"] = from_valence(center, asym_scale(val_s5, pos=0.8, neg=1.0))
    candidates["407_HuberAsymP10N08Valence_CenterArousal"] = from_valence(center, asym_scale(val_s5, pos=1.0, neg=0.8))
    candidates["408_HuberAsymP12N08Valence_CenterArousal"] = from_valence(center, asym_scale(val_s5, pos=1.2, neg=0.8))
    candidates["409_HuberAsymP08N12Valence_CenterArousal"] = from_valence(center, asym_scale(val_s5, pos=0.8, neg=1.2))
    candidates["410_HuberMeanStdCalibValence_CenterArousal"] = from_valence(
        center, mean_std_calibrate(train_s5, y_train_v, val_s5)
    )
    candidates["411_HuberMedianIQRCalibValence_CenterArousal"] = from_valence(
        center, median_iqr_calibrate(train_s5, y_train_v, val_s5)
    )
    candidates["412_HuberMeanShiftCalibValence_CenterArousal"] = from_valence(
        center, val_s5 - float(train_s5.mean()) + float(y_train_v.mean())
    )
    candidates["413_HuberSubjectMeanTo128Valence_CenterArousal"] = from_valence(
        center, group_prediction_center(val_ids, val_s5, mode="subject", target=128.0)
    )
    candidates["414_HuberSubjectMeanToTrainPredValence_CenterArousal"] = from_valence(
        center, group_prediction_center(val_ids, val_s5, mode="subject", target=float(train_s5.mean()))
    )
    candidates["415_HuberRidgeBlend75Valence_CenterArousal"] = from_valence(
        center, 0.75 * val_s5 + 0.25 * ridge_val_s5
    )
    return candidates


def fit_huber_train_val(
    x_train: np.ndarray,
    y_train_v: np.ndarray,
    x_val: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026),
        HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=300),
    )
    model.fit(sanitize(x_train), y_train_v)
    train_pred = np.clip(model.predict(sanitize(x_train)).astype(np.float32), 1.0, 255.0)
    val_pred = np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)
    return train_pred, val_pred


def smooth_1d(sample_ids: list[str], values: np.ndarray, window: int) -> np.ndarray:
    return smooth_predictions(sample_ids, np.asarray(values, dtype=np.float32).reshape(-1, 1), window)[:, 0]


def from_valence(center: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = center.copy()
    out[:, 0] = np.asarray(values, dtype=np.float32)
    return out


def group_key(sample_id: str, mode: str) -> str:
    subject, rest = sample_id.split("_V", 1)
    if mode == "subject":
        return subject
    video = rest.split("_T", 1)[0]
    return f"{subject}_V{video}"


def timestamp(sample_id: str) -> int:
    return int(sample_id.rsplit("_T", 1)[1])


def grouped_indices(sample_ids: list[str], mode: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[group_key(sample_id, mode)].append(index)
    return groups


def median_smooth(sample_ids: list[str], values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    radius = window // 2
    for indices in grouped_indices(sample_ids, "trial").values():
        idx = sorted(indices, key=lambda item: timestamp(sample_ids[item]))
        seq = values[idx]
        med = np.zeros_like(seq)
        for local_index in range(len(seq)):
            start = max(0, local_index - radius)
            stop = min(len(seq), local_index + radius + 1)
            med[local_index] = float(np.median(seq[start:stop]))
        out[idx] = med
    return out


def exp_smooth(sample_ids: list[str], values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, "trial").values():
        idx = sorted(indices, key=lambda item: timestamp(sample_ids[item]))
        seq = values[idx]
        smoothed = np.zeros_like(seq)
        smoothed[0] = seq[0]
        for local_index in range(1, len(seq)):
            smoothed[local_index] = alpha * seq[local_index] + (1.0 - alpha) * smoothed[local_index - 1]
        out[idx] = smoothed
    return out


def soft_tanh(values: np.ndarray, scale: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    return 128.0 + scale * np.tanh(residual / scale)


def hard_clip(values: np.ndarray, cap: float) -> np.ndarray:
    return 128.0 + np.clip(np.asarray(values, dtype=np.float32) - 128.0, -cap, cap)


def asym_scale(values: np.ndarray, pos: float, neg: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    scaled = np.where(residual >= 0.0, pos * residual, neg * residual)
    return 128.0 + scaled


def mean_std_calibrate(train_pred: np.ndarray, y_train_v: np.ndarray, val_pred: np.ndarray) -> np.ndarray:
    pred_std = max(float(np.std(train_pred)), 1e-4)
    target_std = max(float(np.std(y_train_v)), 1e-4)
    return (val_pred - float(np.mean(train_pred))) * (target_std / pred_std) + float(np.mean(y_train_v))


def median_iqr_calibrate(train_pred: np.ndarray, y_train_v: np.ndarray, val_pred: np.ndarray) -> np.ndarray:
    pred_q25, pred_q75 = np.quantile(train_pred, [0.25, 0.75])
    target_q25, target_q75 = np.quantile(y_train_v, [0.25, 0.75])
    pred_iqr = max(float(pred_q75 - pred_q25), 1e-4)
    target_iqr = max(float(target_q75 - target_q25), 1e-4)
    return (val_pred - float(np.median(train_pred))) * (target_iqr / pred_iqr) + float(np.median(y_train_v))


def group_prediction_center(sample_ids: list[str], values: np.ndarray, mode: str, target: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, mode).values():
        idx = np.asarray(indices, dtype=np.int64)
        out[idx] = values[idx] - float(values[idx].mean()) + target
    return out


def candidate_note(name: str) -> str:
    if name.startswith("Reference"):
        return "Reference from previous physiological-only batches."
    if "Smooth" in name or "Median" in name or "Exp" in name:
        return "Temporal smoothing variant over the fixed HuberPCA16 valence trajectory."
    if "Tanh" in name or "Clip" in name:
        return "Soft or hard residual clipping around center."
    if "Asym" in name:
        return "Asymmetric positive/negative residual scaling around center."
    if "Calib" in name:
        return "Train-prediction distribution calibration using only train labels."
    if "SubjectMean" in name:
        return "Unlabeled subject-level prediction centering."
    if "Blend" in name:
        return "Blend Huber with Ridge low-rank physiological valence."
    return "No-video-prior physiological output calibration candidate."


if __name__ == "__main__":
    main()
