from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.cross_fold_no_prior_physio_adaptive_batch import (  # noqa: E402
    add_metric,
    finalize_metric,
    pca_head_predict,
    pca_ridge_predict,
)
from tools.cross_fold_no_prior_physio_calibration_batch import (  # noqa: E402
    exp_smooth,
    median_smooth,
    smooth_1d,
)
from tools.run_iteration_experiments import expand_subjects, load_labels, score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-video-prior asymmetric residual refinement 416-435.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_416_435_no_prior_physio_asym_refine.json")
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
        huber_val_raw = huber_predict(x_train, y_train[:, 0], x_val, components=16)
        huber_s5 = smooth_1d(val_ids, huber_val_raw, 5)
        huber_s7 = smooth_1d(val_ids, huber_val_raw, 7)
        huber_s9 = smooth_1d(val_ids, huber_val_raw, 9)
        huber_exp03 = exp_smooth(val_ids, huber_val_raw, alpha=0.30)
        huber_median5 = median_smooth(val_ids, huber_val_raw, window=5)
        ridge_s5 = smooth_1d(
            val_ids,
            pca_ridge_predict(x_train, y_train, x_val, components=16, alpha=10000.0)[:, 0],
            5,
        )
        elastic_s5 = smooth_1d(
            val_ids,
            pca_head_predict(x_train, y_train[:, 0], x_val, components=16, head="elastic"),
            5,
        )
        train_pred = smooth_1d(train_ids, huber_predict(x_train, y_train[:, 0], x_train, components=16), 5)
        train_mean_shift = float(y_train[:, 0].mean() - train_pred.mean())

        candidates = build_candidates(
            center=center,
            huber_s5=huber_s5,
            huber_s7=huber_s7,
            huber_s9=huber_s9,
            huber_exp03=huber_exp03,
            huber_median5=huber_median5,
            ridge_s5=ridge_s5,
            elastic_s5=elastic_s5,
            train_mean_shift=train_mean_shift,
        )
        references = {
            "Reference_321_Center128_noPrior": center,
            "Reference_407_HuberAsymP10N08Valence_CenterArousal": from_valence(center, asym_scale(huber_s5, 1.0, 0.8)),
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
        "method": "No-video-prior asymmetric residual refinement batch",
        "iteration_range": "416-435",
        "note": (
            "No candidate uses video/time label priors. This batch refines the winning asymmetric "
            "Huber valence correction by varying negative scale, smoothing, blends, and magnitude gates."
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
    huber_s5: np.ndarray,
    huber_s7: np.ndarray,
    huber_s9: np.ndarray,
    huber_exp03: np.ndarray,
    huber_median5: np.ndarray,
    ridge_s5: np.ndarray,
    elastic_s5: np.ndarray,
    train_mean_shift: float,
) -> dict[str, np.ndarray]:
    blend75 = 0.75 * huber_s5 + 0.25 * ridge_s5
    blend50 = 0.50 * huber_s5 + 0.50 * ridge_s5
    blend90 = 0.90 * huber_s5 + 0.10 * ridge_s5
    candidates: dict[str, np.ndarray] = {
        "416_HuberAsymP10N06Valence_CenterArousal": from_valence(center, asym_scale(huber_s5, 1.0, 0.6)),
        "417_HuberAsymP10N07Valence_CenterArousal": from_valence(center, asym_scale(huber_s5, 1.0, 0.7)),
        "418_HuberAsymP10N09Valence_CenterArousal": from_valence(center, asym_scale(huber_s5, 1.0, 0.9)),
        "419_HuberAsymP09N08Valence_CenterArousal": from_valence(center, asym_scale(huber_s5, 0.9, 0.8)),
        "420_HuberAsymP11N08Valence_CenterArousal": from_valence(center, asym_scale(huber_s5, 1.1, 0.8)),
        "421_HuberAsymP10N08Smooth7Valence_CenterArousal": from_valence(center, asym_scale(huber_s7, 1.0, 0.8)),
        "422_HuberAsymP10N08Smooth9Valence_CenterArousal": from_valence(center, asym_scale(huber_s9, 1.0, 0.8)),
        "423_HuberAsymP10N08Exp03Valence_CenterArousal": from_valence(center, asym_scale(huber_exp03, 1.0, 0.8)),
        "424_HuberAsymP10N08Median5Valence_CenterArousal": from_valence(center, asym_scale(huber_median5, 1.0, 0.8)),
        "425_HuberAsymBlend75Valence_CenterArousal": from_valence(center, asym_scale(blend75, 1.0, 0.8)),
        "426_HuberRidgeBlend50Valence_CenterArousal": from_valence(center, blend50),
        "427_HuberRidgeBlend90Valence_CenterArousal": from_valence(center, blend90),
        "428_RidgeAsymP10N08Valence_CenterArousal": from_valence(center, asym_scale(ridge_s5, 1.0, 0.8)),
        "429_ElasticAsymP10N08Valence_CenterArousal": from_valence(center, asym_scale(elastic_s5, 1.0, 0.8)),
        "430_HuberMeanShiftAsymP10N08Valence_CenterArousal": from_valence(
            center, asym_scale(huber_s5 + train_mean_shift, 1.0, 0.8)
        ),
        "431_HuberAsymThenClip40Valence_CenterArousal": from_valence(
            center, hard_clip(asym_scale(huber_s5, 1.0, 0.8), 40.0)
        ),
        "432_HuberNegMagGateAValence_CenterArousal": from_valence(center, neg_magnitude_gate(huber_s5, mild=0.9, strong=0.7, threshold=12.0)),
        "433_HuberNegMagGateBValence_CenterArousal": from_valence(center, neg_magnitude_gate(huber_s5, mild=0.9, strong=0.6, threshold=16.0)),
        "434_HuberPosBoostNegShrinkValence_CenterArousal": from_valence(center, magnitude_asym(huber_s5, pos_small=1.0, pos_large=1.1, neg_small=0.9, neg_large=0.7, threshold=12.0)),
        "435_HuberAsymExpBlendValence_CenterArousal": from_valence(
            center, 0.50 * asym_scale(huber_s5, 1.0, 0.8) + 0.50 * asym_scale(huber_exp03, 1.0, 0.8)
        ),
    }
    return candidates


def huber_predict(x_train: np.ndarray, y_train_v: np.ndarray, x_val: np.ndarray, components: int) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026),
        HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=300),
    )
    model.fit(sanitize(x_train), y_train_v)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def from_valence(center: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = center.copy()
    out[:, 0] = np.asarray(values, dtype=np.float32)
    return out


def asym_scale(values: np.ndarray, pos: float, neg: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    scaled = np.where(residual >= 0.0, pos * residual, neg * residual)
    return 128.0 + scaled


def hard_clip(values: np.ndarray, cap: float) -> np.ndarray:
    return 128.0 + np.clip(np.asarray(values, dtype=np.float32) - 128.0, -cap, cap)


def neg_magnitude_gate(values: np.ndarray, mild: float, strong: float, threshold: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    neg_scale = np.where(np.abs(residual) >= threshold, strong, mild)
    scale = np.where(residual < 0.0, neg_scale, 1.0)
    return 128.0 + scale * residual


def magnitude_asym(
    values: np.ndarray,
    pos_small: float,
    pos_large: float,
    neg_small: float,
    neg_large: float,
    threshold: float,
) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    large = np.abs(residual) >= threshold
    pos_scale = np.where(large, pos_large, pos_small)
    neg_scale = np.where(large, neg_large, neg_small)
    scale = np.where(residual >= 0.0, pos_scale, neg_scale)
    return 128.0 + scale * residual


def candidate_note(name: str) -> str:
    if name.startswith("Reference"):
        return "Reference from previous physiological-only batches."
    if "Smooth" in name or "Exp" in name or "Median" in name:
        return "Asymmetric residual correction after alternative temporal smoothing."
    if "Blend" in name:
        return "Blend Huber with Ridge before or after asymmetric residual correction."
    if "RidgeAsym" in name or "ElasticAsym" in name:
        return "Apply the same asymmetric output rule to a non-Huber head."
    if "MeanShift" in name:
        return "Train mean-shift plus asymmetric residual correction."
    if "Clip" in name or "Mag" in name or "Boost" in name:
        return "Magnitude-dependent asymmetric residual correction."
    return "Refined asymmetric Huber valence residual candidate."


if __name__ == "__main__":
    main()
