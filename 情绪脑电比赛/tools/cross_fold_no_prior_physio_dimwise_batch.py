from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-video-prior physiological dimwise fusion batch 336-355."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_336_355_no_prior_physio_dimwise.json")
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
        train_mean = np.tile(y_train.mean(axis=0, keepdims=True), (len(val_ids), 1)).astype(np.float32)
        base = build_base_predictions(pre, train_idx, val_idx, y_train, val_ids)
        candidates = build_dimwise_candidates(center, train_mean, base)
        references = {
            "Reference_321_Center128_noPrior": center,
            "Reference_322_TrainMean_noPrior": train_mean,
            "Reference_333_PCA8Direct_smooth5": base["pca8_s5"],
            "Reference_333_PCA16Direct_smooth5": base["pca16_s5"],
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
        "method": "No-video-prior physiological dimwise fusion batch",
        "iteration_range": "336-355",
        "note": (
            "No candidate uses VideoTimeMean, PatternPrior, video ID, timestamp, or label-derived "
            "video-time cells. This batch tests whether direct EEG/fNIRS signal should correct "
            "valence only while arousal remains conservative."
        ),
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_base_predictions(
    pre: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
) -> dict[str, np.ndarray]:
    alpha = 10000.0
    bases = {
        "pca8": pca_ridge_predict(pre["early_concat"][train_idx], y_train, pre["early_concat"][val_idx], 8, alpha),
        "pca16": pca_ridge_predict(pre["early_concat"][train_idx], y_train, pre["early_concat"][val_idx], 16, alpha),
        "pls2": pls_predict(pre["early_concat"][train_idx], y_train, pre["early_concat"][val_idx], 2),
        "pls4": pls_predict(pre["early_concat"][train_idx], y_train, pre["early_concat"][val_idx], 4),
        "neuro": ridge_predict(pre["neurovascular"][train_idx], y_train, pre["neurovascular"][val_idx], alpha),
        "coupled": ridge_predict(pre["coupled_slope"][train_idx], y_train, pre["coupled_slope"][val_idx], alpha),
        "fnirs": ridge_predict(pre["fnirs_slow"][train_idx], y_train, pre["fnirs_slow"][val_idx], alpha),
        "eeg": ridge_predict(pre["eeg_lag"][train_idx], y_train, pre["eeg_lag"][val_idx], alpha),
        "hrf": ridge_predict(
            np.concatenate([pre["eeg_hrf"], pre["fnirs_core"], pre["neurovascular"]], axis=1)[train_idx],
            y_train,
            np.concatenate([pre["eeg_hrf"], pre["fnirs_core"], pre["neurovascular"]], axis=1)[val_idx],
            alpha,
        ),
        "early": ridge_predict(pre["early_concat"][train_idx], y_train, pre["early_concat"][val_idx], alpha),
    }
    for name, pred in list(bases.items()):
        bases[f"{name}_s5"] = smooth_predictions(val_ids, pred, 5).astype(np.float32)
    return bases


def build_dimwise_candidates(
    center: np.ndarray,
    train_mean: np.ndarray,
    base: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    source_order = [
        ("336_PCA8Valence_CenterArousal", "pca8_s5"),
        ("337_PCA16Valence_CenterArousal", "pca16_s5"),
        ("338_PLS2Valence_CenterArousal", "pls2_s5"),
        ("339_PLS4Valence_CenterArousal", "pls4_s5"),
        ("340_NeurovascularValence_CenterArousal", "neuro_s5"),
        ("341_CoupledSlopeValence_CenterArousal", "coupled_s5"),
        ("342_FNIRSValence_CenterArousal", "fnirs_s5"),
        ("343_EEGValence_CenterArousal", "eeg_s5"),
        ("344_HRFValence_CenterArousal", "hrf_s5"),
        ("345_EarlyConcatValence_CenterArousal", "early_s5"),
    ]
    for method, key in source_order:
        candidates[method] = dimwise(base[key], center, use_valence=True, use_arousal=False)

    candidates["346_PCA8Valence_TrainMeanArousal"] = dimwise(
        base["pca8_s5"], train_mean, use_valence=True, use_arousal=False
    )
    candidates["347_PCA16Valence_TrainMeanArousal"] = dimwise(
        base["pca16_s5"], train_mean, use_valence=True, use_arousal=False
    )
    candidates["348_PCA16ValenceShrink50_CenterArousal"] = shrink_valence(center, base["pca16_s5"], 0.50)
    candidates["349_PCA16ValenceShrink75_CenterArousal"] = shrink_valence(center, base["pca16_s5"], 0.75)
    candidates["350_PCA8ValenceShrink50_CenterArousal"] = shrink_valence(center, base["pca8_s5"], 0.50)
    candidates["351_PCA8ValenceShrink75_CenterArousal"] = shrink_valence(center, base["pca8_s5"], 0.75)

    ensemble_v = (base["pca8_s5"][:, 0] + base["pca16_s5"][:, 0] + base["pls2_s5"][:, 0]) / 3.0
    pred = center.copy()
    pred[:, 0] = ensemble_v
    candidates["352_LowRankEnsembleValence_CenterArousal"] = pred

    ensemble_v2 = (base["pca16_s5"][:, 0] + base["pls2_s5"][:, 0] + base["neuro_s5"][:, 0]) / 3.0
    pred = center.copy()
    pred[:, 0] = ensemble_v2
    candidates["353_NeuroLowRankEnsembleValence_CenterArousal"] = pred

    disagreement = np.abs(base["pca8_s5"][:, 0] - base["pca16_s5"][:, 0])
    pred = center.copy()
    pred[:, 0] = np.where(disagreement < 8.0, base["pca16_s5"][:, 0], center[:, 0])
    candidates["354_AgreementGatedPCA16Valence_CenterArousal"] = pred

    pred = center.copy()
    signal_v = ensemble_v - center[:, 0]
    gate = np.where(disagreement < 8.0, 0.75, 0.35)
    pred[:, 0] = center[:, 0] + gate * signal_v
    candidates["355_AgreementWeightedEnsembleValence_CenterArousal"] = pred

    return candidates


def dimwise(source: np.ndarray, fallback: np.ndarray, use_valence: bool, use_arousal: bool) -> np.ndarray:
    out = fallback.copy()
    if use_valence:
        out[:, 0] = source[:, 0]
    if use_arousal:
        out[:, 1] = source[:, 1]
    return out


def shrink_valence(center: np.ndarray, source: np.ndarray, scale: float) -> np.ndarray:
    out = center.copy()
    out[:, 0] = center[:, 0] + scale * (source[:, 0] - center[:, 0])
    return out


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(sanitize(x_train), y_train)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def pca_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    components: int,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=components, svd_solver="randomized", random_state=2026),
        Ridge(alpha=alpha, solver="lsqr"),
    )
    model.fit(sanitize(x_train), y_train)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def pls_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, components: int) -> np.ndarray:
    model = make_pipeline(StandardScaler(), PLSRegression(n_components=components, scale=False))
    model.fit(sanitize(x_train), y_train)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def add_metric(
    metric_acc: dict[str, dict[str, object]],
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    notes: str,
) -> None:
    payload = metric_acc.setdefault(
        name,
        {
            "abs_sum": np.zeros(2, dtype=np.float64),
            "sq_sum": np.zeros(2, dtype=np.float64),
            "count": 0,
            "notes": notes,
        },
    )
    diff = y_pred - y_true
    payload["abs_sum"] += np.abs(diff).sum(axis=0)
    payload["sq_sum"] += (diff**2).sum(axis=0)
    payload["count"] += int(y_true.shape[0])


def finalize_metric(name: str, payload: dict[str, object]) -> dict[str, object]:
    count = max(int(payload["count"]), 1)
    abs_sum = np.asarray(payload["abs_sum"], dtype=np.float64)
    sq_sum = np.asarray(payload["sq_sum"], dtype=np.float64)
    return {
        "method": name,
        "overall_mae": round(float(abs_sum.sum() / (2 * count)), 4),
        "valence_mae": round(float(abs_sum[0] / count), 4),
        "arousal_mae": round(float(abs_sum[1] / count), 4),
        "overall_mse": round(float(sq_sum.sum() / (2 * count)), 4),
        "notes": str(payload["notes"]),
    }


def candidate_note(name: str) -> str:
    if name.startswith("Reference"):
        return "Reference from previous no-video-prior physiological batch."
    if "CenterArousal" in name:
        return "No-prior physiological dimwise output: use EEG/fNIRS signal for valence, keep arousal at center."
    if "TrainMeanArousal" in name:
        return "No-prior physiological dimwise output: use EEG/fNIRS signal for valence, train global mean for arousal."
    if "Shrink" in name:
        return "Shrink signal valence toward center to reduce cross-subject overfit; arousal stays conservative."
    if "Ensemble" in name:
        return "Ensemble low-rank physiological valence experts; arousal stays conservative."
    if "Agreement" in name:
        return "Gate physiological valence correction by agreement between low-rank experts."
    return "No-video-prior physiological candidate."


if __name__ == "__main__":
    main()
