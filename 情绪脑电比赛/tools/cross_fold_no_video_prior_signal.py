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
from emotion_merps.features import DEFAULT_FNIRS_TYPES, load_training_features  # noqa: E402
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subject-disjoint no-video-prior EEG/fNIRS signal-only baselines."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_321_335_no_video_prior_signal.json")
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--fnirs-types", default="0,1,2,3,4,5")
    parser.add_argument("--baseline-correction", default="true", choices=["true", "false"])
    parser.add_argument("--alphas", default="100,1000,10000")
    parser.add_argument("--smooth-windows", default="0,5")
    parser.add_argument("--pca-components", default="8,16")
    parser.add_argument("--pls-components", default="2,4")
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    fnirs_types = tuple(parse_ints(args.fnirs_types)) or DEFAULT_FNIRS_TYPES
    baseline_correction = args.baseline_correction == "true"
    labels = load_labels(data_root, subjects)
    alphas = parse_floats(args.alphas)
    smooth_windows = parse_ints(args.smooth_windows)
    pca_components = parse_ints(args.pca_components)
    pls_components = parse_ints(args.pls_components)

    print("[features] loading raw EEG/fNIRS label-aligned features", flush=True)
    eeg, fnirs, _, sample_subjects, subject_names, sample_ids_array = load_training_features(
        data_root,
        subjects=subjects,
        fnirs_types=fnirs_types,
        baseline_correction=baseline_correction,
        include_sample_ids=True,
        verbose=True,
    )
    sample_ids = sample_ids_array.astype(str).tolist()
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    print("[features] loading compact neurovascular features", flush=True)
    compact_ids, pre, compact_shapes = load_or_build_precomputed(
        data_root=data_root,
        subjects=subjects,
        cache_path=Path(args.precompute_cache),
        fnirs_types=fnirs_types,
        feature_normalization="none",
        baseline_correction=baseline_correction,
    )
    compact_index = {sample_id: index for index, sample_id in enumerate(compact_ids)}

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
        x_train_raw, x_val_raw = fold_standardized_flatten(eeg, fnirs, train_idx, val_idx)

        compact_train_idx = np.asarray([compact_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        compact_val_idx = np.asarray([compact_index[sample_id] for sample_id in val_ids], dtype=np.int64)
        views = {
            "325_EEGLagDirectRidge": pre["eeg_lag"],
            "326_FNIRSSlowDirectRidge": pre["fnirs_slow"],
            "327_EarlyConcatDirectRidge": pre["early_concat"],
            "328_NeurovascularDirectRidge": pre["neurovascular"],
            "329_HRFDirectRidge": np.concatenate(
                [pre["eeg_hrf"], pre["fnirs_core"], pre["neurovascular"]], axis=1
            ),
            "330_CoupledSlopeDirectRidge": pre["coupled_slope"],
        }

        fold_predictions: dict[str, np.ndarray] = {
            "321_Center128_noPrior": np.full_like(y_val, 128.0),
            "322_TrainMean_noPrior": np.tile(y_train.mean(axis=0, keepdims=True), (len(val_ids), 1)),
        }

        for alpha in alphas:
            name = f"323_RawFlatDirectRidge_a{format_float(alpha)}"
            fold_predictions[name] = ridge_predict(x_train_raw, y_train, x_val_raw, alpha)
            for view_name, matrix in views.items():
                fold_predictions[f"{view_name}_a{format_float(alpha)}"] = ridge_predict(
                    matrix[compact_train_idx],
                    y_train,
                    matrix[compact_val_idx],
                    alpha,
                )

        for comp in pca_components:
            for alpha in alphas:
                name = f"333_PCAEarlyDirectRidge_c{comp}_a{format_float(alpha)}"
                fold_predictions[name] = pca_ridge_predict(
                    pre["early_concat"][compact_train_idx],
                    y_train,
                    pre["early_concat"][compact_val_idx],
                    comp,
                    alpha,
                )

        for comp in pls_components:
            name = f"334_PLSEarlyDirect_c{comp}"
            fold_predictions[name] = pls_predict(
                pre["early_concat"][compact_train_idx],
                y_train,
                pre["early_concat"][compact_val_idx],
                comp,
            )

        for name, pred in list(fold_predictions.items()):
            for window in smooth_windows:
                if window <= 1:
                    continue
                fold_predictions[f"{name}_SignalSmooth{window}"] = smooth_predictions(val_ids, pred, window)

        fold_results = []
        for name, pred in fold_predictions.items():
            note = no_prior_note(name)
            result = score(name, y_val, np.clip(pred, 1.0, 255.0), note)
            fold_results.append(result)
            add_metric(metric_acc, name, y_val, np.clip(pred, 1.0, 255.0), note)

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
        "method": "No-video-prior signal-only EEG/fNIRS evaluation",
        "note": (
            "No candidate uses VideoTimeMean, PatternPrior, video ID, timestamp, or label-derived "
            "video/time cells. SignalSmooth variants use only the model output order within each trial."
        ),
        "subjects": subjects,
        "fold_size": args.fold_size,
        "fnirs_types": list(fnirs_types),
        "baseline_correction": baseline_correction,
        "raw_feature_shapes": {
            "eeg": list(eeg.shape),
            "fnirs": list(fnirs.shape),
            "subjects": subject_names,
        },
        "compact_feature_shapes": compact_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def fold_standardized_flatten(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eeg_mean = eeg[train_idx].mean(axis=(0, 1), keepdims=True)
    eeg_std = np.maximum(eeg[train_idx].std(axis=(0, 1), keepdims=True), 1e-6)
    fnirs_mean = fnirs[train_idx].mean(axis=(0, 1), keepdims=True)
    fnirs_std = np.maximum(fnirs[train_idx].std(axis=(0, 1), keepdims=True), 1e-6)
    eeg_train = ((eeg[train_idx] - eeg_mean) / eeg_std).reshape(len(train_idx), -1)
    eeg_val = ((eeg[val_idx] - eeg_mean) / eeg_std).reshape(len(val_idx), -1)
    fnirs_train = ((fnirs[train_idx] - fnirs_mean) / fnirs_std).reshape(len(train_idx), -1)
    fnirs_val = ((fnirs[val_idx] - fnirs_mean) / fnirs_std).reshape(len(val_idx), -1)
    return (
        sanitize(np.concatenate([eeg_train, fnirs_train], axis=1).astype(np.float32)),
        sanitize(np.concatenate([eeg_val, fnirs_val], axis=1).astype(np.float32)),
    )


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


def no_prior_note(name: str) -> str:
    if "Center128" in name:
        return "No signal and no video/time prior; constant VA-plane center."
    if "TrainMean" in name:
        return "No signal and no video/time prior; train-subject global mean."
    if "RawFlat" in name:
        return "Direct Ridge on flattened EEG bandpower and fNIRS statistics; no video/time prior."
    if "PCA" in name:
        return "Low-rank direct signal model on EEG/fNIRS compact features; no video/time prior."
    if "PLS" in name:
        return "PLS latent direct signal model on EEG/fNIRS compact features; no video/time prior."
    if "EEGLag" in name:
        return "EEG-only lagged signal features; no label-derived video/time prior."
    if "FNIRSSlow" in name:
        return "fNIRS-only slow/lagged signal features; no label-derived video/time prior."
    if "Neurovascular" in name or "HRF" in name or "Coupled" in name or "EarlyConcat" in name:
        return "EEG-fNIRS signal-derived fusion features; no label-derived video/time prior."
    return "No-video-prior signal-only candidate."


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
