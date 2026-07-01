from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from tools.cross_fold_signal_residual_over_pattern_prior import (  # noqa: E402
    make_leave_subject_out_train_prior,
    make_pattern_prior,
)
from tools.run_iteration_experiments import expand_subjects, load_labels, smooth_predictions  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fold EEG-fNIRS neurovascular fusion modules."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_229_239_neurovascular_fusion.json")
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_baseline.npz")
    parser.add_argument("--alphas", default="100,1000,10000,100000")
    parser.add_argument("--scales", default="0.01,0.03,0.05,0.08,0.12,0.20")
    parser.add_argument("--clips", default="0.5,1,2,4")
    parser.add_argument("--smooth-windows", default="0,5")
    parser.add_argument("--pls-components", default="2,4,6")
    parser.add_argument("--pca-components", default="4,8,12")
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)

    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(data_root, subjects, Path(args.precompute_cache))
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    alphas = parse_floats(args.alphas)
    scales = parse_floats(args.scales)
    clips = parse_floats(args.clips)
    smooth_windows = parse_ints(args.smooth_windows)
    pls_components = parse_ints(args.pls_components)
    pca_components = parse_ints(args.pca_components)

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] train={len(train_subjects)} val={val_subjects}", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        prior_train = make_leave_subject_out_train_prior(labels, train_subjects, train_ids)
        prior_val = make_pattern_prior(train_ids, y_train, val_ids)
        residual_train = (y_train - prior_train).astype(np.float32)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)

        fold_results = []
        fold_results.append(evaluate_candidate(metric_acc, "098_PatternPrior_reference", y_val, prior_val, "Strong non-signal prior."))

        module_residuals = predict_all_modules(
            pre=pre,
            train_idx=train_idx,
            val_idx=val_idx,
            residual_train=residual_train,
            alphas=alphas,
            pls_components=pls_components,
            pca_components=pca_components,
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
        "method": "EEG-fNIRS neurovascular fusion module search",
        "note": (
            "Each module predicts a small residual over PatternPrior_098 using separated EEG and fNIRS "
            "features. The modules differ in fusion logic: single-modality, lag coupling, HRF kernel, "
            "low-rank bilinear, PLS latent fusion, and agreement/confidence gates."
        ),
        "feature_shapes": {
            "eeg": feature_shapes["eeg"],
            "fnirs": feature_shapes["fnirs"],
            "eeg_core": list(pre["eeg_core"].shape),
            "fnirs_core": list(pre["fnirs_core"].shape),
        },
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def load_or_build_precomputed(
    data_root: Path,
    subjects: list[str],
    cache_path: Path,
    fnirs_types: tuple[int, ...] | None = None,
    feature_normalization: str = "none",
    baseline_correction: bool = True,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[int]]]:
    if cache_path.exists():
        print(f"[features] loading neurovascular precompute cache: {cache_path}", flush=True)
        with np.load(cache_path, allow_pickle=False) as data:
            sample_ids = data["sample_ids"].astype(str).tolist()
            feature_shapes = json.loads(str(data["feature_shapes"]))
            pre = {
                key: data[key].astype(np.float32)
                for key in data.files
                if key not in {"sample_ids", "feature_shapes"}
            }
        return sample_ids, pre, feature_shapes

    print("[features] loading separated EEG/fNIRS features", flush=True)
    selected_fnirs_types = tuple(DEFAULT_FNIRS_TYPES if fnirs_types is None else fnirs_types)
    eeg, fnirs, _, _, _, sample_ids_array = load_training_features(
        data_root,
        subjects=subjects,
        include_sample_ids=True,
        fnirs_types=selected_fnirs_types,
        baseline_correction=baseline_correction,
        verbose=True,
    )
    sample_ids = sample_ids_array.astype(str).tolist()
    eeg, fnirs = normalize_raw_feature_sequences(sample_ids, eeg, fnirs, feature_normalization)
    pre = build_precomputed_features(eeg, fnirs, sample_ids)
    feature_shapes = {
        "eeg": list(eeg.shape),
        "fnirs": list(fnirs.shape),
        "fnirs_types": list(selected_fnirs_types),
        "feature_normalization": feature_normalization,
        "baseline_correction": bool(baseline_correction),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        feature_shapes=json.dumps(feature_shapes, ensure_ascii=False),
        **{key: value.astype(np.float32) for key, value in pre.items()},
    )
    return sample_ids, pre, feature_shapes


def normalize_raw_feature_sequences(
    sample_ids: list[str],
    eeg: np.ndarray,
    fnirs: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = mode.strip().lower()
    if mode in {"", "none"}:
        return eeg.astype(np.float32), fnirs.astype(np.float32)
    if mode not in {"trial_zscore", "subject_zscore"}:
        raise ValueError(f"Unsupported feature normalization: {mode}")
    groups: dict[object, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, _ = parse_sample_id(sample_id)
        key: object = subject if mode == "subject_zscore" else (subject, video)
        groups[key].append(index)
    eeg_out = eeg.astype(np.float32).copy()
    fnirs_out = fnirs.astype(np.float32).copy()
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        eeg_seq = eeg_out[idx]
        fnirs_seq = fnirs_out[idx]
        eeg_out[idx] = (eeg_seq - eeg_seq.mean(axis=0, keepdims=True)) / np.maximum(
            eeg_seq.std(axis=0, keepdims=True), 1e-4
        )
        fnirs_out[idx] = (fnirs_seq - fnirs_seq.mean(axis=0, keepdims=True)) / np.maximum(
            fnirs_seq.std(axis=0, keepdims=True), 1e-4
        )
    return sanitize(eeg_out), sanitize(fnirs_out)


def build_precomputed_features(eeg: np.ndarray, fnirs: np.ndarray, sample_ids: list[str]) -> dict[str, np.ndarray]:
    eeg_core = eeg_core_features(eeg)
    fnirs_core = fnirs_core_features(fnirs)
    eeg_scalar = eeg_scalar_features(eeg_core)
    fnirs_scalar = fnirs_scalar_features(fnirs_core)
    eeg_lag = lagged_features(sample_ids, eeg_core, lags=(0, 1, 2, 4), include_delta=True)
    fnirs_slow = np.concatenate(
        [
            fnirs_core,
            rolling_past_features(sample_ids, fnirs_core, windows=(3, 5, 9)),
            lagged_features(sample_ids, fnirs_core, lags=(1, 3, 5), include_delta=True),
        ],
        axis=1,
    )
    eeg_hrf = hrf_kernel_features(sample_ids, eeg_core, lags=(0, 1, 2, 3, 4, 6, 8, 10))
    eeg_hrf_scalar = hrf_kernel_features(sample_ids, eeg_scalar, lags=(0, 1, 2, 3, 4, 6, 8, 10))
    fnirs_roll_scalar = rolling_past_features(sample_ids, fnirs_scalar, windows=(3, 5, 9))
    neurovascular = neurovascular_product_features(eeg_hrf_scalar, fnirs_scalar, fnirs_roll_scalar)
    coupled_slope = coupled_slope_features(sample_ids, eeg_scalar, fnirs_scalar)
    coherence = rolling_coherence_confidence(sample_ids, eeg_hrf_scalar[:, 0], fnirs_scalar[:, 0])
    return {
        "eeg_core": eeg_core,
        "fnirs_core": fnirs_core,
        "eeg_scalar": eeg_scalar,
        "fnirs_scalar": fnirs_scalar,
        "eeg_lag": eeg_lag,
        "fnirs_slow": fnirs_slow,
        "early_concat": np.concatenate([eeg_lag, fnirs_slow], axis=1).astype(np.float32),
        "eeg_hrf": eeg_hrf,
        "eeg_hrf_scalar": eeg_hrf_scalar,
        "fnirs_roll_scalar": fnirs_roll_scalar,
        "neurovascular": neurovascular,
        "coupled_slope": coupled_slope,
        "coherence": coherence.astype(np.float32),
    }


def eeg_core_features(eeg: np.ndarray) -> np.ndarray:
    mean_band = eeg.mean(axis=1)
    std_band = eeg.std(axis=1)
    half = eeg.shape[1] // 2
    spatial_diff = eeg[:, :half].mean(axis=1) - eeg[:, half:].mean(axis=1)
    beta_alpha = mean_band[:, 3:4] - mean_band[:, 2:3]
    gamma_alpha = mean_band[:, 4:5] - mean_band[:, 2:3]
    out = np.concatenate([mean_band, std_band, spatial_diff, beta_alpha, gamma_alpha], axis=1)
    return sanitize(out)


def fnirs_core_features(fnirs: np.ndarray) -> np.ndarray:
    mean_feat = fnirs.mean(axis=1)
    std_feat = fnirs.std(axis=1)
    half = fnirs.shape[1] // 2
    spatial_diff = fnirs[:, :half].mean(axis=1) - fnirs[:, half:].mean(axis=1)
    hbo_hbr = mean_feat[:, 0:1] - mean_feat[:, 1:2]
    hbt_slope = mean_feat[:, 8:9] if mean_feat.shape[1] > 8 else np.zeros_like(hbo_hbr)
    out = np.concatenate([mean_feat, std_feat, spatial_diff, hbo_hbr, hbt_slope], axis=1)
    return sanitize(out)


def eeg_scalar_features(eeg_core: np.ndarray) -> np.ndarray:
    delta = eeg_core[:, 0:1]
    theta = eeg_core[:, 1:2]
    alpha = eeg_core[:, 2:3]
    beta = eeg_core[:, 3:4]
    gamma = eeg_core[:, 4:5]
    activation = beta + gamma - alpha - theta
    spatial_beta = eeg_core[:, 13:14]
    return sanitize(np.concatenate([delta, theta, alpha, beta, gamma, activation, spatial_beta], axis=1))


def fnirs_scalar_features(fnirs_core: np.ndarray) -> np.ndarray:
    hbo = fnirs_core[:, 0:1]
    hbr = fnirs_core[:, 1:2]
    hbt = fnirs_core[:, 2:3]
    hbo_slope = fnirs_core[:, 6:7]
    hbr_slope = fnirs_core[:, 7:8]
    hbt_slope = fnirs_core[:, 8:9]
    oxygenation = hbo - hbr
    return sanitize(np.concatenate([hbo, hbr, hbt, hbo_slope, hbr_slope, hbt_slope, oxygenation], axis=1))


def lagged_features(
    sample_ids: list[str],
    values: np.ndarray,
    lags: tuple[int, ...],
    include_delta: bool,
) -> np.ndarray:
    groups = group_indices(sample_ids)
    lagged = np.zeros((len(sample_ids), values.shape[1] * len(lags)), dtype=np.float32)
    delta_lags = [lag for lag in lags if lag > 0]
    delta_out = np.zeros((len(sample_ids), values.shape[1] * len(delta_lags)), dtype=np.float32)
    for items in groups.values():
        items = sorted(items)
        indices = [index for _, index in items]
        index_array = np.asarray(indices, dtype=np.int64)
        seq = values[indices]
        for lag_pos, lag in enumerate(lags):
            shifted = shift_past(seq, lag)
            lagged[index_array, lag_pos * values.shape[1] : (lag_pos + 1) * values.shape[1]] = shifted
            if include_delta and lag > 0:
                delta_pos = delta_lags.index(lag)
                start = delta_pos * values.shape[1]
                stop = start + values.shape[1]
                delta_out[index_array, start:stop] = seq - shifted
    if include_delta and delta_lags:
        return sanitize(np.concatenate([lagged, delta_out], axis=1))
    return sanitize(lagged)


def rolling_past_features(sample_ids: list[str], values: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    out = np.zeros((len(sample_ids), values.shape[1] * len(windows)), dtype=np.float32)
    for items in group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        cumsum = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(seq, axis=0)], axis=0)
        for win_pos, window in enumerate(windows):
            rolled = np.zeros_like(seq)
            for local_index in range(seq.shape[0]):
                start = max(0, local_index - window + 1)
                stop = local_index + 1
                rolled[local_index] = (cumsum[stop] - cumsum[start]) / float(stop - start)
            out[np.asarray(indices), win_pos * values.shape[1] : (win_pos + 1) * values.shape[1]] = rolled
    return sanitize(out)


def hrf_kernel_features(sample_ids: list[str], values: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    lag_values = np.asarray(lags, dtype=np.float32) + 1.0
    kernel = (lag_values**2) * np.exp(-lag_values / 3.0)
    kernel = kernel / np.maximum(kernel.sum(), 1e-6)
    out = np.zeros_like(values, dtype=np.float32)
    for items in group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        acc = np.zeros_like(seq)
        for weight, lag in zip(kernel, lags):
            acc += float(weight) * shift_past(seq, int(lag))
        out[np.asarray(indices)] = acc
    return sanitize(out)


def neurovascular_product_features(eeg_hrf: np.ndarray, fnirs_now: np.ndarray, fnirs_roll: np.ndarray) -> np.ndarray:
    common = min(eeg_hrf.shape[1], fnirs_now.shape[1])
    e = eeg_hrf[:, :common]
    f = fnirs_now[:, :common]
    product = e * f
    diff = e - f
    abs_diff = np.abs(diff)
    return sanitize(np.concatenate([eeg_hrf, fnirs_now, fnirs_roll, product, diff, abs_diff], axis=1))


def coupled_slope_features(sample_ids: list[str], eeg_scalar: np.ndarray, fnirs_scalar: np.ndarray) -> np.ndarray:
    eeg_delta = lagged_features(sample_ids, eeg_scalar, lags=(1, 2, 4), include_delta=True)
    fnirs_delta = lagged_features(sample_ids, fnirs_scalar, lags=(1, 3, 5), include_delta=True)
    common = min(eeg_delta.shape[1], fnirs_delta.shape[1])
    interaction = eeg_delta[:, :common] * fnirs_delta[:, :common]
    return sanitize(np.concatenate([eeg_delta, fnirs_delta, interaction], axis=1))


def rolling_coherence_confidence(sample_ids: list[str], x: np.ndarray, y: np.ndarray, window: int = 9) -> np.ndarray:
    confidence = np.full((len(sample_ids), 1), 0.5, dtype=np.float32)
    for items in group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        x_seq = x[indices].astype(np.float32)
        y_seq = y[indices].astype(np.float32)
        for local_index, global_index in enumerate(indices):
            start = max(0, local_index - window + 1)
            xs = x_seq[start : local_index + 1]
            ys = y_seq[start : local_index + 1]
            if xs.shape[0] < 3 or float(xs.std()) < 1e-6 or float(ys.std()) < 1e-6:
                confidence[global_index, 0] = 0.5
            else:
                corr = float(np.corrcoef(xs.ravel(), ys.ravel())[0, 1])
                confidence[global_index, 0] = np.clip(0.5 + 0.5 * corr, 0.0, 1.0)
    return confidence


def group_indices(sample_ids: list[str]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    return groups


def shift_past(seq: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return seq.copy()
    shifted = np.empty_like(seq)
    shifted[:lag] = seq[:1]
    shifted[lag:] = seq[:-lag]
    return shifted


def predict_all_modules(
    pre: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    residual_train: np.ndarray,
    alphas: list[float],
    pls_components: list[int],
    pca_components: list[int],
) -> dict[str, np.ndarray]:
    residuals: dict[str, np.ndarray] = {}
    ridge_modules = {
        "229_EEGLagRidge": pre["eeg_lag"],
        "230_FNIRSSlowRidge": pre["fnirs_slow"],
        "231_EarlyConcatLagRidge": pre["early_concat"],
        "232_NeurovascularLagProductRidge": pre["neurovascular"],
        "233_HRFKernelFusionRidge": np.concatenate([pre["eeg_hrf"], pre["fnirs_core"], pre["neurovascular"]], axis=1),
        "234_CoupledSlopeFusionRidge": pre["coupled_slope"],
    }
    for module_name, matrix in ridge_modules.items():
        for alpha in alphas:
            residuals[f"{module_name}_a{format_float(alpha)}"] = ridge_predict(
                matrix[train_idx],
                residual_train,
                matrix[val_idx],
                alpha,
            )

    residuals.update(
        predict_low_rank_pca_products(
            pre=pre,
            train_idx=train_idx,
            val_idx=val_idx,
            residual_train=residual_train,
            alphas=alphas,
            components=pca_components,
        )
    )
    residuals.update(
        predict_pls_latent(
            pre["early_concat"][train_idx],
            residual_train,
            pre["early_concat"][val_idx],
            pls_components,
        )
    )
    residuals.update(
        predict_dual_agreement(
            pre=pre,
            train_idx=train_idx,
            val_idx=val_idx,
            residual_train=residual_train,
            alphas=alphas,
        )
    )
    confidence = pre["coherence"][val_idx]
    for alpha in alphas:
        base = ridge_predict(pre["early_concat"][train_idx], residual_train, pre["early_concat"][val_idx], alpha)
        residuals[f"238_CoherenceConfidenceGate_a{format_float(alpha)}"] = base * (0.25 + 0.75 * confidence)
    return residuals


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, alpha: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(sanitize(x_train), y_train)
    return model.predict(sanitize(x_val)).astype(np.float32)


def predict_low_rank_pca_products(
    pre: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    residual_train: np.ndarray,
    alphas: list[float],
    components: list[int],
) -> dict[str, np.ndarray]:
    out = {}
    eeg_train = pre["eeg_lag"][train_idx]
    eeg_val = pre["eeg_lag"][val_idx]
    fnirs_train = pre["fnirs_slow"][train_idx]
    fnirs_val = pre["fnirs_slow"][val_idx]
    for comp in components:
        eeg_scaler = StandardScaler()
        fnirs_scaler = StandardScaler()
        eeg_pca = PCA(n_components=comp, random_state=17)
        fnirs_pca = PCA(n_components=comp, random_state=23)
        e_train = eeg_pca.fit_transform(eeg_scaler.fit_transform(sanitize(eeg_train)))
        f_train = fnirs_pca.fit_transform(fnirs_scaler.fit_transform(sanitize(fnirs_train)))
        e_val = eeg_pca.transform(eeg_scaler.transform(sanitize(eeg_val)))
        f_val = fnirs_pca.transform(fnirs_scaler.transform(sanitize(fnirs_val)))
        x_train = np.concatenate([e_train, f_train, e_train * f_train, np.abs(e_train - f_train)], axis=1)
        x_val = np.concatenate([e_val, f_val, e_val * f_val, np.abs(e_val - f_val)], axis=1)
        for alpha in alphas:
            out[f"235_LowRankBilinearPCA_c{comp}_a{format_float(alpha)}"] = ridge_predict(
                x_train,
                residual_train,
                x_val,
                alpha,
            )
    return out


def predict_pls_latent(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    x_val: np.ndarray,
    components: list[int],
) -> dict[str, np.ndarray]:
    out = {}
    x_train = sanitize(x_train)
    x_val = sanitize(x_val)
    max_comp = min(x_train.shape[1], x_train.shape[0] - 1)
    for comp in components:
        n_comp = min(comp, max_comp)
        model = PLSRegression(n_components=n_comp, scale=True)
        model.fit(x_train, residual_train)
        out[f"236_PLSJointLatent_c{n_comp}"] = model.predict(x_val).astype(np.float32)
    return out


def predict_dual_agreement(
    pre: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    residual_train: np.ndarray,
    alphas: list[float],
) -> dict[str, np.ndarray]:
    out = {}
    for alpha in alphas:
        eeg_train = pre["eeg_lag"][train_idx]
        eeg_val = pre["eeg_lag"][val_idx]
        fnirs_train = pre["fnirs_slow"][train_idx]
        fnirs_val = pre["fnirs_slow"][val_idx]
        eeg_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        fnirs_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        eeg_model.fit(sanitize(eeg_train), residual_train)
        fnirs_model.fit(sanitize(fnirs_train), residual_train)
        eeg_train_pred = eeg_model.predict(sanitize(eeg_train)).astype(np.float32)
        fnirs_train_pred = fnirs_model.predict(sanitize(fnirs_train)).astype(np.float32)
        eeg_pred = eeg_model.predict(sanitize(eeg_val)).astype(np.float32)
        fnirs_pred = fnirs_model.predict(sanitize(fnirs_val)).astype(np.float32)
        eeg_mse = ((eeg_train_pred - residual_train) ** 2).mean(axis=0) + 1e-6
        fnirs_mse = ((fnirs_train_pred - residual_train) ** 2).mean(axis=0) + 1e-6
        eeg_weight = (1.0 / eeg_mse) / (1.0 / eeg_mse + 1.0 / fnirs_mse)
        combined = eeg_weight[None, :] * eeg_pred + (1.0 - eeg_weight[None, :]) * fnirs_pred
        agree = np.sign(eeg_pred) == np.sign(fnirs_pred)
        gated = np.where(agree, combined, 0.25 * combined)
        out[f"237_DualModalityAgreementGate_a{format_float(alpha)}"] = gated.astype(np.float32)
        out[f"239_ModalityVarianceWeighted_a{format_float(alpha)}"] = combined.astype(np.float32)
    return out


def evaluate_residual_grid(
    metric_acc: dict[str, dict[str, object]],
    base_name: str,
    y_val: np.ndarray,
    prior_val: np.ndarray,
    val_ids: list[str],
    residual_val: np.ndarray,
    scales: list[float],
    clips: list[float],
    smooth_windows: list[int],
) -> list[dict[str, object]]:
    results = []
    for mode in ("v", "a", "both"):
        for scale in scales:
            for clip_value in clips:
                correction = np.zeros_like(prior_val, dtype=np.float32)
                if mode in ("v", "both"):
                    correction[:, 0] = np.clip(scale * residual_val[:, 0], -clip_value, clip_value)
                if mode in ("a", "both"):
                    correction[:, 1] = np.clip(scale * residual_val[:, 1], -clip_value, clip_value)
                pred = np.clip(prior_val + correction, 1.0, 255.0).astype(np.float32)
                raw_name = f"{base_name}_{mode}_s{format_float(scale)}_c{format_float(clip_value)}"
                results.append(
                    evaluate_candidate(
                        metric_acc,
                        raw_name,
                        y_val,
                        pred,
                        "PatternPrior_098 plus EEG-fNIRS residual correction.",
                    )
                )
                for window in smooth_windows:
                    if window <= 1:
                        continue
                    smooth = smooth_predictions(val_ids, pred, window=window).astype(np.float32)
                    results.append(
                        evaluate_candidate(
                            metric_acc,
                            f"{raw_name}_smooth{window}",
                            y_val,
                            smooth,
                            "Smoothed EEG-fNIRS residual correction.",
                        )
                    )
    return results


def evaluate_candidate(
    metric_acc: dict[str, dict[str, object]],
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    notes: str,
) -> dict[str, object]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    payload = metric_acc.setdefault(
        name,
        {
            "sum_abs": np.zeros(2, dtype=np.float64),
            "sum_sq": 0.0,
            "count": 0,
            "notes": notes,
        },
    )
    payload["sum_abs"] = np.asarray(payload["sum_abs"]) + abs_err.sum(axis=0)
    payload["sum_sq"] = float(payload["sum_sq"]) + float((err**2).sum())
    payload["count"] = int(payload["count"]) + int(y_true.shape[0])
    return {
        "method": name,
        "overall_mae": round(float(abs_err.mean()), 4),
        "valence_mae": round(float(abs_err[:, 0].mean()), 4),
        "arousal_mae": round(float(abs_err[:, 1].mean()), 4),
        "overall_mse": round(float((err**2).mean()), 4),
        "notes": notes,
    }


def finalize_metric(name: str, payload: dict[str, object]) -> dict[str, object]:
    count = int(payload["count"])
    sum_abs = np.asarray(payload["sum_abs"], dtype=np.float64)
    val_mae = float(sum_abs[0] / count)
    aro_mae = float(sum_abs[1] / count)
    return {
        "method": name,
        "overall_mae": round(float((val_mae + aro_mae) / 2.0), 4),
        "valence_mae": round(val_mae, 4),
        "arousal_mae": round(aro_mae, 4),
        "overall_mse": round(float(payload["sum_sq"]) / float(count * 2), 4),
        "notes": str(payload["notes"]),
    }


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
