from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array, parse_floats  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    build_oof_training_set,
    make_candidates,
    make_feature_matrix,
    make_pattern_098,
    parse_strings,
    prior_slope_by_trial,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch 105-124: twenty new low-capacity models.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_105_124_batch20_new_models.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
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
        print(f"[fold {fold_index}] building strict OOF features", flush=True)
        x_train, y_train, prior_train, train_rows = build_oof_training_set(
            labels=labels,
            train_subjects=train_subjects,
            candidate_pool=candidate_pool,
            args=args,
        )
        residual_target = (y_train - prior_train).astype(np.float32)

        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        val_candidates = make_candidates(train_ids, y_outer_train, val_ids, args)
        prior_val = make_pattern_098(val_ids, val_candidates)
        x_val = make_feature_matrix(val_ids, val_candidates, candidate_pool, prior_val)

        print(f"[fold {fold_index}] fitting 104 reference HGB models", flush=True)
        ref104, hgb_v_residual, hgb_a_residual = make_reference_104(
            x_train=x_train,
            residual_target=residual_target,
            x_val=x_val,
            prior_val=prior_val,
            val_ids=val_ids,
            seed=args.seed + fold_index * 101,
        )

        fold_predictions: dict[str, np.ndarray] = {
            "098_PatternPrior": prior_val,
            "104_DimwiseOOFMeta_reference": ref104,
        }
        add_batch_105_124(
            fold_predictions=fold_predictions,
            val_ids=val_ids,
            prior_val=prior_val,
            ref104=ref104,
            val_candidates=val_candidates,
            candidate_pool=candidate_pool,
            x_train=x_train,
            y_train=y_train,
            residual_target=residual_target,
            x_val=x_val,
            prior_train=prior_train,
            hgb_v_residual=hgb_v_residual,
            hgb_a_residual=hgb_a_residual,
            seed=args.seed + fold_index * 997,
        )

        fold_results = sorted(
            [score(name, y_val, pred, "Batch 105-124 low-capacity model.") for name, pred in fold_predictions.items()],
            key=lambda item: float(item["overall_mae"]),
        )
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
    dimwise_integration = compose_dimwise_integrations(aggregate_results, top_n=24)

    output = {
        "method": "Batch 105-124: twenty new low-capacity models plus 125 dimwise integration",
        "note": (
            "Each outer fold uses strict inner leave-one-subject-out prior features. "
            "The 20 models are fixed low-capacity adaptations of state-space, graph/diffusion, "
            "robust calibration, and residual meta-learning ideas."
        ),
        "paper_inspirations": [
            "Mamba/state-space time-series models: low-parameter state smoothing and local residual dynamics.",
            "Graph EEG emotion models: Laplacian/diffusion smoothing over temporal or video-time graphs.",
            "Time-series decomposition models: DCT, SSA, Savitzky-Golay, derivative-constrained postprocessing.",
            "Robust MAE-aligned calibration: small HGB/Huber/Bayesian/KNN residual correction.",
        ],
        "fold_size": args.fold_size,
        "candidate_pool_size": len(candidate_pool),
        "aggregate_results": aggregate_results[: args.top_k],
        "dimwise_integration": dimwise_integration[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_reference_104(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    val_ids: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_v = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=120,
            max_leaf_nodes=11,
            min_samples_leaf=90,
            l2_regularization=0.15,
            random_state=seed,
        )
    )
    model_v.fit(x_train, residual_target)
    residual_v = np.asarray(model_v.predict(x_val), dtype=np.float32)

    model_a = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=11,
            min_samples_leaf=70,
            l2_regularization=0.10,
            random_state=seed + 17,
        )
    )
    model_a.fit(x_train, residual_target)
    residual_a = np.asarray(model_a.predict(x_val), dtype=np.float32)

    pred = prior_val.copy()
    pred[:, 0] = np.clip(prior_val[:, 0] + 0.10 * residual_v[:, 0], 1.0, 255.0)
    arousal_raw = np.clip(prior_val + 0.20 * residual_a, 1.0, 255.0)
    pred[:, 1] = smooth_predictions(val_ids, arousal_raw, 5)[:, 1]
    return pred.astype(np.float32), residual_v, residual_a


def add_batch_105_124(
    fold_predictions: dict[str, np.ndarray],
    val_ids: list[str],
    prior_val: np.ndarray,
    ref104: np.ndarray,
    val_candidates: dict[str, np.ndarray],
    candidate_pool: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_train: np.ndarray,
    hgb_v_residual: np.ndarray,
    hgb_a_residual: np.ndarray,
    seed: int,
) -> None:
    candidate_stack = np.stack([val_candidates[name] for name in candidate_pool], axis=0)
    candidate_std = candidate_stack.std(axis=0).astype(np.float32)

    pred105 = alpha_beta_filter(ref104, val_ids, alpha=0.62, beta=0.08)
    pred106 = slope_adaptive_ema(ref104, val_ids, slow=0.38, fast=0.88)
    fold_predictions["105_AlphaBetaStateFilter_104"] = pred105
    fold_predictions["106_SlopeAdaptiveEMA_104"] = pred106
    fold_predictions["107_SavitzkyGolay_w7p2_104"] = savgol_by_trial(ref104, val_ids, window=7, polyorder=2)
    fold_predictions["108_DCTSoftShrink_104"] = dct_soft_shrink_by_trial(ref104, val_ids, keep_ratio=0.18, shrink=0.45)
    fold_predictions["109_HankelSSA_w15r2_104"] = ssa_by_trial(ref104, val_ids, window=15, rank=2)
    fold_predictions["110_DerivativeClip_q90_104"] = derivative_clip_by_train(ref104, val_ids, y_train, train_quantile=90.0)
    fold_predictions["111_KalmanCV_104"] = kalman_cv_filter(ref104, val_ids, process=0.04, measurement=1.0)
    fold_predictions["112_TemporalGraphDiffusion_104"] = temporal_graph_diffusion(ref104, val_ids, lam=0.18, steps=2)
    fold_predictions["113_CrossVideoTimestampDiffusion_104"] = cross_video_timestamp_diffusion(ref104, val_ids, lam=0.06)
    fold_predictions["114_UncertaintyShrink_104_to_098"] = uncertainty_shrink(ref104, prior_val, candidate_std)
    fold_predictions["115_UncertaintySmoothGate_104"] = uncertainty_smooth_gate(ref104, val_ids, candidate_std)
    fold_predictions["116_VideoBiasCalibrated_098"] = bias_calibrate_by_video(prior_val, val_ids, y_train - prior_train)
    fold_predictions["117_TimeBinBiasCalibrated_098"] = bias_calibrate_by_timebin(prior_val, val_ids, y_train - prior_train)
    fold_predictions["118_VideoTimeBinBiasCalibrated_098"] = bias_calibrate_by_video_timebin(prior_val, val_ids, y_train - prior_train)

    add_residual_model(
        fold_predictions,
        "119_KNNResidual_k96_scale0p06",
        make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=96, weights="distance")),
        x_train,
        residual_target,
        x_val,
        prior_val,
        val_ids,
        scale=0.06,
        smooth_window=0,
    )
    add_residual_model(
        fold_predictions,
        "120_NystroemRidgeResidual_rbf_scale0p06",
        make_pipeline(
            StandardScaler(),
            Nystroem(kernel="rbf", gamma=0.015, n_components=256, random_state=seed),
            Ridge(alpha=80.0),
        ),
        x_train,
        residual_target,
        x_val,
        prior_val,
        val_ids,
        scale=0.06,
        smooth_window=0,
    )
    add_residual_model(
        fold_predictions,
        "121_PLSResidual_c8_scale0p06",
        make_pipeline(StandardScaler(), PLSRegression(n_components=8, scale=False)),
        x_train,
        residual_target,
        x_val,
        prior_val,
        val_ids,
        scale=0.06,
        smooth_window=0,
    )
    add_residual_model(
        fold_predictions,
        "122_BayesianRidgeResidual_scale0p05",
        make_pipeline(StandardScaler(), MultiOutputRegressor(BayesianRidge())),
        x_train,
        residual_target,
        x_val,
        prior_val,
        val_ids,
        scale=0.05,
        smooth_window=0,
    )
    add_residual_model(
        fold_predictions,
        "123_HuberResidual_scale0p05",
        make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(HuberRegressor(epsilon=1.35, alpha=0.002, max_iter=160)),
        ),
        x_train,
        residual_target,
        x_val,
        prior_val,
        val_ids,
        scale=0.05,
        smooth_window=0,
    )
    add_residual_model(
        fold_predictions,
        "124_RandomForestResidual_d6_scale0p05",
        RandomForestRegressor(
            n_estimators=140,
            max_depth=6,
            min_samples_leaf=80,
            max_features=0.35,
            random_state=seed + 31,
            n_jobs=-1,
        ),
        x_train,
        residual_target,
        x_val,
        prior_val,
        val_ids,
        scale=0.05,
        smooth_window=0,
    )

    # A fixed cross-module fusion after the 20-model batch: keep the best-known valence residual
    # path, and test whether the strongest arousal smoother from this batch helps.
    fixed = prior_val.copy()
    fixed[:, 0] = np.clip(prior_val[:, 0] + 0.10 * hgb_v_residual[:, 0], 1.0, 255.0)
    arousal_candidate = uncertainty_smooth_gate(ref104, val_ids, candidate_std)
    fixed[:, 1] = arousal_candidate[:, 1]
    fold_predictions["125_FixedBatch20Fusion_VHGB_AUncertaintySmooth"] = fixed.astype(np.float32)

    dimwise_state = ref104.copy()
    dimwise_state[:, 0] = pred105[:, 0]
    dimwise_state[:, 1] = pred106[:, 1]
    fold_predictions["125_DimwiseBatch20_VAlphaBeta_ASlopeEMA"] = dimwise_state.astype(np.float32)


def add_residual_model(
    predictions: dict[str, np.ndarray],
    name: str,
    model,
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    val_ids: list[str],
    scale: float,
    smooth_window: int,
) -> None:
    print(f"  fitting {name}", flush=True)
    model.fit(x_train, residual_target)
    residual = np.asarray(model.predict(x_val), dtype=np.float32)
    pred = np.clip(prior_val + scale * residual, 1.0, 255.0)
    if smooth_window > 1:
        pred = smooth_predictions(val_ids, pred, smooth_window)
    predictions[name] = pred.astype(np.float32)


def alpha_beta_filter(values: np.ndarray, sample_ids: list[str], alpha: float, beta: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices].astype(np.float32)
        x = z[0].copy()
        v = np.zeros(z.shape[1], dtype=np.float32)
        out[indices[0]] = x
        for local_index, global_index in enumerate(indices[1:], start=1):
            pred = x + v
            residual = z[local_index] - pred
            x = pred + alpha * residual
            v = v + beta * residual
            out[global_index] = x
    return clip(out)


def slope_adaptive_ema(values: np.ndarray, sample_ids: list[str], slow: float, fast: float) -> np.ndarray:
    slopes = np.abs(prior_slope_by_trial(sample_ids, values))
    q_low = np.percentile(slopes, 45, axis=0)
    q_high = np.percentile(slopes, 85, axis=0)
    gate = np.clip((slopes - q_low[None, :]) / np.maximum(q_high - q_low, 1e-6)[None, :], 0.0, 1.0)
    out = np.zeros_like(values, dtype=np.float32)
    for indices in trial_indices(sample_ids):
        out[indices[0]] = values[indices[0]]
        for global_index in indices[1:]:
            alpha = slow + (fast - slow) * gate[global_index]
            out[global_index] = alpha * values[global_index] + (1.0 - alpha) * out[indices[indices.index(global_index) - 1]]
    return clip(out)


def savgol_by_trial(values: np.ndarray, sample_ids: list[str], window: int, polyorder: int) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        if len(indices) < window:
            continue
        local_window = window if window % 2 == 1 else window + 1
        if local_window >= len(indices):
            local_window = len(indices) - 1 if len(indices) % 2 == 0 else len(indices)
        if local_window <= polyorder:
            continue
        out[indices] = savgol_filter(values[indices], local_window, polyorder, axis=0, mode="nearest")
    return clip(out)


def dct_soft_shrink_by_trial(values: np.ndarray, sample_ids: list[str], keep_ratio: float, shrink: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices].astype(np.float32)
        coeff = dct(z, type=2, norm="ortho", axis=0)
        keep = max(2, int(np.ceil(len(indices) * keep_ratio)))
        coeff[keep:] *= shrink
        out[indices] = idct(coeff, type=2, norm="ortho", axis=0)
    return clip(out)


def ssa_by_trial(values: np.ndarray, sample_ids: list[str], window: int, rank: int) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        if len(indices) < window + 2:
            continue
        for dim in range(values.shape[1]):
            out[indices, dim] = ssa_1d(values[indices, dim], window=window, rank=rank)
    return clip(out)


def ssa_1d(x: np.ndarray, window: int, rank: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    k = n - window + 1
    hankel = np.column_stack([x[i : i + k] for i in range(window)])
    u, s, vt = np.linalg.svd(hankel, full_matrices=False)
    recon = (u[:, :rank] * s[:rank]) @ vt[:rank]
    y = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    for col in range(window):
        y[col : col + k] += recon[:, col]
        counts[col : col + k] += 1.0
    return y / np.maximum(counts, 1.0)


def derivative_clip_by_train(
    values: np.ndarray,
    sample_ids: list[str],
    y_train: np.ndarray,
    train_quantile: float,
) -> np.ndarray:
    diffs = np.diff(y_train.reshape(-1, 2), axis=0)
    limit = np.percentile(np.abs(diffs), train_quantile, axis=0).astype(np.float32)
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        out[indices[0]] = values[indices[0]]
        for pos in range(1, len(indices)):
            prev = out[indices[pos - 1]]
            target = values[indices[pos]]
            delta = np.clip(target - prev, -limit, limit)
            out[indices[pos]] = prev + delta
    return clip(out)


def kalman_cv_filter(values: np.ndarray, sample_ids: list[str], process: float, measurement: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    transition = np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    observation = np.asarray([[1.0, 0.0]], dtype=np.float32)
    q = process * np.eye(2, dtype=np.float32)
    r = np.asarray([[measurement]], dtype=np.float32)
    for indices in trial_indices(sample_ids):
        for dim in range(values.shape[1]):
            state = np.asarray([values[indices[0], dim], 0.0], dtype=np.float32)
            cov = np.eye(2, dtype=np.float32)
            out[indices[0], dim] = state[0]
            for global_index in indices[1:]:
                state = transition @ state
                cov = transition @ cov @ transition.T + q
                innovation = values[global_index, dim] - float(observation @ state)
                s = observation @ cov @ observation.T + r
                gain = cov @ observation.T @ np.linalg.inv(s)
                state = state + (gain[:, 0] * innovation)
                cov = (np.eye(2, dtype=np.float32) - gain @ observation) @ cov
                out[global_index, dim] = state[0]
    return clip(out)


def temporal_graph_diffusion(values: np.ndarray, sample_ids: list[str], lam: float, steps: int) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for _ in range(steps):
        updated = out.copy()
        for indices in trial_indices(sample_ids):
            z = out[indices]
            if len(indices) < 3:
                continue
            updated[indices[1:-1]] = (1 - 2 * lam) * z[1:-1] + lam * z[:-2] + lam * z[2:]
        out = updated
    return clip(out)


def cross_video_timestamp_diffusion(values: np.ndarray, sample_ids: list[str], lam: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, _, timestamp = parse_sample_id(sample_id)
        groups[(subject, timestamp)].append(index)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        mean = values[indices].mean(axis=0)
        out[indices] = (1.0 - lam) * values[indices] + lam * mean[None, :]
    return clip(out)


def uncertainty_shrink(values: np.ndarray, reference: np.ndarray, candidate_std: np.ndarray) -> np.ndarray:
    low = np.percentile(candidate_std, 45, axis=0)
    high = np.percentile(candidate_std, 90, axis=0)
    gate = np.clip((candidate_std - low[None, :]) / np.maximum(high - low, 1e-6)[None, :], 0.0, 1.0)
    pred = (1.0 - 0.35 * gate) * values + (0.35 * gate) * reference
    return clip(pred)


def uncertainty_smooth_gate(values: np.ndarray, sample_ids: list[str], candidate_std: np.ndarray) -> np.ndarray:
    smooth = smooth_predictions(sample_ids, values, 5)
    low = np.percentile(candidate_std, 50, axis=0)
    high = np.percentile(candidate_std, 85, axis=0)
    gate = np.clip((candidate_std - low[None, :]) / np.maximum(high - low, 1e-6)[None, :], 0.0, 1.0)
    pred = (1.0 - 0.55 * gate) * values + (0.55 * gate) * smooth
    return clip(pred)


def bias_calibrate_by_video(values: np.ndarray, sample_ids: list[str], train_residual: np.ndarray) -> np.ndarray:
    # The OOF training rows are subject-major with 15 trials of 102 samples. Estimate a conservative
    # video bias from that order and apply it to matching validation videos.
    residual_by_video = residual_means_by_video(train_residual)
    rows = []
    for sample_id in sample_ids:
        _, video, _ = parse_sample_id(sample_id)
        rows.append(residual_by_video.get(video, np.zeros(2, dtype=np.float32)))
    return clip(values + 0.08 * np.asarray(rows, dtype=np.float32))


def bias_calibrate_by_timebin(values: np.ndarray, sample_ids: list[str], train_residual: np.ndarray) -> np.ndarray:
    residual_by_bin = residual_means_by_timebin(train_residual, bins=8)
    rows = []
    for sample_id in sample_ids:
        _, _, timestamp = parse_sample_id(sample_id)
        time_bin = min(7, int(timestamp * 8 / 102))
        rows.append(residual_by_bin.get(time_bin, np.zeros(2, dtype=np.float32)))
    return clip(values + 0.08 * np.asarray(rows, dtype=np.float32))


def bias_calibrate_by_video_timebin(values: np.ndarray, sample_ids: list[str], train_residual: np.ndarray) -> np.ndarray:
    residual_by_key = residual_means_by_video_timebin(train_residual, bins=6)
    rows = []
    for sample_id in sample_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        time_bin = min(5, int(timestamp * 6 / 102))
        rows.append(residual_by_key.get((video, time_bin), np.zeros(2, dtype=np.float32)))
    return clip(values + 0.05 * np.asarray(rows, dtype=np.float32))


def residual_means_by_video(train_residual: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, residual in enumerate(train_residual):
        video = (index // 102) % 15 + 1
        groups[video].append(residual)
    return {key: np.asarray(values, dtype=np.float32).mean(axis=0) for key, values in groups.items()}


def residual_means_by_timebin(train_residual: np.ndarray, bins: int) -> dict[int, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, residual in enumerate(train_residual):
        timestamp = index % 102
        time_bin = min(bins - 1, int(timestamp * bins / 102))
        groups[time_bin].append(residual)
    return {key: np.asarray(values, dtype=np.float32).mean(axis=0) for key, values in groups.items()}


def residual_means_by_video_timebin(train_residual: np.ndarray, bins: int) -> dict[tuple[int, int], np.ndarray]:
    groups: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for index, residual in enumerate(train_residual):
        video = (index // 102) % 15 + 1
        timestamp = index % 102
        time_bin = min(bins - 1, int(timestamp * bins / 102))
        groups[(video, time_bin)].append(residual)
    return {key: np.asarray(values, dtype=np.float32).mean(axis=0) for key, values in groups.items()}


def trial_indices(sample_ids: list[str]) -> list[list[int]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    return [[index for _, index in sorted(items)] for items in groups.values()]


def compose_dimwise_integrations(rows: list[dict[str, object]], top_n: int) -> list[dict[str, object]]:
    valence_rows = sorted(rows, key=lambda item: float(item["valence_mae"]))[:top_n]
    arousal_rows = sorted(rows, key=lambda item: float(item["arousal_mae"]))[:top_n]
    results = []
    for v_row in valence_rows:
        for a_row in arousal_rows:
            overall = (float(v_row["valence_mae"]) + float(a_row["arousal_mae"])) / 2.0
            results.append(
                {
                    "method": f"125_DimwiseBatch20_V[{v_row['method']}]__A[{a_row['method']}]",
                    "overall_mae": round(float(overall), 4),
                    "valence_mae": round(float(v_row["valence_mae"]), 4),
                    "arousal_mae": round(float(a_row["arousal_mae"]), 4),
                    "overall_mse": None,
                    "notes": "Metric-composed integration after the 20-model batch.",
                }
            )
    return sorted(results, key=lambda item: float(item["overall_mae"]))


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1.0, 255.0).astype(np.float32)


if __name__ == "__main__":
    main()
