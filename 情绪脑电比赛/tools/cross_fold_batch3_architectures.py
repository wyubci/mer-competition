from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import (  # noqa: E402
    alpha_beta_filter,
    clip,
    make_reference_104,
    slope_adaptive_ema,
    trial_indices,
)
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Batch 147-166: twenty new architecture families.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_147_167_batch3_architectures.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=180)
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
        oof_train_ids = []
        for subject in train_subjects:
            oof_train_ids.extend(ids_for_subjects(labels, [subject]))
        residual_target = (y_train - prior_train).astype(np.float32)

        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        val_candidates = make_candidates(train_ids, y_outer_train, val_ids, args)
        prior_val = make_pattern_098(val_ids, val_candidates)
        x_val = make_feature_matrix(val_ids, val_candidates, candidate_pool, prior_val)
        candidate_stack = np.stack([val_candidates[name] for name in candidate_pool], axis=0).astype(np.float32)
        candidate_mean = candidate_stack.mean(axis=0).astype(np.float32)
        candidate_std = candidate_stack.std(axis=0).astype(np.float32)

        print(f"[fold {fold_index}] fitting 104 reference and 20 architectures", flush=True)
        ref104, _, _ = make_reference_104(
            x_train=x_train,
            residual_target=residual_target,
            x_val=x_val,
            prior_val=prior_val,
            val_ids=val_ids,
            seed=args.seed + fold_index * 173,
        )
        previous_125 = make_previous_125(ref104, val_ids)

        fold_predictions: dict[str, np.ndarray] = {
            "098_PatternPrior": prior_val,
            "104_DimwiseOOFMeta_reference": ref104,
            "125_PreviousStateFusion_reference": previous_125,
        }
        add_architecture_batch(
            fold_predictions=fold_predictions,
            val_ids=val_ids,
            ref104=ref104,
            previous_125=previous_125,
            candidate_stack=candidate_stack,
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            oof_train_ids=oof_train_ids,
            y_train=y_train,
            prior_train=prior_train,
            residual_target=residual_target,
            x_train=x_train,
            x_val=x_val,
            prior_val=prior_val,
            seed=args.seed + fold_index * 1117,
        )

        fold_results = sorted(
            [score(name, y_val, pred, "Batch 147-166 architecture model.") for name, pred in fold_predictions.items()],
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
    dimwise_integration = compose_dimwise_integrations(aggregate_results, top_n=32)

    output = {
        "method": "Batch 147-166: twenty new architecture families plus 167 integration",
        "note": "No parameter-only variants: each numbered item uses a distinct modeling assumption.",
        "architecture_map": architecture_map(),
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


def architecture_map() -> dict[str, str]:
    return {
        "147": "Tukey biweight local M-estimator smoother",
        "148": "Trimmed local attention smoother",
        "149": "Bilateral edge-preserving value-time filter",
        "150": "Robust trend/residual decomposition",
        "151": "Deterministic particle filter",
        "152": "Switching-regime linear dynamical system",
        "153": "Gaussian Markov random field MAP smoother",
        "154": "Laplacian pyramid multiscale blend",
        "155": "Empirical-Bayes shrinkage to candidate consensus",
        "156": "Conformal residual clamp around candidate consensus",
        "157": "Prior-bin rank-preserving bias calibration",
        "158": "Gaussian copula covariance calibration",
        "159": "Trial prototype residual retrieval",
        "160": "FFT phase-preserving high-frequency shrinkage",
        "161": "AR(2) trajectory repair smoother",
        "162": "Monotone micro-segment reversal repair",
        "163": "Change-point piecewise trend model",
        "164": "Dirichlet evidence-weighted expert averaging",
        "165": "Risk-averse center/candidate clamp",
        "166": "Learned phase-flow vector-field correction",
    }


def make_previous_125(ref104: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    valence = alpha_beta_filter(ref104, sample_ids, alpha=0.62, beta=0.08)
    arousal = slope_adaptive_ema(ref104, sample_ids, slow=0.38, fast=0.88)
    pred = ref104.copy()
    pred[:, 0] = valence[:, 0]
    pred[:, 1] = arousal[:, 1]
    return clip(pred)


def add_architecture_batch(
    fold_predictions: dict[str, np.ndarray],
    val_ids: list[str],
    ref104: np.ndarray,
    previous_125: np.ndarray,
    candidate_stack: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    oof_train_ids: list[str],
    y_train: np.ndarray,
    prior_train: np.ndarray,
    residual_target: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    seed: int,
) -> None:
    p147 = tukey_biweight_smoother(previous_125, val_ids)
    p148 = trimmed_local_attention(previous_125, val_ids)
    p149 = bilateral_edge_filter(previous_125, val_ids)
    p150 = robust_trend_residual_decomposition(previous_125, val_ids)
    p151 = deterministic_particle_filter(previous_125, val_ids, candidate_std)
    p152 = switching_regime_dynamics(previous_125, val_ids)
    p153 = gmrf_map_smoother(previous_125, val_ids)
    p154 = laplacian_pyramid_blend(previous_125, val_ids)
    p155 = empirical_bayes_shrink(previous_125, candidate_mean, candidate_std)
    p156 = conformal_residual_clamp(previous_125, candidate_mean, residual_target)
    p157 = prior_bin_bias_calibration(prior_train, y_train, previous_125)
    p158 = gaussian_copula_covariance_calibration(y_train, previous_125)
    p159 = trial_prototype_residual_retrieval(
        oof_train_ids=oof_train_ids,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=val_ids,
        val_pred=previous_125,
    )
    p160 = fft_phase_preserving_shrink(previous_125, val_ids)
    p161 = ar2_trajectory_repair(previous_125, val_ids)
    p162 = monotone_microsegment_repair(previous_125, val_ids)
    p163 = changepoint_piecewise_trend(previous_125, val_ids)
    p164 = dirichlet_expert_averaging(candidate_stack, candidate_std)
    p165 = risk_averse_consensus_clamp(previous_125, candidate_mean, candidate_std)
    p166 = learned_phase_flow_correction(
        train_ids=oof_train_ids,
        y_train=y_train,
        x_train=x_train,
        x_val=x_val,
        prior_val=prior_val,
        val_ids=val_ids,
    )

    fold_predictions["147_TukeyBiweightLocalMEstimator"] = p147
    fold_predictions["148_TrimmedLocalAttention"] = p148
    fold_predictions["149_BilateralEdgePreservingFilter"] = p149
    fold_predictions["150_RobustTrendResidualSplit"] = p150
    fold_predictions["151_DeterministicParticleFilter"] = p151
    fold_predictions["152_SwitchingRegimeDynamics"] = p152
    fold_predictions["153_GMRFMapSmoother"] = p153
    fold_predictions["154_LaplacianPyramidBlend"] = p154
    fold_predictions["155_EmpiricalBayesShrinkage"] = p155
    fold_predictions["156_ConformalResidualClamp"] = p156
    fold_predictions["157_PriorBinBiasCalibration"] = p157
    fold_predictions["158_GaussianCopulaCovarianceCalibration"] = p158
    fold_predictions["159_TrialPrototypeResidualRetrieval"] = p159
    fold_predictions["160_FFTPhasePreservingShrinkage"] = p160
    fold_predictions["161_AR2TrajectoryRepair"] = p161
    fold_predictions["162_MonotoneMicrosegmentRepair"] = p162
    fold_predictions["163_ChangePointPiecewiseTrend"] = p163
    fold_predictions["164_DirichletEvidenceExpertAverage"] = p164
    fold_predictions["165_RiskAverseConsensusClamp"] = p165
    fold_predictions["166_LearnedPhaseFlowCorrection"] = p166

    p167 = previous_125.copy()
    p167[:, 0] = p147[:, 0]
    p167[:, 1] = min_by_disagreement([p148, p149, p153, p165], candidate_mean, candidate_std)[:, 1]
    fold_predictions["167_FixedBatch3Fusion_VTukey_ARiskMOE"] = clip(p167)

    p167_best = previous_125.copy()
    p167_best[:, 0] = p158[:, 0]
    p167_best[:, 1] = p159[:, 1]
    fold_predictions["167_FixedBatch3Fusion_VCopula_APrototype"] = clip(p167_best)


def tukey_biweight_smoother(values: np.ndarray, sample_ids: list[str], radius: int = 4) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        for pos, global_index in enumerate(indices):
            left = max(0, pos - radius)
            right = min(len(indices), pos + radius + 1)
            window = z[left:right]
            median = np.median(window, axis=0)
            mad = np.median(np.abs(window - median[None, :]), axis=0) + 1e-6
            u = (window - median[None, :]) / (4.685 * 1.4826 * mad[None, :])
            weights = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
            denom = np.maximum(weights.sum(axis=0), 1e-6)
            robust = (weights * window).sum(axis=0) / denom
            out[global_index] = 0.72 * z[pos] + 0.28 * robust
    return clip(out)


def trimmed_local_attention(values: np.ndarray, sample_ids: list[str], radius: int = 5) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        for pos, global_index in enumerate(indices):
            left = max(0, pos - radius)
            right = min(len(indices), pos + radius + 1)
            local = z[left:right]
            time = np.arange(left, right, dtype=np.float32) - pos
            distance = np.linalg.norm(local - z[pos][None, :], axis=1) + 0.35 * np.abs(time)
            keep = distance <= np.percentile(distance, 75)
            weights = np.exp(-0.5 * (time[keep] / 3.0) ** 2)
            weights /= max(weights.sum(), 1e-6)
            out[global_index] = 0.65 * z[pos] + 0.35 * (weights @ local[keep])
    return clip(out)


def bilateral_edge_filter(values: np.ndarray, sample_ids: list[str], radius: int = 4) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        for pos, global_index in enumerate(indices):
            left = max(0, pos - radius)
            right = min(len(indices), pos + radius + 1)
            local = z[left:right]
            dt = np.arange(left, right, dtype=np.float32) - pos
            value_dist = np.abs(local - z[pos][None, :])
            weights = np.exp(-0.5 * (dt[:, None] / 2.5) ** 2 - 0.5 * (value_dist / 7.0) ** 2)
            weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-6)
            out[global_index] = 0.55 * z[pos] + 0.45 * (weights * local).sum(axis=0)
    return clip(out)


def robust_trend_residual_decomposition(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        trend = median_filter(z, size=(9, 1), mode="nearest")
        residual = z - trend
        residual_scale = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0) + 1e-6
        damp = np.tanh(residual / (2.5 * residual_scale[None, :])) * (2.5 * residual_scale[None, :])
        out[indices] = trend + 0.72 * damp
    return clip(out)


def deterministic_particle_filter(
    values: np.ndarray,
    sample_ids: list[str],
    candidate_std: np.ndarray,
) -> np.ndarray:
    out = values.copy().astype(np.float32)
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        particles = np.stack([z[0] + offset * 2.0 for offset in offsets], axis=0)
        weights = np.full(3, 1.0 / 3.0, dtype=np.float32)
        out[indices[0]] = z[0]
        for pos, global_index in enumerate(indices[1:], start=1):
            particles = particles + 0.35 * (z[pos] - z[pos - 1])[None, :]
            sigma = 4.0 + 0.25 * candidate_std[global_index].mean()
            likelihood = np.exp(-0.5 * ((particles - z[pos][None, :]) ** 2).sum(axis=1) / (sigma**2))
            weights = weights * likelihood
            weights = weights / max(weights.sum(), 1e-6)
            estimate = weights @ particles
            particles = 0.72 * particles + 0.28 * z[pos][None, :]
            out[global_index] = 0.70 * z[pos] + 0.30 * estimate
    return clip(out)


def switching_regime_dynamics(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    slopes = np.abs(prior_slope_by_trial(sample_ids, values))
    low = np.percentile(slopes, 45, axis=0)
    high = np.percentile(slopes, 80, axis=0)
    slow = smooth_predictions(sample_ids, values, 7)
    fast = alpha_beta_filter(values, sample_ids, alpha=0.66, beta=0.10)
    medium = smooth_predictions(sample_ids, values, 3)
    out = values.copy().astype(np.float32)
    out = np.where(slopes < low[None, :], slow, out)
    out = np.where((slopes >= low[None, :]) & (slopes < high[None, :]), medium, out)
    out = np.where(slopes >= high[None, :], fast, out)
    return clip(out)


def gmrf_map_smoother(values: np.ndarray, sample_ids: list[str], lam: float = 0.22) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        n = len(indices)
        if n < 3:
            continue
        diag = np.ones(n, dtype=np.float32) * (1.0 + 2.0 * lam)
        diag[0] = diag[-1] = 1.0 + lam
        off = np.ones(n - 1, dtype=np.float32) * (-lam)
        mat = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        for dim in range(values.shape[1]):
            out[indices, dim] = np.linalg.solve(mat, values[indices, dim])
    return clip(out)


def laplacian_pyramid_blend(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    low = smooth_predictions(sample_ids, values, 15)
    mid = smooth_predictions(sample_ids, values, 5) - low
    high = values - smooth_predictions(sample_ids, values, 5)
    pred = low + 0.88 * mid + 0.55 * high
    return clip(pred)


def empirical_bayes_shrink(
    values: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> np.ndarray:
    sigma2 = candidate_std**2 + 1.0
    tau2 = np.percentile(sigma2, 60, axis=0)[None, :]
    weight = tau2 / (tau2 + sigma2)
    pred = weight * values + (1.0 - weight) * candidate_mean
    return clip(0.80 * values + 0.20 * pred)


def conformal_residual_clamp(
    values: np.ndarray,
    candidate_mean: np.ndarray,
    residual_target: np.ndarray,
) -> np.ndarray:
    radius = np.percentile(np.abs(residual_target), 88, axis=0)[None, :]
    clipped = np.minimum(np.maximum(values, candidate_mean - radius), candidate_mean + radius)
    return clip(0.86 * values + 0.14 * clipped)


def prior_bin_bias_calibration(prior_train: np.ndarray, y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for dim in range(2):
        edges = np.percentile(prior_train[:, dim], np.linspace(0, 100, 11))
        residual = y_train[:, dim] - prior_train[:, dim]
        bins = np.digitize(prior_train[:, dim], edges[1:-1], right=False)
        bias = np.zeros(10, dtype=np.float32)
        for bucket in range(10):
            mask = bins == bucket
            if mask.any():
                bias[bucket] = np.median(residual[mask])
        pred_bins = np.digitize(pred[:, dim], edges[1:-1], right=False)
        out[:, dim] = pred[:, dim] + 0.08 * bias[pred_bins]
    return clip(out)


def gaussian_copula_covariance_calibration(y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y_mean = y_train.mean(axis=0)
    p_mean = pred.mean(axis=0)
    y_cov = np.cov(y_train.T) + 1e-3 * np.eye(2)
    p_cov = np.cov(pred.T) + 1e-3 * np.eye(2)
    ey, vy = np.linalg.eigh(y_cov)
    ep, vp = np.linalg.eigh(p_cov)
    transform = vy @ np.diag(np.sqrt(np.maximum(ey, 1e-6))) @ vy.T @ vp @ np.diag(1.0 / np.sqrt(np.maximum(ep, 1e-6))) @ vp.T
    mapped = (pred - p_mean[None, :]) @ transform.T + y_mean[None, :]
    return clip(0.92 * pred + 0.08 * mapped)


def trial_prototype_residual_retrieval(
    oof_train_ids: list[str],
    prior_train: np.ndarray,
    residual_target: np.ndarray,
    val_ids: list[str],
    val_pred: np.ndarray,
) -> np.ndarray:
    train_trials = collect_trial_arrays(oof_train_ids, prior_train, residual_target)
    val_trials = collect_trial_arrays(val_ids, val_pred, None)
    pred = val_pred.copy().astype(np.float32)
    for key, val_info in val_trials.items():
        video = key[1]
        same_video = [item for item in train_trials.values() if item["video"] == video]
        pool = same_video if same_video else list(train_trials.values())
        val_feat = downsample_trial(val_info["prior"])
        distances = np.asarray([np.linalg.norm(val_feat - downsample_trial(item["prior"])) for item in pool])
        order = np.argsort(distances)[:5]
        weights = 1.0 / (distances[order] + 1e-3)
        weights = weights / weights.sum()
        residual = np.zeros_like(val_info["prior"])
        for weight, pool_index in zip(weights, order):
            residual += weight * resize_trial(pool[pool_index]["residual"], len(val_info["indices"]))
        pred[val_info["indices"]] = val_info["prior"] + 0.04 * residual
    return clip(pred)


def collect_trial_arrays(
    sample_ids: list[str],
    prior: np.ndarray,
    residual: np.ndarray | None,
) -> dict[tuple[str, int], dict[str, object]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    out = {}
    for key, items in groups.items():
        indices = [index for _, index in sorted(items)]
        out[key] = {
            "video": key[1],
            "indices": indices,
            "prior": prior[indices],
            "residual": np.zeros_like(prior[indices]) if residual is None else residual[indices],
        }
    return out


def downsample_trial(values: np.ndarray, bins: int = 12) -> np.ndarray:
    parts = np.array_split(values, bins, axis=0)
    return np.concatenate([part.mean(axis=0) for part in parts], axis=0)


def resize_trial(values: np.ndarray, n: int) -> np.ndarray:
    source = np.linspace(0, 1, len(values))
    target = np.linspace(0, 1, n)
    return np.stack([np.interp(target, source, values[:, dim]) for dim in range(values.shape[1])], axis=1)


def fft_phase_preserving_shrink(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices].astype(np.float32)
        freq = np.fft.rfft(z, axis=0)
        cutoff = max(3, int(freq.shape[0] * 0.18))
        freq[cutoff:] *= 0.45
        out[indices] = np.fft.irfft(freq, n=len(indices), axis=0)
    return clip(out)


def ar2_trajectory_repair(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        if len(indices) < 3:
            continue
        out[indices[:2]] = z[:2]
        for pos in range(2, len(indices)):
            predicted = 1.65 * out[indices[pos - 1]] - 0.70 * out[indices[pos - 2]]
            out[indices[pos]] = 0.82 * z[pos] + 0.18 * predicted
    return clip(out)


def monotone_microsegment_repair(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    slopes = prior_slope_by_trial(sample_ids, values)
    small = np.abs(slopes) < np.percentile(np.abs(slopes), 35, axis=0)[None, :]
    for indices in trial_indices(sample_ids):
        for pos in range(1, len(indices) - 1):
            idx = indices[pos]
            prev_idx = indices[pos - 1]
            next_idx = indices[pos + 1]
            reversal = np.sign(values[idx] - values[prev_idx]) != np.sign(values[next_idx] - values[idx])
            mask = reversal & small[idx]
            repaired = 0.5 * (values[prev_idx] + values[next_idx])
            out[idx] = np.where(mask, 0.65 * values[idx] + 0.35 * repaired, out[idx])
    return clip(out)


def changepoint_piecewise_trend(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        magnitude = np.linalg.norm(np.diff(z, axis=0), axis=1)
        if len(magnitude) < 8:
            continue
        cuts = sorted(np.argsort(magnitude)[-3:] + 1)
        bounds = [0] + cuts + [len(indices)]
        trend = z.copy()
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end - start < 2:
                continue
            t = np.arange(end - start, dtype=np.float32)
            design = np.stack([np.ones_like(t), t], axis=1)
            for dim in range(2):
                coef = np.linalg.pinv(design) @ z[start:end, dim]
                trend[start:end, dim] = design @ coef
        out[indices] = 0.78 * z + 0.22 * trend
    return clip(out)


def dirichlet_expert_averaging(candidate_stack: np.ndarray, candidate_std: np.ndarray) -> np.ndarray:
    center = np.median(candidate_stack, axis=0)
    distance = np.abs(candidate_stack - center[None, :, :])
    evidence = 1.0 / (1.0 + distance + 0.15 * candidate_std[None, :, :])
    weights = evidence / np.maximum(evidence.sum(axis=0, keepdims=True), 1e-6)
    pred = (weights * candidate_stack).sum(axis=0)
    return clip(pred)


def risk_averse_consensus_clamp(
    values: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> np.ndarray:
    gate = uncertainty_gate(candidate_std, 50, 88)
    lower = candidate_mean - (1.25 + 0.5 * gate) * candidate_std
    upper = candidate_mean + (1.25 + 0.5 * gate) * candidate_std
    clamped = np.minimum(np.maximum(values, lower), upper)
    center = np.asarray([128.0, 128.0], dtype=np.float32)[None, :]
    pred = (1.0 - 0.25 * gate) * clamped + (0.25 * gate) * (0.75 * clamped + 0.25 * center)
    return clip(pred)


def learned_phase_flow_correction(
    train_ids: list[str],
    y_train: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    val_ids: list[str],
) -> np.ndarray:
    features = []
    targets = []
    id_to_index = {sample_id: index for index, sample_id in enumerate(train_ids)}
    groups: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for sample_id in train_ids:
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, sample_id))
    for items in groups.values():
        ordered = [sample_id for _, sample_id in sorted(items)]
        for left, right in zip(ordered[:-1], ordered[1:]):
            left_index = id_to_index[left]
            right_index = id_to_index[right]
            features.append(x_train[left_index])
            targets.append(y_train[right_index] - y_train[left_index])
    if not features:
        return prior_val
    model = make_pipeline(StandardScaler(), Ridge(alpha=500.0))
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32))
    flow = np.asarray(model.predict(x_val), dtype=np.float32)
    repaired = prior_val.copy().astype(np.float32)
    for indices in trial_indices(val_ids):
        for pos in range(1, len(indices)):
            prev = repaired[indices[pos - 1]]
            predicted = prev + flow[indices[pos - 1]]
            repaired[indices[pos]] = 0.92 * prior_val[indices[pos]] + 0.08 * predicted
    return clip(repaired)


def min_by_disagreement(
    predictions: list[np.ndarray],
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> np.ndarray:
    risks = []
    for pred in predictions:
        risks.append(np.abs(pred - candidate_mean) / (candidate_std + 1.0))
    choice = np.stack(risks, axis=0).argmin(axis=0)
    out = np.zeros_like(predictions[0], dtype=np.float32)
    for index, pred in enumerate(predictions):
        out = np.where(choice == index, pred, out)
    return clip(out)


def uncertainty_gate(candidate_std: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    low = np.percentile(candidate_std, q_low, axis=0)
    high = np.percentile(candidate_std, q_high, axis=0)
    return np.clip((candidate_std - low[None, :]) / np.maximum(high - low, 1e-6)[None, :], 0.0, 1.0)


def compose_dimwise_integrations(rows: list[dict[str, object]], top_n: int) -> list[dict[str, object]]:
    valence_rows = sorted(rows, key=lambda item: float(item["valence_mae"]))[:top_n]
    arousal_rows = sorted(rows, key=lambda item: float(item["arousal_mae"]))[:top_n]
    results = []
    for v_row in valence_rows:
        for a_row in arousal_rows:
            overall = (float(v_row["valence_mae"]) + float(a_row["arousal_mae"])) / 2.0
            results.append(
                {
                    "method": f"167_DimwiseBatch3_V[{v_row['method']}]__A[{a_row['method']}]",
                    "overall_mae": round(float(overall), 4),
                    "valence_mae": round(float(v_row["valence_mae"]), 4),
                    "arousal_mae": round(float(a_row["arousal_mae"]), 4),
                    "overall_mse": None,
                    "notes": "Metric-composed integration after the 147-166 architecture batch.",
                }
            )
    return sorted(results, key=lambda item: float(item["overall_mae"]))


if __name__ == "__main__":
    main()
