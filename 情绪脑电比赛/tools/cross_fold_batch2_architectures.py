from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import median_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import (  # noqa: E402
    alpha_beta_filter,
    clip,
    make_reference_104,
    slope_adaptive_ema,
    temporal_graph_diffusion,
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
    parser = argparse.ArgumentParser(description="Batch 126-145: twenty distinct architectures.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_126_146_batch2_architectures.json")
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
        residual_target = (y_train - prior_train).astype(np.float32)

        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        val_candidates = make_candidates(train_ids, y_outer_train, val_ids, args)
        prior_val = make_pattern_098(val_ids, val_candidates)
        x_val = make_feature_matrix(val_ids, val_candidates, candidate_pool, prior_val)
        candidate_stack = np.stack([val_candidates[name] for name in candidate_pool], axis=0)
        candidate_std = candidate_stack.std(axis=0).astype(np.float32)
        candidate_mean = candidate_stack.mean(axis=0).astype(np.float32)

        print(f"[fold {fold_index}] fitting 104 reference and distinct architectures", flush=True)
        ref104, _, _ = make_reference_104(
            x_train=x_train,
            residual_target=residual_target,
            x_val=x_val,
            prior_val=prior_val,
            val_ids=val_ids,
            seed=args.seed + fold_index * 151,
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
            candidate_std=candidate_std,
            candidate_mean=candidate_mean,
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            prior_train=prior_train,
            prior_val=prior_val,
            residual_target=residual_target,
            train_ids=train_ids,
            y_outer_train=y_outer_train,
            seed=args.seed + fold_index * 997,
        )

        fold_results = sorted(
            [score(name, y_val, pred, "Batch 126-145 distinct architecture.") for name, pred in fold_predictions.items()],
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
        "method": "Batch 126-145: twenty distinct architectures plus 146 integration",
        "note": (
            "This batch avoids parameter-only variants. Each numbered model uses a different "
            "architecture family or modeling assumption."
        ),
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
        "126": "Local linear LOESS trajectory smoother",
        "127": "Gaussian-process-style time kernel smoother",
        "128": "Hampel median outlier architecture",
        "129": "Total-variation proximal denoiser",
        "130": "Haar wavelet shrinkage",
        "131": "Piecewise spline/knot trend projection",
        "132": "Discrete HMM Viterbi trajectory decoder",
        "133": "CRF-like anchored temporal graph smoother",
        "134": "Local self-attention smoother",
        "135": "Prototype residual expert via KMeans",
        "136": "Gaussian mixture residual expert",
        "137": "Random Fourier feature kernel residual model",
        "138": "PLS/PCA latent residual reconstruction",
        "139": "Quantile distribution calibration",
        "140": "Isotonic monotone calibration",
        "141": "Huber robust residual calibration",
        "142": "Small MLP residual architecture",
        "143": "Gradient-boosted residual architecture",
        "144": "Linear stacking of prior candidates/features",
        "145": "Rule-based mixture of heterogeneous architecture outputs",
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
    candidate_std: np.ndarray,
    candidate_mean: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    prior_train: np.ndarray,
    prior_val: np.ndarray,
    residual_target: np.ndarray,
    train_ids: list[str],
    y_outer_train: np.ndarray,
    seed: int,
) -> None:
    p126 = local_linear_loess(previous_125, val_ids, radius=4, bandwidth=3.0)
    p127 = gaussian_time_kernel(previous_125, val_ids, sigma=2.4)
    p128 = hampel_median_filter(previous_125, val_ids, window=5, threshold=2.7)
    p129 = total_variation_denoise(previous_125, val_ids, weight=0.08, iterations=18)
    p130 = haar_wavelet_shrink(previous_125, val_ids, threshold_scale=0.18)
    p131 = piecewise_knot_projection(previous_125, val_ids, knot_count=7, blend=0.35)
    p132 = hmm_viterbi_decode(ref104, val_ids, train_ids, y_outer_train, blend=0.16)
    p133 = anchored_graph_smoother(previous_125, val_ids, lam=0.18, steps=8)
    p134 = local_attention_smoother(previous_125, val_ids, radius=5, sigma_time=3.2, sigma_value=9.0)
    p135 = kmeans_residual_expert(x_train, residual_target, x_val, prior_val, seed=seed)
    p136 = gmm_residual_expert(x_train, residual_target, x_val, prior_val, seed=seed)
    p137 = rff_kernel_residual(x_train, residual_target, x_val, prior_val, seed=seed)
    p138 = pca_latent_residual(x_train, residual_target, x_val, prior_val)
    p139 = quantile_distribution_calibration(prior_train, y_train, previous_125)
    p140 = isotonic_calibration(prior_train, y_train, previous_125)
    p141 = huber_residual_calibration(x_train, residual_target, x_val, prior_val)
    p142 = mlp_residual_architecture(x_train, residual_target, x_val, prior_val, seed=seed)
    p143 = gradient_boosted_residual(x_train, residual_target, x_val, prior_val, seed=seed)
    p144 = linear_stacking_architecture(x_train, y_train, x_val, prior_val)
    p145 = heterogeneous_rule_moe(
        previous_125=previous_125,
        candidate_std=candidate_std,
        candidate_mean=candidate_mean,
        p126=p126,
        p130=p130,
        p133=p133,
        p134=p134,
        p139=p139,
    )

    fold_predictions["126_LocalLinearLOESS"] = p126
    fold_predictions["127_GaussianTimeKernelSmoother"] = p127
    fold_predictions["128_HampelMedianFilter"] = p128
    fold_predictions["129_TotalVariationDenoiser"] = p129
    fold_predictions["130_HaarWaveletShrinkage"] = p130
    fold_predictions["131_PiecewiseKnotProjection"] = p131
    fold_predictions["132_HMMViterbiDecoder"] = p132
    fold_predictions["133_AnchoredGraphSmoother"] = p133
    fold_predictions["134_LocalSelfAttentionSmoother"] = p134
    fold_predictions["135_KMeansResidualExpert"] = p135
    fold_predictions["136_GMMResidualExpert"] = p136
    fold_predictions["137_RFFKernelResidual"] = p137
    fold_predictions["138_PCALatentResidual"] = p138
    fold_predictions["139_QuantileDistributionCalibration"] = p139
    fold_predictions["140_IsotonicCalibration"] = p140
    fold_predictions["141_HuberResidualCalibration"] = p141
    fold_predictions["142_MLPResidualArchitecture"] = p142
    fold_predictions["143_GradientBoostedResidual"] = p143
    fold_predictions["144_LinearStackingArchitecture"] = p144
    fold_predictions["145_HeterogeneousRuleMOE"] = p145

    p146 = previous_125.copy()
    p146[:, 0] = min_by_proxy([previous_125, p128, p133, p145], candidate_std)[:, 0]
    p146[:, 1] = p134[:, 1]
    fold_predictions["146_FixedBatch2Fusion_VProxyMOE_ALocalAttention"] = clip(p146)


def local_linear_loess(values: np.ndarray, sample_ids: list[str], radius: int, bandwidth: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        times = np.arange(len(indices), dtype=np.float32)
        for local_pos, global_index in enumerate(indices):
            left = max(0, local_pos - radius)
            right = min(len(indices), local_pos + radius + 1)
            local_times = times[left:right] - times[local_pos]
            weights = np.exp(-0.5 * (local_times / bandwidth) ** 2)
            design = np.stack([np.ones_like(local_times), local_times], axis=1)
            for dim in range(values.shape[1]):
                y = values[indices[left:right], dim]
                weighted = design * weights[:, None]
                coef = np.linalg.pinv(weighted.T @ design + 1e-4 * np.eye(2)) @ weighted.T @ y
                out[global_index, dim] = coef[0]
    return clip(out)


def gaussian_time_kernel(values: np.ndarray, sample_ids: list[str], sigma: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        n = len(indices)
        t = np.arange(n, dtype=np.float32)
        dist = (t[:, None] - t[None, :]) ** 2
        kernel = np.exp(-0.5 * dist / (sigma**2))
        kernel = kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-6)
        out[indices] = kernel @ values[indices]
    return clip(out)


def hampel_median_filter(values: np.ndarray, sample_ids: list[str], window: int, threshold: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    size = (window, 1)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        med = median_filter(z, size=size, mode="nearest")
        mad = median_filter(np.abs(z - med), size=size, mode="nearest") + 1e-6
        mask = np.abs(z - med) > threshold * 1.4826 * mad
        out[indices] = np.where(mask, med, z)
    return clip(out)


def total_variation_denoise(values: np.ndarray, sample_ids: list[str], weight: float, iterations: int) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        x = values[indices].astype(np.float32).copy()
        z = x.copy()
        for _ in range(iterations):
            lap = np.zeros_like(z)
            lap[1:-1] = z[:-2] - 2 * z[1:-1] + z[2:]
            z = z + weight * lap
            z[0] = 0.85 * z[0] + 0.15 * x[0]
            z[-1] = 0.85 * z[-1] + 0.15 * x[-1]
        out[indices] = z
    return clip(out)


def haar_wavelet_shrink(values: np.ndarray, sample_ids: list[str], threshold_scale: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        for dim in range(values.shape[1]):
            out[indices, dim] = haar_denoise_1d(values[indices, dim], threshold_scale)
    return clip(out)


def haar_denoise_1d(x: np.ndarray, threshold_scale: float) -> np.ndarray:
    n = len(x)
    m = 1
    while m * 2 <= n:
        m *= 2
    base = x[:m].astype(np.float32).copy()
    coeffs = []
    current = base
    while len(current) >= 2:
        avg = (current[0::2] + current[1::2]) / 2.0
        detail = (current[0::2] - current[1::2]) / 2.0
        coeffs.append(detail)
        current = avg
    sigma = np.median(np.abs(coeffs[0])) / 0.6745 if coeffs else 0.0
    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(max(m, 2)))
    current = current.copy()
    for detail in reversed(coeffs):
        shrunk = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        up = np.empty(len(shrunk) * 2, dtype=np.float32)
        up[0::2] = current + shrunk
        up[1::2] = current - shrunk
        current = up
    result = x.copy().astype(np.float32)
    result[:m] = current
    if m < n:
        result[m:] = x[m:]
    return result


def piecewise_knot_projection(values: np.ndarray, sample_ids: list[str], knot_count: int, blend: float) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        n = len(indices)
        knots = np.linspace(0, n - 1, knot_count).astype(int)
        for dim in range(values.shape[1]):
            trend = np.interp(np.arange(n), knots, values[np.asarray(indices)[knots], dim])
            out[indices, dim] = (1.0 - blend) * values[indices, dim] + blend * trend
    return clip(out)


def hmm_viterbi_decode(
    emission_values: np.ndarray,
    sample_ids: list[str],
    train_ids: list[str],
    y_train: np.ndarray,
    blend: float,
) -> np.ndarray:
    states = make_hmm_states(y_train, bins=5)
    transition = train_transition_logprob(train_ids, y_train, states)
    out = emission_values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        emissions = emission_values[indices]
        path = viterbi_path(emissions, states, transition, emission_sigma=22.0)
        decoded = states[path]
        out[indices] = (1.0 - blend) * emissions + blend * decoded
    return clip(out)


def make_hmm_states(y_train: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(5, 95, bins)
    val = np.percentile(y_train[:, 0], qs)
    aro = np.percentile(y_train[:, 1], qs)
    grid = np.asarray([[v, a] for v in val for a in aro], dtype=np.float32)
    return grid


def train_transition_logprob(train_ids: list[str], y_train: np.ndarray, states: np.ndarray) -> np.ndarray:
    n_states = len(states)
    counts = np.ones((n_states, n_states), dtype=np.float32) * 0.2
    labels = nearest_state(y_train, states)
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(train_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        ordered = [index for _, index in sorted(items)]
        for left, right in zip(ordered[:-1], ordered[1:]):
            counts[labels[left], labels[right]] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs + 1e-8)


def nearest_state(values: np.ndarray, states: np.ndarray) -> np.ndarray:
    distances = ((values[:, None, :] - states[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def viterbi_path(emissions: np.ndarray, states: np.ndarray, transition: np.ndarray, emission_sigma: float) -> np.ndarray:
    n = len(emissions)
    n_states = len(states)
    emission_log = -((emissions[:, None, :] - states[None, :, :]) ** 2).sum(axis=2) / (2 * emission_sigma**2)
    dp = np.zeros((n, n_states), dtype=np.float32)
    back = np.zeros((n, n_states), dtype=np.int32)
    dp[0] = emission_log[0]
    for t in range(1, n):
        scores = dp[t - 1][:, None] + transition
        back[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + emission_log[t]
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def anchored_graph_smoother(values: np.ndarray, sample_ids: list[str], lam: float, steps: int) -> np.ndarray:
    out = values.copy().astype(np.float32)
    source = values.astype(np.float32)
    for _ in range(steps):
        diffused = temporal_graph_diffusion(out, sample_ids, lam=0.18, steps=1)
        out = (1.0 - lam) * diffused + lam * source
    return clip(out)


def local_attention_smoother(
    values: np.ndarray,
    sample_ids: list[str],
    radius: int,
    sigma_time: float,
    sigma_value: float,
) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        n = len(indices)
        for i, global_index in enumerate(indices):
            left = max(0, i - radius)
            right = min(n, i + radius + 1)
            dt = np.arange(left, right, dtype=np.float32) - i
            dv = np.linalg.norm(z[left:right] - z[i][None, :], axis=1)
            weights = np.exp(-0.5 * (dt / sigma_time) ** 2 - 0.5 * (dv / sigma_value) ** 2)
            weights = weights / max(weights.sum(), 1e-6)
            out[global_index] = weights @ z[left:right]
    return clip(out)


def kmeans_residual_expert(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    model = MiniBatchKMeans(n_clusters=32, random_state=seed, batch_size=2048, n_init=3)
    clusters = model.fit_predict(x_train_scaled)
    residual_means = np.zeros((32, 2), dtype=np.float32)
    for cluster in range(32):
        mask = clusters == cluster
        if mask.any():
            residual_means[cluster] = residual_target[mask].mean(axis=0)
    val_clusters = model.predict(x_val_scaled)
    return clip(prior_val + 0.05 * residual_means[val_clusters])


def gmm_residual_expert(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    pca = PCA(n_components=12, random_state=seed)
    train_latent = pca.fit_transform(scaler.fit_transform(x_train))
    val_latent = pca.transform(scaler.transform(x_val))
    model = GaussianMixture(n_components=10, covariance_type="diag", random_state=seed, reg_covar=1e-3)
    weights = model.fit(train_latent).predict_proba(train_latent)
    denom = weights.sum(axis=0)[:, None] + 1e-6
    component_residual = (weights.T @ residual_target) / denom
    val_weights = model.predict_proba(val_latent)
    residual = val_weights @ component_residual
    return clip(prior_val + 0.05 * residual)


def rff_kernel_residual(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=0.025, n_components=256, random_state=seed),
        Ridge(alpha=120.0),
    )
    model.fit(x_train, residual_target)
    residual = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(prior_val + 0.05 * residual)


def pca_latent_residual(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    pca = PCA(n_components=16, random_state=0)
    train_latent = pca.fit_transform(scaler.fit_transform(x_train))
    val_latent = pca.transform(scaler.transform(x_val))
    ridge = Ridge(alpha=80.0)
    ridge.fit(train_latent, residual_target)
    residual = ridge.predict(val_latent)
    return clip(prior_val + 0.05 * residual)


def quantile_distribution_calibration(prior_train: np.ndarray, y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for dim in range(2):
        transformer_x = QuantileTransformer(n_quantiles=128, output_distribution="uniform", random_state=0)
        transformer_y = QuantileTransformer(n_quantiles=128, output_distribution="uniform", random_state=1)
        transformer_x.fit(prior_train[:, [dim]])
        transformer_y.fit(y_train[:, [dim]])
        quantiles = transformer_x.transform(pred[:, [dim]])
        mapped = transformer_y.inverse_transform(np.clip(quantiles, 0.0, 1.0))
        out[:, dim] = 0.85 * pred[:, dim] + 0.15 * mapped[:, 0]
    return clip(out)


def isotonic_calibration(prior_train: np.ndarray, y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for dim in range(2):
        iso = IsotonicRegression(out_of_bounds="clip")
        order = np.argsort(prior_train[:, dim])
        iso.fit(prior_train[order, dim], y_train[order, dim])
        mapped = iso.predict(pred[:, dim])
        out[:, dim] = 0.88 * pred[:, dim] + 0.12 * mapped
    return clip(out)


def huber_residual_calibration(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(HuberRegressor(epsilon=1.25, alpha=0.004, max_iter=260)),
    )
    model.fit(x_train, residual_target)
    residual = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(prior_val + 0.035 * residual)


def mlp_residual_architecture(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(48,),
            activation="tanh",
            alpha=0.08,
            learning_rate_init=0.002,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=seed,
        ),
    )
    model.fit(x_train, residual_target)
    residual = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(prior_val + 0.035 * residual)


def gradient_boosted_residual(
    x_train: np.ndarray,
    residual_target: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            loss="absolute_error",
            n_estimators=80,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=120,
            random_state=seed,
        )
    )
    model.fit(x_train, residual_target)
    residual = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(prior_val + 0.05 * residual)


def linear_stacking_architecture(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    prior_val: np.ndarray,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=250.0))
    model.fit(x_train, y_train)
    direct = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(0.94 * prior_val + 0.06 * direct)


def heterogeneous_rule_moe(
    previous_125: np.ndarray,
    candidate_std: np.ndarray,
    candidate_mean: np.ndarray,
    p126: np.ndarray,
    p130: np.ndarray,
    p133: np.ndarray,
    p134: np.ndarray,
    p139: np.ndarray,
) -> np.ndarray:
    gate = uncertainty_gate(candidate_std, q_low=45, q_high=85)
    pred = previous_125.copy().astype(np.float32)
    pred[:, 0] = np.where(gate[:, 0] < 0.35, p126[:, 0], p134[:, 0])
    pred[:, 1] = np.where(gate[:, 1] < 0.35, p133[:, 1], p139[:, 1])
    very_uncertain = gate > 0.82
    pred = np.where(very_uncertain, 0.80 * pred + 0.20 * candidate_mean, pred)
    return clip(pred)


def min_by_proxy(predictions: list[np.ndarray], candidate_std: np.ndarray) -> np.ndarray:
    # Use candidate disagreement as a label-free risk proxy: choose the model closest to the
    # low-uncertainty candidate center per sample/dimension.
    center = np.mean(np.stack(predictions, axis=0), axis=0)
    distances = np.stack([np.abs(pred - center) for pred in predictions], axis=0)
    weights = 1.0 / (1.0 + candidate_std[None, :, :])
    risk = distances * weights
    choice = risk.argmin(axis=0)
    out = np.zeros_like(predictions[0], dtype=np.float32)
    for model_index, pred in enumerate(predictions):
        out = np.where(choice == model_index, pred, out)
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
                    "method": f"146_DimwiseBatch2_V[{v_row['method']}]__A[{a_row['method']}]",
                    "overall_mae": round(float(overall), 4),
                    "valence_mae": round(float(v_row["valence_mae"]), 4),
                    "arousal_mae": round(float(a_row["arousal_mae"]), 4),
                    "overall_mse": None,
                    "notes": "Metric-composed integration after the 126-145 distinct architecture batch.",
                }
            )
    return sorted(results, key=lambda item: float(item["overall_mae"]))


if __name__ == "__main__":
    main()
