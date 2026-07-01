from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import fractional_matrix_power
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import clip, make_reference_104, trial_indices  # noqa: E402
from tools.cross_fold_batch3_architectures import (  # noqa: E402
    compose_dimwise_integrations,
    downsample_trial,
    gaussian_copula_covariance_calibration,
    make_previous_125,
    prior_bin_bias_calibration,
    resize_trial,
    trial_prototype_residual_retrieval,
    uncertainty_gate,
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
    parser = argparse.ArgumentParser(description="Experiments 168-200: architecture families and milestone synthesis.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_168_200_architectures.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=220)
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

        print(f"[fold {fold_index}] fitting 104/125/167 references and 168-200 architectures", flush=True)
        ref104, _, _ = make_reference_104(
            x_train=x_train,
            residual_target=residual_target,
            x_val=x_val,
            prior_val=prior_val,
            val_ids=val_ids,
            seed=args.seed + fold_index * 173,
        )
        previous_125 = make_previous_125(ref104, val_ids)
        previous_167 = make_previous_167(
            previous_125=previous_125,
            oof_train_ids=oof_train_ids,
            y_train=y_train,
            prior_train=prior_train,
            residual_target=residual_target,
            val_ids=val_ids,
        )

        fold_predictions: dict[str, np.ndarray] = {
            "098_PatternPrior": prior_val,
            "104_DimwiseOOFMeta_reference": ref104,
            "125_PreviousStateFusion_reference": previous_125,
            "167_PreviousBest_reference": previous_167,
        }
        add_to200_architectures(
            fold_predictions=fold_predictions,
            train_ids=oof_train_ids,
            val_ids=val_ids,
            y_train=y_train,
            prior_train=prior_train,
            residual_target=residual_target,
            prior_val=prior_val,
            previous_125=previous_125,
            previous_167=previous_167,
            candidate_stack=candidate_stack,
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            x_train=x_train,
            x_val=x_val,
            seed=args.seed + fold_index * 1117,
        )

        fold_results = sorted(
            [score(name, y_val, pred, "Experiments 168-200 architecture model.") for name, pred in fold_predictions.items()],
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
    dimwise_integration = compose_dimwise_integrations(aggregate_results, top_n=40)

    output = {
        "method": "Experiments 168-200: new architectures, 188 batch integration, and 200 milestone synthesis",
        "note": "168-187 are the next 20 architecture families; 188 integrates that batch; 189-199 are additional distinct families; 200 is the milestone synthesis.",
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
        "168": "Piecewise affine prior-value calibration",
        "169": "Monotone PCHIP residual-bias calibration",
        "170": "Video-time residual surface correction",
        "171": "Lag-aligned trial prototype retrieval",
        "172": "Uncertainty-gated calibration/prototype router",
        "173": "Linear covariance transport calibration",
        "174": "Uncertainty-aware Kalman smoother",
        "175": "Student-t robust Kalman smoother",
        "176": "Chebyshev temporal graph filter",
        "177": "Lagged ridge temporal-convolution residual decoder",
        "178": "Landmark RBF residual decoder",
        "179": "Video-level residual prior correction",
        "180": "Temporal-rank calibration",
        "181": "Deterministic feature-bagging ridge residual ensemble",
        "182": "ExtraTrees residual decoder",
        "183": "GaussianNB residual-bin expectation decoder",
        "184": "Nystroem kernel ridge residual decoder",
        "185": "Robust Mahalanobis KNN residual retrieval",
        "186": "PLS latent residual decoder",
        "187": "Temporal basis ridge residual model",
        "188": "Batch-4 fixed integration",
        "189": "Adaptive uncertainty candidate mixture",
        "190": "Huberized slope limiter",
        "191": "Residual sign-memory correction",
        "192": "Low-rank video-time residual factorization",
        "193": "Per-video affine copula-lite calibration",
        "194": "Identity-mixed Wasserstein histogram calibration",
        "195": "Conformal median-band projector",
        "196": "Derivative-template residual retrieval",
        "197": "Dimension-coupled arousal-from-valence correction",
        "198": "Multi-resolution expert switch",
        "199": "Jackknife uncertainty shrinkage",
        "200": "Milestone synthesis fusion",
    }


def make_previous_167(
    previous_125: np.ndarray,
    oof_train_ids: list[str],
    y_train: np.ndarray,
    prior_train: np.ndarray,
    residual_target: np.ndarray,
    val_ids: list[str],
) -> np.ndarray:
    p158 = gaussian_copula_covariance_calibration(y_train, previous_125)
    p159 = trial_prototype_residual_retrieval(
        oof_train_ids=oof_train_ids,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=val_ids,
        val_pred=previous_125,
    )
    pred = previous_125.copy()
    pred[:, 0] = p158[:, 0]
    pred[:, 1] = p159[:, 1]
    return clip(pred)


def add_to200_architectures(
    fold_predictions: dict[str, np.ndarray],
    train_ids: list[str],
    val_ids: list[str],
    y_train: np.ndarray,
    prior_train: np.ndarray,
    residual_target: np.ndarray,
    prior_val: np.ndarray,
    previous_125: np.ndarray,
    previous_167: np.ndarray,
    candidate_stack: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    seed: int,
) -> None:
    def log(name: str) -> None:
        print(f"    building {name}", flush=True)

    log("157/168-173 calibration and retrieval family")
    p157 = prior_bin_bias_calibration(prior_train, y_train, previous_125)

    p168 = piecewise_affine_calibration(prior_train, y_train, previous_167)
    p169 = monotone_pchip_bias_calibration(prior_train, residual_target, previous_167)
    p170 = video_time_residual_surface(train_ids, residual_target, val_ids, previous_167)
    p171 = lag_aligned_trial_prototype(train_ids, prior_train, residual_target, val_ids, previous_167)
    p172 = uncertainty_gated_router(previous_167, p157, p169, p171, candidate_std)
    p173 = covariance_transport(y_train, previous_167)
    log("174-180 state, graph, kernel, and temporal families")
    p174 = kalman_uncertainty_smoother(previous_167, val_ids, candidate_std)
    p175 = student_t_kalman_smoother(previous_167, val_ids, candidate_std)
    p176 = chebyshev_graph_filter(previous_167, val_ids)
    p177 = lagged_ridge_decoder(train_ids, val_ids, x_train, x_val, prior_train, prior_val, residual_target, previous_167)
    p178 = landmark_rbf_decoder(x_train, x_val, residual_target, previous_167, seed)
    p179 = video_level_residual_prior(train_ids, residual_target, val_ids, previous_167)
    p180 = temporal_rank_calibration(train_ids, y_train, val_ids, previous_167)
    log("181-188 feature decoders and fixed integration")
    p181 = feature_bagging_ridge(x_train, x_val, residual_target, previous_167, seed)
    p182 = extratrees_residual_decoder(x_train, x_val, residual_target, previous_167, seed)
    p183 = gaussian_nb_residual_bins(x_train, x_val, residual_target, previous_167)
    p184 = nystroem_kernel_ridge(x_train, x_val, residual_target, previous_167, seed)
    p185 = mahalanobis_knn_residual(x_train, x_val, residual_target, previous_167)
    p186 = pls_latent_residual(x_train, x_val, residual_target, previous_167)
    p187 = temporal_basis_ridge(train_ids, val_ids, residual_target, previous_167)

    p188 = previous_167.copy()
    gate = uncertainty_gate(candidate_std, 45, 82)
    p188[:, 0] = (1.0 - 0.35 * gate[:, 0]) * p169[:, 0] + (0.35 * gate[:, 0]) * p173[:, 0]
    p188[:, 1] = (1.0 - 0.30 * gate[:, 1]) * p171[:, 1] + (0.30 * gate[:, 1]) * p174[:, 1]

    log("189-200 final distinct families and milestone synthesis")
    p189 = adaptive_uncertainty_candidate_mixture(previous_167, candidate_mean, candidate_std)
    p190 = huberized_slope_limiter(previous_167, val_ids, train_ids, y_train)
    p191 = residual_sign_memory(train_ids, residual_target, val_ids, previous_167)
    p192 = low_rank_video_time_factor(train_ids, residual_target, val_ids, previous_167)
    p193 = per_video_affine_calibration(train_ids, prior_train, y_train, val_ids, previous_167)
    p194 = identity_mixed_histogram_calibration(y_train, previous_167)
    p195 = conformal_median_band_projector(train_ids, y_train, val_ids, previous_167)
    p196 = derivative_template_retrieval(train_ids, prior_train, residual_target, val_ids, previous_167)
    p197 = dimension_coupled_arousal(y_train, previous_167)
    p198 = multiresolution_expert_switch(previous_167, [p169, p174, p176, p188], val_ids, candidate_std)
    p199 = jackknife_uncertainty_shrink(previous_167, candidate_stack, candidate_std)

    p200 = previous_167.copy()
    p200[:, 0] = p188[:, 0]
    p200[:, 1] = p195[:, 1]

    fold_predictions.update(
        {
            "168_PiecewiseAffineCalibration": clip(p168),
            "169_MonotonePCHIPBiasCalibration": clip(p169),
            "170_VideoTimeResidualSurface": clip(p170),
            "171_LagAlignedTrialPrototype": clip(p171),
            "172_UncertaintyGatedRouter": clip(p172),
            "173_CovarianceTransportCalibration": clip(p173),
            "174_KalmanUncertaintySmoother": clip(p174),
            "175_StudentTRobustKalmanSmoother": clip(p175),
            "176_ChebyshevTemporalGraphFilter": clip(p176),
            "177_LaggedRidgeTCNResidual": clip(p177),
            "178_LandmarkRBFResidualDecoder": clip(p178),
            "179_VideoLevelResidualPrior": clip(p179),
            "180_TemporalRankCalibration": clip(p180),
            "181_FeatureBaggingRidgeResidual": clip(p181),
            "182_ExtraTreesResidualDecoder": clip(p182),
            "183_GaussianNBResidualBinExpectation": clip(p183),
            "184_NystroemKernelRidgeResidual": clip(p184),
            "185_MahalanobisKNNResidualRetrieval": clip(p185),
            "186_PLSLatentResidualDecoder": clip(p186),
            "187_TemporalBasisRidgeResidual": clip(p187),
            "188_FixedBatch4Fusion": clip(p188),
            "189_AdaptiveUncertaintyCandidateMixture": clip(p189),
            "190_HuberizedSlopeLimiter": clip(p190),
            "191_ResidualSignMemoryCorrection": clip(p191),
            "192_LowRankVideoTimeResidualFactor": clip(p192),
            "193_PerVideoAffineCalibration": clip(p193),
            "194_IdentityMixedWassersteinCalibration": clip(p194),
            "195_ConformalMedianBandProjector": clip(p195),
            "196_DerivativeTemplateRetrieval": clip(p196),
            "197_DimensionCoupledArousalCorrection": clip(p197),
            "198_MultiResolutionExpertSwitch": clip(p198),
            "199_JackknifeUncertaintyShrinkage": clip(p199),
            "200_MilestoneSynthesisFusion_VBatch4_AConformalBand": clip(p200),
        }
    )


def piecewise_affine_calibration(prior_train: np.ndarray, y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for dim in range(2):
        x = prior_train[:, dim]
        y = y_train[:, dim]
        edges = np.unique(np.percentile(x, [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]))
        corrected = out[:, dim].copy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x <= hi)
            apply_mask = (pred[:, dim] >= lo) & (pred[:, dim] <= hi)
            if mask.sum() < 24 or not np.any(apply_mask):
                continue
            design = np.stack([np.ones(mask.sum(), dtype=np.float32), x[mask]], axis=1)
            coef = np.linalg.pinv(design) @ y[mask]
            local = coef[0] + coef[1] * pred[apply_mask, dim]
            corrected[apply_mask] = 0.82 * pred[apply_mask, dim] + 0.18 * local
        out[:, dim] = corrected
    return clip(out)


def monotone_pchip_bias_calibration(prior_train: np.ndarray, residual_target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for dim in range(2):
        x = prior_train[:, dim]
        r = residual_target[:, dim]
        qs = np.unique(np.percentile(x, [2, 8, 16, 28, 40, 52, 64, 76, 88, 96, 98]))
        if len(qs) < 4:
            continue
        medians = []
        for left, right in zip(qs[:-1], qs[1:]):
            mask = (x >= left) & (x <= right)
            medians.append(float(np.median(r[mask])) if mask.sum() else 0.0)
        knots = 0.5 * (qs[:-1] + qs[1:])
        try:
            interp = PchipInterpolator(knots, np.asarray(medians, dtype=np.float32), extrapolate=True)
            bias = np.clip(interp(pred[:, dim]), -3.0, 3.0)
            out[:, dim] = pred[:, dim] + bias
        except ValueError:
            continue
    return clip(out)


def video_time_residual_surface(train_ids: list[str], residual: np.ndarray, val_ids: list[str], pred: np.ndarray) -> np.ndarray:
    table: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    video_table: dict[int, list[np.ndarray]] = defaultdict(list)
    for sample_id, r in zip(train_ids, residual):
        _, video, timestamp = parse_sample_id(sample_id)
        table[(video, timestamp)].append(r)
        video_table[video].append(r)
    out = pred.copy().astype(np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, timestamp = parse_sample_id(sample_id)
        if table[(video, timestamp)]:
            corr = np.median(np.asarray(table[(video, timestamp)]), axis=0)
        elif video_table[video]:
            corr = np.median(np.asarray(video_table[video]), axis=0)
        else:
            corr = np.zeros(2, dtype=np.float32)
        out[index] = pred[index] + np.clip(0.22 * corr, -2.5, 2.5)
    return clip(out)


def lag_aligned_trial_prototype(
    train_ids: list[str],
    prior_train: np.ndarray,
    residual: np.ndarray,
    val_ids: list[str],
    pred: np.ndarray,
) -> np.ndarray:
    train_trials = build_trial_library(train_ids, prior_train, residual)
    out = pred.copy().astype(np.float32)
    for indices in trial_indices(val_ids):
        query = downsample_trial(pred[indices], bins=10)
        best: list[tuple[float, np.ndarray]] = []
        for item in train_trials:
            proto = item["prior_ds"]
            best_dist = np.inf
            best_residual = item["residual"]
            for lag in (-2, -1, 0, 1, 2):
                shifted = np.roll(proto, lag, axis=0)
                dist = float(np.mean(np.abs(query - shifted)))
                if dist < best_dist:
                    best_dist = dist
                    best_residual = item["residual"]
            best.append((best_dist, resize_trial(best_residual, len(indices))))
        best = sorted(best, key=lambda item: item[0])[:5]
        weights = np.asarray([1.0 / (1.0 + item[0]) for item in best], dtype=np.float32)
        weights = weights / np.maximum(weights.sum(), 1e-6)
        correction = sum(weight * item[1] for weight, item in zip(weights, best))
        out[indices] = pred[indices] + np.clip(0.16 * correction, -2.2, 2.2)
    return clip(out)


def uncertainty_gated_router(
    base: np.ndarray,
    bin_calibrated: np.ndarray,
    pchip: np.ndarray,
    prototype: np.ndarray,
    candidate_std: np.ndarray,
) -> np.ndarray:
    gate = uncertainty_gate(candidate_std, 35, 88)
    out = base.copy().astype(np.float32)
    out[:, 0] = (1.0 - gate[:, 0]) * pchip[:, 0] + gate[:, 0] * bin_calibrated[:, 0]
    out[:, 1] = (1.0 - 0.5 * gate[:, 1]) * prototype[:, 1] + (0.5 * gate[:, 1]) * base[:, 1]
    return clip(out)


def covariance_transport(y_train: np.ndarray, pred: np.ndarray) -> np.ndarray:
    source_mean = pred.mean(axis=0)
    target_mean = y_train.mean(axis=0)
    source_cov = np.cov(pred.T) + np.eye(2) * 1e-3
    target_cov = np.cov(y_train.T) + np.eye(2) * 1e-3
    try:
        transform = fractional_matrix_power(target_cov, 0.5) @ fractional_matrix_power(source_cov, -0.5)
        transported = (pred - source_mean[None, :]) @ np.real(transform).T + target_mean[None, :]
        return clip(0.84 * pred + 0.16 * transported)
    except Exception:
        return pred


def kalman_uncertainty_smoother(values: np.ndarray, sample_ids: list[str], candidate_std: np.ndarray) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        state = values[indices[0]].astype(np.float32)
        velocity = np.zeros(2, dtype=np.float32)
        for idx in indices:
            measurement = values[idx]
            uncertainty = candidate_std[idx]
            gain = 1.0 / (1.0 + 0.035 * uncertainty)
            predicted = state + 0.35 * velocity
            innovation = measurement - predicted
            state = predicted + gain * innovation
            velocity = 0.80 * velocity + 0.20 * innovation
            out[idx] = state
    return clip(out)


def student_t_kalman_smoother(values: np.ndarray, sample_ids: list[str], candidate_std: np.ndarray) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        state = values[indices[0]].astype(np.float32)
        scale = np.maximum(np.median(candidate_std[indices], axis=0), 1.0)
        for idx in indices:
            innovation = values[idx] - state
            robust_gain = 1.0 / (1.0 + (innovation / (2.5 * scale)) ** 2)
            state = state + 0.42 * robust_gain * innovation
            out[idx] = 0.70 * values[idx] + 0.30 * state
    return clip(out)


def chebyshev_graph_filter(values: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for indices in trial_indices(sample_ids):
        z = values[indices]
        if len(z) < 5:
            continue
        lap = np.zeros_like(z)
        lap[1:-1] = 2.0 * z[1:-1] - z[:-2] - z[2:]
        lap2 = np.zeros_like(z)
        lap2[1:-1] = 2.0 * lap[1:-1] - lap[:-2] - lap[2:]
        out[indices] = z - 0.10 * lap + 0.025 * lap2
    return clip(out)


def lagged_ridge_decoder(
    train_ids: list[str],
    val_ids: list[str],
    x_train: np.ndarray,
    x_val: np.ndarray,
    prior_train: np.ndarray,
    prior_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    train_feat = append_lagged_prediction_features(train_ids, x_train, prior_train)
    val_feat = append_lagged_prediction_features(val_ids, x_val, prior_val)
    model = make_pipeline(StandardScaler(), Ridge(alpha=900.0))
    model.fit(train_feat, residual)
    corr = np.asarray(model.predict(val_feat), dtype=np.float32)
    return clip(base + np.clip(0.12 * corr, -2.5, 2.5))


def landmark_rbf_decoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_landmarks = min(96, len(x_train))
    landmarks = x_train[rng.choice(len(x_train), size=n_landmarks, replace=False)]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    val_scaled = scaler.transform(x_val)
    landmark_scaled = scaler.transform(landmarks)
    gamma = 1.0 / max(train_scaled.shape[1], 1)
    train_phi = np.exp(-gamma * ((train_scaled[:, None, :] - landmark_scaled[None, :, :]) ** 2).sum(axis=2))
    val_phi = np.exp(-gamma * ((val_scaled[:, None, :] - landmark_scaled[None, :, :]) ** 2).sum(axis=2))
    model = Ridge(alpha=800.0)
    model.fit(train_phi, residual)
    corr = np.asarray(model.predict(val_phi), dtype=np.float32)
    return clip(base + np.clip(0.10 * corr, -2.0, 2.0))


def video_level_residual_prior(train_ids: list[str], residual: np.ndarray, val_ids: list[str], base: np.ndarray) -> np.ndarray:
    table: dict[int, list[np.ndarray]] = defaultdict(list)
    for sample_id, r in zip(train_ids, residual):
        _, video, _ = parse_sample_id(sample_id)
        table[video].append(r)
    out = base.copy().astype(np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, _ = parse_sample_id(sample_id)
        if table[video]:
            corr = np.mean(np.asarray(table[video]), axis=0)
            out[index] = base[index] + np.clip(0.18 * corr, -2.0, 2.0)
    return clip(out)


def temporal_rank_calibration(train_ids: list[str], y_train: np.ndarray, val_ids: list[str], base: np.ndarray) -> np.ndarray:
    by_time: dict[int, list[np.ndarray]] = defaultdict(list)
    for sample_id, y in zip(train_ids, y_train):
        _, _, timestamp = parse_sample_id(sample_id)
        by_time[timestamp].append(y)
    out = base.copy().astype(np.float32)
    global_median = np.median(y_train, axis=0)
    for index, sample_id in enumerate(val_ids):
        _, _, timestamp = parse_sample_id(sample_id)
        if by_time[timestamp]:
            target = np.median(np.asarray(by_time[timestamp]), axis=0)
        else:
            target = global_median
        out[index] = 0.92 * base[index] + 0.08 * target
    return clip(out)


def feature_bagging_ridge(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corrections = []
    width = max(12, int(x_train.shape[1] * 0.45))
    for _ in range(7):
        columns = np.sort(rng.choice(x_train.shape[1], size=width, replace=False))
        model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
        model.fit(x_train[:, columns], residual)
        corrections.append(np.asarray(model.predict(x_val[:, columns]), dtype=np.float32))
    corr = np.mean(corrections, axis=0)
    return clip(base + np.clip(0.10 * corr, -2.0, 2.0))


def extratrees_residual_decoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=96,
        max_depth=6,
        min_samples_leaf=80,
        max_features=0.45,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, residual)
    corr = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(base + np.clip(0.08 * corr, -2.0, 2.0))


def gaussian_nb_residual_bins(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xva = scaler.transform(x_val)
    corr = np.zeros_like(base, dtype=np.float32)
    bins = np.asarray([-4.0, -1.5, 0.0, 1.5, 4.0], dtype=np.float32)
    centers = np.asarray([-3.0, -0.75, 0.75, 3.0], dtype=np.float32)
    for dim in range(2):
        labels = np.digitize(np.clip(residual[:, dim], bins[0], bins[-1]), bins[1:-1])
        clf = GaussianNB(var_smoothing=1e-1)
        clf.fit(xtr, labels)
        prob = clf.predict_proba(xva)
        corr[:, dim] = prob @ centers[: prob.shape[1]]
    return clip(base + np.clip(0.10 * corr, -1.5, 1.5))


def nystroem_kernel_ridge(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        Nystroem(kernel="rbf", gamma=0.012, n_components=128, random_state=seed),
        Ridge(alpha=900.0),
    )
    model.fit(x_train, residual)
    corr = np.asarray(model.predict(x_val), dtype=np.float32)
    return clip(base + np.clip(0.08 * corr, -1.8, 1.8))


def mahalanobis_knn_residual(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train).astype(np.float32)
    xva = scaler.transform(x_val).astype(np.float32)
    scale = np.sqrt(np.var(xtr, axis=0) + 0.05).astype(np.float32)
    xtr = xtr / scale[None, :]
    xva = xva / scale[None, :]
    index = NearestNeighbors(n_neighbors=8, algorithm="auto", metric="euclidean", n_jobs=-1)
    index.fit(xtr)
    distances, nn = index.kneighbors(xva, return_distance=True)
    weights = 1.0 / (1.0 + distances.astype(np.float32))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    out_corr = np.sum(weights[:, :, None] * residual[nn], axis=1)
    return clip(base + np.clip(0.07 * out_corr, -1.6, 1.6))


def pls_latent_residual(
    x_train: np.ndarray,
    x_val: np.ndarray,
    residual: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xva = scaler.transform(x_val)
    model = PLSRegression(n_components=min(8, xtr.shape[1] - 1))
    model.fit(xtr, residual)
    corr = np.asarray(model.predict(xva), dtype=np.float32)
    return clip(base + np.clip(0.07 * corr, -1.6, 1.6))


def temporal_basis_ridge(
    train_ids: list[str],
    val_ids: list[str],
    residual: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    train_feat = temporal_basis_features(train_ids)
    val_feat = temporal_basis_features(val_ids)
    model = Ridge(alpha=300.0)
    model.fit(train_feat, residual)
    corr = np.asarray(model.predict(val_feat), dtype=np.float32)
    return clip(base + np.clip(0.12 * corr, -2.2, 2.2))


def adaptive_uncertainty_candidate_mixture(
    base: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> np.ndarray:
    gate = uncertainty_gate(candidate_std, 25, 90)
    median_target = 0.65 * base + 0.35 * candidate_mean
    return clip((1.0 - 0.22 * gate) * base + (0.22 * gate) * median_target)


def huberized_slope_limiter(
    base: np.ndarray,
    val_ids: list[str],
    train_ids: list[str],
    y_train: np.ndarray,
) -> np.ndarray:
    train_slopes = np.abs(prior_slope_by_trial(train_ids, y_train))
    limit = np.percentile(train_slopes, 92, axis=0) + 1.0
    out = base.copy().astype(np.float32)
    for indices in trial_indices(val_ids):
        for pos in range(1, len(indices)):
            prev = out[indices[pos - 1]]
            raw_delta = base[indices[pos]] - prev
            clipped_delta = limit * np.tanh(raw_delta / np.maximum(limit, 1e-3))
            out[indices[pos]] = prev + 0.65 * raw_delta + 0.35 * clipped_delta
    return clip(out)


def residual_sign_memory(train_ids: list[str], residual: np.ndarray, val_ids: list[str], base: np.ndarray) -> np.ndarray:
    table: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for sample_id, r in zip(train_ids, residual):
        _, video, timestamp = parse_sample_id(sample_id)
        table[(video, timestamp)].append(np.sign(r))
    out = base.copy().astype(np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, timestamp = parse_sample_id(sample_id)
        signs = table.get((video, timestamp), [])
        if signs:
            direction = np.mean(np.asarray(signs), axis=0)
            out[index] = base[index] + 0.45 * direction
    return clip(out)


def low_rank_video_time_factor(train_ids: list[str], residual: np.ndarray, val_ids: list[str], base: np.ndarray) -> np.ndarray:
    max_video = max(parse_sample_id(sample_id)[1] for sample_id in train_ids + val_ids)
    max_time = max(parse_sample_id(sample_id)[2] for sample_id in train_ids + val_ids) + 1
    out = base.copy().astype(np.float32)
    for dim in range(2):
        matrix = np.zeros((max_video + 1, max_time), dtype=np.float32)
        count = np.zeros_like(matrix)
        for sample_id, r in zip(train_ids, residual):
            _, video, timestamp = parse_sample_id(sample_id)
            matrix[video, timestamp] += r[dim]
            count[video, timestamp] += 1
        observed = count > 0
        matrix[observed] /= count[observed]
        matrix[~observed] = np.median(matrix[observed]) if np.any(observed) else 0.0
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        low_rank = (u[:, :2] * s[:2]) @ vh[:2]
        for index, sample_id in enumerate(val_ids):
            _, video, timestamp = parse_sample_id(sample_id)
            out[index, dim] += np.clip(0.12 * low_rank[video, timestamp], -1.8, 1.8)
    return clip(out)


def per_video_affine_calibration(
    train_ids: list[str],
    prior_train: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
    base: np.ndarray,
) -> np.ndarray:
    coefs: dict[tuple[int, int], np.ndarray] = {}
    for video in sorted({parse_sample_id(sample_id)[1] for sample_id in train_ids}):
        mask = np.asarray([parse_sample_id(sample_id)[1] == video for sample_id in train_ids])
        for dim in range(2):
            if mask.sum() < 64:
                coefs[(video, dim)] = np.asarray([0.0, 1.0], dtype=np.float32)
                continue
            design = np.stack([np.ones(mask.sum(), dtype=np.float32), prior_train[mask, dim]], axis=1)
            coef = np.linalg.pinv(design) @ y_train[mask, dim]
            coefs[(video, dim)] = coef.astype(np.float32)
    out = base.copy().astype(np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, _ = parse_sample_id(sample_id)
        for dim in range(2):
            coef = coefs.get((video, dim), np.asarray([0.0, 1.0], dtype=np.float32))
            local = coef[0] + coef[1] * base[index, dim]
            out[index, dim] = 0.90 * base[index, dim] + 0.10 * local
    return clip(out)


def identity_mixed_histogram_calibration(y_train: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = base.copy().astype(np.float32)
    probs = np.linspace(2, 98, 25)
    for dim in range(2):
        source = np.percentile(base[:, dim], probs)
        target = np.percentile(y_train[:, dim], probs)
        mapped = np.interp(base[:, dim], source, target)
        out[:, dim] = 0.90 * base[:, dim] + 0.10 * mapped
    return clip(out)


def conformal_median_band_projector(train_ids: list[str], y_train: np.ndarray, val_ids: list[str], base: np.ndarray) -> np.ndarray:
    table: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for sample_id, y in zip(train_ids, y_train):
        _, video, timestamp = parse_sample_id(sample_id)
        bucket = timestamp // 8
        table[(video, bucket)].append(y)
    out = base.copy().astype(np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, timestamp = parse_sample_id(sample_id)
        bucket = timestamp // 8
        vals = table.get((video, bucket), [])
        if len(vals) < 8:
            continue
        arr = np.asarray(vals)
        low = np.percentile(arr, 10, axis=0)
        high = np.percentile(arr, 90, axis=0)
        projected = np.minimum(np.maximum(base[index], low), high)
        out[index] = 0.88 * base[index] + 0.12 * projected
    return clip(out)


def derivative_template_retrieval(
    train_ids: list[str],
    prior_train: np.ndarray,
    residual: np.ndarray,
    val_ids: list[str],
    base: np.ndarray,
) -> np.ndarray:
    train_trials = build_trial_library(train_ids, prior_train, residual)
    out = base.copy().astype(np.float32)
    for indices in trial_indices(val_ids):
        q = np.diff(downsample_trial(base[indices], bins=10), axis=0)
        candidates = []
        for item in train_trials:
            p = np.diff(item["prior_ds"], axis=0)
            dist = float(np.mean(np.abs(q - p)))
            candidates.append((dist, resize_trial(item["residual"], len(indices))))
        top = sorted(candidates, key=lambda item: item[0])[:6]
        weights = np.asarray([1.0 / (1.0 + item[0]) for item in top], dtype=np.float32)
        weights = weights / np.maximum(weights.sum(), 1e-6)
        corr = sum(weight * item[1] for weight, item in zip(weights, top))
        out[indices] = base[indices] + np.clip(0.13 * corr, -2.0, 2.0)
    return clip(out)


def dimension_coupled_arousal(y_train: np.ndarray, base: np.ndarray) -> np.ndarray:
    centered = y_train - y_train.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    beta = float(cov[0, 1] / max(cov[0, 0], 1e-6))
    out = base.copy().astype(np.float32)
    valence_centered = base[:, 0] - np.median(base[:, 0])
    target = np.median(y_train[:, 1]) + beta * valence_centered
    out[:, 1] = 0.94 * base[:, 1] + 0.06 * target
    return clip(out)


def multiresolution_expert_switch(
    base: np.ndarray,
    experts: list[np.ndarray],
    val_ids: list[str],
    candidate_std: np.ndarray,
) -> np.ndarray:
    out = base.copy().astype(np.float32)
    slope = np.abs(prior_slope_by_trial(val_ids, base))
    uncertainty = uncertainty_gate(candidate_std, 30, 88)
    for dim in range(2):
        smooth_expert = experts[2][:, dim]
        calibration_expert = experts[0][:, dim]
        dynamic_expert = experts[3][:, dim]
        threshold = np.percentile(slope[:, dim], 65)
        use_dynamic = slope[:, dim] > threshold
        mixed = np.where(use_dynamic, dynamic_expert, calibration_expert)
        out[:, dim] = (1.0 - 0.18 * uncertainty[:, dim]) * mixed + (0.18 * uncertainty[:, dim]) * smooth_expert
    return clip(out)


def jackknife_uncertainty_shrink(
    base: np.ndarray,
    candidate_stack: np.ndarray,
    candidate_std: np.ndarray,
) -> np.ndarray:
    means = []
    for i in range(candidate_stack.shape[0]):
        leave_one = np.delete(candidate_stack, i, axis=0)
        means.append(leave_one.mean(axis=0))
    jack_std = np.std(np.stack(means, axis=0), axis=0)
    gate = uncertainty_gate(candidate_std + jack_std, 35, 92)
    center = np.median(candidate_stack, axis=0)
    return clip((1.0 - 0.16 * gate) * base + (0.16 * gate) * center)


def build_trial_library(
    sample_ids: list[str],
    prior: np.ndarray,
    residual: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, _ = parse_sample_id(sample_id)
        groups[(subject, video)].append(index)
    library = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: parse_sample_id(sample_ids[i])[2])
        prior_trial = prior[ordered]
        residual_trial = residual[ordered]
        library.append(
            {
                "prior_ds": downsample_trial(prior_trial, bins=10),
                "residual": residual_trial.astype(np.float32),
            }
        )
    return library


def append_lagged_prediction_features(
    sample_ids: list[str],
    features: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    lagged = np.zeros((len(sample_ids), 8), dtype=np.float32)
    for indices in trial_indices(sample_ids):
        z = prediction[indices]
        slopes = np.vstack([np.zeros((1, 2), dtype=np.float32), np.diff(z, axis=0)])
        prev = np.vstack([z[:1], z[:-1]])
        nxt = np.vstack([z[1:], z[-1:]])
        lagged[indices] = np.concatenate([z, slopes, prev - z, nxt - z], axis=1)
    return np.concatenate([features, lagged], axis=1).astype(np.float32)


def temporal_basis_features(sample_ids: list[str]) -> np.ndarray:
    rows = []
    for sample_id in sample_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        t = timestamp / 127.0
        v = video / 15.0
        rows.append(
            [
                1.0,
                t,
                t * t,
                t * t * t,
                np.sin(np.pi * t),
                np.cos(np.pi * t),
                np.sin(2 * np.pi * t),
                np.cos(2 * np.pi * t),
                v,
                v * t,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


if __name__ == "__main__":
    main()
