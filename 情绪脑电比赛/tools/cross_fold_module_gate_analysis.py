from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import clip, make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125, uncertainty_gate  # noqa: E402
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
from tools.cross_fold_to200_architectures import (  # noqa: E402
    conformal_median_band_projector,
    covariance_transport,
    kalman_uncertainty_smoother,
    lag_aligned_trial_prototype,
    make_previous_167,
    monotone_pchip_bias_calibration,
)
from tools.run_iteration_experiments import expand_subjects, load_labels, score  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module mechanism analysis and OOF-safe gate candidates.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_201_module_gate_analysis.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=3031)
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
    diagnostics = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] build references and module gates", flush=True)
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

        p169 = monotone_pchip_bias_calibration(prior_train, residual_target, previous_167)
        p173 = covariance_transport(y_train, previous_167)
        p171 = lag_aligned_trial_prototype(oof_train_ids, prior_train, residual_target, val_ids, previous_167)
        p174 = kalman_uncertainty_smoother(previous_167, val_ids, candidate_std)
        p195 = conformal_median_band_projector(oof_train_ids, y_train, val_ids, previous_167)
        p188 = previous_167.copy()
        base_gate = uncertainty_gate(candidate_std, 45, 82)
        p188[:, 0] = (1.0 - 0.35 * base_gate[:, 0]) * p169[:, 0] + (0.35 * base_gate[:, 0]) * p173[:, 0]
        p188[:, 1] = (1.0 - 0.30 * base_gate[:, 1]) * p171[:, 1] + (0.30 * base_gate[:, 1]) * p174[:, 1]
        p200 = previous_167.copy()
        p200[:, 0] = p188[:, 0]
        p200[:, 1] = p195[:, 1]

        fold_predictions = {
            "167_PreviousBest_reference": previous_167,
            "188_ValenceRiskExpert": clip(p188),
            "195_ConformalBandSafeProjector": clip(p195),
            "200_CurrentManualFusion": clip(p200),
        }
        add_gate_predictions(
            fold_predictions=fold_predictions,
            previous_167=previous_167,
            p188=p188,
            p195=p195,
            train_ids=oof_train_ids,
            y_train=y_train,
            val_ids=val_ids,
            y_val=y_val,
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
        )

        fold_results = sorted(
            [score(name, y_val, pred, "Module gate mechanism analysis.") for name, pred in fold_predictions.items()],
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
        diagnostics.append(fold_diagnostics(val_ids, y_val, previous_167, p188, p195, candidate_mean, candidate_std))

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
    output = {
        "method": "Module mechanism analysis after 200 experiments",
        "note": "No new base architecture. These are deterministic gates/probes over previously discovered modules.",
        "aggregate_results": aggregate_results[: args.top_k],
        "diagnostics": summarize_diagnostics(diagnostics),
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def add_gate_predictions(
    fold_predictions: dict[str, np.ndarray],
    previous_167: np.ndarray,
    p188: np.ndarray,
    p195: np.ndarray,
    train_ids: list[str],
    y_train: np.ndarray,
    val_ids: list[str],
    y_val: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> None:
    base = previous_167.copy()
    correction = np.abs(p188[:, 0] - previous_167[:, 0])
    slope = np.abs(prior_slope_by_trial(val_ids, previous_167)[:, 0])
    uncertainty = candidate_std[:, 0]
    consensus_z = np.abs(p188[:, 0] - candidate_mean[:, 0]) / (candidate_std[:, 0] + 1.0)

    gate_unc_low = 1.0 - normalize_by_quantile(uncertainty, 30, 90)
    gate_unc_high = 1.0 - gate_unc_low
    gate_slope_low = 1.0 - normalize_by_quantile(slope, 30, 90)
    gate_slope_high = 1.0 - gate_slope_low
    gate_small_correction = 1.0 - normalize_by_quantile(correction, 40, 95)
    gate_consensus = 1.0 - normalize_by_quantile(consensus_z, 35, 90)
    gate_hybrid = np.sqrt(np.clip(gate_unc_low * gate_small_correction * gate_consensus, 0.0, 1.0))

    gate_specs = {
        "201_GateV_Trust188WhenUncertaintyLow": gate_unc_low,
        "202_GateV_Trust188WhenUncertaintyHigh": gate_unc_high,
        "203_GateV_Trust188WhenSlopeLow": gate_slope_low,
        "204_GateV_Trust188WhenSlopeHigh": gate_slope_high,
        "205_GateV_Trust188WhenCorrectionSmall": gate_small_correction,
        "206_GateV_Trust188WhenConsensusClose": gate_consensus,
        "207_GateV_HybridLowRisk": gate_hybrid,
    }
    for name, gate in gate_specs.items():
        pred = base.copy()
        pred[:, 0] = gate * p188[:, 0] + (1.0 - gate) * previous_167[:, 0]
        pred[:, 1] = p195[:, 1]
        fold_predictions[name] = clip(pred)

    reference_v = video_time_reference(train_ids, y_train[:, 0], val_ids)
    advantage = np.abs(previous_167[:, 0] - reference_v) - np.abs(p188[:, 0] - reference_v)
    scale = max(float(np.percentile(np.abs(advantage), 75)), 1.0)
    gate_reference_soft = sigmoid(advantage / scale)
    gate_reference_hard = (advantage > 0.0).astype(np.float32)
    gate_reference_conservative = sigmoid((advantage - 0.35 * scale) / scale)
    reference_specs = {
        "209_GateV_TrainVideoTimePriorSoft": gate_reference_soft,
        "210_GateV_TrainVideoTimePriorHard": gate_reference_hard,
        "211_GateV_TrainVideoTimePriorConservative": gate_reference_conservative,
    }
    for name, gate in reference_specs.items():
        pred = base.copy()
        pred[:, 0] = gate * p188[:, 0] + (1.0 - gate) * previous_167[:, 0]
        pred[:, 1] = p195[:, 1]
        fold_predictions[name] = clip(pred)

    oracle = base.copy()
    choose_188 = np.abs(p188[:, 0] - y_val[:, 0]) < np.abs(previous_167[:, 0] - y_val[:, 0])
    oracle[:, 0] = np.where(choose_188, p188[:, 0], previous_167[:, 0])
    oracle[:, 1] = p195[:, 1]
    fold_predictions["208_OracleRowGateV_UpperBound_NotDeployable"] = clip(oracle)


def fold_diagnostics(
    val_ids: list[str],
    y_val: np.ndarray,
    previous_167: np.ndarray,
    p188: np.ndarray,
    p195: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> dict[str, object]:
    delta_v = np.abs(p188[:, 0] - y_val[:, 0]) - np.abs(previous_167[:, 0] - y_val[:, 0])
    delta_a = np.abs(p195[:, 1] - y_val[:, 1]) - np.abs(previous_167[:, 1] - y_val[:, 1])
    features = {
        "candidate_std_v": candidate_std[:, 0],
        "slope_abs_v": np.abs(prior_slope_by_trial(val_ids, previous_167)[:, 0]),
        "correction_abs_v": np.abs(p188[:, 0] - previous_167[:, 0]),
        "consensus_z_v": np.abs(p188[:, 0] - candidate_mean[:, 0]) / (candidate_std[:, 0] + 1.0),
    }
    video_delta: dict[str, list[float]] = defaultdict(list)
    for sample_id, value in zip(val_ids, delta_v):
        _, video, _ = parse_sample_id(sample_id)
        video_delta[str(video)].append(float(value))
    return {
        "delta_v_mean": float(np.mean(delta_v)),
        "delta_a_mean": float(np.mean(delta_a)),
        "delta_v_by_feature_bins": {
            name: binned_delta(values, delta_v) for name, values in features.items()
        },
        "delta_v_by_video": {
            video: round(float(np.mean(values)), 5) for video, values in sorted(video_delta.items(), key=lambda item: int(item[0]))
        },
    }


def summarize_diagnostics(items: list[dict[str, object]]) -> dict[str, object]:
    feature_bins: dict[str, list[list[float]]] = defaultdict(list)
    video_bins: dict[str, list[float]] = defaultdict(list)
    for item in items:
        for name, rows in item["delta_v_by_feature_bins"].items():
            feature_bins[name].append([row["mean_delta"] for row in rows])
        for video, value in item["delta_v_by_video"].items():
            video_bins[video].append(float(value))
    return {
        "interpretation": "Negative delta means the tested module improves absolute error relative to 167.",
        "mean_delta_v_188_vs_167": round(float(np.mean([item["delta_v_mean"] for item in items])), 6),
        "mean_delta_a_195_vs_167": round(float(np.mean([item["delta_a_mean"] for item in items])), 6),
        "feature_bin_mean_delta_v": {
            name: [round(float(x), 6) for x in ragged_column_mean(rows)]
            for name, rows in feature_bins.items()
        },
        "video_mean_delta_v": {
            video: round(float(np.mean(values)), 6) for video, values in sorted(video_bins.items(), key=lambda item: int(item[0]))
        },
    }


def normalize_by_quantile(values: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    low = np.percentile(values, q_low)
    high = np.percentile(values, q_high)
    return np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)


def video_time_reference(train_ids: list[str], values: np.ndarray, val_ids: list[str]) -> np.ndarray:
    by_video_bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_video: dict[int, list[float]] = defaultdict(list)
    global_values = []
    for sample_id, value in zip(train_ids, values):
        _, video, timestamp = parse_sample_id(sample_id)
        bucket = timestamp // 8
        by_video_bucket[(video, bucket)].append(float(value))
        by_video[video].append(float(value))
        global_values.append(float(value))
    global_median = float(np.median(global_values)) if global_values else 128.0
    out = np.zeros(len(val_ids), dtype=np.float32)
    for index, sample_id in enumerate(val_ids):
        _, video, timestamp = parse_sample_id(sample_id)
        bucket = timestamp // 8
        if by_video_bucket[(video, bucket)]:
            out[index] = float(np.median(by_video_bucket[(video, bucket)]))
        elif by_video[video]:
            out[index] = float(np.median(by_video[video]))
        else:
            out[index] = global_median
    return out


def sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))).astype(np.float32)


def binned_delta(values: np.ndarray, delta: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    edges = np.unique(np.percentile(values, np.linspace(0, 100, bins + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        rows.append(
            {
                "low": round(float(lo), 5),
                "high": round(float(hi), 5),
                "count": int(mask.sum()),
                "mean_delta": round(float(np.mean(delta[mask])) if mask.any() else 0.0, 6),
            }
        )
    return rows


def ragged_column_mean(rows: list[list[float]]) -> list[float]:
    width = max((len(row) for row in rows), default=0)
    out = []
    for index in range(width):
        values = [row[index] for row in rows if index < len(row)]
        out.append(float(np.mean(values)) if values else 0.0)
    return out


if __name__ == "__main__":
    main()
