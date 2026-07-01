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
    parser = argparse.ArgumentParser(description="Original hierarchical residual field module.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_212_residual_field_module.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=100)
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
        print(f"[fold {fold_index}] residual field module", flush=True)
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
        p200, parts = make_manual_200(
            previous_167=previous_167,
            oof_train_ids=oof_train_ids,
            y_train=y_train,
            prior_train=prior_train,
            residual_target=residual_target,
            val_ids=val_ids,
            candidate_std=candidate_std,
        )

        field_small = hierarchical_residual_field(
            train_ids=oof_train_ids,
            train_pred=prior_train,
            y_train=y_train,
            val_ids=val_ids,
            base_pred=p200,
            strength=(0.18, 0.10),
            sign_consensus=False,
        )
        field_medium = hierarchical_residual_field(
            train_ids=oof_train_ids,
            train_pred=prior_train,
            y_train=y_train,
            val_ids=val_ids,
            base_pred=p200,
            strength=(0.30, 0.16),
            sign_consensus=False,
        )
        field_sign = hierarchical_residual_field(
            train_ids=oof_train_ids,
            train_pred=prior_train,
            y_train=y_train,
            val_ids=val_ids,
            base_pred=p200,
            strength=(0.28, 0.14),
            sign_consensus=True,
        )
        valence_only = p200.copy()
        valence_only[:, 0] = field_medium[:, 0]
        arousal_only = p200.copy()
        arousal_only[:, 1] = field_medium[:, 1]
        valence_sign = p200.copy()
        valence_sign[:, 0] = field_sign[:, 0]
        valence_consensus = p200.copy()
        delta_medium = field_medium[:, 0] - p200[:, 0]
        delta_sign = field_sign[:, 0] - p200[:, 0]
        agree = (delta_medium * delta_sign) > 0.0
        valence_consensus[:, 0] = p200[:, 0] + np.where(agree, delta_medium, 0.0)
        valence_blend = p200.copy()
        valence_blend[:, 0] = p200[:, 0] + np.where(
            agree,
            0.70 * delta_medium + 0.30 * delta_sign,
            0.35 * delta_sign,
        )
        valence_soft = p200.copy()
        confidence = np.clip(np.abs(delta_sign) / (np.abs(delta_medium) + 1e-3), 0.0, 1.0)
        valence_soft[:, 0] = p200[:, 0] + confidence * delta_medium

        fold_predictions = {
            "167_PreviousBest_reference": previous_167,
            "188_ValenceRiskExpert_reference": parts["p188"],
            "195_ConformalBand_reference": parts["p195"],
            "200_CurrentManualFusion": p200,
            "212_HierResidualField_small": field_small,
            "213_HierResidualField_medium": field_medium,
            "214_HierResidualField_signConsensus": field_sign,
            "215_HierResidualField_valenceOnly": clip(valence_only),
            "216_HierResidualField_arousalOnly": clip(arousal_only),
            "217_HierResidualField_valenceSignOnly": clip(valence_sign),
            "218_HierResidualField_valenceConsensusMask": clip(valence_consensus),
            "219_HierResidualField_valenceConsensusBlend": clip(valence_blend),
            "220_HierResidualField_valenceSoftConfidence": clip(valence_soft),
        }

        fold_results = sorted(
            [score(name, y_val, pred, "Original hierarchical residual field module.") for name, pred in fold_predictions.items()],
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
        diagnostics.append(residual_field_diagnostics(p200, field_medium, y_val))

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
        "method": "Original module: hierarchical residual field",
        "note": "The module learns a heavily shrunk residual field over video/time/value/slope conditions from OOF training residuals.",
        "aggregate_results": aggregate_results[: args.top_k],
        "diagnostics": summarize_diagnostics(diagnostics),
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_manual_200(
    previous_167: np.ndarray,
    oof_train_ids: list[str],
    y_train: np.ndarray,
    prior_train: np.ndarray,
    residual_target: np.ndarray,
    val_ids: list[str],
    candidate_std: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p169 = monotone_pchip_bias_calibration(prior_train, residual_target, previous_167)
    p173 = covariance_transport(y_train, previous_167)
    p171 = lag_aligned_trial_prototype(oof_train_ids, prior_train, residual_target, val_ids, previous_167)
    p174 = kalman_uncertainty_smoother(previous_167, val_ids, candidate_std)
    p195 = conformal_median_band_projector(oof_train_ids, y_train, val_ids, previous_167)
    gate = uncertainty_gate(candidate_std, 45, 82)
    p188 = previous_167.copy()
    p188[:, 0] = (1.0 - 0.35 * gate[:, 0]) * p169[:, 0] + (0.35 * gate[:, 0]) * p173[:, 0]
    p188[:, 1] = (1.0 - 0.30 * gate[:, 1]) * p171[:, 1] + (0.30 * gate[:, 1]) * p174[:, 1]
    p200 = previous_167.copy()
    p200[:, 0] = p188[:, 0]
    p200[:, 1] = p195[:, 1]
    return clip(p200), {"p188": clip(p188), "p195": clip(p195)}


def hierarchical_residual_field(
    train_ids: list[str],
    train_pred: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
    base_pred: np.ndarray,
    strength: tuple[float, float],
    sign_consensus: bool,
) -> np.ndarray:
    out = base_pred.copy().astype(np.float32)
    train_slope = np.abs(prior_slope_by_trial(train_ids, train_pred))
    val_slope = np.abs(prior_slope_by_trial(val_ids, base_pred))
    for dim in range(2):
        residual = (y_train[:, dim] - train_pred[:, dim]).astype(np.float32)
        value_edges = unique_edges(np.percentile(train_pred[:, dim], [0, 10, 25, 40, 55, 70, 85, 100]))
        slope_edges = unique_edges(np.percentile(train_slope[:, dim], [0, 30, 60, 85, 100]))
        train_meta = [meta_tuple(sample_id, train_pred[i, dim], train_slope[i, dim], value_edges, slope_edges) for i, sample_id in enumerate(train_ids)]
        val_meta = [meta_tuple(sample_id, base_pred[i, dim], val_slope[i, dim], value_edges, slope_edges) for i, sample_id in enumerate(val_ids)]

        correction = np.zeros(len(val_ids), dtype=np.float32)
        correction += 0.32 * lookup_group_correction(train_meta, val_meta, residual, lambda m: ("vt", m["video"], m["time_bucket"]), 28, sign_consensus)
        correction += 0.22 * lookup_group_correction(train_meta, val_meta, residual, lambda m: ("video", m["video"]), 48, sign_consensus)
        correction += 0.16 * lookup_group_correction(train_meta, val_meta, residual, lambda m: ("time", m["time_bucket"]), 48, sign_consensus)
        correction += 0.20 * lookup_group_correction(train_meta, val_meta, residual, lambda m: ("value", m["value_bin"]), 52, sign_consensus)
        correction += 0.10 * lookup_group_correction(train_meta, val_meta, residual, lambda m: ("slope", m["slope_bin"]), 64, sign_consensus)
        correction = np.clip(correction, -4.0 if dim == 0 else -2.5, 4.0 if dim == 0 else 2.5)
        out[:, dim] = base_pred[:, dim] + strength[dim] * correction
    return clip(out)


def lookup_group_correction(
    train_meta: list[dict[str, int]],
    val_meta: list[dict[str, int]],
    residual: np.ndarray,
    key_fn,
    shrink_k: float,
    sign_consensus: bool,
) -> np.ndarray:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for meta, value in zip(train_meta, residual):
        groups[key_fn(meta)].append(float(value))
    stats: dict[tuple[object, ...], float] = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) < 3:
            stats[key] = 0.0
            continue
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) + 1e-3
        shrink = len(arr) / (len(arr) + shrink_k)
        shrink *= 1.0 / (1.0 + mad / 24.0)
        if sign_consensus:
            sign_strength = abs(float(np.mean(np.sign(arr))))
            shrink *= max(0.0, (sign_strength - 0.12) / 0.88)
        stats[key] = float(np.clip(shrink * median, -8.0, 8.0))
    return np.asarray([stats.get(key_fn(meta), 0.0) for meta in val_meta], dtype=np.float32)


def meta_tuple(
    sample_id: str,
    value: float,
    slope: float,
    value_edges: np.ndarray,
    slope_edges: np.ndarray,
) -> dict[str, int]:
    _, video, timestamp = parse_sample_id(sample_id)
    return {
        "video": int(video),
        "time_bucket": int(timestamp // 8),
        "value_bin": int(np.searchsorted(value_edges[1:-1], value, side="right")),
        "slope_bin": int(np.searchsorted(slope_edges[1:-1], slope, side="right")),
    }


def unique_edges(values: np.ndarray) -> np.ndarray:
    edges = np.unique(values.astype(np.float32))
    if len(edges) < 2:
        return np.asarray([values[0] - 1.0, values[0] + 1.0], dtype=np.float32)
    return edges


def residual_field_diagnostics(base: np.ndarray, corrected: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    delta = np.abs(corrected - y_true) - np.abs(base - y_true)
    return {
        "delta_overall": float(np.mean(delta)),
        "delta_valence": float(np.mean(delta[:, 0])),
        "delta_arousal": float(np.mean(delta[:, 1])),
        "mean_abs_correction_valence": float(np.mean(np.abs(corrected[:, 0] - base[:, 0]))),
        "mean_abs_correction_arousal": float(np.mean(np.abs(corrected[:, 1] - base[:, 1]))),
    }


def summarize_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(rows[0]) if rows else []
    return {key: round(float(np.mean([row[key] for row in rows])), 6) for key in keys}


if __name__ == "__main__":
    main()
