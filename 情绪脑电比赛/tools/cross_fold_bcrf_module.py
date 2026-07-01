from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import clip, make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125  # noqa: E402
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
from tools.cross_fold_residual_field_module import hierarchical_residual_field, make_manual_200  # noqa: E402
from tools.cross_fold_to200_architectures import make_previous_167  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCRF: Bayesian Credible Residual Field.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_221_230_bcrf_module.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=2027)
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
        print(f"[fold {fold_index}] BCRF module", flush=True)
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
        p218 = make_scrf_218(oof_train_ids, prior_train, y_train, val_ids, p200)

        b_delta, b_conf = bayesian_credible_residual_field(
            train_ids=oof_train_ids,
            train_pred=prior_train,
            y_train=y_train,
            val_ids=val_ids,
            base_pred=p200,
        )
        scrf_delta = p218 - p200
        agree = (b_delta[:, 0] * scrf_delta[:, 0]) > 0.0

        p221 = p200.copy()
        p221[:, 0] = p200[:, 0] + b_delta[:, 0]

        p222 = p218.copy()
        p222[:, 0] = p218[:, 0] + 0.50 * b_conf[:, 0] * b_delta[:, 0]

        p223 = p200.copy()
        p223[:, 0] = p200[:, 0] + np.where(agree, 0.65 * scrf_delta[:, 0] + 0.35 * b_delta[:, 0], 0.0)

        p224 = p200.copy()
        p224[:, 0] = p200[:, 0] + np.where(agree, scrf_delta[:, 0], 0.0)

        p225 = p200.copy()
        p225[:, 0] = p200[:, 0] + b_conf[:, 0] * scrf_delta[:, 0]

        p226 = p200.copy()
        p226[:, 0] = p200[:, 0] + np.where(agree, (0.50 + 0.50 * b_conf[:, 0]) * scrf_delta[:, 0], 0.25 * b_conf[:, 0] * b_delta[:, 0])

        p227 = p200.copy()
        p227[:, 1] = p200[:, 1] + 0.40 * b_conf[:, 1] * b_delta[:, 1]

        p228 = p218.copy()
        p228[:, 1] = p200[:, 1] + 0.25 * b_conf[:, 1] * b_delta[:, 1]

        fold_predictions = {
            "167_PreviousBest_reference": previous_167,
            "188_ValenceRiskExpert_reference": parts["p188"],
            "200_CurrentManualFusion": p200,
            "218_SCRF_reference": p218,
            "221_BCRF_valenceOnly": clip(p221),
            "222_BCRF_onSCRF": clip(p222),
            "223_BCRF_SCRFConsensusBlend": clip(p223),
            "224_BCRF_BrakeSCRFDisagreement": clip(p224),
            "225_BCRF_ConfidenceScaledSCRF": clip(p225),
            "226_BCRF_ReliabilityMixture": clip(p226),
            "227_BCRF_arousalOnly_probe": clip(p227),
            "228_BCRF_SCRF_plusArousal_probe": clip(p228),
        }

        fold_results = sorted(
            [score(name, y_val, pred, "BCRF module over SCRF and 200.") for name, pred in fold_predictions.items()],
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
        diagnostics.append(bcrf_diagnostics(p200, p218, b_delta, b_conf, y_val))

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
        "method": "BCRF: Bayesian Credible Residual Field",
        "note": "BCRF estimates residual correction and reliability from hierarchical OOF residual cells, then combines it with SCRF.",
        "aggregate_results": aggregate_results[: args.top_k],
        "diagnostics": summarize_diagnostics(diagnostics),
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_scrf_218(
    train_ids: list[str],
    train_pred: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
    p200: np.ndarray,
) -> np.ndarray:
    field_medium = hierarchical_residual_field(
        train_ids=train_ids,
        train_pred=train_pred,
        y_train=y_train,
        val_ids=val_ids,
        base_pred=p200,
        strength=(0.30, 0.16),
        sign_consensus=False,
    )
    field_sign = hierarchical_residual_field(
        train_ids=train_ids,
        train_pred=train_pred,
        y_train=y_train,
        val_ids=val_ids,
        base_pred=p200,
        strength=(0.28, 0.14),
        sign_consensus=True,
    )
    delta_medium = field_medium[:, 0] - p200[:, 0]
    delta_sign = field_sign[:, 0] - p200[:, 0]
    out = p200.copy()
    out[:, 0] = p200[:, 0] + np.where(delta_medium * delta_sign > 0.0, delta_medium, 0.0)
    return clip(out)


def bayesian_credible_residual_field(
    train_ids: list[str],
    train_pred: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
    base_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    correction = np.zeros_like(base_pred, dtype=np.float32)
    confidence = np.zeros_like(base_pred, dtype=np.float32)
    train_slope = np.abs(prior_slope_by_trial(train_ids, train_pred))
    val_slope = np.abs(prior_slope_by_trial(val_ids, base_pred))
    for dim in range(2):
        residual = (y_train[:, dim] - train_pred[:, dim]).astype(np.float32)
        value_edges = unique_edges(np.percentile(train_pred[:, dim], [0, 12, 25, 38, 50, 62, 75, 88, 100]))
        slope_edges = unique_edges(np.percentile(train_slope[:, dim], [0, 35, 65, 88, 100]))
        train_meta = [meta_tuple(sample_id, train_pred[i, dim], train_slope[i, dim], value_edges, slope_edges) for i, sample_id in enumerate(train_ids)]
        val_meta = [meta_tuple(sample_id, base_pred[i, dim], val_slope[i, dim], value_edges, slope_edges) for i, sample_id in enumerate(val_ids)]
        views = [
            (lambda m: ("vt", m["video"], m["time_bucket"]), 22.0),
            (lambda m: ("vv", m["video"], m["value_bin"]), 30.0),
            (lambda m: ("tv", m["time_bucket"], m["value_bin"]), 36.0),
            (lambda m: ("vs", m["value_bin"], m["slope_bin"]), 42.0),
            (lambda m: ("v", m["video"]), 52.0),
            (lambda m: ("t", m["time_bucket"]), 58.0),
        ]
        view_corrs = []
        view_weights = []
        for key_fn, prior_k in views:
            corr, weight = credible_lookup(train_meta, val_meta, residual, key_fn, prior_k)
            view_corrs.append(corr)
            view_weights.append(weight)
        corr_stack = np.stack(view_corrs, axis=0)
        weight_stack = np.stack(view_weights, axis=0)
        weight_sum = np.maximum(weight_stack.sum(axis=0), 1e-6)
        mean_corr = (corr_stack * weight_stack).sum(axis=0) / weight_sum
        sign_agreement = np.abs((np.sign(corr_stack) * weight_stack).sum(axis=0)) / weight_sum
        dispersion = np.sqrt(np.maximum(((corr_stack - mean_corr[None, :]) ** 2 * weight_stack).sum(axis=0) / weight_sum, 0.0))
        dispersion_gate = 1.0 / (1.0 + dispersion / 3.0)
        total_conf = np.clip((weight_sum / (weight_sum + 1.2)) * sign_agreement * dispersion_gate, 0.0, 1.0)
        strength = 0.34 if dim == 0 else 0.10
        correction[:, dim] = strength * total_conf * np.clip(mean_corr, -5.0 if dim == 0 else -2.5, 5.0 if dim == 0 else 2.5)
        confidence[:, dim] = total_conf.astype(np.float32)
    return correction.astype(np.float32), confidence.astype(np.float32)


def credible_lookup(
    train_meta: list[dict[str, int]],
    val_meta: list[dict[str, int]],
    residual: np.ndarray,
    key_fn,
    prior_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for meta, value in zip(train_meta, residual):
        groups[key_fn(meta)].append(float(value))
    stats: dict[tuple[object, ...], tuple[float, float]] = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) < 4:
            stats[key] = (0.0, 0.0)
            continue
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) + 1e-3
        se = 1.4826 * mad / np.sqrt(len(arr))
        z = abs(median) / (se + 1.0)
        sign_strength = abs(float(np.mean(np.sign(arr))))
        n_shrink = len(arr) / (len(arr) + prior_k)
        credible_gate = sigmoid(z - 0.85)
        noise_gate = 1.0 / (1.0 + mad / 22.0)
        weight = n_shrink * sign_strength * credible_gate * noise_gate
        corr = weight * median
        stats[key] = (float(np.clip(corr, -8.0, 8.0)), float(np.clip(weight, 0.0, 1.0)))
    corr = np.asarray([stats.get(key_fn(meta), (0.0, 0.0))[0] for meta in val_meta], dtype=np.float32)
    weight = np.asarray([stats.get(key_fn(meta), (0.0, 0.0))[1] for meta in val_meta], dtype=np.float32)
    return corr, weight


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


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def bcrf_diagnostics(
    p200: np.ndarray,
    p218: np.ndarray,
    b_delta: np.ndarray,
    b_conf: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    p221 = p200.copy()
    p221[:, 0] = p200[:, 0] + b_delta[:, 0]
    delta_scrf = np.abs(p218[:, 0] - y_true[:, 0]) - np.abs(p200[:, 0] - y_true[:, 0])
    delta_bcrf = np.abs(p221[:, 0] - y_true[:, 0]) - np.abs(p200[:, 0] - y_true[:, 0])
    return {
        "scrf_delta_valence": float(np.mean(delta_scrf)),
        "bcrf_delta_valence": float(np.mean(delta_bcrf)),
        "mean_conf_valence": float(np.mean(b_conf[:, 0])),
        "mean_conf_arousal": float(np.mean(b_conf[:, 1])),
        "mean_abs_bcrf_delta_valence": float(np.mean(np.abs(b_delta[:, 0]))),
        "mean_abs_bcrf_delta_arousal": float(np.mean(np.abs(b_delta[:, 1]))),
    }


def summarize_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(rows[0]) if rows else []
    return {key: round(float(np.mean([row[key] for row in rows])), 6) for key in keys}


if __name__ == "__main__":
    main()
