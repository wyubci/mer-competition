from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import (  # noqa: E402
    evaluate_candidate,
    evaluate_residual_grid,
    finalize_metric,
    load_or_build_precomputed,
    sanitize,
)
from tools.cross_fold_neurovascular_oof_gate import nested_expert_predictions  # noqa: E402
from tools.cross_fold_signal_residual_over_pattern_prior import (  # noqa: E402
    make_leave_subject_out_train_prior,
    make_pattern_prior,
)
from tools.run_iteration_experiments import expand_subjects, load_labels  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOVA-v2: diverse EEG-fNIRS fusion modules.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_baseline.npz")
    parser.add_argument("--output", default="experiments/results/iteration_247_262_neurovascular_fusion_v2.json")
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--meta-alpha", type=float, default=300.0)
    parser.add_argument("--scales", default="0.03,0.05,0.08,0.12,0.16,0.20,0.24,0.30")
    parser.add_argument("--clips", default="0.5,1,2,4")
    parser.add_argument("--smooth-windows", default="0,5")
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
    scales = parse_floats(args.scales)
    clips = parse_floats(args.clips)
    smooth_windows = parse_ints(args.smooth_windows)

    views = {
        "eeg": pre["eeg_lag"],
        "fnirs": pre["fnirs_slow"],
        "nv": pre["neurovascular"],
        "early": pre["early_concat"],
    }

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] NOVA-v2 train={len(train_subjects)} val={val_subjects}", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        prior_train = make_leave_subject_out_train_prior(labels, train_subjects, train_ids)
        prior_val = make_pattern_prior(train_ids, y_train, val_ids)
        residual_train = (y_train - prior_train).astype(np.float32)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)

        fold_results = [
            evaluate_candidate(metric_acc, "098_PatternPrior_reference", y_val, prior_val, "Strong non-signal prior.")
        ]
        expert_oof, expert_val = nested_expert_predictions(
            views=views,
            train_idx=train_idx,
            val_idx=val_idx,
            train_ids=train_ids,
            train_subjects=train_subjects,
            residual_train=residual_train,
            alpha=args.alpha,
        )
        modules = build_nova_v2_modules(
            train_ids=train_ids,
            val_ids=val_ids,
            prior_train=prior_train,
            prior_val=prior_val,
            residual_train=residual_train,
            expert_oof=expert_oof,
            expert_val=expert_val,
            coherence_val=pre["coherence"][val_idx],
            meta_alpha=args.meta_alpha,
        )
        for name, residual_val in modules.items():
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
        "method": "NOVA-v2: neurovascular agreement module search",
        "note": (
            "This batch tests distinct EEG-fNIRS fusion rules over PatternPrior_098: reliability softmax, "
            "trimmed agreement, geometric sign consensus, orthogonal neurovascular evidence, temporal "
            "delta agreement, state-dependent helpfulness, and lightweight attention over modality experts."
        ),
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_nova_v2_modules(
    train_ids: list[str],
    val_ids: list[str],
    prior_train: np.ndarray,
    prior_val: np.ndarray,
    residual_train: np.ndarray,
    expert_oof: dict[str, np.ndarray],
    expert_val: dict[str, np.ndarray],
    coherence_val: np.ndarray,
    meta_alpha: float,
) -> dict[str, np.ndarray]:
    eeg_oof, fnirs_oof, nv_oof, early_oof = (
        expert_oof["eeg"],
        expert_oof["fnirs"],
        expert_oof["nv"],
        expert_oof["early"],
    )
    eeg_val, fnirs_val, nv_val, early_val = (
        expert_val["eeg"],
        expert_val["fnirs"],
        expert_val["nv"],
        expert_val["early"],
    )

    soft_train = reliability_softmax([eeg_oof, fnirs_oof, nv_oof], residual_train, [eeg_oof, fnirs_oof, nv_oof])
    soft_val = reliability_softmax([eeg_oof, fnirs_oof, nv_oof], residual_train, [eeg_val, fnirs_val, nv_val])
    tri_train = trimodal_intersection(eeg_oof, fnirs_oof, nv_oof)
    tri_val = trimodal_intersection(eeg_val, fnirs_val, nv_val)
    min_train = min_magnitude_agreement(eeg_oof, fnirs_oof)
    min_val = min_magnitude_agreement(eeg_val, fnirs_val)
    geom_train = signed_geometric_consensus(eeg_oof, fnirs_oof)
    geom_val = signed_geometric_consensus(eeg_val, fnirs_val)
    orth_train, orth_val = orthogonal_neurovascular(
        eeg_oof,
        fnirs_oof,
        nv_oof,
        eeg_val,
        fnirs_val,
        nv_val,
        alpha=meta_alpha,
    )
    attention_train = attention_consensus([eeg_oof, fnirs_oof, nv_oof, early_oof])
    attention_val = attention_consensus([eeg_val, fnirs_val, nv_val, early_val])
    delta_train = temporal_delta_agreement(train_ids, eeg_oof, fnirs_oof)
    delta_val = temporal_delta_agreement(val_ids, eeg_val, fnirs_val)
    lead_train = slow_fast_confirm(train_ids, eeg_oof, fnirs_oof, lag=3)
    lead_val = slow_fast_confirm(val_ids, eeg_val, fnirs_val, lag=3)
    coherence_val = np.clip(coherence_val.astype(np.float32), 0.0, 1.0)

    state_soft_conf = state_helpfulness_confidence(
        train_ids, prior_train, residual_train, soft_train, val_ids, prior_val
    )
    state_tri_conf = state_helpfulness_confidence(
        train_ids, prior_train, residual_train, tri_train, val_ids, prior_val
    )

    modules = {
        "247_ReliabilitySoftmax3": soft_val,
        "248_TriModalSignIntersection": tri_val,
        "249_MinMagnitudeAgreement": min_val,
        "250_SignedGeometricConsensus": geom_val,
        "251_OrthogonalNeurovascular": sign_consensus_blend(soft_val, orth_val, weight=0.35),
        "252_RatioReliabilityGate": ratio_reliability(eeg_val, fnirs_val),
        "253_TemporalDeltaAgreement": delta_val,
        "254_EEGLeadFNIRSConfirm": lead_val,
        "255_CoherenceWeightedSoftmax": soft_val * (0.20 + 0.80 * coherence_val),
        "256_StateHelpfulSoftmax": soft_val * state_soft_conf,
        "257_StateHelpfulTriIntersection": tri_val * state_tri_conf,
        "258_TrimmedMeanConsensus4": trimmed_mean_consensus([eeg_val, fnirs_val, nv_val, early_val]),
        "259_HuberConsensus4": huber_consensus([eeg_val, fnirs_val, nv_val, early_val]),
        "260_AttentionConsensus4": attention_val,
        "261_AttentionStateHelpful": attention_val
        * state_helpfulness_confidence(train_ids, prior_train, residual_train, attention_train, val_ids, prior_val),
        "262_ValenceOnlyNOVA": valence_only(0.45 * soft_val + 0.35 * tri_val + 0.20 * attention_val),
    }

    # The first 15 modules expose both dimensions to the grid, but this branch bakes in the observation
    # that MER-PS signal residuals have repeatedly helped valence more than arousal.
    for name, residual in list(modules.items()):
        if name != "262_ValenceOnlyNOVA":
            modules[f"{name}_VOnly"] = valence_only(residual)
    return {name: sanitize(value) for name, value in modules.items()}


def reliability_softmax(
    train_preds: list[np.ndarray],
    residual_train: np.ndarray,
    val_preds: list[np.ndarray],
    temperature: float = 16.0,
) -> np.ndarray:
    mse = np.stack([((pred - residual_train) ** 2).mean(axis=0) for pred in train_preds], axis=0) + 1e-6
    logits = -mse / float(temperature)
    logits = logits - logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-6)
    out = np.zeros_like(val_preds[0], dtype=np.float32)
    for weight, pred in zip(weights, val_preds):
        out += weight[None, :] * pred
    return out.astype(np.float32)


def trimodal_intersection(eeg: np.ndarray, fnirs: np.ndarray, nv: np.ndarray) -> np.ndarray:
    stack = np.stack([eeg, fnirs, nv], axis=0)
    sign_sum = np.abs(np.sign(stack).sum(axis=0))
    same_sign = sign_sum >= 3.0
    ratio = np.min(np.abs(stack), axis=0) / (np.max(np.abs(stack), axis=0) + 1e-3)
    median = np.median(stack, axis=0)
    return np.where(same_sign, ratio * median, 0.0).astype(np.float32)


def min_magnitude_agreement(eeg: np.ndarray, fnirs: np.ndarray) -> np.ndarray:
    same = np.sign(eeg) == np.sign(fnirs)
    magnitude = np.minimum(np.abs(eeg), np.abs(fnirs))
    direction = np.sign(eeg + fnirs)
    return np.where(same, direction * magnitude, 0.0).astype(np.float32)


def signed_geometric_consensus(eeg: np.ndarray, fnirs: np.ndarray) -> np.ndarray:
    same = np.sign(eeg) == np.sign(fnirs)
    magnitude = np.sqrt(np.abs(eeg * fnirs) + 1e-6)
    direction = np.sign(eeg + fnirs)
    ratio = np.minimum(np.abs(eeg), np.abs(fnirs)) / (np.maximum(np.abs(eeg), np.abs(fnirs)) + 1e-3)
    return np.where(same, direction * magnitude * ratio, 0.0).astype(np.float32)


def orthogonal_neurovascular(
    eeg_train: np.ndarray,
    fnirs_train: np.ndarray,
    nv_train: np.ndarray,
    eeg_val: np.ndarray,
    fnirs_val: np.ndarray,
    nv_val: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_base = np.concatenate([eeg_train, fnirs_train, eeg_train * fnirs_train, np.abs(eeg_train - fnirs_train)], axis=1)
    val_base = np.concatenate([eeg_val, fnirs_val, eeg_val * fnirs_val, np.abs(eeg_val - fnirs_val)], axis=1)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(sanitize(train_base), nv_train)
    train_orth = (nv_train - model.predict(sanitize(train_base))).astype(np.float32)
    val_orth = (nv_val - model.predict(sanitize(val_base))).astype(np.float32)
    return train_orth, val_orth


def sign_consensus_blend(base: np.ndarray, addon: np.ndarray, weight: float) -> np.ndarray:
    same = np.sign(base) == np.sign(addon)
    return np.where(same, base + float(weight) * addon, base).astype(np.float32)


def ratio_reliability(eeg: np.ndarray, fnirs: np.ndarray) -> np.ndarray:
    same = (np.sign(eeg) == np.sign(fnirs)).astype(np.float32)
    ratio = np.minimum(np.abs(eeg), np.abs(fnirs)) / (np.maximum(np.abs(eeg), np.abs(fnirs)) + 1e-3)
    confidence = same * (ratio**0.5)
    return (confidence * (0.5 * eeg + 0.5 * fnirs)).astype(np.float32)


def temporal_delta_agreement(sample_ids: list[str], eeg: np.ndarray, fnirs: np.ndarray) -> np.ndarray:
    eeg_delta = temporal_delta(sample_ids, eeg, lag=1)
    fnirs_delta = temporal_delta(sample_ids, fnirs, lag=3)
    same_level = np.sign(eeg) == np.sign(fnirs)
    same_delta = np.sign(eeg_delta) == np.sign(fnirs_delta)
    consensus = 0.55 * eeg + 0.45 * fnirs
    confidence = same_level.astype(np.float32) * (0.45 + 0.55 * same_delta.astype(np.float32))
    return (confidence * consensus).astype(np.float32)


def slow_fast_confirm(sample_ids: list[str], eeg: np.ndarray, fnirs: np.ndarray, lag: int) -> np.ndarray:
    fnirs_past = temporal_shift(sample_ids, fnirs, lag=lag)
    same = np.sign(eeg) == np.sign(fnirs_past)
    ratio = np.minimum(np.abs(eeg), np.abs(fnirs_past)) / (np.maximum(np.abs(eeg), np.abs(fnirs_past)) + 1e-3)
    return np.where(same, (0.70 * eeg + 0.30 * fnirs_past) * (0.35 + 0.65 * ratio), 0.0).astype(np.float32)


def state_helpfulness_confidence(
    train_ids: list[str],
    prior_train: np.ndarray,
    residual_train: np.ndarray,
    module_train: np.ndarray,
    val_ids: list[str],
    prior_val: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(prior_val, dtype=np.float32)
    train_slope = np.abs(temporal_delta(train_ids, prior_train, lag=1))
    val_slope = np.abs(temporal_delta(val_ids, prior_val, lag=1))
    for dim in range(2):
        helpful = (
            np.abs(residual_train[:, dim] - module_train[:, dim]) < np.abs(residual_train[:, dim])
        ).astype(np.float32)
        global_rate = float(helpful.mean())
        value_edges = unique_edges(np.percentile(prior_train[:, dim], [0, 20, 40, 60, 80, 100]))
        slope_edges = unique_edges(np.percentile(train_slope[:, dim], [0, 50, 80, 100]))
        table: dict[tuple[int, int], list[float]] = defaultdict(list)
        for i in range(prior_train.shape[0]):
            key = (
                bin_index(prior_train[i, dim], value_edges),
                bin_index(train_slope[i, dim], slope_edges),
            )
            table[key].append(float(helpful[i]))
        for i in range(prior_val.shape[0]):
            key = (
                bin_index(prior_val[i, dim], value_edges),
                bin_index(val_slope[i, dim], slope_edges),
            )
            values = table.get(key, [])
            if not values:
                out[i, dim] = global_rate
            else:
                n = len(values)
                out[i, dim] = (sum(values) + 3.0 * global_rate) / (n + 3.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def trimmed_mean_consensus(preds: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(preds, axis=0)
    signs = np.sign(stack)
    majority = np.abs(signs.sum(axis=0)) >= 2.0
    sorted_values = np.sort(stack, axis=0)
    trimmed = sorted_values[1:-1].mean(axis=0)
    disagreement = stack.std(axis=0)
    confidence = 1.0 / (1.0 + disagreement / 8.0)
    return np.where(majority, confidence * trimmed, 0.0).astype(np.float32)


def huber_consensus(preds: list[np.ndarray], delta: float = 2.0) -> np.ndarray:
    stack = np.stack(preds, axis=0)
    center = np.median(stack, axis=0)
    distance = np.abs(stack - center)
    weights = 1.0 / np.maximum(1.0, distance / float(delta))
    weighted = (weights * stack).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-6)
    confidence = 1.0 / (1.0 + stack.std(axis=0) / 6.0)
    return (confidence * weighted).astype(np.float32)


def attention_consensus(preds: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(preds, axis=0)
    center = np.median(stack, axis=0, keepdims=True)
    logits = -np.abs(stack - center) / 3.0
    logits = logits - logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-6)
    consensus = (weights * stack).sum(axis=0)
    sign_mass = np.abs(np.sign(stack).mean(axis=0))
    return (sign_mass * consensus).astype(np.float32)


def valence_only(residual: np.ndarray) -> np.ndarray:
    out = np.zeros_like(residual, dtype=np.float32)
    out[:, 0] = residual[:, 0]
    return out


def temporal_delta(sample_ids: list[str], values: np.ndarray, lag: int) -> np.ndarray:
    shifted = temporal_shift(sample_ids, values, lag=lag)
    return (values - shifted).astype(np.float32)


def temporal_shift(sample_ids: list[str], values: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for items in group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        if lag <= 0:
            shifted = seq.copy()
        else:
            shifted = np.empty_like(seq)
            shifted[:lag] = seq[:1]
            shifted[lag:] = seq[:-lag]
        out[np.asarray(indices)] = shifted
    return out


def group_indices(sample_ids: list[str]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    return groups


def unique_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float32)
    edges[0] -= 1e-3
    edges[-1] += 1e-3
    return np.unique(edges)


def bin_index(value: float, edges: np.ndarray) -> int:
    if edges.shape[0] <= 2:
        return 0
    return int(np.clip(np.searchsorted(edges[1:-1], value, side="right"), 0, edges.shape[0] - 2))


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
