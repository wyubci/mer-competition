from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
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
from tools.cross_fold_neurovascular_fusion_v2 import (  # noqa: E402
    temporal_delta,
    temporal_shift,
    valence_only,
)
from tools.cross_fold_neurovascular_oof_gate import nested_expert_predictions  # noqa: E402
from tools.cross_fold_signal_residual_over_pattern_prior import (  # noqa: E402
    make_leave_subject_out_train_prior,
    make_pattern_prior,
)
from tools.run_iteration_experiments import expand_subjects, load_labels  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCMI: conservative cross-modal intersection modules.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_baseline.npz")
    parser.add_argument("--output", default="experiments/results/iteration_263_274_ccmi_neurovascular.json")
    parser.add_argument("--fnirs-types", default="0,1,2")
    parser.add_argument("--feature-normalization", default="none", choices=["none", "trial_zscore", "subject_zscore"])
    parser.add_argument("--baseline-correction", default="true", choices=["true", "false"])
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--cal-alpha", type=float, default=20.0)
    parser.add_argument("--scales", default="0.16,0.20,0.24,0.30,0.36,0.45,0.60")
    parser.add_argument("--clips", default="2,4,6,8")
    parser.add_argument("--smooth-windows", default="0,5")
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root,
        subjects,
        Path(args.precompute_cache),
        fnirs_types=tuple(parse_ints(args.fnirs_types)),
        feature_normalization=args.feature_normalization,
        baseline_correction=args.baseline_correction == "true",
    )
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
        print(f"[fold {fold_index}] CCMI train={len(train_subjects)} val={val_subjects}", flush=True)
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
        modules = build_ccmi_modules(
            train_ids=train_ids,
            val_ids=val_ids,
            prior_train=prior_train,
            prior_val=prior_val,
            residual_train=residual_train,
            expert_oof=expert_oof,
            expert_val=expert_val,
            cal_alpha=args.cal_alpha,
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
        "method": "CCMI: Conservative Cross-Modal Intersection",
        "note": (
            "CCMI treats EEG and fNIRS residual experts as two noisy measurements of the same hidden "
            "affective correction. A residual is trusted when signs agree, amplitudes overlap, and optional "
            "neurovascular/temporal gates do not veto it."
        ),
        "fnirs_types": parse_ints(args.fnirs_types),
        "feature_normalization": args.feature_normalization,
        "baseline_correction": args.baseline_correction == "true",
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_ccmi_modules(
    train_ids: list[str],
    val_ids: list[str],
    prior_train: np.ndarray,
    prior_val: np.ndarray,
    residual_train: np.ndarray,
    expert_oof: dict[str, np.ndarray],
    expert_val: dict[str, np.ndarray],
    cal_alpha: float,
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
    base_train = ccmi_core(eeg_oof, fnirs_oof, ratio_power=0.0)
    base_val = ccmi_core(eeg_val, fnirs_val, ratio_power=0.0)
    ratio05_train = ccmi_core(eeg_oof, fnirs_oof, ratio_power=0.5)
    ratio05_val = ccmi_core(eeg_val, fnirs_val, ratio_power=0.5)
    ratio2_train = ccmi_core(eeg_oof, fnirs_oof, ratio_power=2.0)
    ratio2_val = ccmi_core(eeg_val, fnirs_val, ratio_power=2.0)
    nv_confirm_train = nv_confirm(base_train, nv_oof, hard=False)
    nv_confirm_val = nv_confirm(base_val, nv_val, hard=False)
    nv_veto_train = nv_veto(base_train, nv_oof)
    nv_veto_val = nv_veto(base_val, nv_val)
    early_veto_train = nv_veto(base_train, early_oof)
    early_veto_val = nv_veto(base_val, early_val)
    calibrated_val = oof_calibrated(base_train, residual_train, base_val, cal_alpha)
    helpful_val = helpful_probability_gate(
        train_ids=train_ids,
        val_ids=val_ids,
        prior_train=prior_train,
        prior_val=prior_val,
        residual_train=residual_train,
        candidate_train=base_train,
        candidate_val=base_val,
        eeg_train=eeg_oof,
        fnirs_train=fnirs_oof,
        nv_train=nv_oof,
        eeg_val=eeg_val,
        fnirs_val=fnirs_val,
        nv_val=nv_val,
    )
    leaky_val = leaky_integrator(val_ids, base_val, alpha=0.55)
    var_gate_val = local_variance_gate(val_ids, base_val)
    slope_gate_val = prior_slope_gate(train_ids, prior_train, base_train, residual_train, val_ids, prior_val, base_val)
    hrf_val = ccmi_core(eeg_val, temporal_shift(val_ids, fnirs_val, lag=3), ratio_power=0.0)

    modules = {
        "263_CCMI_MinOverlap": base_val,
        "264_CCMI_RatioPower05": ratio05_val,
        "265_CCMI_RatioPower2": ratio2_val,
        "266_CCMI_NeurovascularConfirm": nv_confirm_val,
        "267_CCMI_NeurovascularVeto": nv_veto_val,
        "268_CCMI_EarlyExpertVeto": early_veto_val,
        "269_CCMI_OOFCalibrated": calibrated_val,
        "270_CCMI_HelpfulProbabilityGate": helpful_val,
        "271_CCMI_LeakyIntegrator": leaky_val,
        "272_CCMI_LocalVarianceGate": var_gate_val,
        "273_CCMI_PriorSlopeGate": slope_gate_val,
        "274_CCMI_HRFDelayedFNIRS": hrf_val,
        "275_CCMI_ConfirmCalibratedBlend": 0.65 * nv_confirm_val + 0.35 * calibrated_val,
        "276_CCMI_VetoHelpfulBlend": 0.55 * nv_veto_val + 0.45 * helpful_val,
    }
    for name, residual in list(modules.items()):
        modules[f"{name}_VOnly"] = valence_only(residual)
    return {name: sanitize(value) for name, value in modules.items()}


def ccmi_core(eeg: np.ndarray, fnirs: np.ndarray, ratio_power: float) -> np.ndarray:
    same = np.sign(eeg) == np.sign(fnirs)
    min_abs = np.minimum(np.abs(eeg), np.abs(fnirs))
    max_abs = np.maximum(np.abs(eeg), np.abs(fnirs))
    ratio = min_abs / (max_abs + 1e-3)
    if ratio_power > 0.0:
        min_abs = min_abs * np.power(ratio, ratio_power)
    direction = np.sign(eeg + fnirs)
    return np.where(same, direction * min_abs, 0.0).astype(np.float32)


def nv_confirm(base: np.ndarray, nv: np.ndarray, hard: bool) -> np.ndarray:
    same = np.sign(base) == np.sign(nv)
    if hard:
        return np.where(same, base, 0.0).astype(np.float32)
    ratio = np.minimum(np.abs(base), np.abs(nv)) / (np.maximum(np.abs(base), np.abs(nv)) + 1e-3)
    gain = np.where(same, 0.70 + 0.30 * ratio, 0.35)
    return (gain * base).astype(np.float32)


def nv_veto(base: np.ndarray, nv: np.ndarray) -> np.ndarray:
    opposite = (np.sign(base) != np.sign(nv)) & (np.abs(nv) > np.abs(base))
    weak_opposite = (np.sign(base) != np.sign(nv)) & ~opposite
    out = base.copy()
    out[opposite] = 0.0
    out[weak_opposite] *= 0.55
    return out.astype(np.float32)


def oof_calibrated(train_candidate: np.ndarray, residual_train: np.ndarray, val_candidate: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(val_candidate, dtype=np.float32)
    for dim in range(2):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        x_train = train_candidate[:, dim : dim + 1]
        x_val = val_candidate[:, dim : dim + 1]
        model.fit(x_train, residual_train[:, dim])
        out[:, dim] = model.predict(x_val).astype(np.float32)
    return out


def helpful_probability_gate(
    train_ids: list[str],
    val_ids: list[str],
    prior_train: np.ndarray,
    prior_val: np.ndarray,
    residual_train: np.ndarray,
    candidate_train: np.ndarray,
    candidate_val: np.ndarray,
    eeg_train: np.ndarray,
    fnirs_train: np.ndarray,
    nv_train: np.ndarray,
    eeg_val: np.ndarray,
    fnirs_val: np.ndarray,
    nv_val: np.ndarray,
) -> np.ndarray:
    x_train = gate_features(train_ids, prior_train, candidate_train, eeg_train, fnirs_train, nv_train)
    x_val = gate_features(val_ids, prior_val, candidate_val, eeg_val, fnirs_val, nv_val)
    out = np.zeros_like(candidate_val, dtype=np.float32)
    for dim in range(2):
        helpful = (
            np.abs(residual_train[:, dim] - candidate_train[:, dim]) < np.abs(residual_train[:, dim])
        ).astype(int)
        if helpful.min() == helpful.max():
            prob = np.full(candidate_val.shape[0], float(helpful.mean()), dtype=np.float32)
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, solver="lbfgs"),
            )
            model.fit(x_train, helpful)
            prob = model.predict_proba(x_val)[:, 1].astype(np.float32)
        out[:, dim] = prob * candidate_val[:, dim]
    return out


def gate_features(
    sample_ids: list[str],
    prior: np.ndarray,
    candidate: np.ndarray,
    eeg: np.ndarray,
    fnirs: np.ndarray,
    nv: np.ndarray,
) -> np.ndarray:
    prior_slope = np.abs(temporal_delta(sample_ids, prior, lag=1))
    cand_slope = np.abs(temporal_delta(sample_ids, candidate, lag=1))
    min_abs = np.minimum(np.abs(eeg), np.abs(fnirs))
    max_abs = np.maximum(np.abs(eeg), np.abs(fnirs))
    ratio = min_abs / (max_abs + 1e-3)
    sign_agree = (np.sign(eeg) == np.sign(fnirs)).astype(np.float32)
    nv_agree = (np.sign(candidate) == np.sign(nv)).astype(np.float32)
    return sanitize(
        np.concatenate(
            [
                np.abs(candidate),
                min_abs,
                max_abs,
                ratio,
                sign_agree,
                nv_agree,
                prior,
                prior_slope,
                cand_slope,
            ],
            axis=1,
        )
    )


def leaky_integrator(sample_ids: list[str], values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for items in group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        state = seq[0].copy()
        smoothed = np.zeros_like(seq)
        for i, value in enumerate(seq):
            state = float(alpha) * value + (1.0 - float(alpha)) * state
            smoothed[i] = state
        out[np.asarray(indices)] = smoothed
    return out


def local_variance_gate(sample_ids: list[str], values: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for items in group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        gated = np.zeros_like(seq)
        for i in range(seq.shape[0]):
            start = max(0, i - window + 1)
            local = seq[start : i + 1]
            std = local.std(axis=0)
            confidence = 1.0 / (1.0 + std / 2.5)
            gated[i] = confidence * seq[i]
        out[np.asarray(indices)] = gated
    return out


def prior_slope_gate(
    train_ids: list[str],
    prior_train: np.ndarray,
    candidate_train: np.ndarray,
    residual_train: np.ndarray,
    val_ids: list[str],
    prior_val: np.ndarray,
    candidate_val: np.ndarray,
) -> np.ndarray:
    train_slope = np.abs(temporal_delta(train_ids, prior_train, lag=1))
    val_slope = np.abs(temporal_delta(val_ids, prior_val, lag=1))
    out = np.zeros_like(candidate_val, dtype=np.float32)
    for dim in range(2):
        helpful = np.abs(residual_train[:, dim] - candidate_train[:, dim]) < np.abs(residual_train[:, dim])
        edges = np.percentile(train_slope[:, dim], [0, 33, 66, 100]).astype(np.float32)
        edges[0] -= 1e-3
        edges[-1] += 1e-3
        rates = []
        for bucket in range(3):
            mask = (train_slope[:, dim] >= edges[bucket]) & (train_slope[:, dim] < edges[bucket + 1])
            rates.append(float(helpful[mask].mean()) if mask.any() else float(helpful.mean()))
        global_rate = max(float(helpful.mean()), 1e-3)
        gains = np.asarray([rate / global_rate for rate in rates], dtype=np.float32)
        gains = np.clip(gains, 0.25, 1.25)
        val_bucket = np.clip(np.searchsorted(edges[1:-1], val_slope[:, dim], side="right"), 0, 2)
        out[:, dim] = gains[val_bucket] * candidate_val[:, dim]
    return out


def group_indices(sample_ids: list[str]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    return groups


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
