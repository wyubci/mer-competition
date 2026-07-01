from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-video-prior physiological adaptive batch 356-375."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_356_375_no_prior_physio_adaptive.json")
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root=data_root,
        subjects=subjects,
        cache_path=Path(args.precompute_cache),
        fnirs_types=(0, 1, 2, 3, 4, 5),
        feature_normalization="none",
        baseline_correction=True,
    )
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] train={len(train_subjects)} val={val_subjects}", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)
        center = np.full_like(y_val, 128.0)

        x_raw_train = pre["early_concat"][train_idx]
        x_raw_val = pre["early_concat"][val_idx]
        x_eeg_train = pre["eeg_lag"][train_idx]
        x_eeg_val = pre["eeg_lag"][val_idx]
        x_fnirs_train = pre["fnirs_slow"][train_idx]
        x_fnirs_val = pre["fnirs_slow"][val_idx]
        x_neuro_train = pre["neurovascular"][train_idx]
        x_neuro_val = pre["neurovascular"][val_idx]

        base = build_reference_predictions(x_raw_train, x_raw_val, y_train, val_ids)
        candidates = build_candidates(
            center=center,
            train_ids=train_ids,
            val_ids=val_ids,
            y_train=y_train,
            x_raw_train=x_raw_train,
            x_raw_val=x_raw_val,
            x_eeg_train=x_eeg_train,
            x_eeg_val=x_eeg_val,
            x_fnirs_train=x_fnirs_train,
            x_fnirs_val=x_fnirs_val,
            x_neuro_train=x_neuro_train,
            x_neuro_val=x_neuro_val,
            base=base,
        )
        references = {
            "Reference_321_Center128_noPrior": center,
            "Reference_337_PCA16Valence_CenterArousal": dimwise(base["pca16_s5"], center, True, False),
            "Reference_354_AgreementGatedPCA16Valence_CenterArousal": agreement_gate(
                center, base["pca8_s5"], base["pca16_s5"], threshold=8.0
            ),
        }
        fold_predictions = {**references, **candidates}

        fold_results = []
        for name, pred in fold_predictions.items():
            pred = np.clip(pred.astype(np.float32), 1.0, 255.0)
            note = candidate_note(name)
            fold_results.append(score(name, y_val, pred, note))
            add_metric(metric_acc, name, y_val, pred, note)
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
        "method": "No-video-prior physiological adaptive module batch",
        "iteration_range": "356-375",
        "note": (
            "No candidate uses video/time label priors. This batch tests unsupervised subject/trial "
            "feature alignment, robust heads, residual mean constraints, temporal difference features, "
            "and conservative arousal gates."
        ),
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_reference_predictions(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
) -> dict[str, np.ndarray]:
    base = {
        "pca8": pca_ridge_predict(x_train, y_train, x_val, components=8, alpha=10000.0),
        "pca16": pca_ridge_predict(x_train, y_train, x_val, components=16, alpha=10000.0),
    }
    for name, pred in list(base.items()):
        base[f"{name}_s5"] = smooth_predictions(val_ids, pred, 5).astype(np.float32)
    return base


def build_candidates(
    center: np.ndarray,
    train_ids: list[str],
    val_ids: list[str],
    y_train: np.ndarray,
    x_raw_train: np.ndarray,
    x_raw_val: np.ndarray,
    x_eeg_train: np.ndarray,
    x_eeg_val: np.ndarray,
    x_fnirs_train: np.ndarray,
    x_fnirs_val: np.ndarray,
    x_neuro_train: np.ndarray,
    x_neuro_val: np.ndarray,
    base: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}

    train_subject_center, val_subject_center = group_center(x_raw_train, train_ids, x_raw_val, val_ids, "subject")
    train_subject_z, val_subject_z = group_zscore(x_raw_train, train_ids, x_raw_val, val_ids, "subject")
    train_trial_center, val_trial_center = group_center(x_raw_train, train_ids, x_raw_val, val_ids, "trial")
    train_trial_z, val_trial_z = group_zscore(x_raw_train, train_ids, x_raw_val, val_ids, "trial")
    train_delta = temporal_delta(x_raw_train, train_ids)
    val_delta = temporal_delta(x_raw_val, val_ids)
    train_level_delta = np.concatenate([x_raw_train, train_delta], axis=1)
    val_level_delta = np.concatenate([x_raw_val, val_delta], axis=1)

    candidates["356_SubjectCenterPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(train_subject_center, y_train, val_subject_center, 16, 10000.0), val_ids
    )
    candidates["357_SubjectZPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(train_subject_z, y_train, val_subject_z, 16, 10000.0), val_ids
    )
    candidates["358_TrialCenterPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(train_trial_center, y_train, val_trial_center, 16, 10000.0), val_ids
    )
    candidates["359_TrialZPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(train_trial_z, y_train, val_trial_z, 16, 10000.0), val_ids
    )
    candidates["360_SubjectCenterPLS2Valence_CenterArousal"] = valence_center(
        center, pls_predict(train_subject_center, y_train[:, 0], val_subject_center, 2), val_ids
    )
    candidates["361_TrialCenterPLS2Valence_CenterArousal"] = valence_center(
        center, pls_predict(train_trial_center, y_train[:, 0], val_trial_center, 2), val_ids
    )
    candidates["362_RobustScalerPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(x_raw_train, y_train, x_raw_val, 16, 10000.0, robust=True), val_ids
    )
    candidates["363_ZeroMeanResidualPCA16Valence_CenterArousal"] = residual_valence_center(
        center, pca_ridge_residual_predict(x_raw_train, y_train[:, 0] - 128.0, x_raw_val, 16, 10000.0, no_intercept=True), val_ids
    )
    candidates["364_HuberPCA16Valence_CenterArousal"] = valence_center(
        center, pca_head_predict(x_raw_train, y_train[:, 0], x_raw_val, 16, "huber"), val_ids
    )
    candidates["365_BayesianPCA16Valence_CenterArousal"] = valence_center(
        center, pca_head_predict(x_raw_train, y_train[:, 0], x_raw_val, 16, "bayes"), val_ids
    )
    candidates["366_ElasticNetPCA16Valence_CenterArousal"] = valence_center(
        center, pca_head_predict(x_raw_train, y_train[:, 0], x_raw_val, 16, "elastic"), val_ids
    )
    candidates["367_HistGBPCA16Valence_CenterArousal"] = valence_center(
        center, pca_head_predict(x_raw_train, y_train[:, 0], x_raw_val, 16, "histgb"), val_ids
    )
    candidates["368_NystroemRBFValence_CenterArousal"] = valence_center(
        center, nystroem_ridge_predict(x_raw_train, y_train[:, 0], x_raw_val), val_ids
    )
    candidates["369_DeltaPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(train_delta, y_train, val_delta, 16, 10000.0), val_ids
    )
    candidates["370_LevelDeltaPCA16Valence_CenterArousal"] = valence_center(
        center, pca_ridge_predict(train_level_delta, y_train, val_level_delta, 16, 10000.0), val_ids
    )
    candidates["371_EEGFNIRSAgreementValence_CenterArousal"] = agreement_average(
        center,
        valence_model(x_eeg_train, y_train[:, 0], x_eeg_val, val_ids),
        valence_model(x_fnirs_train, y_train[:, 0], x_fnirs_val, val_ids),
        threshold=10.0,
    )
    candidates["372_RawSubjectAgreementValence_CenterArousal"] = agreement_average(
        center,
        smooth_predictions(val_ids, pca_ridge_predict(x_raw_train, y_train, x_raw_val, 16, 10000.0), 5)[:, 0],
        smooth_predictions(val_ids, pca_ridge_predict(train_subject_center, y_train, val_subject_center, 16, 10000.0), 5)[:, 0],
        threshold=8.0,
    )
    candidates["373_NeurovascularAgreementValence_CenterArousal"] = agreement_average(
        center,
        smooth_predictions(val_ids, pca_ridge_predict(x_raw_train, y_train, x_raw_val, 16, 10000.0), 5)[:, 0],
        valence_model(x_neuro_train, y_train[:, 0], x_neuro_val, val_ids),
        threshold=10.0,
    )
    candidates["374_PCA16Valence_ArousalTinyAgreementGate"] = arousal_tiny_gate(
        center, base["pca8_s5"], base["pca16_s5"], valence_source=base["pca16_s5"], threshold=6.0, scale=0.20
    )
    candidates["375_PCA16Valence_ArousalSignedSmallGate"] = arousal_tiny_gate(
        center, base["pca8_s5"], base["pca16_s5"], valence_source=base["pca16_s5"], threshold=4.0, scale=0.10
    )

    return candidates


def valence_model(x_train: np.ndarray, y_train_v: np.ndarray, x_val: np.ndarray, val_ids: list[str]) -> np.ndarray:
    pred = pca_head_predict(x_train, y_train_v, x_val, components=min(8, x_train.shape[1]), head="bayes")
    return smooth_predictions(val_ids, pred.reshape(-1, 1), 5)[:, 0]


def dimwise(source: np.ndarray, fallback: np.ndarray, use_valence: bool, use_arousal: bool) -> np.ndarray:
    out = fallback.copy()
    if use_valence:
        out[:, 0] = source[:, 0]
    if use_arousal:
        out[:, 1] = source[:, 1]
    return out


def valence_center(center: np.ndarray, source: np.ndarray, val_ids: list[str]) -> np.ndarray:
    pred = source
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.shape[1] == 1:
        smoothed_v = smooth_predictions(val_ids, pred, 5)[:, 0]
    else:
        smoothed_v = smooth_predictions(val_ids, pred, 5)[:, 0]
    out = center.copy()
    out[:, 0] = smoothed_v
    return out


def residual_valence_center(center: np.ndarray, residual_v: np.ndarray, val_ids: list[str]) -> np.ndarray:
    residual_v = np.asarray(residual_v, dtype=np.float32).reshape(-1, 1)
    smoothed_residual = smooth_predictions(val_ids, residual_v, 5)[:, 0]
    out = center.copy()
    out[:, 0] = center[:, 0] + smoothed_residual
    return out


def agreement_gate(center: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, threshold: float) -> np.ndarray:
    out = center.copy()
    disagreement = np.abs(pred_a[:, 0] - pred_b[:, 0])
    out[:, 0] = np.where(disagreement < threshold, pred_b[:, 0], center[:, 0])
    return out


def agreement_average(center: np.ndarray, pred_a_v: np.ndarray, pred_b_v: np.ndarray, threshold: float) -> np.ndarray:
    out = center.copy()
    disagreement = np.abs(pred_a_v - pred_b_v)
    avg = 0.5 * (pred_a_v + pred_b_v)
    out[:, 0] = np.where(disagreement < threshold, avg, center[:, 0])
    return out


def arousal_tiny_gate(
    center: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    valence_source: np.ndarray,
    threshold: float,
    scale: float,
) -> np.ndarray:
    out = center.copy()
    out[:, 0] = valence_source[:, 0]
    disagreement = np.abs(pred_a[:, 1] - pred_b[:, 1])
    residual = pred_b[:, 1] - center[:, 1]
    out[:, 1] = np.where(disagreement < threshold, center[:, 1] + scale * residual, center[:, 1])
    return out


def pca_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    components: int,
    alpha: float,
    robust: bool = False,
) -> np.ndarray:
    scaler = RobustScaler(quantile_range=(10.0, 90.0)) if robust else StandardScaler()
    model = make_pipeline(
        scaler,
        PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026),
        Ridge(alpha=alpha, solver="lsqr"),
    )
    model.fit(sanitize(x_train), y_train)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def pca_ridge_residual_predict(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    x_val: np.ndarray,
    components: int,
    alpha: float,
    no_intercept: bool,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026),
        Ridge(alpha=alpha, solver="lsqr", fit_intercept=not no_intercept),
    )
    model.fit(sanitize(x_train), residual_train)
    return model.predict(sanitize(x_val)).astype(np.float32)


def pca_head_predict(
    x_train: np.ndarray,
    y_train_v: np.ndarray,
    x_val: np.ndarray,
    components: int,
    head: str,
) -> np.ndarray:
    steps = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026)),
    ]
    if head == "huber":
        reg = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=300)
    elif head == "bayes":
        reg = BayesianRidge()
    elif head == "elastic":
        reg = ElasticNet(alpha=0.02, l1_ratio=0.15, max_iter=3000, random_state=2026)
    elif head == "histgb":
        reg = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=1.0,
            max_leaf_nodes=15,
            random_state=2026,
        )
    else:
        raise ValueError(f"Unknown head: {head}")
    model = make_pipeline(*(step for _, step in steps), reg)
    model.fit(sanitize(x_train), y_train_v)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def pls_predict(x_train: np.ndarray, y_train_v: np.ndarray, x_val: np.ndarray, components: int) -> np.ndarray:
    model = make_pipeline(StandardScaler(), PLSRegression(n_components=components, scale=False))
    model.fit(sanitize(x_train), y_train_v)
    pred = model.predict(sanitize(x_val))
    return np.clip(np.asarray(pred).reshape(-1).astype(np.float32), 1.0, 255.0)


def nystroem_ridge_predict(x_train: np.ndarray, y_train_v: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=16, svd_solver="randomized", random_state=2026),
        Nystroem(kernel="rbf", gamma=0.08, n_components=64, random_state=2026),
        Ridge(alpha=1000.0, solver="lsqr"),
    )
    model.fit(sanitize(x_train), y_train_v)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def group_key(sample_id: str, mode: str) -> str:
    subject, rest = sample_id.split("_V", 1)
    if mode == "subject":
        return subject
    video = rest.split("_T", 1)[0]
    return f"{subject}_V{video}"


def group_center(
    x_train: np.ndarray,
    train_ids: list[str],
    x_val: np.ndarray,
    val_ids: list[str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_group_transform(x_train, train_ids, mode, "center"), apply_group_transform(x_val, val_ids, mode, "center")


def group_zscore(
    x_train: np.ndarray,
    train_ids: list[str],
    x_val: np.ndarray,
    val_ids: list[str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_group_transform(x_train, train_ids, mode, "zscore"), apply_group_transform(x_val, val_ids, mode, "zscore")


def apply_group_transform(x: np.ndarray, sample_ids: list[str], mode: str, transform: str) -> np.ndarray:
    out = sanitize(x).copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[group_key(sample_id, mode)].append(index)
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        seq = out[idx]
        mean = seq.mean(axis=0, keepdims=True)
        if transform == "center":
            out[idx] = seq - mean
        elif transform == "zscore":
            std = np.maximum(seq.std(axis=0, keepdims=True), 1e-4)
            out[idx] = (seq - mean) / std
        else:
            raise ValueError(transform)
    return sanitize(out)


def temporal_delta(x: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[group_key(sample_id, "trial")].append(index)
    for indices in groups.values():
        idx = np.asarray(sorted(indices), dtype=np.int64)
        seq = x[idx]
        delta = np.zeros_like(seq, dtype=np.float32)
        delta[1:] = seq[1:] - seq[:-1]
        out[idx] = delta
    return sanitize(out)


def add_metric(
    metric_acc: dict[str, dict[str, object]],
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    notes: str,
) -> None:
    payload = metric_acc.setdefault(
        name,
        {
            "abs_sum": np.zeros(2, dtype=np.float64),
            "sq_sum": np.zeros(2, dtype=np.float64),
            "count": 0,
            "notes": notes,
        },
    )
    diff = y_pred - y_true
    payload["abs_sum"] += np.abs(diff).sum(axis=0)
    payload["sq_sum"] += (diff**2).sum(axis=0)
    payload["count"] += int(y_true.shape[0])


def finalize_metric(name: str, payload: dict[str, object]) -> dict[str, object]:
    count = max(int(payload["count"]), 1)
    abs_sum = np.asarray(payload["abs_sum"], dtype=np.float64)
    sq_sum = np.asarray(payload["sq_sum"], dtype=np.float64)
    return {
        "method": name,
        "overall_mae": round(float(abs_sum.sum() / (2 * count)), 4),
        "valence_mae": round(float(abs_sum[0] / count), 4),
        "arousal_mae": round(float(abs_sum[1] / count), 4),
        "overall_mse": round(float(sq_sum.sum() / (2 * count)), 4),
        "notes": str(payload["notes"]),
    }


def candidate_note(name: str) -> str:
    if name.startswith("Reference"):
        return "Reference from previous no-video-prior physiological batches."
    if "Subject" in name:
        return "Unsupervised subject-level signal alignment; uses no labels from validation subjects."
    if "Trial" in name:
        return "Unsupervised trial-level signal alignment; tests whether within-trial dynamics matter more than level."
    if "Residual" in name:
        return "Zero-mean residual head around center to reduce label-prior leakage and cross-subject bias."
    if "Huber" in name or "Bayesian" in name or "ElasticNet" in name or "HistGB" in name or "Nystroem" in name:
        return "Alternative robust/nonlinear valence head over low-rank physiological features."
    if "Delta" in name:
        return "Temporal-difference physiological features; tests whether change is safer than absolute level."
    if "Agreement" in name:
        return "Conservative agreement gate between independently transformed physiological views."
    if "Arousal" in name:
        return "Tiny arousal residual gate; tests whether arousal can be corrected without damaging center baseline."
    return "No-video-prior physiological adaptive candidate."


if __name__ == "__main__":
    main()
