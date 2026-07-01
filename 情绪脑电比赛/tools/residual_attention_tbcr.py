from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dual_tbcr_blend import parse_modes
from tools.run_iteration_experiments import expand_subjects, load_labels, predict_video_time_mean, score
from tools.scew_tbcr_blend import build_scew_prediction, format_float, insert_top
from tools.trial_basis_residual import (
    build_trials,
    fit_basis_coefficients,
    load_feature_cache,
    order_predictions,
    parse_floats,
    parse_ints,
    parse_sample_id,
    reconstruct_predictions,
    trial_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual attention refinement over Dual-TBCR corrections."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_086_residual_attention_tbcr.json")
    parser.add_argument(
        "--valence-checkpoints",
        nargs=2,
        default=[
            "experiments/checkpoints/graph_mamba/moddrop010_seed123.pt",
            "experiments/checkpoints/graph_mamba/itransformer_hybrid_159.pt",
        ],
    )
    parser.add_argument(
        "--arousal-checkpoints",
        nargs=2,
        default=[
            "experiments/checkpoints/graph_mamba/nobase_itransformer_arousal_159.pt",
            "experiments/checkpoints/graph_mamba/scalegated_msgm_arousal.pt",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--basis-counts", default="4")
    parser.add_argument("--alphas", default="0.01,0.05,0.25")
    parser.add_argument("--valence-mode", default="mean_std")
    parser.add_argument("--arousal-mode", default="mean_std_slope")
    parser.add_argument("--valence-weight", type=float, default=0.75)
    parser.add_argument("--arousal-weight", type=float, default=0.10)
    parser.add_argument("--valence-clip", type=float, default=4.2)
    parser.add_argument("--arousal-clip", type=float, default=5.5)
    parser.add_argument("--gammas", default="-0.5,-0.25,0,0.25,0.5,0.75,1")
    parser.add_argument("--temperatures", default="0.5,1,2,4")
    parser.add_argument("--distance-sigmas", default="0.15,0.3,0.6,1")
    parser.add_argument("--attention-modes", default="interp,residual_add")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    data_root = Path(args.data_root)
    labels = load_labels(data_root, train_subjects + val_subjects)
    train_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
    ]
    val_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in val_subjects
    ]
    y_train = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(np.float32)
    y_val = np.stack([labels[sample_id] for sample_id in val_label_ids]).astype(np.float32)
    prior_val = predict_video_time_mean(train_label_ids, y_train, val_label_ids)
    scew_pred = build_scew_prediction(args, prior_val, y_val)

    cache = load_feature_cache(Path(args.feature_cache))
    trials = build_trials(cache, labels, train_label_ids, y_train)
    train_trials = [trial for trial in trials if trial["subject"] in train_subjects]
    val_trials = [trial for trial in trials if trial["subject"] in val_subjects]

    basis_counts = parse_ints(args.basis_counts)
    alphas = parse_floats(args.alphas)
    valence_modes = parse_modes(args.valence_mode)
    arousal_modes = parse_modes(args.arousal_mode)
    gammas = parse_floats(args.gammas)
    temperatures = parse_floats(args.temperatures)
    distance_sigmas = parse_floats(args.distance_sigmas)
    attention_modes = [item.strip() for item in args.attention_modes.split(",") if item.strip()]

    residual_cache: dict[tuple[int, float, str], np.ndarray] = {}
    for basis_count in basis_counts:
        y_coeff_train = np.stack(
            [fit_basis_coefficients(trial["residual"], basis_count) for trial in train_trials],
            axis=0,
        ).reshape(len(train_trials), -1)
        for mode in sorted(set(valence_modes + arousal_modes)):
            x_train = np.stack([trial_features(trial["x"], mode) for trial in train_trials], axis=0)
            x_val = np.stack([trial_features(trial["x"], mode) for trial in val_trials], axis=0)
            for alpha in alphas:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x_train, y_coeff_train)
                coeff_pred = model.predict(x_val).reshape(len(val_trials), basis_count, 2)
                pred_by_id = reconstruct_predictions(val_trials, coeff_pred, basis_count)
                tbcr_pred = order_predictions(val_label_ids, pred_by_id)
                residual_cache[(basis_count, alpha, mode)] = tbcr_pred - prior_val

    top: list[dict[str, object]] = []
    reference = score("SCEW077_reference", y_val, scew_pred, "Reconstructed current best SCEW prediction.")
    baseline_items = []
    for basis_count in basis_counts:
        for alpha_v in alphas:
            for alpha_a in alphas:
                rv = residual_cache[(basis_count, alpha_v, valence_modes[0])][:, 0]
                ra = residual_cache[(basis_count, alpha_a, arousal_modes[0])][:, 1]
                base_cv = np.clip(args.valence_weight * rv, -args.valence_clip, args.valence_clip)
                base_ca = np.clip(args.arousal_weight * ra, -args.arousal_clip, args.arousal_clip)
                base_pred = np.clip(scew_pred + np.stack([base_cv, base_ca], axis=1), 1.0, 255.0)
                baseline = score(
                    f"DualTBCR_base_k{basis_count}_av{format_float(alpha_v)}_aa{format_float(alpha_a)}",
                    y_val,
                    base_pred,
                    "Dual-TBCR correction before residual attention.",
                )
                baseline["basis_count"] = basis_count
                baseline["alpha_valence"] = alpha_v
                baseline["alpha_arousal"] = alpha_a
                baseline_items.append(baseline)
                insert_top(top, baseline, args.top_k)

                for mode in attention_modes:
                    if mode not in {"interp", "residual_add"}:
                        raise ValueError(f"Unknown attention mode: {mode}")
                    for gamma_v in gammas:
                        for gamma_a in gammas:
                            if gamma_v == 0 and gamma_a == 0:
                                continue
                            for temperature in temperatures:
                                for distance_sigma in distance_sigmas:
                                    cv = residual_attention_by_trial(
                                        sample_ids=val_label_ids,
                                        prior=prior_val[:, 0],
                                        correction=base_cv,
                                        gamma=gamma_v,
                                        temperature=temperature,
                                        distance_sigma=distance_sigma,
                                        mode=mode,
                                    )
                                    ca = residual_attention_by_trial(
                                        sample_ids=val_label_ids,
                                        prior=prior_val[:, 1],
                                        correction=base_ca,
                                        gamma=gamma_a,
                                        temperature=temperature,
                                        distance_sigma=distance_sigma,
                                        mode=mode,
                                    )
                                    cv = np.clip(cv, -args.valence_clip, args.valence_clip)
                                    ca = np.clip(ca, -args.arousal_clip, args.arousal_clip)
                                    pred = np.clip(scew_pred + np.stack([cv, ca], axis=1), 1.0, 255.0)
                                    item = score(
                                        (
                                            f"LRAG_k{basis_count}_av{format_float(alpha_v)}_aa{format_float(alpha_a)}"
                                            f"_mode{mode}_gv{format_float(gamma_v)}_ga{format_float(gamma_a)}"
                                            f"_temp{format_float(temperature)}_dist{format_float(distance_sigma)}"
                                        ),
                                        y_val,
                                        pred,
                                        "Latent residual attention refinement over clipped Dual-TBCR corrections.",
                                    )
                                    item["basis_count"] = basis_count
                                    item["alpha_valence"] = alpha_v
                                    item["alpha_arousal"] = alpha_a
                                    item["attention_mode"] = mode
                                    item["gamma_valence"] = gamma_v
                                    item["gamma_arousal"] = gamma_a
                                    item["temperature"] = temperature
                                    item["distance_sigma"] = distance_sigma
                                    insert_top(top, item, args.top_k)

    output = {
        "method": "LRAG: latent residual attention gate over Dual-TBCR corrections",
        "formula": (
            "Within each trial, build low-dimensional token z_t=[time, sin(time), cos(time), "
            "zscore(prior), zscore(prior_slope), zscore(correction)]. "
            "A=softmax(z_t z_s^T / sqrt(d) / temperature - distance_penalty); "
            "correction'=correction + gamma*(A correction - correction) for interp, "
            "or correction'=correction + gamma*A correction for residual_add."
        ),
        "feature_cache": args.feature_cache,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_trials": len(train_trials),
            "val_trials": len(val_trials),
            "val_samples": len(val_label_ids),
        },
        "reference": reference,
        "dual_tbcr_baselines": sorted(baseline_items, key=lambda item: float(item["overall_mae"]))[:20],
        "results": top,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def residual_attention_by_trial(
    sample_ids: list[str],
    prior: np.ndarray,
    correction: np.ndarray,
    gamma: float,
    temperature: float,
    distance_sigma: float,
    mode: str,
) -> np.ndarray:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))

    refined = correction.astype(np.float32).copy()
    for items in groups.values():
        ordered = sorted(items)
        indices = np.asarray([index for _, index in ordered], dtype=np.int64)
        if len(indices) < 3:
            continue
        local_prior = prior[indices].astype(np.float32)
        local_corr = correction[indices].astype(np.float32)
        tokens = make_attention_tokens(local_prior, local_corr)
        logits = tokens @ tokens.T / math.sqrt(tokens.shape[1])
        logits = logits / max(temperature, 1e-6)
        t = np.arange(len(indices), dtype=np.float32)
        distance = np.abs(t[:, None] - t[None, :]) / max(len(indices) - 1, 1)
        logits -= (distance / max(distance_sigma, 1e-6)) ** 2
        weights = softmax(logits, axis=1)
        attended = weights @ local_corr
        if mode == "interp":
            refined[indices] = local_corr + gamma * (attended - local_corr)
        else:
            refined[indices] = local_corr + gamma * attended
    return refined.astype(np.float32)


def make_attention_tokens(prior: np.ndarray, correction: np.ndarray) -> np.ndarray:
    length = len(prior)
    time = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    slope = np.gradient(prior).astype(np.float32)
    features = np.stack(
        [
            time,
            np.sin(np.pi * time),
            np.cos(np.pi * time),
            zscore(prior),
            zscore(slope),
            zscore(correction),
        ],
        axis=1,
    )
    return features.astype(np.float32)


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=axis, keepdims=True)).astype(np.float32)


if __name__ == "__main__":
    main()
