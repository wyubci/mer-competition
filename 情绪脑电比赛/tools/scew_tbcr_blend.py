from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import expand_subjects, load_labels, predict_video_time_mean, score
from tools.scew_smoothing_search import build_state_residual
from tools.trial_basis_residual import (  # noqa: E402
    build_trials,
    cosine_basis,
    fit_basis_coefficients,
    load_feature_cache,
    order_predictions,
    parse_floats,
    parse_ints,
    reconstruct_predictions,
    trial_features,
)
from tools.consensus_gated_residual import load_payloads  # noqa: E402

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend current SCEW residual route with trial-basis residual corrections."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_080_scew_tbcr_blend.json")
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
    parser.add_argument("--basis-counts", default="2,3,4,6,8")
    parser.add_argument("--alphas", default="1,10,100,1000")
    parser.add_argument("--feature-mode", choices=("mean", "mean_std", "mean_std_slope"), default="mean_std")
    parser.add_argument("--blend-weights", default="-0.5,-0.25,-0.1,0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--clips", default="0,2,5,8,10")
    parser.add_argument("--top-k", type=int, default=80)
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
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(
        np.float32
    )
    y_val = np.stack([labels[sample_id] for sample_id in val_label_ids]).astype(np.float32)
    prior_val = predict_video_time_mean(train_label_ids, y_train_label_order, val_label_ids)
    scew_pred = build_scew_prediction(args, prior_val, y_val)

    cache = load_feature_cache(Path(args.feature_cache))
    trials = build_trials(cache, labels, train_label_ids, y_train_label_order)
    train_trials = [trial for trial in trials if trial["subject"] in train_subjects]
    val_trials = [trial for trial in trials if trial["subject"] in val_subjects]

    results: list[dict[str, object]] = [
        score("SCEW077_reference", y_val, scew_pred, "Reconstructed current best SCEW prediction."),
    ]
    basis_counts = parse_ints(args.basis_counts)
    alphas = parse_floats(args.alphas)
    blend_weights = parse_floats(args.blend_weights)
    clips = parse_floats(args.clips)

    top: list[dict[str, object]] = []
    for basis_count in basis_counts:
        x_train = np.stack(
            [trial_features(trial["x"], args.feature_mode) for trial in train_trials], axis=0
        )
        x_val = np.stack(
            [trial_features(trial["x"], args.feature_mode) for trial in val_trials], axis=0
        )
        y_coeff_train = np.stack(
            [fit_basis_coefficients(trial["residual"], basis_count) for trial in train_trials],
            axis=0,
        ).reshape(len(train_trials), -1)
        for alpha in alphas:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, y_coeff_train)
            coeff_pred = model.predict(x_val).reshape(len(val_trials), basis_count, 2)
            pred_by_id = reconstruct_predictions(val_trials, coeff_pred, basis_count)
            tbcr_pred = order_predictions(val_label_ids, pred_by_id)
            tbcr_residual = tbcr_pred - prior_val
            for w_v in blend_weights:
                for w_a in blend_weights:
                    blended_residual = np.stack(
                        [w_v * tbcr_residual[:, 0], w_a * tbcr_residual[:, 1]], axis=1
                    )
                    for clip in clips:
                        correction = blended_residual
                        if clip > 0:
                            correction = np.clip(correction, -clip, clip)
                        pred = np.clip(scew_pred + correction, 1.0, 255.0)
                        item = score(
                            f"SCEW_TBCR_k{basis_count}_a{format_float(alpha)}_wv{format_float(w_v)}_wa{format_float(w_a)}_clip{format_float(clip)}",
                            y_val,
                            pred,
                            "Current SCEW prediction plus clipped TBCR low-frequency correction.",
                        )
                        item["basis_count"] = basis_count
                        item["alpha"] = alpha
                        item["w_valence"] = w_v
                        item["w_arousal"] = w_a
                        item["clip"] = clip
                        insert_top(top, item, args.top_k)

    output = {
        "method": "SCEW + TBCR low-frequency correction blend",
        "feature_cache": args.feature_cache,
        "feature_mode": args.feature_mode,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_trials": len(train_trials),
            "val_trials": len(val_trials),
            "val_samples": len(val_label_ids),
        },
        "reference": results[0],
        "results": top,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_scew_prediction(args: argparse.Namespace, prior_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valence_payloads = load_payloads(args.valence_checkpoints, args.batch_size, device)
    arousal_payloads = load_payloads(args.arousal_checkpoints, args.batch_size, device)
    reference = valence_payloads[0]
    sample_ids = reference["sample_ids"]
    prior = reference["prior"]
    if not np.allclose(prior, prior_val):
        print("[warn] checkpoint prior order differs from label prior; using checkpoint prior", flush=True)
    v_stack = np.stack(
        [payload["y_pred"][:, 0] - prior[:, 0] for payload in valence_payloads],
        axis=0,
    )
    a_stack = np.stack(
        [payload["y_pred"][:, 1] - prior[:, 1] for payload in arousal_payloads],
        axis=0,
    )
    valence_residual = build_state_residual(
        sample_ids=sample_ids,
        prior=prior[:, 0],
        residual_stack=v_stack,
        slope_quantile=45.0,
        state_lags={"stable": -2, "rising": 0, "falling": -14},
        state_multipliers={"stable": 1.25, "rising": 1.10, "falling": 0.75},
        state_first_expert_weights={"stable": 0.75, "rising": 0.175, "falling": 0.925},
        cdg_params={
            "sigma_multiplier": 4.0,
            "min_gate": 0.0,
            "max_gate": 1.5,
            "sign_penalty": 1.0,
        },
    )
    arousal_residual = build_state_residual(
        sample_ids=sample_ids,
        prior=prior[:, 1],
        residual_stack=a_stack,
        slope_quantile=50.0,
        state_lags={"stable": -16, "rising": -12, "falling": -10},
        state_multipliers={"stable": 0.75, "rising": 0.75, "falling": 1.50},
        state_first_expert_weights={"stable": 1.0, "rising": 1.0, "falling": 0.0},
        cdg_params={
            "sigma_multiplier": 2.0,
            "min_gate": 0.5,
            "max_gate": 1.5,
            "sign_penalty": 0.25,
        },
    )
    from tools.consensus_gated_residual import finalize_prediction

    pred_v = finalize_prediction(sample_ids, prior[:, 0], valence_residual, 11.5, 10.0, 25, 10.0)
    pred_a = finalize_prediction(sample_ids, prior[:, 1], arousal_residual, 0.2, 0.0, 15, 15.0)
    return np.stack([pred_v, pred_a], axis=1).astype(np.float32)


def insert_top(top: list[dict[str, object]], entry: dict[str, object], top_k: int) -> None:
    top.append(entry)
    top.sort(key=lambda item: float(item["overall_mae"]))
    if len(top) > top_k:
        top.pop()


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
