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
from tools.scew_tbcr_blend import build_scew_prediction, format_float, insert_top
from tools.trial_basis_residual import (
    build_trials,
    fit_basis_coefficients,
    load_feature_cache,
    order_predictions,
    parse_floats,
    parse_ints,
    reconstruct_predictions,
    trial_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dimension-specific TBCR blend: use different trial features for valence and arousal."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_084_dual_tbcr_blend.json")
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
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1,2")
    parser.add_argument("--valence-modes", default="mean_std")
    parser.add_argument("--arousal-modes", default="mean_std_slope")
    parser.add_argument("--valence-weights", default="0,0.05,0.1,0.25,0.5,0.75,1,1.25,1.5")
    parser.add_argument("--arousal-weights", default="0,0.025,0.05,0.075,0.1,0.125,0.15,0.2,0.25")
    parser.add_argument("--valence-clips", default="3,3.5,4,4.5,5,6")
    parser.add_argument("--arousal-clips", default="3,4,5,6,7,8,10")
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
    valence_modes = parse_modes(args.valence_modes)
    arousal_modes = parse_modes(args.arousal_modes)
    valence_weights = parse_floats(args.valence_weights)
    arousal_weights = parse_floats(args.arousal_weights)
    valence_clips = parse_floats(args.valence_clips)
    arousal_clips = parse_floats(args.arousal_clips)

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
    for basis_count in basis_counts:
        for alpha_v in alphas:
            for alpha_a in alphas:
                for mode_v in valence_modes:
                    rv = residual_cache[(basis_count, alpha_v, mode_v)][:, 0]
                    for mode_a in arousal_modes:
                        ra = residual_cache[(basis_count, alpha_a, mode_a)][:, 1]
                        for w_v in valence_weights:
                            for clip_v in valence_clips:
                                cv = w_v * rv
                                if clip_v > 0:
                                    cv = np.clip(cv, -clip_v, clip_v)
                                for w_a in arousal_weights:
                                    for clip_a in arousal_clips:
                                        ca = w_a * ra
                                        if clip_a > 0:
                                            ca = np.clip(ca, -clip_a, clip_a)
                                        pred = np.clip(
                                            scew_pred + np.stack([cv, ca], axis=1),
                                            1.0,
                                            255.0,
                                        )
                                        item = score(
                                            (
                                                f"DualTBCR_k{basis_count}"
                                                f"_v{mode_v}_a{format_float(alpha_v)}_wv{format_float(w_v)}_cv{format_float(clip_v)}"
                                                f"_a{mode_a}_a{format_float(alpha_a)}_wa{format_float(w_a)}_ca{format_float(clip_a)}"
                                            ),
                                            y_val,
                                            pred,
                                            "Dimension-specific TBCR correction on top of SCEW.",
                                        )
                                        item["basis_count"] = basis_count
                                        item["valence_mode"] = mode_v
                                        item["arousal_mode"] = mode_a
                                        item["alpha_valence"] = alpha_v
                                        item["alpha_arousal"] = alpha_a
                                        item["w_valence"] = w_v
                                        item["w_arousal"] = w_a
                                        item["clip_valence"] = clip_v
                                        item["clip_arousal"] = clip_a
                                        insert_top(top, item, args.top_k)

    output = {
        "method": "Dual TBCR dimension-specific correction blend",
        "feature_cache": args.feature_cache,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_trials": len(train_trials),
            "val_trials": len(val_trials),
            "val_samples": len(val_label_ids),
        },
        "reference": reference,
        "results": top,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"mean", "mean_std", "mean_std_slope"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown feature modes: {unknown}")
    return modes


if __name__ == "__main__":
    main()
