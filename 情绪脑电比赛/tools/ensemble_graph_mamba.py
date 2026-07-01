from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.graph_mamba import GraphMambaResidualRegressor  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions
from tools.train_graph_mamba import (  # noqa: E402
    TrialDataset,
    collate_trials,
    evaluate,
    load_trial_examples,
    parse_windows,
)


DEFAULT_CHECKPOINTS = (
    "experiments/checkpoints/graph_mamba/scalegated_msgm_159.pt",
    "experiments/checkpoints/graph_mamba/moddrop010.pt",
    "experiments/checkpoints/graph_mamba/best_seed7.pt",
    "experiments/checkpoints/graph_mamba/best_seed123.pt",
    "experiments/checkpoints/graph_mamba/moddrop010_seed7.pt",
    "experiments/checkpoints/graph_mamba/moddrop010_seed123.pt",
    "experiments/checkpoints/graph_mamba/hybrid_functional_graph.pt",
    "experiments/checkpoints/graph_mamba/hybrid_functional_moddrop010.pt",
    "experiments/checkpoints/graph_mamba/convmixer_159.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual ensembles of Graph-Mamba checkpoints.")
    parser.add_argument("--checkpoints", nargs="*", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--output", default="experiments/results/iteration_023_checkpoint_ensemble.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-ensemble-size", type=int, default=4)
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,2.25,2.5,2.75,3,3.5,4")
    parser.add_argument(
        "--clips",
        default="0",
        help="Comma-separated absolute residual clips after scaling; 0 disables clipping.",
    )
    parser.add_argument("--smooth-windows", default="0,3,5,9")
    parser.add_argument(
        "--pair-weight-step",
        type=float,
        default=0.0,
        help="If positive, additionally sweep weighted two-checkpoint residual ensembles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [Path(path) for path in args.checkpoints if Path(path).exists()]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint paths exist.")

    predictions = []
    names = []
    reference: dict[str, object] | None = None
    single_results = []
    for checkpoint in checkpoints:
        payload = predict_checkpoint(checkpoint, args.batch_size, device)
        if reference is None:
            reference = {
                "sample_ids": payload["sample_ids"],
                "y_true": payload["y_true"],
                "prior": payload["prior"],
            }
        else:
            if payload["sample_ids"] != reference["sample_ids"]:
                raise ValueError(f"sample_id order mismatch for {checkpoint}")
        names.append(checkpoint.stem)
        predictions.append(payload["y_pred"])
        single_results.append(payload["stats"] | {"checkpoint": str(checkpoint)})

    assert reference is not None
    y_true = reference["y_true"]
    prior = reference["prior"]
    sample_ids = reference["sample_ids"]
    residuals = [prediction - prior for prediction in predictions]
    scales = [float(value) for value in args.scales.split(",") if value.strip()]
    clips = [float(value) for value in args.clips.split(",") if value.strip()]
    smooth_windows = [int(value) for value in args.smooth_windows.split(",") if value.strip()]

    ensemble_results = []
    max_size = min(args.max_ensemble_size, len(residuals))
    for size in range(1, max_size + 1):
        for indices in itertools.combinations(range(len(residuals)), size):
            residual = np.mean([residuals[index] for index in indices], axis=0)
            label = "+".join(names[index] for index in indices)
            append_postprocess_results(
                ensemble_results,
                sample_ids,
                y_true,
                prior,
                residual,
                f"Ensemble_{label}",
                scales,
                clips,
                smooth_windows,
                "Equal-weight checkpoint residual ensemble over VideoTimeMean prior.",
            )
            if size == 2 and args.pair_weight_step > 0:
                steps = int(round(1.0 / args.pair_weight_step))
                first, second = indices
                for step in range(steps + 1):
                    first_weight = step / steps
                    weighted_residual = (
                        first_weight * residuals[first] + (1.0 - first_weight) * residuals[second]
                    )
                    append_postprocess_results(
                        ensemble_results,
                        sample_ids,
                        y_true,
                        prior,
                        weighted_residual,
                        f"WeightedEnsemble_{names[first]}_{first_weight:.2f}+{names[second]}_{1.0 - first_weight:.2f}",
                        scales,
                        clips,
                        smooth_windows,
                        "Weighted two-checkpoint residual ensemble over VideoTimeMean prior.",
                    )

    ensemble_results = sorted(ensemble_results, key=lambda item: float(item["overall_mae"]))
    output = {
        "device": str(device),
        "checkpoints": [str(path) for path in checkpoints],
        "single_results": sorted(single_results, key=lambda item: float(item["overall_mae"])),
        "ensemble_results": ensemble_results[:100],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def predict_checkpoint(
    checkpoint_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    train_subjects = expand_subjects(ckpt_args["train_subjects"])
    val_subjects = expand_subjects(ckpt_args["val_subjects"])
    data_root = Path(ckpt_args["data_root"])
    labels = load_labels(data_root, train_subjects + val_subjects)
    train_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
    ]
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(
        np.float32
    )
    multiscale_windows = parse_windows(ckpt_args.get("multiscale_windows", "1"))
    examples, summary = load_trial_examples(
        Path(ckpt_args["feature_cache"]),
        train_subjects,
        val_subjects,
        train_label_ids,
        y_train_label_order,
        multiscale_windows,
        ckpt_args.get("feature_norm", "none"),
    )
    val_examples = [example for example in examples if example["subject"] in val_subjects]
    loader = DataLoader(
        TrialDataset(val_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trials,
        num_workers=0,
    )
    model = GraphMambaResidualRegressor(
        eeg_nodes=int(summary.get("eeg_nodes", 64)),
        eeg_features=int(summary["eeg_feature_dim"]),
        fnirs_nodes=int(summary.get("fnirs_nodes", 51)),
        fnirs_features=int(summary["fnirs_feature_dim"]),
        graph_hidden=int(ckpt_args.get("graph_hidden", 32)),
        d_model=int(ckpt_args.get("d_model", 128)),
        cheb_order=int(ckpt_args.get("cheb_order", 2)),
        mamba_layers=int(ckpt_args.get("mamba_layers", 2)),
        temporal_block=ckpt_args.get("temporal_block", "gated_ssm"),
        d_state=int(ckpt_args.get("d_state", 16)),
        dropout=float(ckpt_args.get("dropout", 0.15)),
        graph_encoder=ckpt_args.get("graph_encoder", "adaptive"),
        fusion_mode=ckpt_args.get("fusion_mode", "pool"),
        eeg_scale_count=len(multiscale_windows) if ckpt_args.get("scale_fusion") == "gated" else 1,
        fnirs_scale_count=len(multiscale_windows) if ckpt_args.get("scale_fusion") == "gated" else 1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    stats, payload = evaluate(
        model,
        loader,
        device,
        ckpt_args.get("target_mode", "valence"),
        ckpt_args.get("input_modality", "both"),
    )
    return {
        "stats": stats,
        "sample_ids": payload["sample_ids"],
        "y_true": payload["y_true"],
        "y_pred": payload["y_pred"],
        "prior": payload["prior"],
    }


def append_postprocess_results(
    results: list[dict[str, object]],
    sample_ids: list[str],
    y_true: np.ndarray,
    prior: np.ndarray,
    residual: np.ndarray,
    name: str,
    scales: list[float],
    clips: list[float],
    smooth_windows: list[int],
    notes: str,
) -> None:
    for scale_value in scales:
        scaled_residual = scale_value * residual
        for clip_value in clips:
            if clip_value > 0:
                adjusted = np.clip(prior + np.clip(scaled_residual, -clip_value, clip_value), 1.0, 255.0)
                clip_suffix = f"_clip{clip_value:.0f}"
            else:
                adjusted = np.clip(prior + scaled_residual, 1.0, 255.0)
                clip_suffix = ""
            for window in smooth_windows:
                if window > 1:
                    pred = smooth_predictions(sample_ids, adjusted, window=window)
                    suffix = f"_scale{scale_value:.2f}{clip_suffix}_smooth{window}"
                else:
                    pred = adjusted
                    suffix = f"_scale{scale_value:.2f}{clip_suffix}"
                results.append(score(f"{name}{suffix}", y_true, pred, notes))


if __name__ == "__main__":
    main()
