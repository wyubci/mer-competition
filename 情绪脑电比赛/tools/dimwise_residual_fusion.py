from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ensemble_graph_mamba import predict_checkpoint  # noqa: E402
from tools.run_iteration_experiments import score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dimension-wise residual fusion for MER-PS.")
    parser.add_argument(
        "--valence-checkpoints",
        nargs="+",
        default=[
            "experiments/checkpoints/graph_mamba/moddrop010_seed123.pt",
            "experiments/checkpoints/graph_mamba/itransformer_hybrid_159.pt",
        ],
    )
    parser.add_argument(
        "--arousal-checkpoints",
        nargs="+",
        default=["experiments/checkpoints/graph_mamba/nobase_itransformer_arousal_159.pt"],
    )
    parser.add_argument("--output", default="experiments/results/iteration_064_dimwise_fusion.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valence-weight-step", type=float, default=0.02)
    parser.add_argument("--valence-scales", default="6,8,10,12")
    parser.add_argument("--valence-clips", default="0,5,10,15")
    parser.add_argument("--valence-smooth-windows", default="5,9")
    parser.add_argument("--arousal-scales", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--arousal-clips", default="0,2,5,10")
    parser.add_argument("--arousal-smooth-windows", default="3,5,9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valence_payloads = load_payloads(args.valence_checkpoints, args.batch_size, device)
    arousal_payloads = load_payloads(args.arousal_checkpoints, args.batch_size, device)
    reference = valence_payloads[0]
    for payload in valence_payloads[1:] + arousal_payloads:
        if payload["sample_ids"] != reference["sample_ids"]:
            raise ValueError("sample_id order mismatch")

    sample_ids = reference["sample_ids"]
    y_true = reference["y_true"]
    prior = reference["prior"]
    valence_residuals = [payload["y_pred"][:, 0] - prior[:, 0] for payload in valence_payloads]
    arousal_residuals = [payload["y_pred"][:, 1] - prior[:, 1] for payload in arousal_payloads]
    arousal_residual = np.mean(arousal_residuals, axis=0)

    valence_scales = parse_float_list(args.valence_scales)
    valence_clips = parse_float_list(args.valence_clips)
    valence_windows = parse_int_list(args.valence_smooth_windows)
    arousal_scales = parse_float_list(args.arousal_scales)
    arousal_clips = parse_float_list(args.arousal_clips)
    arousal_windows = parse_int_list(args.arousal_smooth_windows)
    valence_weights = make_weights(len(valence_residuals), args.valence_weight_step)

    results: list[dict[str, object]] = []
    for weights in valence_weights:
        valence_residual = np.zeros_like(valence_residuals[0])
        for weight, residual in zip(weights, valence_residuals):
            valence_residual += weight * residual
        weight_label = "+".join(
            f"{Path(path).stem}_{weight:.2f}"
            for path, weight in zip(args.valence_checkpoints, weights)
        )

        for v_scale in valence_scales:
            for v_clip in valence_clips:
                valence_base = apply_residual(prior[:, 0], valence_residual, v_scale, v_clip)
                for v_window in valence_windows:
                    valence_pred = smooth_1d(sample_ids, valence_base, v_window)
                    for a_scale in arousal_scales:
                        for a_clip in arousal_clips:
                            arousal_base = apply_residual(
                                prior[:, 1], arousal_residual, a_scale, a_clip
                            )
                            for a_window in arousal_windows:
                                arousal_pred = smooth_1d(sample_ids, arousal_base, a_window)
                                pred = np.stack([valence_pred, arousal_pred], axis=1)
                                name = (
                                    f"Dimwise_{weight_label}"
                                    f"_vscale{v_scale:.2f}_vclip{v_clip:.0f}_vsmooth{v_window}"
                                    f"_ascale{a_scale:.2f}_aclip{a_clip:.0f}_asmooth{a_window}"
                                )
                                results.append(
                                    score(
                                        name,
                                        y_true,
                                        pred,
                                        "Dimension-wise residual fusion with independent valence and arousal postprocess.",
                                    )
                                )

    output = {
        "device": str(device),
        "valence_checkpoints": args.valence_checkpoints,
        "arousal_checkpoints": args.arousal_checkpoints,
        "results": sorted(results, key=lambda item: float(item["overall_mae"]))[:100],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def load_payloads(
    checkpoint_paths: list[str],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    payloads = []
    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payloads.append(predict_checkpoint(path, batch_size, device))
    return payloads


def make_weights(count: int, step: float) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    if count != 2:
        weight = 1.0 / count
        return [tuple(weight for _ in range(count))]
    steps = int(round(1.0 / step))
    return [(index / steps, 1.0 - index / steps) for index in range(steps + 1)]


def apply_residual(
    prior: np.ndarray,
    residual: np.ndarray,
    scale: float,
    clip: float,
) -> np.ndarray:
    scaled = scale * residual
    if clip > 0:
        scaled = np.clip(scaled, -clip, clip)
    return np.clip(prior + scaled, 1.0, 255.0)


def smooth_1d(sample_ids: list[str], values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    two_dim = np.stack([values, values], axis=1)
    return smooth_predictions(sample_ids, two_dim, window=window)[:, 0]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
