from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ensemble_graph_mamba import predict_checkpoint  # noqa: E402
from tools.run_iteration_experiments import score, smooth_predictions  # noqa: E402


SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prior-aware bilateral smoothing for DPRG residuals.")
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
    parser.add_argument("--output", default="experiments/results/iteration_071_prior_aware_smoothing.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valence-weights", default="0.725,0.275")
    parser.add_argument("--arousal-weights", default="0.3,0.7")
    parser.add_argument("--valence-scale", type=float, default=11.5)
    parser.add_argument("--valence-clip", type=float, default=10.0)
    parser.add_argument("--arousal-scale", type=float, default=0.2)
    parser.add_argument("--arousal-clip", type=float, default=0.0)
    parser.add_argument("--valence-lag", type=int, default=-8)
    parser.add_argument("--arousal-lag", type=int, default=-12)
    parser.add_argument("--windows", default="5,7,9,11,13,17")
    parser.add_argument("--sigma-priors", default="2,5,10,20,40,80,1000000")
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
    v_weights = np.asarray(parse_floats(args.valence_weights), dtype=np.float32)
    a_weights = np.asarray(parse_floats(args.arousal_weights), dtype=np.float32)
    v_residual = np.tensordot(
        v_weights,
        np.stack([payload["y_pred"][:, 0] - prior[:, 0] for payload in valence_payloads], axis=0),
        axes=(0, 0),
    )
    a_residual = np.tensordot(
        a_weights,
        np.stack([payload["y_pred"][:, 1] - prior[:, 1] for payload in arousal_payloads], axis=0),
        axes=(0, 0),
    )
    v_residual = shift_by_trial(sample_ids, v_residual, args.valence_lag)
    a_residual = shift_by_trial(sample_ids, a_residual, args.arousal_lag)
    raw_v = apply_residual(prior[:, 0], v_residual, args.valence_scale, args.valence_clip)
    raw_a = apply_residual(prior[:, 1], a_residual, args.arousal_scale, args.arousal_clip)

    results: list[dict[str, object]] = []
    windows = parse_ints(args.windows)
    sigma_priors = parse_floats(args.sigma_priors)
    for window in windows:
        moving = np.stack(
            [
                smooth_1d(sample_ids, raw_v, window),
                smooth_1d(sample_ids, raw_a, window),
            ],
            axis=1,
        )
        item = score(
            f"MovingAverage_window{window}",
            y_true,
            moving,
            "Fixed per-trial moving average smoothing.",
        )
        item["window"] = window
        item["sigma_prior"] = None
        results.append(item)
        for sigma_prior in sigma_priors:
            pred = np.stack(
                [
                    bilateral_smooth(sample_ids, raw_v, prior[:, 0], window, sigma_prior),
                    bilateral_smooth(sample_ids, raw_a, prior[:, 1], window, sigma_prior),
                ],
                axis=1,
            )
            item = score(
                f"PABS_window{window}_sigma{format_float(sigma_prior)}",
                y_true,
                pred,
                "Prior-aware bilateral smoothing over each trial.",
            )
            item["window"] = window
            item["sigma_prior"] = sigma_prior
            results.append(item)

    output = {
        "device": str(device),
        "method": "PA-BS: Prior-Aware Bilateral Smoothing",
        "lagged_dprg": {
            "valence_lag": args.valence_lag,
            "arousal_lag": args.arousal_lag,
            "valence_scale": args.valence_scale,
            "valence_clip": args.valence_clip,
            "arousal_scale": args.arousal_scale,
            "arousal_clip": args.arousal_clip,
        },
        "results": sorted(results, key=lambda item: float(item["overall_mae"]))[:100],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def load_payloads(paths: list[str], batch_size: int, device: torch.device) -> list[dict[str, object]]:
    payloads = []
    for checkpoint_path in paths:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payloads.append(predict_checkpoint(path, batch_size, device))
    return payloads


def bilateral_smooth(
    sample_ids: list[str],
    values: np.ndarray,
    prior: np.ndarray,
    window: int,
    sigma_prior: float,
) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    sigma_time = max(radius / 2.0, 1.0)
    out = values.copy()
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        trial_values = values[indices]
        trial_prior = prior[indices]
        length = len(indices)
        for local_index, global_index in enumerate(indices):
            start = max(0, local_index - radius)
            stop = min(length, local_index + radius + 1)
            offsets = np.arange(start, stop) - local_index
            time_weight = np.exp(-(offsets.astype(np.float32) ** 2) / (2.0 * sigma_time**2))
            prior_diff = trial_prior[start:stop] - trial_prior[local_index]
            prior_weight = np.exp(-(prior_diff.astype(np.float32) ** 2) / (2.0 * sigma_prior**2))
            weight = time_weight * prior_weight
            denom = float(weight.sum())
            if denom > 0:
                out[global_index] = float(np.sum(weight * trial_values[start:stop]) / denom)
    return out


def shift_by_trial(sample_ids: list[str], values: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return values.copy()
    shifted = values.copy()
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        trial_values = values[indices]
        source = np.arange(len(indices)) - lag
        source = np.clip(source, 0, len(indices) - 1)
        shifted[indices] = trial_values[source]
    return shifted


def apply_residual(prior: np.ndarray, residual: np.ndarray, scale: float, clip: float) -> np.ndarray:
    scaled = scale * residual
    if clip > 0:
        scaled = np.clip(scaled, -clip, clip)
    return np.clip(prior + scaled, 1.0, 255.0)


def smooth_1d(sample_ids: list[str], values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    two_dim = np.stack([values, values], axis=1)
    return smooth_predictions(sample_ids, two_dim, window=window)[:, 0]


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    if value >= 100000:
        return "inf"
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
