from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Lag sweep for DPRG residuals.")
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
    parser.add_argument("--output", default="experiments/results/iteration_069_lagged_dprg.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valence-weights", default="0.725,0.275")
    parser.add_argument("--arousal-weights", default="0.3,0.7")
    parser.add_argument("--valence-scale", type=float, default=11.5)
    parser.add_argument("--valence-clip", type=float, default=10.0)
    parser.add_argument("--arousal-scale", type=float, default=0.2)
    parser.add_argument("--arousal-clip", type=float, default=0.0)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--lags", default="-12:12")
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
    valence_weights = np.asarray(parse_floats(args.valence_weights), dtype=np.float32)
    arousal_weights = np.asarray(parse_floats(args.arousal_weights), dtype=np.float32)
    lags = parse_lags(args.lags)

    valence_stack = np.stack(
        [payload["y_pred"][:, 0] - prior[:, 0] for payload in valence_payloads],
        axis=0,
    )
    arousal_stack = np.stack(
        [payload["y_pred"][:, 1] - prior[:, 1] for payload in arousal_payloads],
        axis=0,
    )
    valence_residual = np.tensordot(valence_weights, valence_stack, axes=(0, 0))
    arousal_residual = np.tensordot(arousal_weights, arousal_stack, axes=(0, 0))

    results: list[dict[str, object]] = []
    for lag_v in lags:
        shifted_v = shift_by_trial(sample_ids, valence_residual, lag_v)
        pred_v = apply_residual(
            prior[:, 0],
            shifted_v,
            scale=args.valence_scale,
            clip=args.valence_clip,
        )
        pred_v = smooth_1d(sample_ids, pred_v, args.smooth_window)
        for lag_a in lags:
            shifted_a = shift_by_trial(sample_ids, arousal_residual, lag_a)
            pred_a = apply_residual(
                prior[:, 1],
                shifted_a,
                scale=args.arousal_scale,
                clip=args.arousal_clip,
            )
            pred_a = smooth_1d(sample_ids, pred_a, args.smooth_window)
            pred = np.stack([pred_v, pred_a], axis=1)
            item = score(
                f"LaggedDPRG_vlag{lag_v:+d}_alag{lag_a:+d}",
                y_true,
                pred,
                "DPRG residuals shifted within each trial before scale/clip/smooth.",
            )
            item["valence_lag"] = lag_v
            item["arousal_lag"] = lag_a
            results.append(item)

    output = {
        "device": str(device),
        "lag_definition": "positive lag uses an earlier residual r(t-lag), negative lag uses a later residual r(t-lag)",
        "valence_checkpoints": args.valence_checkpoints,
        "arousal_checkpoints": args.arousal_checkpoints,
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


def parse_lags(value: str) -> list[int]:
    if ":" not in value:
        return [int(item) for item in value.split(",") if item.strip()]
    start, stop = (int(item) for item in value.split(":", 1))
    return list(range(start, stop + 1))


if __name__ == "__main__":
    main()
