from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ensemble_graph_mamba import predict_checkpoint  # noqa: E402
from tools.run_iteration_experiments import score, smooth_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize dimension-wise residual gates independently for MER-PS."
    )
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
        default=[
            "experiments/checkpoints/graph_mamba/nobase_itransformer_arousal_159.pt",
            "experiments/checkpoints/graph_mamba/itransformer_arousal_159.pt",
            "experiments/checkpoints/graph_mamba/scalegated_msgm_arousal.pt",
        ],
    )
    parser.add_argument("--output", default="experiments/results/iteration_066_dprg_search.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--valence-weight-step", type=float, default=0.005)
    parser.add_argument("--arousal-weight-step", type=float, default=0.1)
    parser.add_argument("--valence-weight-range", default="0.66,0.78")
    parser.add_argument("--valence-scales", default="8.5:11.5:0.25")
    parser.add_argument("--valence-clips", default="8:12:1")
    parser.add_argument("--valence-smooth-windows", default="7,9,11,13")
    parser.add_argument("--arousal-scales", default="0.20:0.50:0.025")
    parser.add_argument("--arousal-clips", default="0,2,5")
    parser.add_argument("--arousal-smooth-windows", default="7,9,11,13")
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
    y_v = y_true[:, 0]
    y_a = y_true[:, 1]

    valence_residuals = [payload["y_pred"][:, 0] - prior[:, 0] for payload in valence_payloads]
    arousal_residuals = [payload["y_pred"][:, 1] - prior[:, 1] for payload in arousal_payloads]

    valence_weights = valence_weight_grid(
        len(valence_residuals),
        args.valence_weight_step,
        args.valence_weight_range,
    )
    arousal_weights = simplex_weights(len(arousal_residuals), args.arousal_weight_step)

    valence_results = search_dimension(
        name="valence",
        sample_ids=sample_ids,
        y_true=y_v,
        prior=prior[:, 0],
        residuals=valence_residuals,
        checkpoint_names=[Path(path).stem for path in args.valence_checkpoints],
        weight_grid=valence_weights,
        scales=parse_number_grid(args.valence_scales),
        clips=parse_number_grid(args.valence_clips),
        smooth_windows=parse_int_list(args.valence_smooth_windows),
        top_k=args.top_k,
    )
    arousal_results = search_dimension(
        name="arousal",
        sample_ids=sample_ids,
        y_true=y_a,
        prior=prior[:, 1],
        residuals=arousal_residuals,
        checkpoint_names=[Path(path).stem for path in args.arousal_checkpoints],
        weight_grid=arousal_weights,
        scales=parse_number_grid(args.arousal_scales),
        clips=parse_number_grid(args.arousal_clips),
        smooth_windows=parse_int_list(args.arousal_smooth_windows),
        top_k=args.top_k,
    )

    best_v = valence_results[0]
    best_a = arousal_results[0]
    pred = np.stack([best_v.pop("_prediction"), best_a.pop("_prediction")], axis=1)
    best_combined = score("DPRG_dimwise_best", y_true, pred, "Dimension-wise prior residual gate.")
    best_combined["valence_config"] = best_v
    best_combined["arousal_config"] = best_a

    for item in valence_results:
        item.pop("_prediction", None)
    for item in arousal_results:
        item.pop("_prediction", None)

    output = {
        "device": str(device),
        "framework": "DPRG: dimension-wise prior residual gate",
        "objective": "minimize valence MAE and arousal MAE independently, then average",
        "valence_checkpoints": args.valence_checkpoints,
        "arousal_checkpoints": args.arousal_checkpoints,
        "best_combined": best_combined,
        "top_valence": valence_results,
        "top_arousal": arousal_results,
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


def search_dimension(
    name: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    prior: np.ndarray,
    residuals: list[np.ndarray],
    checkpoint_names: list[str],
    weight_grid: list[tuple[float, ...]],
    scales: list[float],
    clips: list[float],
    smooth_windows: list[int],
    top_k: int,
) -> list[dict[str, object]]:
    top: list[dict[str, object]] = []
    residual_stack = np.stack(residuals, axis=0)
    for weights in weight_grid:
        residual = np.tensordot(np.asarray(weights, dtype=np.float32), residual_stack, axes=(0, 0))
        for scale in scales:
            scaled = scale * residual
            for clip in clips:
                if clip > 0:
                    adjusted = np.clip(prior + np.clip(scaled, -clip, clip), 1.0, 255.0)
                else:
                    adjusted = np.clip(prior + scaled, 1.0, 255.0)
                for window in smooth_windows:
                    pred = smooth_1d(sample_ids, adjusted, window)
                    mae = float(np.mean(np.abs(pred - y_true)))
                    entry = {
                        "dimension": name,
                        "mae": round(mae, 4),
                        "weights": {
                            checkpoint: round(float(weight), 6)
                            for checkpoint, weight in zip(checkpoint_names, weights)
                        },
                        "scale": round(float(scale), 6),
                        "clip": round(float(clip), 6),
                        "smooth": int(window),
                        "_prediction": pred,
                    }
                    insert_top(top, entry, top_k)
    return top


def insert_top(top: list[dict[str, object]], entry: dict[str, object], top_k: int) -> None:
    top.append(entry)
    top.sort(key=lambda item: float(item["mae"]))
    if len(top) > top_k:
        top.pop()


def smooth_1d(sample_ids: list[str], values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    two_dim = np.stack([values, values], axis=1)
    return smooth_predictions(sample_ids, two_dim, window=window)[:, 0]


def valence_weight_grid(count: int, step: float, value_range: str) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    if count != 2:
        return simplex_weights(count, step)
    low, high = [float(item) for item in value_range.split(",", 1)]
    n_low = int(round(low / step))
    n_high = int(round(high / step))
    return [(index * step, 1.0 - index * step) for index in range(n_low, n_high + 1)]


def simplex_weights(count: int, step: float) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    units = int(round(1.0 / step))
    weights: list[tuple[float, ...]] = []
    for combo in integer_simplex(count, units):
        weights.append(tuple(value / units for value in combo))
    return weights


def integer_simplex(count: int, total: int) -> Iterable[tuple[int, ...]]:
    if count == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for rest in integer_simplex(count - 1, total - value):
            yield (value,) + rest


def parse_number_grid(value: str) -> list[float]:
    if ":" not in value:
        return [float(item) for item in value.split(",") if item.strip()]
    start, stop, step = (float(item) for item in value.split(":"))
    values = []
    current = start
    while current <= stop + step * 0.5:
        values.append(round(current, 10))
        current += step
    return values


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
