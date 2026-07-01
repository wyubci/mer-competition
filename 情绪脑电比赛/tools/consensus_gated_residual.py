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
from tools.run_iteration_experiments import score  # noqa: E402


SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consensus-driven residual gating over DPRG experts."
    )
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
    parser.add_argument("--output", default="experiments/results/iteration_072_consensus_gate.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valence-weights", default="0.725,0.275")
    parser.add_argument("--arousal-weights", default="0.3,0.7")
    parser.add_argument("--valence-scale", type=float, default=11.5)
    parser.add_argument("--valence-clip", type=float, default=10.0)
    parser.add_argument("--arousal-scale", type=float, default=0.2)
    parser.add_argument("--arousal-clip", type=float, default=0.0)
    parser.add_argument("--valence-lag", type=int, default=-8)
    parser.add_argument("--arousal-lag", type=int, default=-12)
    parser.add_argument("--pabs-window", type=int, default=17)
    parser.add_argument("--pabs-sigma-prior", type=float, default=10.0)
    parser.add_argument("--sigma-multipliers", default="0.25,0.5,1,2,4,8,16")
    parser.add_argument("--min-gates", default="0,0.25,0.5,0.75")
    parser.add_argument("--max-gates", default="0.75,1,1.25,1.5")
    parser.add_argument("--sign-penalties", default="0.25,0.5,0.75,1")
    parser.add_argument("--top-k", type=int, default=50)
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

    v_stack = np.stack(
        [payload["y_pred"][:, 0] - prior[:, 0] for payload in valence_payloads],
        axis=0,
    )
    a_stack = np.stack(
        [payload["y_pred"][:, 1] - prior[:, 1] for payload in arousal_payloads],
        axis=0,
    )
    v_stack = shift_stack_by_trial(sample_ids, v_stack, args.valence_lag)
    a_stack = shift_stack_by_trial(sample_ids, a_stack, args.arousal_lag)

    grid = {
        "sigma_multipliers": parse_floats(args.sigma_multipliers),
        "min_gates": parse_floats(args.min_gates),
        "max_gates": parse_floats(args.max_gates),
        "sign_penalties": parse_floats(args.sign_penalties),
    }
    valence_results = search_dimension(
        name="valence",
        sample_ids=sample_ids,
        y_true=y_true[:, 0],
        prior=prior[:, 0],
        residual_stack=v_stack,
        weights=np.asarray(parse_floats(args.valence_weights), dtype=np.float32),
        scale=args.valence_scale,
        clip=args.valence_clip,
        pabs_window=args.pabs_window,
        pabs_sigma_prior=args.pabs_sigma_prior,
        grid=grid,
        top_k=args.top_k,
    )
    arousal_results = search_dimension(
        name="arousal",
        sample_ids=sample_ids,
        y_true=y_true[:, 1],
        prior=prior[:, 1],
        residual_stack=a_stack,
        weights=np.asarray(parse_floats(args.arousal_weights), dtype=np.float32),
        scale=args.arousal_scale,
        clip=args.arousal_clip,
        pabs_window=args.pabs_window,
        pabs_sigma_prior=args.pabs_sigma_prior,
        grid=grid,
        top_k=args.top_k,
    )

    best_v = dict(valence_results[0])
    best_a = dict(arousal_results[0])
    pred = np.stack([best_v.pop("_prediction"), best_a.pop("_prediction")], axis=1)
    best_combined = score(
        "CDG_PABS_dimwise_best",
        y_true,
        pred,
        "Consensus-driven residual gate followed by prior-aware bilateral smoothing.",
    )
    best_combined["valence_config"] = strip_prediction(valence_results[0])
    best_combined["arousal_config"] = strip_prediction(arousal_results[0])

    output = {
        "device": str(device),
        "method": "CDG: Consensus-Driven residual Gate + PA-BS",
        "formula": (
            "base=sum_i w_i r_i; disagreement=mean_pairwise_abs(r_i-r_j); "
            "confidence=exp(-disagreement/sigma); gate=min_gate+(max_gate-min_gate)*confidence; "
            "if expert signs disagree, confidence is multiplied by sign_penalty"
        ),
        "lagged_dprg": {
            "valence_lag": args.valence_lag,
            "arousal_lag": args.arousal_lag,
            "valence_scale": args.valence_scale,
            "valence_clip": args.valence_clip,
            "arousal_scale": args.arousal_scale,
            "arousal_clip": args.arousal_clip,
        },
        "pabs": {
            "window": args.pabs_window,
            "sigma_prior": args.pabs_sigma_prior,
        },
        "best_combined": best_combined,
        "top_valence": [strip_prediction(item) for item in valence_results],
        "top_arousal": [strip_prediction(item) for item in arousal_results],
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
    residual_stack: np.ndarray,
    weights: np.ndarray,
    scale: float,
    clip: float,
    pabs_window: int,
    pabs_sigma_prior: float,
    grid: dict[str, list[float]],
    top_k: int,
) -> list[dict[str, object]]:
    base = np.tensordot(weights, residual_stack, axes=(0, 0))
    disagreement = mean_pairwise_abs(residual_stack)
    disagreement_floor = nonzero_median(disagreement)
    base_pred = finalize_prediction(
        sample_ids,
        prior,
        base,
        scale,
        clip,
        pabs_window,
        pabs_sigma_prior,
    )
    top = [
        {
            "dimension": name,
            "mae": round(float(np.mean(np.abs(base_pred - y_true))), 4),
            "mode": "baseline_no_gate",
            "disagreement_median": round(float(disagreement_floor), 8),
            "_prediction": base_pred,
        }
    ]
    sign_disagree = has_sign_disagreement(residual_stack, base)
    for sigma_multiplier in grid["sigma_multipliers"]:
        sigma = max(disagreement_floor * sigma_multiplier, 1e-8)
        raw_confidence = np.exp(-disagreement / sigma)
        for sign_penalty in grid["sign_penalties"]:
            confidence = raw_confidence.copy()
            confidence[sign_disagree] *= sign_penalty
            for min_gate in grid["min_gates"]:
                for max_gate in grid["max_gates"]:
                    if max_gate < min_gate:
                        continue
                    gate = min_gate + (max_gate - min_gate) * confidence
                    gated_residual = base * gate
                    pred = finalize_prediction(
                        sample_ids,
                        prior,
                        gated_residual,
                        scale,
                        clip,
                        pabs_window,
                        pabs_sigma_prior,
                    )
                    entry = {
                        "dimension": name,
                        "mae": round(float(np.mean(np.abs(pred - y_true))), 4),
                        "mode": "consensus_gate",
                        "sigma_multiplier": round(float(sigma_multiplier), 6),
                        "sigma": round(float(sigma), 8),
                        "min_gate": round(float(min_gate), 6),
                        "max_gate": round(float(max_gate), 6),
                        "sign_penalty": round(float(sign_penalty), 6),
                        "confidence_mean": round(float(confidence.mean()), 6),
                        "gate_mean": round(float(gate.mean()), 6),
                        "disagreement_median": round(float(disagreement_floor), 8),
                        "_prediction": pred,
                    }
                    insert_top(top, entry, top_k)
    return top


def finalize_prediction(
    sample_ids: list[str],
    prior: np.ndarray,
    residual: np.ndarray,
    scale: float,
    clip: float,
    pabs_window: int,
    pabs_sigma_prior: float,
) -> np.ndarray:
    adjusted = apply_residual(prior, residual, scale, clip)
    return bilateral_smooth(sample_ids, adjusted, prior, pabs_window, pabs_sigma_prior)


def mean_pairwise_abs(stack: np.ndarray) -> np.ndarray:
    if stack.shape[0] <= 1:
        return np.zeros(stack.shape[1], dtype=np.float32)
    values = []
    for left in range(stack.shape[0]):
        for right in range(left + 1, stack.shape[0]):
            values.append(np.abs(stack[left] - stack[right]))
    return np.mean(values, axis=0)


def nonzero_median(values: np.ndarray) -> float:
    nonzero = values[values > 1e-8]
    if nonzero.size == 0:
        return 1.0
    return float(np.median(nonzero))


def has_sign_disagreement(stack: np.ndarray, base: np.ndarray) -> np.ndarray:
    base_sign = np.sign(base)
    expert_signs = np.sign(stack)
    disagreement = np.zeros(base.shape, dtype=bool)
    for expert_sign in expert_signs:
        disagreement |= (base_sign != 0) & (expert_sign != 0) & (expert_sign != base_sign)
    return disagreement


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


def shift_stack_by_trial(sample_ids: list[str], stack: np.ndarray, lag: int) -> np.ndarray:
    return np.stack([shift_by_trial(sample_ids, values, lag) for values in stack], axis=0)


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


def insert_top(top: list[dict[str, object]], entry: dict[str, object], top_k: int) -> None:
    top.append(entry)
    top.sort(key=lambda item: float(item["mae"]))
    if len(top) > top_k:
        top.pop()


def strip_prediction(item: dict[str, object]) -> dict[str, object]:
    clean = dict(item)
    clean.pop("_prediction", None)
    return clean


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
