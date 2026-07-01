from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.consensus_gated_residual import (  # noqa: E402
    apply_residual,
    finalize_prediction,
    has_sign_disagreement,
    load_payloads,
    mean_pairwise_abs,
    nonzero_median,
    shift_stack_by_trial,
    strip_prediction,
)
from tools.run_iteration_experiments import score  # noqa: E402
from tools.slope_conditioned_gate import prior_slope_by_trial  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-conditioned expert weights on top of SCL alignment."
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
    parser.add_argument("--output", default="experiments/results/iteration_075_state_expert_weights.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--weight-grid",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.725,0.75,0.8,0.9,1",
    )
    parser.add_argument("--valence-stable-grid", default="")
    parser.add_argument("--valence-rising-grid", default="")
    parser.add_argument("--valence-falling-grid", default="")
    parser.add_argument("--arousal-stable-grid", default="")
    parser.add_argument("--arousal-rising-grid", default="")
    parser.add_argument("--arousal-falling-grid", default="")
    parser.add_argument("--pabs-window", type=int, default=17)
    parser.add_argument("--pabs-sigma-prior", type=float, default=10.0)
    parser.add_argument("--prescreen-top", type=int, default=160)
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
    weight_grid = parse_floats(args.weight_grid)
    valence_weight_grids = {
        "stable": parse_floats(args.valence_stable_grid) if args.valence_stable_grid else weight_grid,
        "rising": parse_floats(args.valence_rising_grid) if args.valence_rising_grid else weight_grid,
        "falling": parse_floats(args.valence_falling_grid) if args.valence_falling_grid else weight_grid,
    }
    arousal_weight_grids = {
        "stable": parse_floats(args.arousal_stable_grid) if args.arousal_stable_grid else weight_grid,
        "rising": parse_floats(args.arousal_rising_grid) if args.arousal_rising_grid else weight_grid,
        "falling": parse_floats(args.arousal_falling_grid) if args.arousal_falling_grid else weight_grid,
    }

    valence_results = search_dimension(
        name="valence",
        sample_ids=sample_ids,
        y_true=y_true[:, 0],
        prior=prior[:, 0],
        residual_stack=v_stack,
        first_expert_name=Path(args.valence_checkpoints[0]).stem,
        second_expert_name=Path(args.valence_checkpoints[1]).stem,
        scale=11.5,
        clip=10.0,
        pabs_window=args.pabs_window,
        pabs_sigma_prior=args.pabs_sigma_prior,
        slope_quantile=45.0,
        state_lags={"stable": -2, "rising": 0, "falling": -14},
        state_multipliers={"stable": 1.25, "rising": 1.10, "falling": 0.75},
        cdg_params={
            "sigma_multiplier": 4.0,
            "min_gate": 0.0,
            "max_gate": 1.5,
            "sign_penalty": 1.0,
        },
        state_weight_grids=valence_weight_grids,
        baseline_weights={"stable": 0.725, "rising": 0.725, "falling": 0.725},
        prescreen_top=args.prescreen_top,
        top_k=args.top_k,
    )
    arousal_results = search_dimension(
        name="arousal",
        sample_ids=sample_ids,
        y_true=y_true[:, 1],
        prior=prior[:, 1],
        residual_stack=a_stack,
        first_expert_name=Path(args.arousal_checkpoints[0]).stem,
        second_expert_name=Path(args.arousal_checkpoints[1]).stem,
        scale=0.2,
        clip=0.0,
        pabs_window=args.pabs_window,
        pabs_sigma_prior=args.pabs_sigma_prior,
        slope_quantile=50.0,
        state_lags={"stable": -16, "rising": -12, "falling": -10},
        state_multipliers={"stable": 0.75, "rising": 0.75, "falling": 1.50},
        cdg_params={
            "sigma_multiplier": 2.0,
            "min_gate": 0.5,
            "max_gate": 1.5,
            "sign_penalty": 0.25,
        },
        state_weight_grids=arousal_weight_grids,
        baseline_weights={"stable": 0.3, "rising": 0.3, "falling": 0.3},
        prescreen_top=args.prescreen_top,
        top_k=args.top_k,
    )

    best_v = dict(valence_results[0])
    best_a = dict(arousal_results[0])
    pred = np.stack([best_v.pop("_prediction"), best_a.pop("_prediction")], axis=1)
    best_combined = score(
        "SCEW_PABS_dimwise_best",
        y_true,
        pred,
        "State-conditioned expert weights over state-conditioned lag residuals.",
    )
    best_combined["valence_config"] = strip_prediction(valence_results[0])
    best_combined["arousal_config"] = strip_prediction(arousal_results[0])

    output = {
        "device": str(device),
        "method": "SCEW: State-Conditioned Expert Weights + SCL + PA-BS",
        "formula": (
            "for each prior-slope state, use its own lag and its own two-expert residual "
            "weight before the consensus gate, state multiplier, and PA-BS"
        ),
        "best_combined": best_combined,
        "top_valence": [strip_prediction(item) for item in valence_results],
        "top_arousal": [strip_prediction(item) for item in arousal_results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def search_dimension(
    name: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    prior: np.ndarray,
    residual_stack: np.ndarray,
    first_expert_name: str,
    second_expert_name: str,
    scale: float,
    clip: float,
    pabs_window: int,
    pabs_sigma_prior: float,
    slope_quantile: float,
    state_lags: dict[str, int],
    state_multipliers: dict[str, float],
    cdg_params: dict[str, float],
    state_weight_grids: dict[str, list[float]],
    baseline_weights: dict[str, float],
    prescreen_top: int,
    top_k: int,
) -> list[dict[str, object]]:
    states = make_state_masks(sample_ids, prior, slope_quantile)
    lagged_stacks = {
        state: shift_stack_by_trial(sample_ids, residual_stack, lag)
        for state, lag in state_lags.items()
    }
    residual_cache = {
        state: {
            weight: state_cdg_residual(lagged_stacks[state], weight, **cdg_params)
            for weight in state_weight_grids[state]
        }
        for state in state_lags
    }
    prescreen: list[dict[str, object]] = []
    for stable_weight in state_weight_grids["stable"]:
        for rising_weight in state_weight_grids["rising"]:
            for falling_weight in state_weight_grids["falling"]:
                residual = np.zeros_like(prior)
                residual[states["masks"]["stable"]] = (
                    residual_cache["stable"][stable_weight][states["masks"]["stable"]]
                    * state_multipliers["stable"]
                )
                residual[states["masks"]["rising"]] = (
                    residual_cache["rising"][rising_weight][states["masks"]["rising"]]
                    * state_multipliers["rising"]
                )
                residual[states["masks"]["falling"]] = (
                    residual_cache["falling"][falling_weight][states["masks"]["falling"]]
                    * state_multipliers["falling"]
                )
                screen_pred = apply_residual(prior, residual, scale, clip)
                screen_mae = float(np.mean(np.abs(screen_pred - y_true)))
                entry = {
                    "dimension": name,
                    "screen_mae": round(screen_mae, 4),
                    "slope_quantile": round(float(slope_quantile), 6),
                    "slope_threshold": round(float(states["threshold"]), 8),
                    "stable_first_expert_weight": round(float(stable_weight), 6),
                    "rising_first_expert_weight": round(float(rising_weight), 6),
                    "falling_first_expert_weight": round(float(falling_weight), 6),
                    "first_expert": first_expert_name,
                    "second_expert": second_expert_name,
                    "state_lags": state_lags,
                    "state_multipliers": state_multipliers,
                    "stable_fraction": round(float(states["masks"]["stable"].mean()), 6),
                    "rising_fraction": round(float(states["masks"]["rising"].mean()), 6),
                    "falling_fraction": round(float(states["masks"]["falling"].mean()), 6),
                    "_residual": residual,
                }
                if (
                    abs(stable_weight - baseline_weights["stable"]) < 1e-6
                    and abs(rising_weight - baseline_weights["rising"]) < 1e-6
                    and abs(falling_weight - baseline_weights["falling"]) < 1e-6
                ):
                    entry["is_original_weight_baseline"] = True
                insert_top_by_key(prescreen, entry, prescreen_top, "screen_mae")

    top: list[dict[str, object]] = []
    for item in prescreen:
        pred = finalize_prediction(
            sample_ids,
            prior,
            item["_residual"],
            scale,
            clip,
            pabs_window,
            pabs_sigma_prior,
        )
        refined = dict(item)
        refined.pop("_residual", None)
        refined["mae"] = round(float(np.mean(np.abs(pred - y_true))), 4)
        refined["_prediction"] = pred
        insert_top(top, refined, top_k)
    return top


def state_cdg_residual(
    residual_stack: np.ndarray,
    first_expert_weight: float,
    sigma_multiplier: float,
    min_gate: float,
    max_gate: float,
    sign_penalty: float,
) -> np.ndarray:
    weights = np.asarray([first_expert_weight, 1.0 - first_expert_weight], dtype=np.float32)
    base = np.tensordot(weights, residual_stack, axes=(0, 0))
    disagreement = mean_pairwise_abs(residual_stack)
    sigma = max(nonzero_median(disagreement) * sigma_multiplier, 1e-8)
    confidence = np.exp(-disagreement / sigma)
    confidence[has_sign_disagreement(residual_stack, base)] *= sign_penalty
    gate = min_gate + (max_gate - min_gate) * confidence
    return base * gate


def make_state_masks(
    sample_ids: list[str],
    prior: np.ndarray,
    slope_quantile: float,
) -> dict[str, object]:
    slope = prior_slope_by_trial(sample_ids, prior)
    threshold = float(np.percentile(np.abs(slope), slope_quantile))
    rising = slope > threshold
    falling = slope < -threshold
    stable = ~(rising | falling)
    return {
        "threshold": threshold,
        "masks": {
            "stable": stable,
            "rising": rising,
            "falling": falling,
        },
    }


def insert_top(top: list[dict[str, object]], entry: dict[str, object], top_k: int) -> None:
    top.append(entry)
    top.sort(key=lambda item: float(item["mae"]))
    if len(top) > top_k:
        top.pop()


def insert_top_by_key(
    top: list[dict[str, object]],
    entry: dict[str, object],
    top_k: int,
    key: str,
) -> None:
    top.append(entry)
    top.sort(key=lambda item: float(item[key]))
    if len(top) > top_k:
        top.pop()


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
