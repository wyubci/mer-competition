from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.consensus_gated_residual import (  # noqa: E402
    finalize_prediction,
    load_payloads,
    shift_stack_by_trial,
    strip_prediction,
)
from tools.run_iteration_experiments import score  # noqa: E402
from tools.state_conditioned_expert_weights import (  # noqa: E402
    make_state_masks,
    state_cdg_residual,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search PA-BS smoothing parameters on top of the best SCEW residuals."
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
    parser.add_argument("--output", default="experiments/results/iteration_077_scew_smoothing_search.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows", default="9,11,13,15,17,19,21,25")
    parser.add_argument("--sigma-priors", default="5,7.5,10,12.5,15,20,30,40,80,1000000")
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

    windows = parse_ints(args.windows)
    sigma_priors = parse_floats(args.sigma_priors)
    valence_results = search_dimension(
        name="valence",
        sample_ids=sample_ids,
        y_true=y_true[:, 0],
        prior=prior[:, 0],
        residual=valence_residual,
        scale=11.5,
        clip=10.0,
        windows=windows,
        sigma_priors=sigma_priors,
        top_k=args.top_k,
    )
    arousal_results = search_dimension(
        name="arousal",
        sample_ids=sample_ids,
        y_true=y_true[:, 1],
        prior=prior[:, 1],
        residual=arousal_residual,
        scale=0.2,
        clip=0.0,
        windows=windows,
        sigma_priors=sigma_priors,
        top_k=args.top_k,
    )

    best_v = dict(valence_results[0])
    best_a = dict(arousal_results[0])
    pred = np.stack([best_v.pop("_prediction"), best_a.pop("_prediction")], axis=1)
    best_combined = score(
        "SCEW_PABS_smoothing_best",
        y_true,
        pred,
        "Smoothing parameter search on top of fixed SCEW residuals.",
    )
    best_combined["valence_config"] = strip_prediction(valence_results[0])
    best_combined["arousal_config"] = strip_prediction(arousal_results[0])

    output = {
        "device": str(device),
        "method": "SCEW + PA-BS smoothing search",
        "best_combined": best_combined,
        "top_valence": [strip_prediction(item) for item in valence_results],
        "top_arousal": [strip_prediction(item) for item in arousal_results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_state_residual(
    sample_ids: list[str],
    prior: np.ndarray,
    residual_stack: np.ndarray,
    slope_quantile: float,
    state_lags: dict[str, int],
    state_multipliers: dict[str, float],
    state_first_expert_weights: dict[str, float],
    cdg_params: dict[str, float],
) -> np.ndarray:
    states = make_state_masks(sample_ids, prior, slope_quantile)
    residual = np.zeros_like(prior)
    for state, lag in state_lags.items():
        lagged_stack = shift_stack_by_trial(sample_ids, residual_stack, lag)
        state_residual = state_cdg_residual(
            lagged_stack,
            state_first_expert_weights[state],
            **cdg_params,
        )
        mask = states["masks"][state]
        residual[mask] = state_residual[mask] * state_multipliers[state]
    return residual


def search_dimension(
    name: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    prior: np.ndarray,
    residual: np.ndarray,
    scale: float,
    clip: float,
    windows: list[int],
    sigma_priors: list[float],
    top_k: int,
) -> list[dict[str, object]]:
    top: list[dict[str, object]] = []
    for window in windows:
        for sigma_prior in sigma_priors:
            pred = finalize_prediction(
                sample_ids,
                prior,
                residual,
                scale,
                clip,
                window,
                sigma_prior,
            )
            entry = {
                "dimension": name,
                "mae": round(float(np.mean(np.abs(pred - y_true))), 4),
                "window": int(window),
                "sigma_prior": round(float(sigma_prior), 6),
                "_prediction": pred,
            }
            insert_top(top, entry, top_k)
    return top


def insert_top(top: list[dict[str, object]], entry: dict[str, object], top_k: int) -> None:
    top.append(entry)
    top.sort(key=lambda item: float(item["mae"]))
    if len(top) > top_k:
        top.pop()


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
