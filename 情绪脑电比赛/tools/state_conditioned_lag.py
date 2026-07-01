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
    parse_floats,
    shift_stack_by_trial,
    strip_prediction,
)
from tools.run_iteration_experiments import score  # noqa: E402
from tools.slope_conditioned_gate import consensus_residual, prior_slope_by_trial  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-conditioned lag alignment for consensus residuals."
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
    parser.add_argument("--output", default="experiments/results/iteration_074_state_conditioned_lag.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valence-weights", default="0.725,0.275")
    parser.add_argument("--arousal-weights", default="0.3,0.7")
    parser.add_argument("--valence-scale", type=float, default=11.5)
    parser.add_argument("--valence-clip", type=float, default=10.0)
    parser.add_argument("--arousal-scale", type=float, default=0.2)
    parser.add_argument("--arousal-clip", type=float, default=0.0)
    parser.add_argument("--pabs-window", type=int, default=17)
    parser.add_argument("--pabs-sigma-prior", type=float, default=10.0)
    parser.add_argument("--valence-lags", default="-14,-12,-10,-8,-6,-4,-2,0")
    parser.add_argument("--arousal-lags", default="-18,-16,-14,-12,-10,-8,-6,-4")
    parser.add_argument("--slope-quantiles", default="45,50,55")
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

    valence_results = search_dimension(
        name="valence",
        sample_ids=sample_ids,
        y_true=y_true[:, 0],
        prior=prior[:, 0],
        residual_stack=v_stack,
        weights=np.asarray(parse_floats(args.valence_weights), dtype=np.float32),
        lags=parse_ints(args.valence_lags),
        scale=args.valence_scale,
        clip=args.valence_clip,
        pabs_window=args.pabs_window,
        pabs_sigma_prior=args.pabs_sigma_prior,
        slope_quantiles=parse_floats(args.slope_quantiles),
        state_multipliers={"stable": 1.25, "rising": 1.10, "falling": 0.75},
        cdg_params={
            "sigma_multiplier": 4.0,
            "min_gate": 0.0,
            "max_gate": 1.5,
            "sign_penalty": 1.0,
        },
        top_k=args.top_k,
    )
    arousal_results = search_dimension(
        name="arousal",
        sample_ids=sample_ids,
        y_true=y_true[:, 1],
        prior=prior[:, 1],
        residual_stack=a_stack,
        weights=np.asarray(parse_floats(args.arousal_weights), dtype=np.float32),
        lags=parse_ints(args.arousal_lags),
        scale=args.arousal_scale,
        clip=args.arousal_clip,
        pabs_window=args.pabs_window,
        pabs_sigma_prior=args.pabs_sigma_prior,
        slope_quantiles=parse_floats(args.slope_quantiles),
        state_multipliers={"stable": 0.75, "rising": 0.75, "falling": 1.50},
        cdg_params={
            "sigma_multiplier": 2.0,
            "min_gate": 0.5,
            "max_gate": 1.5,
            "sign_penalty": 0.25,
        },
        top_k=args.top_k,
    )

    best_v = dict(valence_results[0])
    best_a = dict(arousal_results[0])
    pred = np.stack([best_v.pop("_prediction"), best_a.pop("_prediction")], axis=1)
    best_combined = score(
        "SCL_PABS_dimwise_best",
        y_true,
        pred,
        "State-conditioned lag alignment over consensus residuals.",
    )
    best_combined["valence_config"] = strip_prediction(valence_results[0])
    best_combined["arousal_config"] = strip_prediction(arousal_results[0])

    output = {
        "device": str(device),
        "method": "SCL: State-Conditioned Lag alignment + SCDG + PA-BS",
        "formula": (
            "derive stable/rising/falling from prior slope; each state selects its own lagged "
            "consensus residual before the SCDG state multiplier and PA-BS smoothing"
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
    weights: np.ndarray,
    lags: list[int],
    scale: float,
    clip: float,
    pabs_window: int,
    pabs_sigma_prior: float,
    slope_quantiles: list[float],
    state_multipliers: dict[str, float],
    cdg_params: dict[str, float],
    top_k: int,
) -> list[dict[str, object]]:
    residual_by_lag = {
        lag: consensus_residual(
            residual_stack=shift_stack_by_trial(sample_ids, residual_stack, lag),
            weights=weights,
            **cdg_params,
        )
        for lag in lags
    }
    top: list[dict[str, object]] = []
    slope = prior_slope_by_trial(sample_ids, prior)
    abs_slope = np.abs(slope)
    for quantile in slope_quantiles:
        threshold = float(np.percentile(abs_slope, quantile))
        rising = slope > threshold
        falling = slope < -threshold
        stable = ~(rising | falling)
        for stable_lag in lags:
            for rising_lag in lags:
                for falling_lag in lags:
                    residual = np.zeros_like(prior)
                    residual[stable] = (
                        residual_by_lag[stable_lag][stable] * state_multipliers["stable"]
                    )
                    residual[rising] = (
                        residual_by_lag[rising_lag][rising] * state_multipliers["rising"]
                    )
                    residual[falling] = (
                        residual_by_lag[falling_lag][falling] * state_multipliers["falling"]
                    )
                    pred = finalize_prediction(
                        sample_ids,
                        prior,
                        residual,
                        scale,
                        clip,
                        pabs_window,
                        pabs_sigma_prior,
                    )
                    entry = {
                        "dimension": name,
                        "mae": round(float(np.mean(np.abs(pred - y_true))), 4),
                        "slope_quantile": round(float(quantile), 6),
                        "slope_threshold": round(float(threshold), 8),
                        "stable_lag": int(stable_lag),
                        "rising_lag": int(rising_lag),
                        "falling_lag": int(falling_lag),
                        "stable_multiplier": round(float(state_multipliers["stable"]), 6),
                        "rising_multiplier": round(float(state_multipliers["rising"]), 6),
                        "falling_multiplier": round(float(state_multipliers["falling"]), 6),
                        "stable_fraction": round(float(stable.mean()), 6),
                        "rising_fraction": round(float(rising.mean()), 6),
                        "falling_fraction": round(float(falling.mean()), 6),
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


if __name__ == "__main__":
    main()
