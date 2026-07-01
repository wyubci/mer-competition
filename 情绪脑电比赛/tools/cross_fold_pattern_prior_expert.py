from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import build_candidates, build_prior_stats
from tools.run_iteration_experiments import expand_subjects, load_labels, score
from tools.trial_basis_residual import parse_sample_id


DEFAULT_POOL = [
    "RobustMedian_lag-2_smooth11",
    "AltRobustMedian_lag-1_smooth9",
    "StablePriorEnsemble_baseW0p5",
    "UncertaintyBlend_smooth61_q15p0-45p0_g0p25",
    "UncertaintyBlend_smooth61_q15p0-45p0_g0p35",
    "UncertaintyBlend_smooth61_q15p0-45p0_g0p45",
    "UncertaintyBlend_smooth61_q15p0-50p0_g0p35",
    "UncertaintyBlend_smooth51_q15p0-45p0_g0p35",
    "UncertaintyBlend_smooth51_q15p0-45p0_g0p45",
    "UncertaintyBlend_smooth51_q20p0-45p0_g0p45",
    "UncertaintyBlend_smooth51_q20p0-45p0_g0p5",
    "UncertaintyBlend_smooth43_q20p0-45p0_g0p45",
    "UncertaintyBlend_meanPrior_q15p0-45p0_g0p25",
    "UncertaintyBlend_meanPrior_q15p0-45p0_g0p35",
    "UncertaintyBlend_meanPrior_q15p0-45p0_g0p45",
    "UncertaintyBlend_meanPrior_q20p0-45p0_g0p35",
    "UncertaintyBlend_meanPrior_q20p0-45p0_g0p45",
    "UncertaintyBlend_meanPrior_q20p0-50p0_g0p5",
    "UncertaintyBlend_meanPrior_q20p0-55p0_g0p45",
    "UncertaintyBlend_meanPrior_q20p0-55p0_g0p5",
    "UncertaintyBlend_meanPrior_q20p0-55p0_g0p55",
    "UncertaintyBlend_meanPrior_q20p0-60p0_g0p5",
    "UncertaintyBlend_meanPrior_q20p0-60p0_g0p55",
    "UncertaintyBlend_meanPrior_q20p0-70p0_g0p55",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pattern-specific prior experts over confidence-fusion candidates."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_098_pattern_prior_expert.json")
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--state-quantiles", default="35,45,55,65,75")
    parser.add_argument("--expert-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--dimwise-top-n", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    expert_pool = parse_strings(args.expert_pool)
    state_quantiles = parse_floats(args.state_quantiles)

    single_sums: dict[str, dict[str, float]] = defaultdict(new_metric_sum)
    pattern_sums: dict[str, dict[str, float]] = defaultdict(new_metric_sum)
    fold_outputs = []
    missing_by_fold = []
    total_samples = 0

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        stats = build_prior_stats(train_ids, y_train)
        candidates = build_candidates(
            stats=stats,
            val_ids=val_ids,
            base_lag=-2,
            base_smooth=11,
            alt_lag=-1,
            alt_smooth=9,
            q_lows=parse_floats(args.quantile_lows),
            q_highs=parse_floats(args.quantile_highs),
            max_gates=parse_floats(args.max_gates),
            long_smooths=parse_ints(args.long_smooths),
            ensemble_weights=parse_floats(args.ensemble_weights),
        )
        missing = sorted(set(expert_pool).difference(candidates))
        if missing:
            missing_by_fold.append({"fold": fold_index, "missing": missing})
            raise KeyError(f"Missing configured experts: {missing}")

        base = candidates["RobustMedian_lag-2_smooth11"]
        slopes = np.stack(
            [
                prior_slope_by_trial(val_ids, base[:, 0]),
                prior_slope_by_trial(val_ids, base[:, 1]),
            ],
            axis=1,
        )

        fold_pattern_results = []
        for name in expert_pool:
            update_metric_sum(single_sums[name], y_val, candidates[name])

        for dim, dim_name in enumerate(["valence", "arousal"]):
            y_dim = y_val[:, dim]
            abs_slope = np.abs(slopes[:, dim])
            for quantile in state_quantiles:
                threshold = float(np.percentile(abs_slope, quantile))
                stable = abs_slope <= threshold
                for stable_name in expert_pool:
                    stable_values = candidates[stable_name][:, dim]
                    for dynamic_name in expert_pool:
                        dynamic_values = candidates[dynamic_name][:, dim]
                        pred = np.where(stable, stable_values, dynamic_values).astype(np.float32)
                        method = (
                            f"PatternExpert_{dim_name}_q{format_float(quantile)}"
                            f"_S[{stable_name}]_D[{dynamic_name}]"
                        )
                        update_dim_sum(pattern_sums[method], y_dim, pred, dim)
                        if fold_index == len(folds):
                            pass
        total_samples += len(val_ids)

        # A small fold-local summary for sanity; full pattern grid is aggregated metric-only.
        fold_single = [
            score(name, y_val, candidates[name], "Pattern expert candidate pool single expert.")
            for name in expert_pool
        ]
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "expert_pool_size": len(expert_pool),
                "single_results": sorted(fold_single, key=lambda item: float(item["overall_mae"]))[
                    : args.top_k
                ],
            }
        )

    single_results = sorted(
        [finalize_single_metric(name, sums) for name, sums in single_sums.items()],
        key=lambda item: float(item["overall_mae"]),
    )
    pattern_results = [finalize_pattern_metric(name, sums) for name, sums in pattern_sums.items()]
    valence_results = sorted(
        [item for item in pattern_results if item["dimension"] == "valence"],
        key=lambda item: float(item["mae"]),
    )
    arousal_results = sorted(
        [item for item in pattern_results if item["dimension"] == "arousal"],
        key=lambda item: float(item["mae"]),
    )

    dimwise_results = []
    for v_item in valence_results[: args.dimwise_top_n]:
        for a_item in arousal_results[: args.dimwise_top_n]:
            overall_mae = (float(v_item["mae"]) + float(a_item["mae"])) / 2.0
            overall_mse = (float(v_item["mse"]) + float(a_item["mse"])) / 2.0
            dimwise_results.append(
                {
                    "method": f"PatternDimwise_V[{v_item['method']}]__A[{a_item['method']}]",
                    "overall_mae": round(overall_mae, 4),
                    "valence_mae": round(float(v_item["mae"]), 4),
                    "arousal_mae": round(float(a_item["mae"]), 4),
                    "overall_mse": round(overall_mse, 4),
                    "notes": "Metric-composed pattern-specific dimwise prior experts.",
                }
            )
    dimwise_results = sorted(dimwise_results, key=lambda item: float(item["overall_mae"]))

    output = {
        "method": "Pattern-specific prior experts",
        "note": (
            "A TimeMixer/Pathformer-style state expert: stable and dynamic segments, defined "
            "by prior-slope quantiles, may use different fixed confidence-fusion experts."
        ),
        "fold_size": args.fold_size,
        "total_samples": total_samples,
        "state_quantiles": state_quantiles,
        "expert_pool": expert_pool,
        "missing_by_fold": missing_by_fold,
        "single_results": single_results[: args.top_k],
        "top_valence_pattern": valence_results[: args.top_k],
        "top_arousal_pattern": arousal_results[: args.top_k],
        "dimwise_results": dimwise_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def prior_slope_by_trial(sample_ids: list[str], values: np.ndarray) -> np.ndarray:
    slopes = np.zeros_like(values, dtype=np.float32)
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        trial_values = values[indices].astype(np.float32)
        if len(indices) <= 1:
            continue
        local = np.gradient(trial_values).astype(np.float32)
        slopes[indices] = local
    return slopes


def new_metric_sum() -> dict[str, float]:
    return {"n": 0.0, "abs_v": 0.0, "abs_a": 0.0, "sq_v": 0.0, "sq_a": 0.0}


def update_metric_sum(sums: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    sums["n"] += float(y_true.shape[0])
    sums["abs_v"] += float(abs_diff[:, 0].sum())
    sums["abs_a"] += float(abs_diff[:, 1].sum())
    sums["sq_v"] += float((diff[:, 0] ** 2).sum())
    sums["sq_a"] += float((diff[:, 1] ** 2).sum())


def update_dim_sum(sums: dict[str, float], y_true: np.ndarray, y_pred: np.ndarray, dim: int) -> None:
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    key_abs = "abs_v" if dim == 0 else "abs_a"
    key_sq = "sq_v" if dim == 0 else "sq_a"
    sums["n"] += float(y_true.shape[0])
    sums[key_abs] += float(abs_diff.sum())
    sums[key_sq] += float((diff**2).sum())


def finalize_single_metric(name: str, sums: dict[str, float]) -> dict[str, object]:
    n = max(sums["n"], 1.0)
    valence_mae = sums["abs_v"] / n
    arousal_mae = sums["abs_a"] / n
    valence_mse = sums["sq_v"] / n
    arousal_mse = sums["sq_a"] / n
    return {
        "method": name,
        "overall_mae": round(float((valence_mae + arousal_mae) / 2.0), 4),
        "valence_mae": round(float(valence_mae), 4),
        "arousal_mae": round(float(arousal_mae), 4),
        "overall_mse": round(float((valence_mse + arousal_mse) / 2.0), 4),
        "notes": "Weighted aggregate across subject-disjoint folds.",
    }


def finalize_pattern_metric(name: str, sums: dict[str, float]) -> dict[str, object]:
    n = max(sums["n"], 1.0)
    if "_valence_" in name:
        mae = sums["abs_v"] / n
        mse = sums["sq_v"] / n
        dimension = "valence"
    else:
        mae = sums["abs_a"] / n
        mse = sums["sq_a"] / n
        dimension = "arousal"
    return {
        "method": name,
        "dimension": dimension,
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
    }


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
