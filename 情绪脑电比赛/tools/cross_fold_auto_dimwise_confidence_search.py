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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-grid metric search for dimwise confidence-aware prior fusion."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_097_auto_dimwise_confidence_search.json")
    parser.add_argument("--base-lag", type=int, default=-2)
    parser.add_argument("--base-smooth", type=int, default=11)
    parser.add_argument("--alt-lag", type=int, default=-1)
    parser.add_argument("--alt-smooth", type=int, default=9)
    parser.add_argument("--quantile-lows", default="0,10,15,20,25,30,35,40,45,50,55,60")
    parser.add_argument("--quantile-highs", default="45,50,55,60,65,70,75,80,85,90")
    parser.add_argument("--max-gates", default="0.15,0.25,0.35,0.45,0.5,0.55,0.65,0.75,0.9,1.0")
    parser.add_argument("--long-smooths", default="21,31,37,43,51,61")
    parser.add_argument("--ensemble-weights", default="0.25,0.4,0.5,0.6,0.75")
    parser.add_argument("--dimwise-top-n", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)

    metric_sums: dict[str, dict[str, float]] = defaultdict(new_metric_sum)
    fold_outputs = []
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
            base_lag=args.base_lag,
            base_smooth=args.base_smooth,
            alt_lag=args.alt_lag,
            alt_smooth=args.alt_smooth,
            q_lows=parse_floats(args.quantile_lows),
            q_highs=parse_floats(args.quantile_highs),
            max_gates=parse_floats(args.max_gates),
            long_smooths=parse_ints(args.long_smooths),
            ensemble_weights=parse_floats(args.ensemble_weights),
        )
        fold_results = []
        for name, pred in candidates.items():
            update_metric_sum(metric_sums[name], y_val, pred)
            fold_results.append(score(name, y_val, pred, "Full-grid confidence prior candidate."))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "candidate_count": len(candidates),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[: args.top_k],
            }
        )
        total_samples += len(val_ids)

    single_results = [
        finalize_single_metric(name, sums) for name, sums in metric_sums.items()
    ]
    single_results = sorted(single_results, key=lambda item: float(item["overall_mae"]))
    top_valence = sorted(single_results, key=lambda item: float(item["valence_mae"]))[
        : args.dimwise_top_n
    ]
    top_arousal = sorted(single_results, key=lambda item: float(item["arousal_mae"]))[
        : args.dimwise_top_n
    ]

    dimwise_results = []
    for v_item in top_valence:
        for a_item in top_arousal:
            v_sums = metric_sums[str(v_item["method"])]
            a_sums = metric_sums[str(a_item["method"])]
            n = max(v_sums["n"], 1.0)
            valence_mae = v_sums["abs_v"] / n
            arousal_mae = a_sums["abs_a"] / n
            valence_mse = v_sums["sq_v"] / n
            arousal_mse = a_sums["sq_a"] / n
            dimwise_results.append(
                {
                    "method": f"AutoDimwise_V[{v_item['method']}]__A[{a_item['method']}]",
                    "overall_mae": round(float((valence_mae + arousal_mae) / 2.0), 4),
                    "valence_mae": round(float(valence_mae), 4),
                    "arousal_mae": round(float(arousal_mae), 4),
                    "overall_mse": round(float((valence_mse + arousal_mse) / 2.0), 4),
                    "notes": "Metric-composed fixed dimwise candidate across subject-disjoint folds.",
                }
            )
    dimwise_results = sorted(dimwise_results, key=lambda item: float(item["overall_mae"]))

    output = {
        "method": "Auto dimwise confidence-aware prior search",
        "note": (
            "This evaluates the full candidate grid, ranks candidates separately for valence "
            "and arousal, then composes fixed dimwise predictions at the metric level."
        ),
        "fold_size": args.fold_size,
        "total_samples": total_samples,
        "grid": {
            "quantile_lows": parse_floats(args.quantile_lows),
            "quantile_highs": parse_floats(args.quantile_highs),
            "max_gates": parse_floats(args.max_gates),
            "long_smooths": parse_ints(args.long_smooths),
            "ensemble_weights": parse_floats(args.ensemble_weights),
        },
        "single_results": single_results[: args.top_k],
        "top_valence": top_valence,
        "top_arousal": top_arousal,
        "dimwise_results": dimwise_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


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


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
