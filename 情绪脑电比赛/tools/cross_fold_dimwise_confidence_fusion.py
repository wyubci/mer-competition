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


DEFAULT_VALENCE_METHODS = [
    "UncertaintyBlend_smooth51_q20p0-60p0_g0p35",
    "UncertaintyBlend_smooth51_q20p0-60p0_g0p45",
    "UncertaintyBlend_smooth51_q20p0-60p0_g0p5",
    "UncertaintyBlend_smooth43_q20p0-60p0_g0p45",
    "UncertaintyBlend_smooth43_q20p0-60p0_g0p5",
    "UncertaintyBlend_smooth37_q20p0-60p0_g0p45",
    "UncertaintyBlend_smooth37_q20p0-60p0_g0p5",
    "StablePriorEnsemble_baseW0p5",
    "RobustMedian_lag-2_smooth11",
]

DEFAULT_AROUSAL_METHODS = [
    "UncertaintyBlend_meanPrior_q20p0-60p0_g0p35",
    "UncertaintyBlend_meanPrior_q20p0-60p0_g0p45",
    "UncertaintyBlend_meanPrior_q20p0-60p0_g0p5",
    "UncertaintyBlend_meanPrior_q20p0-65p0_g0p35",
    "UncertaintyBlend_meanPrior_q20p0-65p0_g0p45",
    "UncertaintyBlend_smooth51_q20p0-60p0_g0p45",
    "UncertaintyBlend_smooth51_q20p0-60p0_g0p5",
    "RobustMedian_lag-2_smooth11",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dimwise combination of confidence-aware robust prior fusion candidates."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_096_dimwise_confidence_fusion.json")
    parser.add_argument("--base-lag", type=int, default=-2)
    parser.add_argument("--base-smooth", type=int, default=11)
    parser.add_argument("--alt-lag", type=int, default=-1)
    parser.add_argument("--alt-smooth", type=int, default=9)
    parser.add_argument("--quantile-lows", default="20")
    parser.add_argument("--quantile-highs", default="60,65")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="37,43,51")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--valence-methods", default=",".join(DEFAULT_VALENCE_METHODS))
    parser.add_argument("--arousal-methods", default=",".join(DEFAULT_AROUSAL_METHODS))
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    valence_methods = parse_strings(args.valence_methods)
    arousal_methods = parse_strings(args.arousal_methods)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []
    missing_by_fold = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        stats = build_prior_stats(train_ids, y_train)
        base_candidates = build_candidates(
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
        missing = sorted(
            set(valence_methods + arousal_methods).difference(base_candidates)
        )
        if missing:
            missing_by_fold.append({"fold": fold_index, "missing": missing})
            raise KeyError(f"Missing configured candidate(s): {missing}")

        fold_predictions = dict(base_candidates)
        for v_name in valence_methods:
            for a_name in arousal_methods:
                name = f"Dimwise_V[{v_name}]__A[{a_name}]"
                fold_predictions[name] = np.stack(
                    [base_candidates[v_name][:, 0], base_candidates[a_name][:, 1]],
                    axis=1,
                ).astype(np.float32)

        fold_results = [
            score(name, y_val, pred, "Dimwise confidence-aware prior fusion.")
            for name, pred in fold_predictions.items()
        ]
        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": fold_results[: args.top_k],
            }
        )
        aggregate_truth.append(y_val)
        for name, pred in fold_predictions.items():
            aggregate_predictions[name].append(pred)

    y_all = np.concatenate(aggregate_truth, axis=0)
    aggregate_results = []
    for name, parts in aggregate_predictions.items():
        if len(parts) != len(folds):
            continue
        aggregate_results.append(
            score(
                name,
                y_all,
                np.concatenate(parts, axis=0),
                "Weighted aggregate across subject-disjoint folds.",
            )
        )
    aggregate_results = sorted(aggregate_results, key=lambda item: float(item["overall_mae"]))
    output = {
        "method": "Dimwise confidence-aware robust prior fusion",
        "note": (
            "Valence and arousal columns can use different fixed confidence-fusion candidates. "
            "No validation labels are used inside a fold to build predictions."
        ),
        "fold_size": args.fold_size,
        "valence_methods": valence_methods,
        "arousal_methods": arousal_methods,
        "missing_by_fold": missing_by_fold,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


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


if __name__ == "__main__":
    main()
