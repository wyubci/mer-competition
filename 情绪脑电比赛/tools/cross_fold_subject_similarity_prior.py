from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_robust_prior_tbcr import robust_median_prior
from tools.run_iteration_experiments import expand_subjects, load_labels, score
from tools.trial_basis_residual import load_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subject-similarity robust median priors using unlabeled EEG/fNIRS features."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_093_subject_similarity_prior.json")
    parser.add_argument("--prior-lag", type=int, default=-2)
    parser.add_argument("--prior-smooth", type=int, default=11)
    parser.add_argument("--neighbors", default="3,5,8,10,15,20")
    parser.add_argument("--feature-mode", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    cache = load_feature_cache(Path(args.feature_cache))
    neighbors = parse_ints(args.neighbors)
    subject_features = build_subject_features(cache, subjects, args.feature_mode)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = {}
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        y_train = labels_to_array(labels, train_ids)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_val = labels_to_array(labels, val_ids)
        baseline = robust_median_prior(
            reference_ids=train_ids,
            reference_y=y_train,
            target_ids=val_ids,
            lag=args.prior_lag,
            smooth_window=args.prior_smooth,
        )
        predictions = {"RobustMedianPrior_all20": baseline}
        neighbor_log = {}

        for k in neighbors:
            pred_by_subject = []
            chosen_by_subject = {}
            for subject in val_subjects:
                selected = nearest_subjects(subject, train_subjects, subject_features, k)
                chosen_by_subject[subject] = selected
                ref_ids = ids_for_subjects(labels, selected)
                ref_y = labels_to_array(labels, ref_ids)
                target_ids = ids_for_subjects(labels, [subject])
                pred = robust_median_prior(
                    reference_ids=ref_ids,
                    reference_y=ref_y,
                    target_ids=target_ids,
                    lag=args.prior_lag,
                    smooth_window=args.prior_smooth,
                )
                pred_by_subject.append((target_ids, pred))
            ordered_pred = order_subject_parts(val_ids, pred_by_subject)
            name = f"SubjectSimilarityMedian_k{k}"
            predictions[name] = ordered_pred
            neighbor_log[name] = chosen_by_subject

        fold_results = [
            score(name, y_val, pred, "Subject-similarity robust median prior.")
            for name, pred in predictions.items()
        ]
        fold_results = sorted(fold_results, key=lambda item: float(item["overall_mae"]))
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "neighbor_log": neighbor_log,
                "results": fold_results[: args.top_k],
            }
        )
        aggregate_truth.append(y_val)
        for name, pred in predictions.items():
            aggregate_predictions.setdefault(name, []).append(pred)

    y_all = np.concatenate(aggregate_truth, axis=0)
    aggregate_results = []
    for name, parts in aggregate_predictions.items():
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
        "method": "Subject-similarity robust median prior",
        "note": (
            "Validation subject labels are not used. Unlabeled EEG/fNIRS feature summaries select "
            "nearest training subjects, then robust median video-time prior is built from those subjects."
        ),
        "fold_size": args.fold_size,
        "feature_mode": args.feature_mode,
        "prior": {
            "estimator": "median",
            "lag": args.prior_lag,
            "smooth_window": args.prior_smooth,
        },
        "neighbors": neighbors,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_subject_features(
    cache: dict[str, np.ndarray],
    subjects: list[str],
    mode: str,
) -> dict[str, np.ndarray]:
    x = cache["x"].astype(np.float32)
    sample_subjects = cache["sample_subjects"].astype(str)
    features = {}
    for subject in subjects:
        subject_x = x[sample_subjects == subject]
        if subject_x.size == 0:
            raise ValueError(f"No feature rows for {subject}")
        mean = subject_x.mean(axis=0)
        if mode == "mean":
            features[subject] = mean.astype(np.float32)
        else:
            features[subject] = np.concatenate([mean, subject_x.std(axis=0)], axis=0).astype(np.float32)
    return features


def nearest_subjects(
    subject: str,
    train_subjects: list[str],
    subject_features: dict[str, np.ndarray],
    k: int,
) -> list[str]:
    train_x = np.stack([subject_features[item] for item in train_subjects], axis=0)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_z = (train_x - mean) / std
    target_z = (subject_features[subject] - mean) / std
    distances = np.sqrt(np.mean((train_z - target_z[None, :]) ** 2, axis=1))
    order = np.argsort(distances)
    selected_count = min(k, len(train_subjects))
    return [train_subjects[index] for index in order[:selected_count]]


def order_subject_parts(
    ordered_ids: list[str],
    parts: list[tuple[list[str], np.ndarray]],
) -> np.ndarray:
    lookup = {}
    for ids, pred in parts:
        for sample_id, value in zip(ids, pred):
            lookup[sample_id] = value
    return np.stack([lookup[sample_id] for sample_id in ordered_ids], axis=0).astype(np.float32)


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
