from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import build_candidates, build_prior_stats
from tools.cross_fold_pattern_prior_expert import prior_slope_by_trial
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions
from tools.trial_basis_residual import load_feature_cache


BEST_098 = {
    "valence": {
        "quantile": 65.0,
        "stable": "UncertaintyBlend_meanPrior_q15p0-45p0_g0p25",
        "dynamic": "UncertaintyBlend_smooth61_q15p0-45p0_g0p45",
    },
    "arousal": {
        "quantile": 55.0,
        "stable": "UncertaintyBlend_smooth51_q20p0-45p0_g0p5",
        "dynamic": "UncertaintyBlend_meanPrior_q20p0-55p0_g0p55",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen EEG/fNIRS feature residual probe over PatternPrior_098."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_099_signal_residual_pattern_prior.json")
    parser.add_argument("--alphas", default="100,1000,10000,100000,1000000")
    parser.add_argument("--valence-scales", default="0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--arousal-scales", default="0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--valence-clips", default="0.5,1,2,4")
    parser.add_argument("--arousal-clips", default="0.5,1,2,4")
    parser.add_argument("--smooth-windows", default="0,5,9")
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    cache = load_feature_cache(Path(args.feature_cache))
    sample_ids_cache = cache["sample_ids"].astype(str).tolist()
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_cache)}

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)

        prior_val = make_pattern_prior(train_ids, y_train, val_ids)
        prior_train = make_leave_subject_out_train_prior(labels, train_subjects, train_ids)

        x_train = feature_rows(cache, feature_index, train_ids)
        x_val = feature_rows(cache, feature_index, val_ids)
        residual_train = y_train - prior_train

        fold_predictions = {"PatternPrior_098": prior_val}
        for alpha in parse_floats(args.alphas):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, residual_train)
            residual_val = model.predict(x_val).astype(np.float32)
            for scale_v in parse_floats(args.valence_scales):
                for scale_a in parse_floats(args.arousal_scales):
                    for clip_v in parse_floats(args.valence_clips):
                        v_corr = np.clip(scale_v * residual_val[:, 0], -clip_v, clip_v)
                        for clip_a in parse_floats(args.arousal_clips):
                            a_corr = np.clip(scale_a * residual_val[:, 1], -clip_a, clip_a)
                            pred = np.clip(
                                prior_val + np.stack([v_corr, a_corr], axis=1),
                                1.0,
                                255.0,
                            ).astype(np.float32)
                            base_name = (
                                f"SignalResidualRidge_a{format_float(alpha)}"
                                f"_sv{format_float(scale_v)}_sa{format_float(scale_a)}"
                                f"_cv{format_float(clip_v)}_ca{format_float(clip_a)}"
                            )
                            fold_predictions[base_name] = pred
                            for window in parse_ints(args.smooth_windows):
                                if window <= 1:
                                    continue
                                smooth = smooth_predictions(val_ids, pred, window).astype(np.float32)
                                fold_predictions[f"{base_name}_smooth{window}"] = smooth

        fold_results = [
            score(name, y_val, pred, "PatternPrior_098 plus frozen-feature Ridge residual probe.")
            for name, pred in fold_predictions.items()
        ]
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[
                    : args.top_k
                ],
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
        "method": "Strong-prior EEG/fNIRS signal residual probe",
        "note": (
            "A frozen-feature linear probe over ASAC EEG/fNIRS features predicts residuals "
            "relative to PatternPrior_098. Train residual targets use leave-subject-out prior."
        ),
        "fold_size": args.fold_size,
        "feature_cache": str(args.feature_cache),
        "best_098_config": BEST_098,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_leave_subject_out_train_prior(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    train_ids: list[str],
) -> np.ndarray:
    prior_by_id: dict[str, np.ndarray] = {}
    train_set = set(train_subjects)
    for subject in train_subjects:
        subject_ids = [sample_id for sample_id in train_ids if sample_id.startswith(subject + "_V")]
        ref_ids = [
            sample_id
            for sample_id in train_ids
            if sample_id.split("_V", 1)[0] in train_set and not sample_id.startswith(subject + "_V")
        ]
        ref_y = labels_to_array(labels, ref_ids)
        subject_prior = make_pattern_prior(ref_ids, ref_y, subject_ids)
        for sample_id, value in zip(subject_ids, subject_prior):
            prior_by_id[sample_id] = value
    return np.stack([prior_by_id[sample_id] for sample_id in train_ids], axis=0).astype(np.float32)


def make_pattern_prior(reference_ids: list[str], reference_y: np.ndarray, target_ids: list[str]) -> np.ndarray:
    stats = build_prior_stats(reference_ids, reference_y)
    candidates = build_candidates(
        stats=stats,
        val_ids=target_ids,
        base_lag=-2,
        base_smooth=11,
        alt_lag=-1,
        alt_smooth=9,
        q_lows=[15.0, 20.0],
        q_highs=[45.0, 50.0, 55.0, 60.0, 70.0],
        max_gates=[0.25, 0.35, 0.45, 0.5, 0.55],
        long_smooths=[43, 51, 61],
        ensemble_weights=[0.5],
    )
    base = candidates["RobustMedian_lag-2_smooth11"]
    out = np.empty_like(base, dtype=np.float32)
    for dim, dim_name in enumerate(["valence", "arousal"]):
        cfg = BEST_098[dim_name]
        slope = np.abs(prior_slope_by_trial(target_ids, base[:, dim]))
        threshold = float(np.percentile(slope, float(cfg["quantile"])))
        stable = slope <= threshold
        out[:, dim] = np.where(
            stable,
            candidates[str(cfg["stable"])][:, dim],
            candidates[str(cfg["dynamic"])][:, dim],
        )
    return np.clip(out, 1.0, 255.0).astype(np.float32)


def feature_rows(
    cache: dict[str, np.ndarray],
    feature_index: dict[str, int],
    sample_ids: list[str],
) -> np.ndarray:
    indices = np.asarray([feature_index[sample_id] for sample_id in sample_ids], dtype=np.int64)
    return cache["x"][indices].astype(np.float32)


def ids_for_subjects(labels: dict[str, np.ndarray], subjects: list[str]) -> list[str]:
    subject_set = set(subjects)
    return [sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in subject_set]


def labels_to_array(labels: dict[str, np.ndarray], sample_ids: list[str]) -> np.ndarray:
    return np.stack([labels[sample_id] for sample_id in sample_ids]).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
