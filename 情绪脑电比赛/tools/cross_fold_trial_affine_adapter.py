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
from tools.cross_fold_adaptive_stimulus_prior import make_p200_fold  # noqa: E402
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    make_candidates as make_existing_candidates,
    make_pattern_098,
    parse_strings,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.cross_fold_signal_residual_over_pattern_prior import (  # noqa: E402
    make_leave_subject_out_train_prior,
    make_pattern_prior,
)
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


VIEW_NAMES = (
    "eeg_scalar",
    "fnirs_scalar",
    "eeg_hrf_scalar",
    "fnirs_roll_scalar",
    "neurovascular",
    "coherence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trial-level affine physiological adapter over strong video priors."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_baseline.npz")
    parser.add_argument("--output", default="experiments/results/iteration_538_trial_affine_adapter.json")
    parser.add_argument("--alphas", default="30,100,300,1000,3000,10000,30000,100000")
    parser.add_argument("--offset-scales", default="0,0.02,0.05,0.08,0.12,0.18,0.25,0.35")
    parser.add_argument("--slope-scales", default="0,0.02,0.05,0.08,0.12,0.18,0.25,0.35")
    parser.add_argument("--offset-clips", default="0.5,1,2,4,6")
    parser.add_argument("--slope-clips", default="0.25,0.5,1,2,4")
    parser.add_argument("--smooth-windows", default="0,5")
    parser.add_argument("--include-p200", action="store_true")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root, subjects, Path(args.precompute_cache)
    )
    sample_ids_all = [str(sample_id) for sample_id in sample_ids_all]
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}
    candidate_pool = parse_strings(args.candidate_pool)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] trial-affine adapter", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)

        prior_train = make_leave_subject_out_train_prior(labels, train_subjects, train_ids)
        prior_val = make_pattern_prior(train_ids, y_train, val_ids)
        x_train, target_train = build_trial_training_set(
            pre=pre,
            feature_index=feature_index,
            sample_ids=train_ids,
            y=y_train,
            prior=prior_train,
        )
        x_val, trial_meta_val = build_trial_feature_set(
            pre=pre,
            feature_index=feature_index,
            sample_ids=val_ids,
            prior=prior_val,
        )

        bases = {"098_PatternPrior_reference": prior_val}
        if args.include_p200:
            existing_candidates = make_existing_candidates(train_ids, y_train, val_ids, args)
            pattern_098 = make_pattern_098(val_ids, existing_candidates)
            p200 = make_p200_fold(
                labels=labels,
                train_subjects=train_subjects,
                train_ids=train_ids,
                y_train=y_train,
                val_ids=val_ids,
                pattern_098=pattern_098,
                existing_candidates=existing_candidates,
                candidate_pool=candidate_pool,
                args=args,
                seed=args.seed + fold_index * 211,
            )
            bases["200_CurrentManualFusion_reference"] = p200

        fold_predictions: dict[str, np.ndarray] = dict(bases)
        for alpha in parse_floats(args.alphas):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, target_train)
            coeff_val = model.predict(x_val).astype(np.float32)
            for base_name, base_pred in bases.items():
                correction_bank = reconstruct_trial_correction(val_ids, base_pred, trial_meta_val, coeff_val)
                add_grid_predictions(
                    fold_predictions=fold_predictions,
                    base_name=base_name,
                    base_pred=base_pred,
                    correction_bank=correction_bank,
                    val_ids=val_ids,
                    alpha=alpha,
                    args=args,
                )

        fold_results = [
            score(name, y_val, pred, "Trial-level affine physiological adapter.")
            for name, pred in fold_predictions.items()
        ]
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "trial_count_train": int(x_train.shape[0]),
                "trial_feature_dim": int(x_train.shape[1]),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[: args.top_k],
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
        "method": "Trial-Affine Physiological Adapter",
        "note": (
            "Physiology predicts one offset and one prior-amplitude scale per trial/dimension. "
            "This constrains EEG/fNIRS to low-frequency calibration instead of per-second residuals."
        ),
        "feature_shapes": feature_shapes,
        "views": VIEW_NAMES,
        "include_p200": bool(args.include_p200),
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_trial_training_set(
    pre: dict[str, np.ndarray],
    feature_index: dict[str, int],
    sample_ids: list[str],
    y: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_by_id = {sample_id: y[index] for index, sample_id in enumerate(sample_ids)}
    prior_by_id = {sample_id: prior[index] for index, sample_id in enumerate(sample_ids)}
    x_rows = []
    targets = []
    for _, trial_ids in grouped_trial_ids(sample_ids):
        x_rows.append(trial_feature(pre, feature_index, trial_ids, prior_by_id))
        y_trial = np.stack([y_by_id[sample_id] for sample_id in trial_ids], axis=0).astype(np.float32)
        prior_trial = np.stack([prior_by_id[sample_id] for sample_id in trial_ids], axis=0).astype(np.float32)
        targets.append(trial_affine_target(y_trial, prior_trial))
    return np.stack(x_rows, axis=0).astype(np.float32), np.stack(targets, axis=0).astype(np.float32)


def build_trial_feature_set(
    pre: dict[str, np.ndarray],
    feature_index: dict[str, int],
    sample_ids: list[str],
    prior: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    prior_by_id = {sample_id: prior[index] for index, sample_id in enumerate(sample_ids)}
    x_rows = []
    meta = []
    for (subject, video), trial_ids in grouped_trial_ids(sample_ids):
        x_rows.append(trial_feature(pre, feature_index, trial_ids, prior_by_id))
        meta.append({"subject": subject, "video": video, "sample_ids": trial_ids})
    return np.stack(x_rows, axis=0).astype(np.float32), meta


def trial_affine_target(y_trial: np.ndarray, prior_trial: np.ndarray) -> np.ndarray:
    residual = y_trial - prior_trial
    offset = residual.mean(axis=0)
    centered_prior = prior_trial - prior_trial.mean(axis=0, keepdims=True)
    denom = np.maximum((centered_prior * centered_prior).sum(axis=0), 1e-3)
    scale_delta = (residual * centered_prior).sum(axis=0) / denom
    return np.concatenate([offset, np.clip(scale_delta, -2.0, 2.0)], axis=0).astype(np.float32)


def trial_feature(
    pre: dict[str, np.ndarray],
    feature_index: dict[str, int],
    trial_ids: list[str],
    prior_by_id: dict[str, np.ndarray],
) -> np.ndarray:
    indices = np.asarray([feature_index[sample_id] for sample_id in trial_ids], dtype=np.int64)
    parts = []
    for view_name in VIEW_NAMES:
        view = pre[view_name][indices].astype(np.float32)
        parts.extend([view.mean(axis=0), view.std(axis=0), view[-1] - view[0]])
    prior = np.stack([prior_by_id[sample_id] for sample_id in trial_ids], axis=0).astype(np.float32)
    parts.extend(
        [
            prior.mean(axis=0),
            prior.std(axis=0),
            prior[-1] - prior[0],
            np.percentile(prior, 25.0, axis=0),
            np.percentile(prior, 75.0, axis=0),
        ]
    )
    _, video, _ = parse_sample_id(trial_ids[0])
    video_one_hot = np.zeros(15, dtype=np.float32)
    video_one_hot[video - 1] = 1.0
    parts.append(video_one_hot)
    return np.concatenate([np.ravel(part).astype(np.float32) for part in parts], axis=0)


def reconstruct_trial_correction(
    val_ids: list[str],
    base_pred: np.ndarray,
    trial_meta: list[dict[str, object]],
    coeff_val: np.ndarray,
) -> dict[str, np.ndarray]:
    id_to_index = {sample_id: index for index, sample_id in enumerate(val_ids)}
    offset_corr = np.zeros_like(base_pred, dtype=np.float32)
    slope_corr = np.zeros_like(base_pred, dtype=np.float32)
    for trial_index, meta in enumerate(trial_meta):
        trial_ids = list(meta["sample_ids"])
        indices = np.asarray([id_to_index[sample_id] for sample_id in trial_ids], dtype=np.int64)
        prior_trial = base_pred[indices].astype(np.float32)
        centered_prior = prior_trial - prior_trial.mean(axis=0, keepdims=True)
        offset = coeff_val[trial_index, 0:2]
        scale_delta = coeff_val[trial_index, 2:4]
        offset_corr[indices] = offset[None, :]
        slope_corr[indices] = centered_prior * scale_delta[None, :]
    return {"offset": offset_corr, "slope": slope_corr}


def add_grid_predictions(
    fold_predictions: dict[str, np.ndarray],
    base_name: str,
    base_pred: np.ndarray,
    correction_bank: dict[str, np.ndarray],
    val_ids: list[str],
    alpha: float,
    args: argparse.Namespace,
) -> None:
    offset = correction_bank["offset"]
    slope = correction_bank["slope"]
    offset_scales = parse_floats(args.offset_scales)
    slope_scales = parse_floats(args.slope_scales)
    offset_clips = parse_floats(args.offset_clips)
    slope_clips = parse_floats(args.slope_clips)
    smooth_windows = parse_ints(args.smooth_windows)

    for oscale in offset_scales:
        for sscale in slope_scales:
            if oscale == 0.0 and sscale == 0.0:
                continue
            for oclip in offset_clips:
                offset_term = np.clip(oscale * offset, -oclip, oclip)
                for sclip in slope_clips:
                    slope_term = np.clip(sscale * slope, -sclip, sclip)
                    for mode, dims in {"both": (0, 1), "v": (0,), "a": (1,)}.items():
                        correction = np.zeros_like(base_pred, dtype=np.float32)
                        for dim in dims:
                            correction[:, dim] = offset_term[:, dim] + slope_term[:, dim]
                        pred = clip(base_pred + correction)
                        name = (
                            f"{base_name}_TrialAffine_{mode}_a{fmt(alpha)}"
                            f"_os{fmt(oscale)}_ss{fmt(sscale)}_oc{fmt(oclip)}_sc{fmt(sclip)}"
                        )
                        fold_predictions[name] = pred
                        for window in smooth_windows:
                            if window <= 1:
                                continue
                            fold_predictions[f"{name}_smooth{window}"] = smooth_predictions(
                                val_ids, pred, window
                            ).astype(np.float32)


def grouped_trial_ids(sample_ids: list[str]) -> list[tuple[tuple[str, int], list[str]]]:
    groups: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for sample_id in sample_ids:
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, sample_id))
    return [
        (key, [sample_id for _, sample_id in sorted(items)])
        for key, items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def clip(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1.0, 255.0).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def fmt(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
