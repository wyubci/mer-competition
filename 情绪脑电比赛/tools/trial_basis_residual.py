from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import (  # noqa: E402
    expand_subjects,
    load_labels,
    predict_video_time_mean,
    score,
    smooth_predictions,
)


SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trial-basis residual regression for MER-PS low-frequency emotion decoding."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_067_tbcr.json")
    parser.add_argument("--basis-counts", default="2,3,4,6,8,10,12")
    parser.add_argument("--alphas", default="1,10,100,1000,10000")
    parser.add_argument("--feature-mode", choices=("mean", "mean_std", "mean_std_slope"), default="mean_std")
    parser.add_argument("--scales", default="0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--clips", default="0,5,10,15,20")
    parser.add_argument("--smooth-windows", default="0,5,9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    basis_counts = parse_ints(args.basis_counts)
    alphas = parse_floats(args.alphas)
    scales = parse_floats(args.scales)
    clips = parse_floats(args.clips)
    smooth_windows = parse_ints(args.smooth_windows)

    labels = load_labels(data_root, train_subjects + val_subjects)
    train_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
    ]
    val_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in val_subjects
    ]
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(
        np.float32
    )
    y_val = np.stack([labels[sample_id] for sample_id in val_label_ids]).astype(np.float32)
    prior_val = predict_video_time_mean(train_label_ids, y_train_label_order, val_label_ids)

    cache = load_feature_cache(Path(args.feature_cache))
    trials = build_trials(cache, labels, train_label_ids, y_train_label_order)
    train_trials = [trial for trial in trials if trial["subject"] in train_subjects]
    val_trials = [trial for trial in trials if trial["subject"] in val_subjects]

    results = [
        score("VideoTimeMean", y_val, prior_val, "Mean trajectory by video and timestamp."),
    ]
    predictions: dict[str, np.ndarray] = {"VideoTimeMean": prior_val}

    for basis_count in basis_counts:
        x_train = np.stack(
            [trial_features(trial["x"], args.feature_mode) for trial in train_trials], axis=0
        )
        x_val = np.stack(
            [trial_features(trial["x"], args.feature_mode) for trial in val_trials], axis=0
        )
        y_coeff_train = np.stack(
            [fit_basis_coefficients(trial["residual"], basis_count) for trial in train_trials],
            axis=0,
        ).reshape(len(train_trials), -1)
        for alpha in alphas:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, y_coeff_train)
            coeff_pred = model.predict(x_val).reshape(len(val_trials), basis_count, 2)
            pred_by_id = reconstruct_predictions(val_trials, coeff_pred, basis_count)
            pred = order_predictions(val_label_ids, pred_by_id)
            base_name = f"TBCR_{args.feature_mode}_k{basis_count}_a{format_alpha(alpha)}"
            for scale in scales:
                residual = pred - prior_val
                scaled_residual = scale * residual
                for clip in clips:
                    if clip > 0:
                        adjusted = np.clip(
                            prior_val + np.clip(scaled_residual, -clip, clip),
                            1.0,
                            255.0,
                        )
                        clip_suffix = f"_clip{clip:.0f}"
                    else:
                        adjusted = np.clip(prior_val + scaled_residual, 1.0, 255.0)
                        clip_suffix = ""
                    for window in smooth_windows:
                        if window > 1:
                            final_pred = smooth_predictions(val_label_ids, adjusted, window=window)
                            suffix = f"_scale{scale:.2f}{clip_suffix}_smooth{window}"
                        else:
                            final_pred = adjusted
                            suffix = f"_scale{scale:.2f}{clip_suffix}"
                        name = base_name + suffix
                        predictions[name] = final_pred
                        results.append(
                            score(
                                name,
                                y_val,
                                final_pred,
                                "Trial-level cosine-basis residual coefficients predicted from aggregated EEG/fNIRS features.",
                            )
                        )

    results = sorted(results, key=lambda item: float(item["overall_mae"]))
    output = {
        "method": "TBCR: Trial-Basis Coefficient Regression",
        "feature_cache": str(args.feature_cache),
        "feature_mode": args.feature_mode,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_trials": len(train_trials),
            "val_trials": len(val_trials),
            "val_samples": len(val_label_ids),
        },
        "results": results[:100],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def load_feature_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "x": data["x"].astype(np.float32),
            "y_raw": data["y_raw"].astype(np.float32),
            "sample_ids": data["sample_ids"].astype(str),
            "sample_subjects": data["sample_subjects"].astype(str),
        }


def build_trials(
    cache: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    train_label_ids: list[str],
    y_train_label_order: np.ndarray,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[tuple[int, int, str]]] = defaultdict(list)
    for index, sample_id in enumerate(cache["sample_ids"].astype(str).tolist()):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index, sample_id))

    trials: list[dict[str, object]] = []
    for (subject, video), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        items = sorted(items)
        sample_ids = [sample_id for _, _, sample_id in items]
        indices = np.asarray([index for _, index, _ in items], dtype=np.int64)
        y = np.stack([labels[sample_id] for sample_id in sample_ids], axis=0).astype(np.float32)
        prior = predict_video_time_mean(train_label_ids, y_train_label_order, sample_ids)
        trials.append(
            {
                "subject": subject,
                "video": video,
                "sample_ids": sample_ids,
                "x": cache["x"][indices],
                "y": y,
                "prior": prior,
                "residual": y - prior,
            }
        )
    return trials


def trial_features(x: np.ndarray, mode: str) -> np.ndarray:
    mean = x.mean(axis=0)
    if mode == "mean":
        return mean.astype(np.float32)
    std = x.std(axis=0)
    if mode == "mean_std":
        return np.concatenate([mean, std], axis=0).astype(np.float32)
    half = max(1, x.shape[0] // 2)
    slope = x[-half:].mean(axis=0) - x[:half].mean(axis=0)
    return np.concatenate([mean, std, slope], axis=0).astype(np.float32)


def fit_basis_coefficients(residual: np.ndarray, basis_count: int) -> np.ndarray:
    basis = cosine_basis(residual.shape[0], basis_count)
    coeff, *_ = np.linalg.lstsq(basis, residual, rcond=None)
    return coeff.astype(np.float32)


def reconstruct_predictions(
    trials: list[dict[str, object]],
    coeff_pred: np.ndarray,
    basis_count: int,
) -> dict[str, np.ndarray]:
    pred_by_id: dict[str, np.ndarray] = {}
    for trial, coeff in zip(trials, coeff_pred):
        prior = np.asarray(trial["prior"], dtype=np.float32)
        basis = cosine_basis(prior.shape[0], basis_count)
        pred = np.clip(prior + basis @ coeff, 1.0, 255.0)
        for sample_id, value in zip(trial["sample_ids"], pred):
            pred_by_id[str(sample_id)] = value.astype(np.float32)
    return pred_by_id


def order_predictions(sample_ids: list[str], pred_by_id: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([pred_by_id[sample_id] for sample_id in sample_ids], axis=0).astype(np.float32)


def cosine_basis(length: int, basis_count: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)[:, None]
    k = np.arange(basis_count, dtype=np.float32)[None, :]
    basis = np.cos(math.pi * (t + 0.5) * k / float(length))
    basis[:, 0] = 1.0
    return basis.astype(np.float32)


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def format_alpha(alpha: float) -> str:
    return str(alpha).replace(".", "p")


if __name__ == "__main__":
    main()
