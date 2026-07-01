from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Temporal-lag Ridge residual correction for MER-PS.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_070_tlrc.json")
    parser.add_argument(
        "--lag-sets",
        default="0;-8,0,8;-12,-8,-4,0,4,8,12;-16,-12,-8,-4,0,4,8,12,16",
        help="Semicolon-separated lag sets. Negative lag uses future features x(t-lag).",
    )
    parser.add_argument("--alphas", default="100,1000,10000")
    parser.add_argument("--target-mode", choices=("all", "valence", "arousal"), default="all")
    parser.add_argument("--scales", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--clips", default="0,2,5,10")
    parser.add_argument("--smooth-windows", default="0,5,9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    labels = load_labels(data_root, train_subjects + val_subjects)
    cache = load_cache(Path(args.feature_cache))
    sample_ids = cache["sample_ids"].astype(str).tolist()
    sample_subjects = cache["sample_subjects"].astype(str)
    y_raw = cache["y_raw"].astype(np.float32)

    train_label_ids = [sample_id for sample_id in sample_ids if subject_of(sample_id) in train_subjects]
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(
        np.float32
    )
    prior = predict_video_time_mean(train_label_ids, y_train_label_order, sample_ids)
    residual_target = y_raw - prior
    train_idx = np.flatnonzero(np.isin(sample_subjects, train_subjects))
    val_idx = np.flatnonzero(np.isin(sample_subjects, val_subjects))
    val_ids = [sample_ids[index] for index in val_idx]
    y_val = y_raw[val_idx]
    prior_val = prior[val_idx]

    results = [score("VideoTimeMean", y_val, prior_val, "Mean trajectory by video and timestamp.")]
    lag_sets = parse_lag_sets(args.lag_sets)
    alphas = parse_floats(args.alphas)
    scales = parse_floats(args.scales)
    clips = parse_floats(args.clips)
    smooth_windows = parse_ints(args.smooth_windows)

    for lags in lag_sets:
        x_lag = make_lagged_features(cache["x"], sample_ids, lags)
        x_train = x_lag[train_idx]
        x_val = x_lag[val_idx]
        y_train = residual_target[train_idx].copy()
        if args.target_mode == "valence":
            y_train[:, 1] = 0.0
        elif args.target_mode == "arousal":
            y_train[:, 0] = 0.0

        for alpha in alphas:
            model = make_pipeline(
                StandardScaler(copy=False),
                Ridge(alpha=alpha, solver="lsqr", max_iter=2000),
            )
            model.fit(x_train, y_train)
            residual_pred = model.predict(x_val).astype(np.float32)
            if args.target_mode == "valence":
                residual_pred[:, 1] = 0.0
            elif args.target_mode == "arousal":
                residual_pred[:, 0] = 0.0
            base_name = f"TLRC_lags{lag_label(lags)}_a{format_alpha(alpha)}_{args.target_mode}"
            for scale in scales:
                scaled = scale * residual_pred
                for clip in clips:
                    if clip > 0:
                        adjusted = np.clip(prior_val + np.clip(scaled, -clip, clip), 1.0, 255.0)
                        clip_suffix = f"_clip{clip:.0f}"
                    else:
                        adjusted = np.clip(prior_val + scaled, 1.0, 255.0)
                        clip_suffix = ""
                    for window in smooth_windows:
                        if window > 1:
                            pred = smooth_predictions(val_ids, adjusted, window=window)
                            suffix = f"_scale{scale:.2f}{clip_suffix}_smooth{window}"
                        else:
                            pred = adjusted
                            suffix = f"_scale{scale:.2f}{clip_suffix}"
                        results.append(
                            score(
                                base_name + suffix,
                                y_val,
                                pred,
                                "Ridge residual model over concatenated temporal-lag EEG/fNIRS features.",
                            )
                        )
        del x_lag

    output = {
        "method": "TLRC: Temporal-Lag Ridge Correction",
        "feature_cache": str(args.feature_cache),
        "target_mode": args.target_mode,
        "lag_definition": "negative lag uses future features x(t-lag); positive lag uses earlier features",
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
        },
        "results": sorted(results, key=lambda item: float(item["overall_mae"]))[:100],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "x": data["x"].astype(np.float32),
            "y_raw": data["y_raw"].astype(np.float32),
            "sample_ids": data["sample_ids"].astype(str),
            "sample_subjects": data["sample_subjects"].astype(str),
        }


def make_lagged_features(x: np.ndarray, sample_ids: list[str], lags: list[int]) -> np.ndarray:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    out = np.empty((x.shape[0], x.shape[1] * len(lags)), dtype=np.float32)
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        trial_x = x[indices]
        length = len(indices)
        for lag_index, lag in enumerate(lags):
            source = np.arange(length) - lag
            source = np.clip(source, 0, length - 1)
            start = lag_index * x.shape[1]
            stop = start + x.shape[1]
            out[np.asarray(indices), start:stop] = trial_x[source]
    return out


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def subject_of(sample_id: str) -> str:
    return sample_id.split("_V", 1)[0]


def parse_lag_sets(value: str) -> list[list[int]]:
    return [parse_ints(chunk) for chunk in value.split(";") if chunk.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def lag_label(lags: list[int]) -> str:
    return "_".join(f"{lag:+d}" for lag in lags).replace("+", "p").replace("-", "m")


def format_alpha(alpha: float) -> str:
    return str(alpha).replace(".", "p")


if __name__ == "__main__":
    main()
