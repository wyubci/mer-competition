from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.consensus_gated_residual import bilateral_smooth  # noqa: E402
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
        description="Train nonlinear tabular residual experts on cached EEG/fNIRS features."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_078_tree_residual_expert.json")
    parser.add_argument("--models", default="hgb_l1,hgb_l2,extratrees")
    parser.add_argument("--top-feature-count", type=int, default=220)
    parser.add_argument("--smooth-windows", default="0,5,9,13,17")
    parser.add_argument("--pabs-windows", default="13,17,21")
    parser.add_argument("--pabs-sigmas", default="7.5,10,15,20")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    data_root = Path(args.data_root)
    labels = load_labels(data_root, train_subjects + val_subjects)
    train_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
    ]
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(
        np.float32
    )

    with np.load(args.feature_cache, allow_pickle=False) as data:
        x = data["x"].astype(np.float32)
        y_raw = data["y_raw"].astype(np.float32)
        sample_subjects = data["sample_subjects"].astype(str)
        sample_ids = data["sample_ids"].astype(str)
        summary = json.loads(str(data["summary"].item()))

    train_idx = np.flatnonzero(np.isin(sample_subjects, train_subjects))
    val_idx = np.flatnonzero(np.isin(sample_subjects, val_subjects))
    train_ids = sample_ids[train_idx].astype(str).tolist()
    val_ids = sample_ids[val_idx].astype(str).tolist()
    x_train_raw = x[train_idx]
    x_val_raw = x[val_idx]
    y_train = y_raw[train_idx]
    y_val = y_raw[val_idx]
    train_prior = predict_video_time_mean(train_label_ids, y_train_label_order, train_ids)
    val_prior = predict_video_time_mean(train_label_ids, y_train_label_order, val_ids)

    x_train_context = build_context_features(train_ids, train_prior)
    x_val_context = build_context_features(val_ids, val_prior)
    selected = select_features_by_residual_corr(
        x_train_raw,
        y_train - train_prior,
        top_k=args.top_feature_count,
    )
    x_train = np.concatenate([x_train_raw[:, selected], x_train_context], axis=1).astype(np.float32)
    x_val = np.concatenate([x_val_raw[:, selected], x_val_context], axis=1).astype(np.float32)
    y_residual = (y_train - train_prior).astype(np.float32)

    results: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    prior_stats = score("VideoTimeMean", y_val, val_prior, "No signal prior.")
    results.append(prior_stats)
    predictions["VideoTimeMean"] = val_prior

    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        model = make_model(model_name, args.seed)
        print(
            f"[tree] fitting {model_name} x_train={x_train.shape} selected={len(selected)}",
            flush=True,
        )
        model.fit(x_train, y_residual)
        residual = np.asarray(model.predict(x_val), dtype=np.float32)
        evaluate_residual_family(
            results,
            predictions,
            model_name,
            val_ids,
            y_val,
            val_prior,
            residual,
            smooth_windows=parse_ints(args.smooth_windows),
            pabs_windows=parse_ints(args.pabs_windows),
            pabs_sigmas=parse_floats(args.pabs_sigmas),
        )

    output = {
        "method": "Tree residual expert over selected EEG/fNIRS tabular features",
        "feature_cache": args.feature_cache,
        "feature_summary": summary,
        "selected_feature_count": int(len(selected)),
        "context_feature_dim": int(x_train_context.shape[1]),
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


def make_model(name: str, seed: int):
    if name == "hgb_l1":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.045,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                l2_regularization=0.05,
                random_state=seed,
            )
        )
    if name == "hgb_l2":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_iter=220,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                l2_regularization=0.10,
                random_state=seed + 1,
            )
        )
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=320,
            max_depth=8,
            min_samples_leaf=18,
            max_features=0.35,
            bootstrap=False,
            random_state=seed + 2,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_residual_family(
    results: list[dict[str, object]],
    predictions: dict[str, np.ndarray],
    model_name: str,
    val_ids: list[str],
    y_val: np.ndarray,
    val_prior: np.ndarray,
    residual: np.ndarray,
    smooth_windows: list[int],
    pabs_windows: list[int],
    pabs_sigmas: list[float],
) -> None:
    for scale in (0.25, 0.5, 0.75, 1.0, 1.25):
        pred = np.clip(val_prior + scale * residual, 1.0, 255.0)
        name = f"{model_name}_residual_scale{format_float(scale)}"
        predictions[name] = pred
        results.append(score(name, y_val, pred, "Tree residual expert over VideoTimeMean prior."))
        for window in smooth_windows:
            if window <= 1:
                continue
            smooth = smooth_predictions(val_ids, pred, window=window)
            smooth_name = f"{name}_smooth{window}"
            predictions[smooth_name] = smooth
            results.append(
                score(smooth_name, y_val, smooth, "Tree residual expert with moving average.")
            )
        for window in pabs_windows:
            for sigma in pabs_sigmas:
                pabs = np.stack(
                    [
                        bilateral_smooth(val_ids, pred[:, 0], val_prior[:, 0], window, sigma),
                        bilateral_smooth(val_ids, pred[:, 1], val_prior[:, 1], window, sigma),
                    ],
                    axis=1,
                )
                pabs_name = f"{name}_pabs{window}_sigma{format_float(sigma)}"
                predictions[pabs_name] = pabs
                results.append(
                    score(pabs_name, y_val, pabs, "Tree residual expert with PA-BS smoothing.")
                )


def build_context_features(sample_ids: list[str], prior: np.ndarray) -> np.ndarray:
    video = []
    timestamp = []
    for sample_id in sample_ids:
        _, v, t = parse_sample_id(sample_id)
        video.append(v)
        timestamp.append(t)
    video_arr = np.asarray(video, dtype=np.float32)
    time_arr = np.asarray(timestamp, dtype=np.float32)
    time_max_by_video: dict[int, float] = defaultdict(float)
    for v, t in zip(video, timestamp):
        time_max_by_video[int(v)] = max(time_max_by_video[int(v)], float(t))
    time_norm = np.asarray(
        [t / max(time_max_by_video[int(v)], 1.0) for v, t in zip(video, timestamp)],
        dtype=np.float32,
    )
    video_onehot = np.zeros((len(sample_ids), 15), dtype=np.float32)
    for index, v in enumerate(video):
        if 1 <= int(v) <= 15:
            video_onehot[index, int(v) - 1] = 1.0
    prior_slope = prior_slope_by_trial(sample_ids, prior)
    context = [
        prior,
        prior_slope,
        time_norm[:, None],
        np.sin(2 * np.pi * time_norm)[:, None],
        np.cos(2 * np.pi * time_norm)[:, None],
        (video_arr[:, None] - 8.0) / 7.0,
        video_onehot,
    ]
    return np.concatenate(context, axis=1).astype(np.float32)


def prior_slope_by_trial(sample_ids: list[str], prior: np.ndarray) -> np.ndarray:
    slope = np.zeros_like(prior, dtype=np.float32)
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        subject, video, timestamp = parse_sample_id(sample_id)
        groups[(subject, video)].append((timestamp, index))
    for items in groups.values():
        indices = [index for _, index in sorted(items)]
        values = prior[indices]
        if len(indices) < 2:
            continue
        local = np.zeros_like(values)
        local[0] = values[1] - values[0]
        local[-1] = values[-1] - values[-2]
        if len(indices) > 2:
            local[1:-1] = 0.5 * (values[2:] - values[:-2])
        slope[indices] = local
    return slope


def select_features_by_residual_corr(
    x_train: np.ndarray,
    residual: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if top_k <= 0 or top_k >= x_train.shape[1]:
        return np.arange(x_train.shape[1])
    x_centered = x_train - x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_centered.std(axis=0), 1e-6)
    y_centered = residual - residual.mean(axis=0, keepdims=True)
    y_std = np.maximum(y_centered.std(axis=0), 1e-6)
    corr = np.abs((x_centered / x_std).T @ (y_centered / y_std) / max(x_train.shape[0] - 1, 1))
    score_by_feature = corr.max(axis=1)
    return np.argsort(score_by_feature)[-top_k:]


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
