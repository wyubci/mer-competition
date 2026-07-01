from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array, parse_floats  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    build_oof_training_set,
    make_candidates,
    make_feature_matrix,
    make_pattern_098,
    parse_strings,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score, smooth_predictions  # noqa: E402


DEFAULT_CONFIGS = [
    "l1,0.025,100,7,120,0.30",
    "l1,0.030,140,9,100,0.20",
    "l1,0.035,120,11,90,0.15",
    "l1,0.040,120,11,70,0.10",
    "l1,0.035,160,15,100,0.20",
    "l1,0.045,100,15,120,0.30",
    "l2,0.030,140,9,120,0.30",
    "l2,0.035,120,11,100,0.20",
    "l2,0.040,100,15,120,0.30",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune OOF HGB residual meta-calibrator.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_103_oof_hgb_tuner.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--configs", default=";".join(DEFAULT_CONFIGS))
    parser.add_argument("--residual-scales", default="0,0.06,0.08,0.1,0.12,0.15,0.18,0.2,0.25,0.3")
    parser.add_argument("--smooth-windows", default="0,3,5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    candidate_pool = parse_strings(args.candidate_pool)
    configs = parse_configs(args.configs)
    residual_scales = parse_floats(args.residual_scales)
    smooth_windows = [int(item) for item in args.smooth_windows.split(",") if item.strip()]

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] building shared OOF features", flush=True)
        x_train, y_train, prior_train, train_rows = build_oof_training_set(
            labels=labels,
            train_subjects=train_subjects,
            candidate_pool=candidate_pool,
            args=args,
        )
        residual_target = (y_train - prior_train).astype(np.float32)

        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        val_candidates = make_candidates(train_ids, y_outer_train, val_ids, args)
        prior_val = make_pattern_098(val_ids, val_candidates)
        x_val = make_feature_matrix(val_ids, val_candidates, candidate_pool, prior_val)

        fold_predictions: dict[str, np.ndarray] = {"PatternPrior_098": prior_val}
        for config_index, config in enumerate(configs, start=1):
            print(f"[fold {fold_index}] fitting {config['name']}", flush=True)
            model = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    loss=config["loss"],
                    learning_rate=config["learning_rate"],
                    max_iter=config["max_iter"],
                    max_leaf_nodes=config["max_leaf_nodes"],
                    min_samples_leaf=config["min_samples_leaf"],
                    l2_regularization=config["l2_regularization"],
                    random_state=args.seed + 100 * fold_index + config_index,
                )
            )
            model.fit(x_train, residual_target)
            residual = np.asarray(model.predict(x_val), dtype=np.float32)
            add_residual_predictions(
                fold_predictions,
                config["name"],
                prior_val,
                residual,
                val_ids,
                residual_scales,
                smooth_windows,
            )

        fold_results = sorted(
            [score(name, y_val, pred, "OOF HGB residual meta-calibrator tuning.") for name, pred in fold_predictions.items()],
            key=lambda item: float(item["overall_mae"]),
        )
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "train_oof_rows": train_rows,
                "val_samples": len(val_ids),
                "feature_dim": int(x_train.shape[1]),
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
        "method": "OOF HGB residual meta-calibrator tuning",
        "note": "Reuses strict inner leave-one-subject-out prior features from iteration 102.",
        "fold_size": args.fold_size,
        "candidate_pool_size": len(candidate_pool),
        "configs": configs,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def parse_configs(value: str) -> list[dict[str, object]]:
    configs = []
    for raw in value.split(";"):
        if not raw.strip():
            continue
        loss_key, lr, max_iter, leaf_nodes, min_leaf, l2 = [item.strip() for item in raw.split(",")]
        loss = "absolute_error" if loss_key == "l1" else "squared_error"
        name = (
            f"HGBTune_{loss_key}_lr{format_float(float(lr))}_it{int(max_iter)}"
            f"_leaf{int(leaf_nodes)}_min{int(min_leaf)}_l2{format_float(float(l2))}"
        )
        configs.append(
            {
                "name": name,
                "loss": loss,
                "learning_rate": float(lr),
                "max_iter": int(max_iter),
                "max_leaf_nodes": int(leaf_nodes),
                "min_samples_leaf": int(min_leaf),
                "l2_regularization": float(l2),
            }
        )
    return configs


def add_residual_predictions(
    predictions: dict[str, np.ndarray],
    config_name: str,
    prior: np.ndarray,
    residual: np.ndarray,
    sample_ids: list[str],
    scales: list[float],
    smooth_windows: list[int],
) -> None:
    for scale in scales:
        pred = np.clip(prior + scale * residual, 1.0, 255.0).astype(np.float32)
        name = f"{config_name}_scale{format_float(scale)}"
        predictions[name] = pred
        for window in smooth_windows:
            if window <= 1:
                continue
            predictions[f"{name}_smooth{window}"] = smooth_predictions(sample_ids, pred, window)


def format_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
