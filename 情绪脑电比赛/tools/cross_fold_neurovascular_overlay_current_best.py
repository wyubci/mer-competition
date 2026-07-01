from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import clip, make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125  # noqa: E402
from tools.cross_fold_bcrf_module import bayesian_credible_residual_field, make_scrf_218  # noqa: E402
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import (  # noqa: E402
    evaluate_candidate,
    evaluate_residual_grid,
    finalize_metric,
    load_or_build_precomputed,
)
from tools.cross_fold_neurovascular_oof_gate import agreement_weighted, nested_expert_predictions  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    build_oof_training_set,
    make_candidates,
    make_feature_matrix,
    make_pattern_098,
    parse_strings,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.cross_fold_residual_field_module import make_manual_200  # noqa: E402
from tools.cross_fold_to200_architectures import make_previous_167  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay neurovascular agreement residual on current best bases.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_baseline.npz")
    parser.add_argument("--output", default="experiments/results/iteration_247_252_neurovascular_overlay_current_best.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--scales", default="0.005,0.01,0.02,0.03,0.05,0.08,0.12")
    parser.add_argument("--clips", default="0.25,0.5,1,2")
    parser.add_argument("--smooth-windows", default="0,5")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(data_root, subjects, Path(args.precompute_cache))
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}
    candidate_pool = parse_strings(args.candidate_pool)
    scales = parse_floats(args.scales)
    clips = parse_floats(args.clips)
    smooth_windows = parse_ints(args.smooth_windows)
    views = {"eeg": pre["eeg_lag"], "fnirs": pre["fnirs_slow"]}

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] overlay current best with neurovascular residual", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)

        bases, residual_train, oof_train_ids = build_current_bases(
            labels=labels,
            train_subjects=train_subjects,
            train_ids=train_ids,
            y_outer_train=y_outer_train,
            val_ids=val_ids,
            candidate_pool=candidate_pool,
            args=args,
            fold_index=fold_index,
        )
        expert_oof, expert_val = nested_expert_predictions(
            views=views,
            train_idx=train_idx,
            val_idx=val_idx,
            train_ids=oof_train_ids,
            train_subjects=train_subjects,
            residual_train=residual_train,
            alpha=args.alpha,
        )
        agreement = agreement_weighted(
            expert_oof["eeg"],
            expert_oof["fnirs"],
            expert_val["eeg"],
            expert_val["fnirs"],
            residual_train,
        )
        confidence = confidence_shrink(expert_val["eeg"], expert_val["fnirs"])
        fold_results = []
        for base_name, base_pred in bases.items():
            fold_results.append(
                evaluate_candidate(metric_acc, base_name, y_val, base_pred, "Current non-signal base reference.")
            )
            for module_name, residual_val in {
                "247_AgreementOverlay": agreement,
                "248_ConfidenceShrinkOverlay": confidence,
            }.items():
                fold_results.extend(
                    evaluate_residual_grid(
                        metric_acc=metric_acc,
                        base_name=f"{base_name}_{module_name}",
                        y_val=y_val,
                        prior_val=base_pred,
                        val_ids=val_ids,
                        residual_val=residual_val,
                        scales=scales,
                        clips=clips,
                        smooth_windows=smooth_windows,
                    )
                )
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[: args.top_k],
            }
        )

    aggregate_results = sorted(
        [finalize_metric(name, payload) for name, payload in metric_acc.items()],
        key=lambda item: float(item["overall_mae"]),
    )
    output = {
        "method": "Neurovascular agreement overlay on current best bases",
        "note": "Tests whether the best EEG-fNIRS agreement residual can improve 200/218/222 bases.",
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_current_bases(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    train_ids: list[str],
    y_outer_train: np.ndarray,
    val_ids: list[str],
    candidate_pool: list[str],
    args: argparse.Namespace,
    fold_index: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    x_train, y_train, prior_train, _ = build_oof_training_set(
        labels=labels,
        train_subjects=train_subjects,
        candidate_pool=candidate_pool,
        args=args,
    )
    oof_train_ids = []
    for subject in train_subjects:
        oof_train_ids.extend(ids_for_subjects(labels, [subject]))
    residual_target = (y_train - prior_train).astype(np.float32)
    val_candidates = make_candidates(train_ids, y_outer_train, val_ids, args)
    prior_val = make_pattern_098(val_ids, val_candidates)
    x_val = make_feature_matrix(val_ids, val_candidates, candidate_pool, prior_val)
    candidate_stack = np.stack([val_candidates[name] for name in candidate_pool], axis=0).astype(np.float32)
    candidate_std = candidate_stack.std(axis=0).astype(np.float32)
    ref104, _, _ = make_reference_104(
        x_train=x_train,
        residual_target=residual_target,
        x_val=x_val,
        prior_val=prior_val,
        val_ids=val_ids,
        seed=args.seed + fold_index * 173,
    )
    previous_125 = make_previous_125(ref104, val_ids)
    previous_167 = make_previous_167(
        previous_125=previous_125,
        oof_train_ids=oof_train_ids,
        y_train=y_train,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=val_ids,
    )
    p200, _ = make_manual_200(
        previous_167=previous_167,
        oof_train_ids=oof_train_ids,
        y_train=y_train,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=val_ids,
        candidate_std=candidate_std,
    )
    p218 = make_scrf_218(oof_train_ids, prior_train, y_train, val_ids, p200)
    b_delta, b_conf = bayesian_credible_residual_field(
        train_ids=oof_train_ids,
        train_pred=prior_train,
        y_train=y_train,
        val_ids=val_ids,
        base_pred=p200,
    )
    p222 = p218.copy()
    p222[:, 0] = p218[:, 0] + 0.50 * b_conf[:, 0] * b_delta[:, 0]
    return (
        {
            "200_CurrentManualFusion": clip(p200),
            "218_SCRF_reference": clip(p218),
            "222_BCRF_onSCRF_reference": clip(p222),
        },
        residual_target,
        oof_train_ids,
    )


def confidence_shrink(eeg_val: np.ndarray, fnirs_val: np.ndarray) -> np.ndarray:
    consensus = 0.5 * (eeg_val + fnirs_val)
    ratio = np.minimum(np.abs(eeg_val), np.abs(fnirs_val)) / (
        np.maximum(np.abs(eeg_val), np.abs(fnirs_val)) + 1e-3
    )
    agreement = (np.sign(eeg_val) == np.sign(fnirs_val)).astype(np.float32)
    return (agreement * ratio * consensus).astype(np.float32)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
