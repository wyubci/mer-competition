from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_batch20_new_models import clip, make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125  # noqa: E402
from tools.cross_fold_bcrf_module import (  # noqa: E402
    bayesian_credible_residual_field,
    make_scrf_218,
)
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    build_oof_training_set as build_oof_training_set_reference,
    make_candidates as make_candidates_reference,
    make_feature_matrix,
    make_pattern_098,
    parse_strings,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.cross_fold_residual_field_module import make_manual_200  # noqa: E402
from tools.cross_fold_to200_architectures import make_previous_167  # noqa: E402
from tools.cross_fold_trimmed_prior import (  # noqa: E402
    build_robust_stats,
    make_dispersion,
    make_prior as make_robust_prior,
    uncertainty_blend,
)
from tools.run_iteration_experiments import expand_subjects, load_labels, score  # noqa: E402


MEAN_BLEND_RE = re.compile(
    r"^UncertaintyBlend_meanPrior_q(?P<low>[0-9]+p[0-9]+)-(?P<high>[0-9]+p[0-9]+)_g(?P<gate>[0-9]+p[0-9]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject robust mean-prior estimators into the full 200/SCRF/BCRF route."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/iteration_541_bcrf_robust_meanprior.json")
    parser.add_argument("--candidate-pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--quantile-lows", default="15,20")
    parser.add_argument("--quantile-highs", default="45,50,55,60,70")
    parser.add_argument("--max-gates", default="0.25,0.35,0.45,0.5,0.55")
    parser.add_argument("--long-smooths", default="43,51,61")
    parser.add_argument("--ensemble-weights", default="0.5")
    parser.add_argument("--estimators", default="winsor5,winsor10,drop1,trim5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    labels = load_labels(Path(args.data_root), subjects)
    candidate_pool = parse_strings(args.candidate_pool)
    estimators = parse_strings(args.estimators)

    aggregate_truth: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    fold_outputs = []

    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] reference + robust mean-prior BCRF", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_outer_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)

        fold_predictions: dict[str, np.ndarray] = {}
        reference = build_route(
            labels=labels,
            train_subjects=train_subjects,
            train_ids=train_ids,
            y_outer_train=y_outer_train,
            val_ids=val_ids,
            candidate_pool=candidate_pool,
            args=args,
            estimator=None,
            seed=args.seed + fold_index * 173,
        )
        for name, pred in reference.items():
            fold_predictions[f"reference_{name}"] = pred

        for estimator in estimators:
            robust = build_route(
                labels=labels,
                train_subjects=train_subjects,
                train_ids=train_ids,
                y_outer_train=y_outer_train,
                val_ids=val_ids,
                candidate_pool=candidate_pool,
                args=args,
                estimator=estimator,
                seed=args.seed + fold_index * 173,
            )
            for name, pred in robust.items():
                fold_predictions[f"{estimator}_{name}"] = pred
            for name, pred in robust.items():
                if name not in reference:
                    continue
                v_robust = reference[name].copy()
                v_robust[:, 0] = pred[:, 0]
                fold_predictions[f"{estimator}_Vrobust_Areference_{name}"] = clip(v_robust)

                a_robust = reference[name].copy()
                a_robust[:, 1] = pred[:, 1]
                fold_predictions[f"{estimator}_Vreference_Arobust_{name}"] = clip(a_robust)

        fold_results = sorted(
            [score(name, y_val, pred, "Full route with robust mean-prior replacement.") for name, pred in fold_predictions.items()],
            key=lambda item: float(item["overall_mae"]),
        )
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
        "method": "Full BCRF route with robust mean-prior replacement",
        "note": (
            "The replacement is applied consistently to OOF training priors and validation priors. "
            "Only UncertaintyBlend_meanPrior_* references are changed; the robust median base, smooth references, "
            "and downstream 200/SCRF/BCRF logic stay unchanged."
        ),
        "estimators": estimators,
        "fold_size": args.fold_size,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_route(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    train_ids: list[str],
    y_outer_train: np.ndarray,
    val_ids: list[str],
    candidate_pool: list[str],
    args: argparse.Namespace,
    estimator: str | None,
    seed: int,
) -> dict[str, np.ndarray]:
    if estimator is None:
        x_train, y_train, prior_train, _ = build_oof_training_set_reference(
            labels=labels,
            train_subjects=train_subjects,
            candidate_pool=candidate_pool,
            args=args,
        )
        val_candidates = make_candidates_reference(train_ids, y_outer_train, val_ids, args)
    else:
        x_train, y_train, prior_train, _ = build_oof_training_set_robust(
            labels=labels,
            train_subjects=train_subjects,
            candidate_pool=candidate_pool,
            args=args,
            estimator=estimator,
        )
        val_candidates = make_candidates_robust(train_ids, y_outer_train, val_ids, args, estimator)

    oof_train_ids: list[str] = []
    for subject in train_subjects:
        oof_train_ids.extend(ids_for_subjects(labels, [subject]))
    residual_target = (y_train - prior_train).astype(np.float32)

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
        seed=seed,
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
    scrf_delta = p218 - p200
    agree = (b_delta[:, 0] * scrf_delta[:, 0]) > 0.0

    p222 = p218.copy()
    p222[:, 0] = p218[:, 0] + 0.50 * b_conf[:, 0] * b_delta[:, 0]

    p224 = p200.copy()
    p224[:, 0] = p200[:, 0] + np.where(agree, scrf_delta[:, 0], 0.0)

    return {
        "098_PatternPrior": clip(prior_val),
        "200_MilestoneSynthesisFusion": clip(p200),
        "218_SCRF": clip(p218),
        "222_BCRF_onSCRF": clip(p222),
        "224_BCRF_Brake": clip(p224),
    }


def build_oof_training_set_robust(
    labels: dict[str, np.ndarray],
    train_subjects: list[str],
    candidate_pool: list[str],
    args: argparse.Namespace,
    estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x_parts = []
    y_parts = []
    prior_parts = []
    for subject in train_subjects:
        fit_subjects = [item for item in train_subjects if item != subject]
        fit_ids = ids_for_subjects(labels, fit_subjects)
        target_ids = ids_for_subjects(labels, [subject])
        y_fit = labels_to_array(labels, fit_ids)
        y_target = labels_to_array(labels, target_ids)
        candidates = make_candidates_robust(fit_ids, y_fit, target_ids, args, estimator)
        prior = make_pattern_098(target_ids, candidates)
        x_parts.append(make_feature_matrix(target_ids, candidates, candidate_pool, prior))
        y_parts.append(y_target)
        prior_parts.append(prior)
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    prior = np.concatenate(prior_parts, axis=0).astype(np.float32)
    return x, y, prior, int(y.shape[0])


def make_candidates_robust(
    train_ids: list[str],
    y_train: np.ndarray,
    target_ids: list[str],
    args: argparse.Namespace,
    estimator: str,
) -> dict[str, np.ndarray]:
    candidates = make_candidates_reference(train_ids, y_train, target_ids, args)
    stats = build_robust_stats(train_ids, y_train)
    base_raw = make_robust_prior(stats, target_ids, estimator="median", lag=-2, smooth=0)
    base = make_robust_prior(stats, target_ids, estimator="median", lag=-2, smooth=11)
    dispersion = make_dispersion(stats, target_ids, lag=-2)
    from tools.run_iteration_experiments import smooth_predictions  # local import avoids another top-level alias

    dispersion = smooth_predictions(target_ids, dispersion, 11).astype(np.float32)
    robust_ref = make_robust_prior(stats, target_ids, estimator=estimator, lag=-2, smooth=11)
    # Keep long smooth references anchored to the same median raw trajectory.
    _ = base_raw

    for name in list(candidates):
        match = MEAN_BLEND_RE.match(name)
        if not match:
            continue
        q_low = parse_p_float(match.group("low"))
        q_high = parse_p_float(match.group("high"))
        max_gate = parse_p_float(match.group("gate"))
        candidates[name] = uncertainty_blend(base, robust_ref, dispersion, q_low, q_high, max_gate)
    return candidates


def parse_p_float(value: str) -> float:
    return float(value.replace("p", "."))


if __name__ == "__main__":
    main()
