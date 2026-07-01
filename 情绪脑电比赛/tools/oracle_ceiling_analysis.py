from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import expand_subjects, load_labels, predict_video_time_mean, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle ceiling diagnostics for MER-PS priors.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_079_oracle_ceiling.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    subjects = train_subjects + val_subjects
    labels = load_labels(data_root, subjects)
    all_ids = list(labels)
    all_y = np.stack([labels[sample_id] for sample_id in all_ids]).astype(np.float32)
    val_ids = [sample_id for sample_id in all_ids if subject_of(sample_id) in val_subjects]
    y_val = np.stack([labels[sample_id] for sample_id in val_ids]).astype(np.float32)

    train_ids = [sample_id for sample_id in all_ids if subject_of(sample_id) in train_subjects]
    y_train = np.stack([labels[sample_id] for sample_id in train_ids]).astype(np.float32)
    train_prior = predict_video_time_mean(train_ids, y_train, val_ids)

    loo_prior = leave_one_subject_video_time_mean(all_ids, all_y, val_ids)
    oracle_all_subject_prior = all_subject_video_time_mean(all_ids, all_y, val_ids)
    subject_offset_oracle = add_subject_mean_offset(train_prior, labels, train_ids, val_ids)
    trial_offset_oracle = add_trial_mean_offset(train_prior, labels, val_ids)
    linear_affine_oracle = fit_subject_affine_oracle(train_prior, labels, val_ids)

    results = [
        score("Train20_VideoTimeMean", y_val, train_prior, "Allowed train-subject video-time prior."),
        score(
            "LOSO24_VideoTimeMean_oracle",
            y_val,
            loo_prior,
            "Oracle: use all other public subjects including validation subjects.",
        ),
        score(
            "All24_VideoTimeMean_leaky_oracle",
            y_val,
            oracle_all_subject_prior,
            "Leaky oracle: average includes the target subject.",
        ),
        score(
            "SubjectMeanOffset_oracle",
            y_val,
            subject_offset_oracle,
            "Oracle: add each validation subject's mean residual offset.",
        ),
        score(
            "TrialMeanOffset_oracle",
            y_val,
            trial_offset_oracle,
            "Oracle: add each validation trial's mean residual offset.",
        ),
        score(
            "SubjectAffine_oracle",
            y_val,
            linear_affine_oracle,
            "Oracle: per-validation-subject affine calibration against labels.",
        ),
    ]

    output = {
        "method": "Oracle ceiling diagnostics",
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "val_samples": len(val_ids),
        },
        "results": sorted(results, key=lambda item: float(item["overall_mae"])),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def leave_one_subject_video_time_mean(
    all_ids: list[str],
    all_y: np.ndarray,
    target_ids: list[str],
) -> np.ndarray:
    by_key: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
    for sample_id, y in zip(all_ids, all_y):
        by_key[video_time_of(sample_id)].append((subject_of(sample_id), y))
    global_mean = all_y.mean(axis=0)
    pred = []
    for sample_id in target_ids:
        subject = subject_of(sample_id)
        values = [y for other_subject, y in by_key[video_time_of(sample_id)] if other_subject != subject]
        if values:
            pred.append(np.mean(values, axis=0))
        else:
            pred.append(global_mean)
    return np.asarray(pred, dtype=np.float32)


def all_subject_video_time_mean(
    all_ids: list[str],
    all_y: np.ndarray,
    target_ids: list[str],
) -> np.ndarray:
    by_key: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for sample_id, y in zip(all_ids, all_y):
        by_key[video_time_of(sample_id)].append(y)
    global_mean = all_y.mean(axis=0)
    return np.asarray(
        [np.mean(by_key.get(video_time_of(sample_id), [global_mean]), axis=0) for sample_id in target_ids],
        dtype=np.float32,
    )


def add_subject_mean_offset(
    train_prior: np.ndarray,
    labels: dict[str, np.ndarray],
    train_ids: list[str],
    val_ids: list[str],
) -> np.ndarray:
    y_train = np.stack([labels[sample_id] for sample_id in train_ids]).astype(np.float32)
    global_offset = y_train.mean(axis=0) - train_prior.mean(axis=0)
    pred = train_prior.copy()
    for subject in sorted({subject_of(sample_id) for sample_id in val_ids}):
        indices = [index for index, sample_id in enumerate(val_ids) if subject_of(sample_id) == subject]
        y = np.stack([labels[val_ids[index]] for index in indices]).astype(np.float32)
        offset = y.mean(axis=0) - train_prior[indices].mean(axis=0)
        pred[indices] += offset - global_offset
    return np.clip(pred, 1.0, 255.0)


def add_trial_mean_offset(
    train_prior: np.ndarray,
    labels: dict[str, np.ndarray],
    val_ids: list[str],
) -> np.ndarray:
    pred = train_prior.copy()
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample_id in enumerate(val_ids):
        groups[(subject_of(sample_id), video_time_of(sample_id)[0])].append(index)
    for indices in groups.values():
        y = np.stack([labels[val_ids[index]] for index in indices]).astype(np.float32)
        pred[indices] += y.mean(axis=0) - train_prior[indices].mean(axis=0)
    return np.clip(pred, 1.0, 255.0)


def fit_subject_affine_oracle(
    train_prior: np.ndarray,
    labels: dict[str, np.ndarray],
    val_ids: list[str],
) -> np.ndarray:
    pred = train_prior.copy()
    for subject in sorted({subject_of(sample_id) for sample_id in val_ids}):
        indices = [index for index, sample_id in enumerate(val_ids) if subject_of(sample_id) == subject]
        y = np.stack([labels[val_ids[index]] for index in indices]).astype(np.float32)
        x = train_prior[indices]
        x_aug = np.concatenate([x, np.ones((len(indices), 1), dtype=np.float32)], axis=1)
        for dim in range(2):
            coef, *_ = np.linalg.lstsq(x_aug, y[:, dim], rcond=None)
            pred[indices, dim] = x_aug @ coef
    return np.clip(pred, 1.0, 255.0)


def subject_of(sample_id: str) -> str:
    return sample_id.split("_V", 1)[0]


def video_time_of(sample_id: str) -> tuple[int, int]:
    rest = sample_id.split("_V", 1)[1]
    video_text, time_text = rest.split("_T", 1)
    return int(video_text), int(time_text)


if __name__ == "__main__":
    main()
