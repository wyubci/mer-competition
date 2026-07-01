from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cross_fold_batch20_new_models import clip, make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125  # noqa: E402
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_oof_prior_stacking import (  # noqa: E402
    build_oof_training_set,
    make_candidates,
    make_feature_matrix,
    make_pattern_098,
    parse_strings,
)
from tools.cross_fold_pattern_prior_expert import DEFAULT_POOL  # noqa: E402
from tools.cross_fold_residual_field_module import make_manual_200  # noqa: E402
from tools.cross_fold_bcrf_module import bayesian_credible_residual_field, make_scrf_218  # noqa: E402
from tools.cross_fold_to200_architectures import make_previous_167  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels  # noqa: E402
from tools.trial_basis_residual import parse_sample_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the video-prior 222_BCRF_onSCRF Codabench submission."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--package-dir",
        default="submissions/video_prior_222_bcrf",
        help="Directory containing submission model.py and generated artifact.",
    )
    parser.add_argument(
        "--output",
        default="submissions/video_prior_222_bcrf.zip",
        help="Final zip path. The zip root contains model.py and the artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    labels = load_labels(Path(args.data_root), subjects)
    candidate_pool = parse_strings(",".join(DEFAULT_POOL))
    config = SimpleNamespace(
        candidate_pool=",".join(DEFAULT_POOL),
        quantile_lows="15,20",
        quantile_highs="45,50,55,60,70",
        max_gates="0.25,0.35,0.45,0.5,0.55",
        long_smooths="43,51,61",
        ensemble_weights="0.5",
        seed=args.seed,
    )

    train_ids = ids_for_subjects(labels, subjects)
    y_train_full = labels_to_array(labels, train_ids)
    target_ids = build_target_ids(labels)

    x_train, y_train, prior_train, train_rows = build_oof_training_set(
        labels=labels,
        train_subjects=subjects,
        candidate_pool=candidate_pool,
        args=config,
    )
    oof_train_ids: list[str] = []
    for subject in subjects:
        oof_train_ids.extend(ids_for_subjects(labels, [subject]))
    residual_target = (y_train - prior_train).astype(np.float32)

    val_candidates = make_candidates(train_ids, y_train_full, target_ids, config)
    prior_val = make_pattern_098(target_ids, val_candidates)
    x_val = make_feature_matrix(target_ids, val_candidates, candidate_pool, prior_val)
    candidate_stack = np.stack([val_candidates[name] for name in candidate_pool], axis=0).astype(np.float32)
    candidate_std = candidate_stack.std(axis=0).astype(np.float32)

    ref104, _, _ = make_reference_104(
        x_train=x_train,
        residual_target=residual_target,
        x_val=x_val,
        prior_val=prior_val,
        val_ids=target_ids,
        seed=args.seed,
    )
    previous_125 = make_previous_125(ref104, target_ids)
    previous_167 = make_previous_167(
        previous_125=previous_125,
        oof_train_ids=oof_train_ids,
        y_train=y_train,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=target_ids,
    )
    p200, parts = make_manual_200(
        previous_167=previous_167,
        oof_train_ids=oof_train_ids,
        y_train=y_train,
        prior_train=prior_train,
        residual_target=residual_target,
        val_ids=target_ids,
        candidate_std=candidate_std,
    )
    p218 = make_scrf_218(oof_train_ids, prior_train, y_train, target_ids, p200)
    b_delta, b_conf = bayesian_credible_residual_field(
        train_ids=oof_train_ids,
        train_pred=prior_train,
        y_train=y_train,
        val_ids=target_ids,
        base_pred=p200,
    )
    p222 = p218.copy()
    p222[:, 0] = p218[:, 0] + 0.50 * b_conf[:, 0] * b_delta[:, 0]
    p222 = clip(p222).astype(np.float32)

    videos, timestamps = split_video_time(target_ids)
    package_dir = Path(args.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = package_dir / "video_prior_222_bcrf_artifact.npz"
    np.savez_compressed(
        artifact_path,
        videos=videos.astype(np.int16),
        timestamps=timestamps.astype(np.int16),
        predictions=p222.astype(np.float32),
        global_prediction=y_train_full.mean(axis=0).astype(np.float32),
        train_subjects=np.asarray(subjects).astype(str),
        method=np.asarray(["222_BCRF_onSCRF"]),
        seed=np.asarray([args.seed], dtype=np.int32),
    )

    metadata = {
        "method": "222_BCRF_onSCRF",
        "local_cv_overall_mae": 28.6868,
        "local_cv_valence_mae": 26.8958,
        "local_cv_arousal_mae": 30.4777,
        "artifact_rows": int(len(target_ids)),
        "train_rows": int(train_rows),
        "seed": int(args.seed),
        "note": "Video-time prior trained from public trainval labels; inference maps sample_id video/time to the learned trajectory.",
        "component_reference": {
            "p200": "manual milestone fusion",
            "p218": "SCRF valence consensus residual field",
            "p222": "BCRF credible residual on SCRF",
            "p188": "valence risk expert",
            "p195": "arousal conformal median-band projector",
        },
    }
    (package_dir / "submission_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(package_dir / "model.py", "model.py")
        archive.write(artifact_path, "video_prior_222_bcrf_artifact.npz")
        archive.write(package_dir / "submission_metadata.json", "submission_metadata.json")
    print(f"Wrote artifact: {artifact_path}")
    print(f"Wrote submission: {output} ({output.stat().st_size / 1024:.1f} KB)")


def build_target_ids(labels: dict[str, np.ndarray]) -> list[str]:
    max_time_by_video: dict[int, int] = {}
    for sample_id in labels:
        _, video, timestamp = parse_sample_id(sample_id)
        max_time_by_video[video] = max(max_time_by_video.get(video, -1), timestamp)
    target_ids: list[str] = []
    for video in sorted(max_time_by_video):
        for timestamp in range(max_time_by_video[video] + 1):
            target_ids.append(f"predict1_V{video:02d}_T{timestamp:03d}")
    return target_ids


def split_video_time(sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    videos = []
    timestamps = []
    for sample_id in sample_ids:
        _, video, timestamp = parse_sample_id(sample_id)
        videos.append(video)
        timestamps.append(timestamp)
    return np.asarray(videos, dtype=np.int16), np.asarray(timestamps, dtype=np.int16)


if __name__ == "__main__":
    main()
