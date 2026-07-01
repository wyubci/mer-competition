from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_video_prior_222_submission import build_target_ids, split_video_time  # noqa: E402
from tools.cross_fold_batch20_new_models import clip, make_reference_104  # noqa: E402
from tools.cross_fold_batch3_architectures import make_previous_125  # noqa: E402
from tools.cross_fold_bcrf_module import bayesian_credible_residual_field, make_scrf_218  # noqa: E402
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.cross_fold_neurovascular_oof_gate import nested_expert_predictions  # noqa: E402
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


ARTIFACT_NAME = "neuro_overlay_sub9_artifact.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sub9: 222_BCRF video prior plus valence-only neurovascular overlay."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--overlay-scale", type=float, default=0.05)
    parser.add_argument("--overlay-clip", type=float, default=1.0)
    parser.add_argument(
        "--precompute-cache",
        default="experiments/features/neurovascular_precompute_baseline.npz",
    )
    parser.add_argument("--package-dir", default="submissions/sub9")
    parser.add_argument("--output", default="submissions/sub9.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    subjects = expand_subjects(args.subjects)
    labels = load_labels(data_root, subjects)
    config = SimpleNamespace(
        candidate_pool=",".join(DEFAULT_POOL),
        quantile_lows="15,20",
        quantile_highs="45,50,55,60,70",
        max_gates="0.25,0.35,0.45,0.5,0.55",
        long_smooths="43,51,61",
        ensemble_weights="0.5",
        seed=args.seed,
    )
    candidate_pool = parse_strings(config.candidate_pool)

    train_ids = ids_for_subjects(labels, subjects)
    y_train_full = labels_to_array(labels, train_ids)
    target_ids = build_target_ids(labels)

    x_train_meta, y_oof, prior_oof, train_rows = build_oof_training_set(
        labels=labels,
        train_subjects=subjects,
        candidate_pool=candidate_pool,
        args=config,
    )
    oof_train_ids: list[str] = []
    for subject in subjects:
        oof_train_ids.extend(ids_for_subjects(labels, [subject]))
    residual_target = (y_oof - prior_oof).astype(np.float32)

    p222 = build_p222_predictions(
        train_ids=train_ids,
        y_train_full=y_train_full,
        target_ids=target_ids,
        x_train_meta=x_train_meta,
        y_oof=y_oof,
        prior_oof=prior_oof,
        residual_target=residual_target,
        oof_train_ids=oof_train_ids,
        candidate_pool=candidate_pool,
        args=config,
        seed=args.seed,
    )

    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root=data_root,
        subjects=subjects,
        cache_path=Path(args.precompute_cache),
    )
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}
    feature_train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
    views = {"eeg": pre["eeg_lag"], "fnirs": pre["fnirs_slow"]}
    expert_oof, _ = nested_expert_predictions(
        views=views,
        train_idx=feature_train_idx,
        val_idx=feature_train_idx[:1],
        train_ids=oof_train_ids,
        train_subjects=subjects,
        residual_train=residual_target,
        alpha=float(args.alpha),
    )
    eeg_mse = ((expert_oof["eeg"] - residual_target) ** 2).mean(axis=0) + 1e-6
    fnirs_mse = ((expert_oof["fnirs"] - residual_target) ** 2).mean(axis=0) + 1e-6
    eeg_weight = (1.0 / eeg_mse) / (1.0 / eeg_mse + 1.0 / fnirs_mse)

    eeg_model = fit_linear_model(pre["eeg_lag"][feature_train_idx], residual_target, float(args.alpha))
    fnirs_model = fit_linear_model(pre["fnirs_slow"][feature_train_idx], residual_target, float(args.alpha))

    videos, timestamps = split_video_time(target_ids)
    package_dir = Path(args.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = package_dir / ARTIFACT_NAME
    np.savez_compressed(
        artifact_path,
        videos=videos.astype(np.int16),
        timestamps=timestamps.astype(np.int16),
        predictions=p222.astype(np.float32),
        global_prediction=y_train_full.mean(axis=0).astype(np.float32),
        eeg_weight=eeg_weight.astype(np.float32),
        overlay_scale=np.asarray([args.overlay_scale], dtype=np.float32),
        overlay_clip=np.asarray([args.overlay_clip], dtype=np.float32),
        eeg_x_mean=eeg_model["x_mean"],
        eeg_x_scale=eeg_model["x_scale"],
        eeg_coef=eeg_model["coef"],
        eeg_intercept=eeg_model["intercept"],
        fnirs_x_mean=fnirs_model["x_mean"],
        fnirs_x_scale=fnirs_model["x_scale"],
        fnirs_coef=fnirs_model["coef"],
        fnirs_intercept=fnirs_model["intercept"],
        train_subjects=np.asarray(subjects).astype(str),
        method=np.asarray(["222_BCRF_onSCRF_plus_247_AgreementOverlay_v_s0p05_c1p0"]),
    )

    metadata = {
        "method": "222_BCRF_onSCRF_plus_247_AgreementOverlay_v_s0p05_c1p0",
        "local_cv_overall_mae": 28.6830,
        "local_cv_valence_mae": 26.8882,
        "local_cv_arousal_mae": 30.4778,
        "overlay": {
            "scale": float(args.overlay_scale),
            "clip": float(args.overlay_clip),
            "mode": "valence_only",
            "alpha": float(args.alpha),
            "eeg_weight": [float(item) for item in eeg_weight],
        },
        "artifact_rows": int(len(target_ids)),
        "train_rows": int(train_rows),
        "feature_shapes": feature_shapes,
        "note": (
            "Video-prior 222 is the base. EEG/fNIRS residual experts are trained from public "
            "trainval OOF residuals; inference applies a tiny valence-only agreement overlay."
        ),
    }
    (package_dir / "submission_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    copy_support_files(root, package_dir)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                archive.write(path, path.relative_to(package_dir).as_posix())
    print(f"Wrote artifact: {artifact_path}")
    print(f"Wrote submission: {output} ({output.stat().st_size / 1024:.1f} KB)")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def build_p222_predictions(
    train_ids: list[str],
    y_train_full: np.ndarray,
    target_ids: list[str],
    x_train_meta: np.ndarray,
    y_oof: np.ndarray,
    prior_oof: np.ndarray,
    residual_target: np.ndarray,
    oof_train_ids: list[str],
    candidate_pool: list[str],
    args: SimpleNamespace,
    seed: int,
) -> np.ndarray:
    val_candidates = make_candidates(train_ids, y_train_full, target_ids, args)
    prior_val = make_pattern_098(target_ids, val_candidates)
    x_val = make_feature_matrix(target_ids, val_candidates, candidate_pool, prior_val)
    candidate_stack = np.stack([val_candidates[name] for name in candidate_pool], axis=0).astype(np.float32)
    candidate_std = candidate_stack.std(axis=0).astype(np.float32)
    ref104, _, _ = make_reference_104(
        x_train=x_train_meta,
        residual_target=residual_target,
        x_val=x_val,
        prior_val=prior_val,
        val_ids=target_ids,
        seed=seed,
    )
    previous_125 = make_previous_125(ref104, target_ids)
    previous_167 = make_previous_167(
        previous_125=previous_125,
        oof_train_ids=oof_train_ids,
        y_train=y_oof,
        prior_train=prior_oof,
        residual_target=residual_target,
        val_ids=target_ids,
    )
    p200, _ = make_manual_200(
        previous_167=previous_167,
        oof_train_ids=oof_train_ids,
        y_train=y_oof,
        prior_train=prior_oof,
        residual_target=residual_target,
        val_ids=target_ids,
        candidate_std=candidate_std,
    )
    p218 = make_scrf_218(oof_train_ids, prior_oof, y_oof, target_ids, p200)
    b_delta, b_conf = bayesian_credible_residual_field(
        train_ids=oof_train_ids,
        train_pred=prior_oof,
        y_train=y_oof,
        val_ids=target_ids,
        base_pred=p200,
    )
    p222 = p218.copy()
    p222[:, 0] = p218[:, 0] + 0.50 * b_conf[:, 0] * b_delta[:, 0]
    return clip(p222).astype(np.float32)


def fit_linear_model(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    model.fit(sanitize(x_train), y_train)
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    return {
        "x_mean": scaler.mean_.astype(np.float32),
        "x_scale": np.maximum(scaler.scale_.astype(np.float32), 1e-6),
        "coef": ridge.coef_.astype(np.float32),
        "intercept": ridge.intercept_.astype(np.float32),
    }


def copy_support_files(root: Path, package_dir: Path) -> None:
    source_model = root / "submissions" / "sub9" / "model.py"
    target_model = package_dir / "model.py"
    if source_model.resolve() != target_model.resolve():
        shutil.copy2(source_model, target_model)
    support_dir = package_dir / "emotion_merps"
    support_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "emotion_merps" / "__init__.py", support_dir / "__init__.py")
    shutil.copy2(root / "emotion_merps" / "features.py", support_dir / "features.py")


if __name__ == "__main__":
    main()
