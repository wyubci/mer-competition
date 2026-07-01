from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.features import load_training_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small teacher-cache NPZ from public labels. "
            "This is a smoke-test teacher for the distillation pipeline, not a hidden-test input."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--output", default="experiments/checkpoints/label_teacher_cache.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, y_unit, _, _, sample_ids = load_training_features(
        args.data_root,
        include_sample_ids=True,
        verbose=True,
    )
    valence = y_unit[:, 0:1] * 254.0 + 1.0
    arousal = y_unit[:, 1:2] * 254.0 + 1.0
    centered = np.concatenate([(valence - 128.0) / 127.0, (arousal - 128.0) / 127.0], axis=1)
    radius = np.linalg.norm(centered, axis=1, keepdims=True)
    angle = np.arctan2(centered[:, 1:2], centered[:, 0:1])
    quadrants = np.concatenate(
        [
            (centered[:, 0:1] >= 0) & (centered[:, 1:2] >= 0),
            (centered[:, 0:1] < 0) & (centered[:, 1:2] >= 0),
            (centered[:, 0:1] < 0) & (centered[:, 1:2] < 0),
            (centered[:, 0:1] >= 0) & (centered[:, 1:2] < 0),
        ],
        axis=1,
    ).astype(np.float32)
    emotion = np.concatenate(
        [
            centered.astype(np.float32),
            radius.astype(np.float32),
            np.sin(angle).astype(np.float32),
            np.cos(angle).astype(np.float32),
            quadrants,
        ],
        axis=1,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sample_ids=sample_ids.astype(str), emotion=emotion)
    print(f"Wrote {output} with emotion teacher shape {emotion.shape}")


if __name__ == "__main__":
    main()
