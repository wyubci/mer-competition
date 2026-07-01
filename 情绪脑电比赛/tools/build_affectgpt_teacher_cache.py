from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.features import discover_subjects  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, parse_sample_id  # noqa: E402


EMOTION_PROTOTYPES = {
    0: (0.0, 0.0),  # neutral
    1: (1.0, 1.0),  # happy
    2: (-1.0, 1.0),  # fear
    3: (-1.0, -1.0),  # sad
    4: (1.0, -1.0),  # relax
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sample-wise teacher embeddings from AffectGPT LoRA checkpoints and MER-PS metadata."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--models-root", default="models/AffectGPT")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--output", default="experiments/teacher_cache/affectgpt_teacher_cache.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_subjects = expand_subjects(args.train_subjects)
    all_subjects = discover_subjects(data_root)
    labels = load_labels(data_root, all_subjects)
    sample_ids = np.asarray(list(labels.keys())).astype(str)
    max_time_by_video = compute_max_time_by_video(sample_ids)
    targeted = read_targeted_emotions(data_root / "Targeted_emotions.txt")
    sam_by_video = read_sam_video_means(data_root / "SAM_score.csv", train_subjects)
    affectgpt_vector, checkpoint_info = affectgpt_lora_vector(Path(args.models_root))

    emotion_rows = []
    sam_rows = []
    affectgpt_rows = []
    semantic_rows = []
    for sample_id in sample_ids:
        _, video, timestamp = parse_sample_id(str(sample_id))
        target = targeted[video - 1]
        one_hot = np.zeros(5, dtype=np.float32)
        one_hot[target] = 1.0
        prototype = np.asarray(EMOTION_PROTOTYPES[target], dtype=np.float32)
        max_time = max(max_time_by_video[video], 1)
        t = np.float32(timestamp / max_time)
        time_features = np.asarray(
            [
                t,
                np.sin(np.pi * t),
                np.cos(np.pi * t),
                np.sin(2.0 * np.pi * t),
                np.cos(2.0 * np.pi * t),
            ],
            dtype=np.float32,
        )
        video_one_hot = np.zeros(15, dtype=np.float32)
        video_one_hot[video - 1] = 1.0
        sam = sam_by_video.get(video, np.zeros(4, dtype=np.float32))
        affect = affectgpt_vector.astype(np.float32)
        emotion = np.concatenate([one_hot, prototype, video_one_hot, time_features])
        semantic = np.concatenate([emotion, sam, affect])
        emotion_rows.append(emotion)
        sam_rows.append(sam)
        affectgpt_rows.append(affect)
        semantic_rows.append(semantic)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=sample_ids,
        emotion=np.stack(emotion_rows).astype(np.float32),
        sam=np.stack(sam_rows).astype(np.float32),
        affectgpt=np.stack(affectgpt_rows).astype(np.float32),
        semantic=np.stack(semantic_rows).astype(np.float32),
        checkpoint_info=json.dumps(checkpoint_info, ensure_ascii=False),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": int(sample_ids.shape[0]),
                "emotion_dim": int(len(emotion_rows[0])),
                "sam_dim": int(len(sam_rows[0])),
                "affectgpt_dim": int(len(affectgpt_rows[0])),
                "semantic_dim": int(len(semantic_rows[0])),
                "checkpoints": checkpoint_info,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def compute_max_time_by_video(sample_ids: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    for sample_id in sample_ids:
        _, video, timestamp = parse_sample_id(str(sample_id))
        out[video] = max(out[video], timestamp)
    return dict(out)


def read_targeted_emotions(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"([0-4](?:\s+[0-4]){14})\s*;", text)
    if not match:
        raise ValueError(f"Could not parse targeted emotions from {path}")
    values = [int(item) for item in match.group(1).split()]
    if len(values) != 15:
        raise ValueError(f"Expected 15 targeted emotion ids, got {len(values)}")
    return values


def read_sam_video_means(path: Path, train_subjects: list[str]) -> dict[int, np.ndarray]:
    train_set = set(train_subjects)
    sums = {video: np.zeros(4, dtype=np.float64) for video in range(1, 16)}
    counts = {video: 0 for video in range(1, 16)}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sub_id"] not in train_set:
                continue
            for video in range(1, 16):
                values = np.asarray(
                    [
                        float(row[f"Video_{video}_Valence"]),
                        float(row[f"Video_{video}_Arousal"]),
                        float(row[f"Video_{video}_Dominance"]),
                        float(row[f"Video_{video}_Familiarity"]),
                    ],
                    dtype=np.float64,
                )
                sums[video] += values
                counts[video] += 1
    return {
        video: (((sums[video] / max(counts[video], 1)) - 5.0) / 4.0).astype(np.float32)
        for video in range(1, 16)
    }


def affectgpt_lora_vector(models_root: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    checkpoints = sorted(models_root.rglob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth AffectGPT checkpoints found under {models_root}")
    parts = []
    info = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        tensor_items = [(key, value.float()) for key, value in state.items() if torch.is_tensor(value)]
        vector = lora_statistics(tensor_items)
        parts.append(vector)
        config = payload.get("config", {})
        info.append(
            {
                "path": str(checkpoint),
                "arch": config.get("model", {}).get("arch"),
                "dataset": list(config.get("datasets", {}).keys()),
                "epoch": payload.get("epoch"),
                "dim": int(vector.shape[0]),
            }
        )
    merged = np.concatenate(parts).astype(np.float32)
    merged = (merged - merged.mean()) / (merged.std() + 1e-6)
    return merged, info


def lora_statistics(tensor_items: list[tuple[str, torch.Tensor]]) -> np.ndarray:
    groups = {
        "all": tensor_items,
        "attn": [(key, value) for key, value in tensor_items if "self_attn" in key],
        "mlp": [(key, value) for key, value in tensor_items if ".mlp." in key],
        "lora_a": [(key, value) for key, value in tensor_items if "lora_A" in key],
        "lora_b": [(key, value) for key, value in tensor_items if "lora_B" in key],
        "q_proj": [(key, value) for key, value in tensor_items if "q_proj" in key],
        "k_proj": [(key, value) for key, value in tensor_items if "k_proj" in key],
        "v_proj": [(key, value) for key, value in tensor_items if "v_proj" in key],
        "o_proj": [(key, value) for key, value in tensor_items if "o_proj" in key],
        "gate_proj": [(key, value) for key, value in tensor_items if "gate_proj" in key],
        "up_proj": [(key, value) for key, value in tensor_items if "up_proj" in key],
        "down_proj": [(key, value) for key, value in tensor_items if "down_proj" in key],
    }
    stats = []
    for _, items in groups.items():
        if not items:
            stats.extend([0.0] * 6)
            continue
        flat = torch.cat([value.reshape(-1) for _, value in items])
        stats.extend(
            [
                float(flat.mean()),
                float(flat.std(unbiased=False)),
                float(flat.abs().mean()),
                float(flat.square().mean().sqrt()),
                float(flat.min()),
                float(flat.max()),
            ]
        )
    return np.asarray(stats, dtype=np.float32)


if __name__ == "__main__":
    main()
