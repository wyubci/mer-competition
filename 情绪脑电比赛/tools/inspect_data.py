from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.features import discover_subjects, read_mat_v5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MER-PS .mat file shapes.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1", help="Comma-separated subjects or 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    subjects = discover_subjects(data_root) if args.subjects == "all" else split_csv(args.subjects)
    for subject in subjects:
        subject_dir = data_root / "data" / subject
        labels = read_mat_v5(data_root / "annotations" / f"{subject}_label.mat")
        eeg = read_mat_v5(subject_dir / "EEG_videos.mat")
        fnirs = read_mat_v5(subject_dir / "fNIRS_videos.mat")
        print(f"\n{subject}")
        for key in sorted(labels, key=video_sort_key):
            label_shape = tuple(labels[key].shape)
            eeg_shape = tuple(eeg[key].shape)
            fnirs_shape = tuple(fnirs[key].shape)
            print(f"  {key}: label={label_shape} eeg={eeg_shape} fnirs={fnirs_shape}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def video_sort_key(key: str) -> int:
    return int(key.split("_", 1)[1]) if key.startswith("video_") else 10_000


if __name__ == "__main__":
    main()
