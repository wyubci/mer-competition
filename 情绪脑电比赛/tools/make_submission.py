from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


DEFAULT_PACKAGE_FILES = (
    "model.py",
    "emotion_merps/__init__.py",
    "emotion_merps/features.py",
    "emotion_merps/model.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MER-PS Codabench submission zip.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained best_model.pt.")
    parser.add_argument("--output", default="submissions/emotion_mtdp_submission.zip")
    parser.add_argument(
        "--include-distill",
        action="store_true",
        help="Also package distillation utilities for reproducibility. Inference does not need them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    package_files = list(DEFAULT_PACKAGE_FILES)
    if args.include_distill:
        package_files.append("emotion_merps/distill.py")

    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in package_files:
            path = Path(relative)
            if not path.exists():
                raise FileNotFoundError(path)
            archive.write(path, relative)
        archive.write(checkpoint, "best_model.pt")

    print(f"Wrote {output} ({format_size(output.stat().st_size)})")

    # Keep a plain copy nearby for local smoke tests against model.py.
    local_checkpoint = Path("best_model.pt")
    if checkpoint.resolve() != local_checkpoint.resolve():
        shutil.copy2(checkpoint, local_checkpoint)
        print(f"Copied {checkpoint} -> {local_checkpoint}")


def format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


if __name__ == "__main__":
    main()
