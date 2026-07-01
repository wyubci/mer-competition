import csv
import os
from pathlib import Path

import numpy as np
import torch

from emotion_merps.features import (
    apply_standardization,
    build_prediction_features,
    read_sample_rows,
)
from emotion_merps.model import ASACRegressor


CHECKPOINT_NAME = os.environ.get("MERPS_CHECKPOINT_NAME", "best_model.pt")


def predict(input_dir, output_dir):
    """Load the trained ASAC baseline checkpoint and predict hidden MER_PS samples."""
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    submission_dir = Path(__file__).resolve().parent
    checkpoint_path = submission_dir / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing required ASAC checkpoint: {checkpoint_path}")

    rows = read_sample_rows(input_dir)
    sample_ids, eeg, fnirs = build_prediction_features(input_dir, rows)
    checkpoint = _load_checkpoint(checkpoint_path)
    eeg, fnirs = apply_standardization(eeg, fnirs, checkpoint["standardization"])
    prediction = _run_model(checkpoint, eeg, fnirs)
    _write_predictions(sample_ids, prediction, output_dir)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _run_model(checkpoint, eeg, fnirs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASACRegressor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    outputs = []
    batch_size = int(os.environ.get("MERPS_BATCH_SIZE", "256"))
    with torch.no_grad():
        for start in range(0, eeg.shape[0], batch_size):
            stop = min(start + batch_size, eeg.shape[0])
            eeg_batch = torch.from_numpy(np.ascontiguousarray(eeg[start:stop])).float().to(device)
            fnirs_batch = torch.from_numpy(np.ascontiguousarray(fnirs[start:stop])).float().to(device)
            unit_prediction, _ = model(eeg_batch, fnirs_batch)
            raw = unit_prediction.cpu().numpy() * 254.0 + 1.0
            outputs.append(np.clip(raw, 1.0, 255.0))
    return np.concatenate(outputs, axis=0)


def _write_predictions(sample_ids, prediction, output_dir):
    output_path = Path(output_dir) / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.rint(prediction).astype(np.int16)
    prediction = np.clip(prediction, 1, 255)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "valence", "arousal"])
        writer.writeheader()
        for sample_id, values in zip(sample_ids, prediction):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "valence": int(values[0]),
                    "arousal": int(values[1]),
                }
            )
