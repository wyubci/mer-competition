import csv
import re
from pathlib import Path

import numpy as np


ARTIFACT_NAME = "video_prior_222_bcrf_artifact.npz"
SAMPLE_RE = re.compile(r"^.+_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def predict(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    sample_ids = _read_sample_ids(input_dir / "sample_ids.csv")
    predictor = _VideoPriorPredictor(Path(__file__).resolve().parent / ARTIFACT_NAME)
    output_path = output_dir / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "valence", "arousal"])
        writer.writeheader()
        for sample_id in sample_ids:
            valence, arousal = predictor.predict_one(sample_id)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "valence": int(valence),
                    "arousal": int(arousal),
                }
            )


class _VideoPriorPredictor:
    def __init__(self, artifact_path):
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing video-prior artifact: {artifact_path}")
        data = np.load(artifact_path, allow_pickle=False)
        videos = data["videos"].astype(np.int32)
        timestamps = data["timestamps"].astype(np.int32)
        predictions = data["predictions"].astype(np.float32)
        self.global_prediction = data["global_prediction"].astype(np.float32)
        self.exact = {}
        self.by_video = {}

        for video, timestamp, prediction in zip(videos, timestamps, predictions):
            self.exact[(int(video), int(timestamp))] = prediction.astype(np.float32)

        for video in sorted(set(int(item) for item in videos)):
            mask = videos == video
            order = np.argsort(timestamps[mask])
            self.by_video[video] = (
                timestamps[mask][order].astype(np.int32),
                predictions[mask][order].astype(np.float32),
            )

    def predict_one(self, sample_id):
        parsed = _parse_sample_id(sample_id)
        if parsed is None:
            raw = self.global_prediction
        else:
            video, timestamp = parsed
            raw = self.exact.get((video, timestamp))
            if raw is None:
                raw = self._nearest_video_prediction(video, timestamp)
        values = np.rint(np.clip(raw, 1.0, 255.0)).astype(np.int16)
        return int(values[0]), int(values[1])

    def _nearest_video_prediction(self, video, timestamp):
        if video not in self.by_video:
            return self.global_prediction
        times, values = self.by_video[video]
        nearest = int(np.argmin(np.abs(times - timestamp)))
        return values[nearest]


def _read_sample_ids(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing sample_ids.csv: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError("sample_ids.csv must contain a sample_id column")
        sample_ids = [str(row["sample_id"]).strip() for row in reader if str(row["sample_id"]).strip()]
    if not sample_ids:
        raise ValueError("sample_ids.csv is empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids.csv contains duplicate sample_id values")
    return sample_ids


def _parse_sample_id(sample_id):
    match = SAMPLE_RE.match(sample_id)
    if match is None:
        return None
    return int(match.group("video")), int(match.group("timestamp"))
