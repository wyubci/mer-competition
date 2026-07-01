from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from emotion_merps.features import build_prediction_features, read_sample_rows


ARTIFACT_NAME = "neuro_overlay_sub9_artifact.npz"
SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def predict(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rows = read_sample_rows(input_dir)
    sample_ids = [str(row["sample_id"]) for row in rows]
    _, eeg, fnirs = build_prediction_features(input_dir, rows)

    artifact = _Artifact(Path(__file__).resolve().parent / ARTIFACT_NAME)
    base = artifact.base_predictions(sample_ids)
    residual = artifact.neurovascular_residual(sample_ids, eeg, fnirs)
    pred = base.copy()
    pred[:, 0] = np.clip(
        pred[:, 0] + np.clip(artifact.overlay_scale * residual[:, 0], -artifact.overlay_clip, artifact.overlay_clip),
        1.0,
        255.0,
    )
    pred = np.rint(np.clip(pred, 1.0, 255.0)).astype(np.int16)

    output_path = output_dir / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "valence", "arousal"])
        writer.writeheader()
        for sample_id, values in zip(sample_ids, pred):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "valence": int(values[0]),
                    "arousal": int(values[1]),
                }
            )


class _Artifact:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        data = np.load(path, allow_pickle=False)
        self.videos = data["videos"].astype(np.int32)
        self.timestamps = data["timestamps"].astype(np.int32)
        self.predictions = data["predictions"].astype(np.float32)
        self.global_prediction = data["global_prediction"].astype(np.float32)
        self.eeg_weight = data["eeg_weight"].astype(np.float32)
        self.overlay_scale = float(data["overlay_scale"][0])
        self.overlay_clip = float(data["overlay_clip"][0])
        self.eeg_model = _LinearModel(data, "eeg")
        self.fnirs_model = _LinearModel(data, "fnirs")
        self.exact: dict[tuple[int, int], np.ndarray] = {}
        self.by_video: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for video, timestamp, prediction in zip(self.videos, self.timestamps, self.predictions):
            self.exact[(int(video), int(timestamp))] = prediction.astype(np.float32)
        for video in sorted(set(int(item) for item in self.videos)):
            mask = self.videos == video
            order = np.argsort(self.timestamps[mask])
            self.by_video[video] = (
                self.timestamps[mask][order].astype(np.int32),
                self.predictions[mask][order].astype(np.float32),
            )

    def base_predictions(self, sample_ids: list[str]) -> np.ndarray:
        rows = []
        for sample_id in sample_ids:
            parsed = _parse_sample_id(sample_id)
            if parsed is None:
                rows.append(self.global_prediction)
                continue
            _, video, timestamp = parsed
            value = self.exact.get((video, timestamp))
            if value is None:
                value = self._nearest_video_prediction(video, timestamp)
            rows.append(value)
        return np.stack(rows, axis=0).astype(np.float32)

    def neurovascular_residual(self, sample_ids: list[str], eeg: np.ndarray, fnirs: np.ndarray) -> np.ndarray:
        features = _build_precomputed_features(eeg, fnirs, sample_ids)
        eeg_pred = self.eeg_model.predict(features["eeg_lag"])
        fnirs_pred = self.fnirs_model.predict(features["fnirs_slow"])
        consensus = self.eeg_weight[None, :] * eeg_pred + (1.0 - self.eeg_weight[None, :]) * fnirs_pred
        agreement = (np.sign(eeg_pred) == np.sign(fnirs_pred)).astype(np.float32)
        ratio = np.minimum(np.abs(eeg_pred), np.abs(fnirs_pred)) / (
            np.maximum(np.abs(eeg_pred), np.abs(fnirs_pred)) + 1e-3
        )
        confidence = agreement * (0.35 + 0.65 * ratio)
        return (confidence * consensus).astype(np.float32)

    def _nearest_video_prediction(self, video: int, timestamp: int) -> np.ndarray:
        if video not in self.by_video:
            return self.global_prediction
        times, values = self.by_video[video]
        nearest = int(np.argmin(np.abs(times - timestamp)))
        return values[nearest]


class _LinearModel:
    def __init__(self, data: np.lib.npyio.NpzFile, prefix: str):
        self.x_mean = data[f"{prefix}_x_mean"].astype(np.float32)
        self.x_scale = np.maximum(data[f"{prefix}_x_scale"].astype(np.float32), 1e-6)
        self.coef = data[f"{prefix}_coef"].astype(np.float32)
        self.intercept = data[f"{prefix}_intercept"].astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (_sanitize(x) - self.x_mean[None, :]) / self.x_scale[None, :]
        return (z @ self.coef.T + self.intercept[None, :]).astype(np.float32)


def _build_precomputed_features(eeg: np.ndarray, fnirs: np.ndarray, sample_ids: list[str]) -> dict[str, np.ndarray]:
    eeg_core = _eeg_core_features(eeg)
    fnirs_core = _fnirs_core_features(fnirs)
    eeg_lag = _lagged_features(sample_ids, eeg_core, lags=(0, 1, 2, 4), include_delta=True)
    fnirs_slow = np.concatenate(
        [
            fnirs_core,
            _rolling_past_features(sample_ids, fnirs_core, windows=(3, 5, 9)),
            _lagged_features(sample_ids, fnirs_core, lags=(1, 3, 5), include_delta=True),
        ],
        axis=1,
    )
    return {"eeg_lag": _sanitize(eeg_lag), "fnirs_slow": _sanitize(fnirs_slow)}


def _eeg_core_features(eeg: np.ndarray) -> np.ndarray:
    mean_band = eeg.mean(axis=1)
    std_band = eeg.std(axis=1)
    half = eeg.shape[1] // 2
    spatial_diff = eeg[:, :half].mean(axis=1) - eeg[:, half:].mean(axis=1)
    beta_alpha = mean_band[:, 3:4] - mean_band[:, 2:3]
    gamma_alpha = mean_band[:, 4:5] - mean_band[:, 2:3]
    return _sanitize(np.concatenate([mean_band, std_band, spatial_diff, beta_alpha, gamma_alpha], axis=1))


def _fnirs_core_features(fnirs: np.ndarray) -> np.ndarray:
    mean_feat = fnirs.mean(axis=1)
    std_feat = fnirs.std(axis=1)
    half = fnirs.shape[1] // 2
    spatial_diff = fnirs[:, :half].mean(axis=1) - fnirs[:, half:].mean(axis=1)
    hbo_hbr = mean_feat[:, 0:1] - mean_feat[:, 1:2]
    hbt_slope = mean_feat[:, 8:9] if mean_feat.shape[1] > 8 else np.zeros_like(hbo_hbr)
    return _sanitize(np.concatenate([mean_feat, std_feat, spatial_diff, hbo_hbr, hbt_slope], axis=1))


def _lagged_features(
    sample_ids: list[str],
    values: np.ndarray,
    lags: tuple[int, ...],
    include_delta: bool,
) -> np.ndarray:
    groups = _group_indices(sample_ids)
    lagged = np.zeros((len(sample_ids), values.shape[1] * len(lags)), dtype=np.float32)
    delta_lags = [lag for lag in lags if lag > 0]
    delta_out = np.zeros((len(sample_ids), values.shape[1] * len(delta_lags)), dtype=np.float32)
    for items in groups.values():
        items = sorted(items)
        indices = [index for _, index in items]
        index_array = np.asarray(indices, dtype=np.int64)
        seq = values[indices]
        for lag_pos, lag in enumerate(lags):
            shifted = _shift_past(seq, lag)
            lagged[index_array, lag_pos * values.shape[1] : (lag_pos + 1) * values.shape[1]] = shifted
            if include_delta and lag > 0:
                delta_pos = delta_lags.index(lag)
                start = delta_pos * values.shape[1]
                stop = start + values.shape[1]
                delta_out[index_array, start:stop] = seq - shifted
    if include_delta and delta_lags:
        return _sanitize(np.concatenate([lagged, delta_out], axis=1))
    return _sanitize(lagged)


def _rolling_past_features(sample_ids: list[str], values: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    out = np.zeros((len(sample_ids), values.shape[1] * len(windows)), dtype=np.float32)
    for items in _group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        cumsum = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(seq, axis=0)], axis=0)
        for win_pos, window in enumerate(windows):
            rolled = np.zeros_like(seq)
            for local_index in range(seq.shape[0]):
                start = max(0, local_index - window + 1)
                stop = local_index + 1
                rolled[local_index] = (cumsum[stop] - cumsum[start]) / float(stop - start)
            out[np.asarray(indices), win_pos * values.shape[1] : (win_pos + 1) * values.shape[1]] = rolled
    return _sanitize(out)


def _group_indices(sample_ids: list[str]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        parsed = _parse_sample_id(sample_id)
        if parsed is None:
            continue
        subject, video, timestamp = parsed
        groups[(subject, video)].append((timestamp, index))
    return groups


def _shift_past(seq: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return seq.copy()
    shifted = np.empty_like(seq)
    shifted[:lag] = seq[:1]
    shifted[lag:] = seq[:-lag]
    return shifted


def _parse_sample_id(sample_id: str) -> tuple[str, int, int] | None:
    match = SAMPLE_RE.match(sample_id)
    if match is None:
        return None
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def _sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
