from __future__ import annotations

import csv
import re
import struct
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EEG_BANDS: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
)
DEFAULT_FNIRS_TYPES = (0, 1, 2)
SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")
MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15
MI_TO_DTYPE = {
    MI_INT8: np.dtype("i1"),
    MI_UINT8: np.dtype("u1"),
    MI_INT16: np.dtype("<i2"),
    MI_UINT16: np.dtype("<u2"),
    MI_INT32: np.dtype("<i4"),
    MI_UINT32: np.dtype("<u4"),
    MI_SINGLE: np.dtype("<f4"),
    MI_DOUBLE: np.dtype("<f8"),
    MI_INT64: np.dtype("<i8"),
    MI_UINT64: np.dtype("<u8"),
}


def read_sample_rows(input_dir: str | Path) -> list[dict[str, object]]:
    sample_path = Path(input_dir) / "sample_ids.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample_ids.csv in {input_dir}")

    rows: list[dict[str, object]] = []
    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError("sample_ids.csv must contain a sample_id column")
        for row in reader:
            sample_id = row["sample_id"].strip()
            match = SAMPLE_RE.match(sample_id)
            if not match:
                raise ValueError(f"Invalid sample_id format: {sample_id}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject": match.group("subject"),
                    "video": int(match.group("video")),
                    "timestamp": int(match.group("timestamp")),
                }
            )
    if not rows:
        raise ValueError("sample_ids.csv has no rows")
    return rows


def build_prediction_features(
    input_dir: str | Path,
    rows: Sequence[dict[str, object]],
    fnirs_types: Sequence[int] = DEFAULT_FNIRS_TYPES,
    baseline_correction: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    input_dir = Path(input_dir)
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["subject"]), int(row["video"])), []).append(row)

    feature_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for subject in sorted({subject for subject, _ in grouped}):
        subject_dir = input_dir / "data" / subject
        mats = _load_subject_mats(subject_dir, baseline_correction)
        for _, video in sorted(key for key in grouped if key[0] == subject):
            key = (subject, video)
            max_timestamp = max(int(item["timestamp"]) for item in grouped[key])
            n_labels = max_timestamp + 1
            feature_cache[key] = _features_for_trial_from_mats(
                mats,
                f"video_{video}",
                n_labels,
                fnirs_types=fnirs_types,
                baseline_correction=baseline_correction,
            )

    sample_ids: list[str] = []
    eeg_samples: list[np.ndarray] = []
    fnirs_samples: list[np.ndarray] = []

    for row in rows:
        subject = str(row["subject"])
        video = int(row["video"])
        timestamp = int(row["timestamp"])
        key = (subject, video)
        eeg_by_second, fnirs_by_second = feature_cache[key]
        if timestamp >= eeg_by_second.shape[0]:
            raise ValueError(f"{row['sample_id']} timestamp exceeds available signal length")
        sample_ids.append(str(row["sample_id"]))
        eeg_samples.append(eeg_by_second[timestamp])
        fnirs_samples.append(fnirs_by_second[timestamp])

    return (
        sample_ids,
        np.stack(eeg_samples, axis=0).astype(np.float32),
        np.stack(fnirs_samples, axis=0).astype(np.float32),
    )


def load_training_features(
    data_root: str | Path,
    subjects: Iterable[str] | None = None,
    fnirs_types: Sequence[int] = DEFAULT_FNIRS_TYPES,
    baseline_correction: bool = True,
    include_sample_ids: bool = False,
    verbose: bool = False,
) -> tuple[object, ...]:
    data_root = Path(data_root)
    subject_names = list(subjects) if subjects is not None else discover_subjects(data_root)
    if not subject_names:
        raise ValueError(f"No subject folders found under {data_root / 'data'}")

    eeg_samples: list[np.ndarray] = []
    fnirs_samples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_subjects: list[str] = []
    sample_ids: list[str] = []

    for index, subject in enumerate(subject_names, start=1):
        if verbose:
            print(f"[features] loading subject {index}/{len(subject_names)}: {subject}", flush=True)
        subject_dir = data_root / "data" / subject
        labels = _load_mat(data_root / "annotations" / f"{subject}_label.mat")
        mats = _load_subject_mats(subject_dir, baseline_correction)
        video_keys = _video_keys(labels)
        for video_index, video_key in enumerate(video_keys, start=1):
            label = _label_matrix(labels[video_key])
            n_labels = int(label.shape[1])
            eeg_by_second, fnirs_by_second = _features_for_trial_from_mats(
                mats,
                video_key,
                n_labels,
                fnirs_types=fnirs_types,
                baseline_correction=baseline_correction,
            )
            y_by_second = np.clip((label - 1.0) / 254.0, 0.0, 1.0).T.astype(np.float32)
            n = min(eeg_by_second.shape[0], fnirs_by_second.shape[0], y_by_second.shape[0])
            eeg_samples.append(eeg_by_second[:n])
            fnirs_samples.append(fnirs_by_second[:n])
            targets.append(y_by_second[:n])
            sample_subjects.extend([subject] * n)
            if include_sample_ids:
                video_number = int(video_key.split("_", 1)[1])
                sample_ids.extend(
                    f"{subject}_V{video_number:02d}_T{timestamp:03d}"
                    for timestamp in range(n)
                )
            if verbose and (video_index == len(video_keys) or video_index % 5 == 0):
                print(
                    f"[features] {subject}: processed {video_index}/{len(video_keys)} videos",
                    flush=True,
                )

    result: tuple[object, ...] = (
        np.concatenate(eeg_samples, axis=0).astype(np.float32),
        np.concatenate(fnirs_samples, axis=0).astype(np.float32),
        np.concatenate(targets, axis=0).astype(np.float32),
        np.asarray(sample_subjects),
        subject_names,
    )
    if include_sample_ids:
        result = result + (np.asarray(sample_ids),)
    return result


def discover_subjects(data_root: str | Path) -> list[str]:
    data_dir = Path(data_root) / "data"
    subjects = [
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"test_\d+", path.name)
    ]
    return sorted(subjects, key=lambda item: int(item.split("_", 1)[1]))


def standardize_from_train(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    eeg_mean = eeg[train_idx].mean(axis=(0, 1), keepdims=True)
    eeg_std = np.maximum(eeg[train_idx].std(axis=(0, 1), keepdims=True), 1e-6)
    fnirs_mean = fnirs[train_idx].mean(axis=(0, 1), keepdims=True)
    fnirs_std = np.maximum(fnirs[train_idx].std(axis=(0, 1), keepdims=True), 1e-6)
    stats = {
        "eeg_mean": eeg_mean.astype(np.float32),
        "eeg_std": eeg_std.astype(np.float32),
        "fnirs_mean": fnirs_mean.astype(np.float32),
        "fnirs_std": fnirs_std.astype(np.float32),
    }
    return apply_standardization(eeg, fnirs, stats) + (stats,)


def apply_standardization(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    eeg_mean = np.asarray(stats["eeg_mean"], dtype=np.float32)
    eeg_std = np.maximum(np.asarray(stats["eeg_std"], dtype=np.float32), 1e-6)
    fnirs_mean = np.asarray(stats["fnirs_mean"], dtype=np.float32)
    fnirs_std = np.maximum(np.asarray(stats["fnirs_std"], dtype=np.float32), 1e-6)
    eeg_out = ((eeg - eeg_mean) / eeg_std).astype(np.float32)
    fnirs_out = ((fnirs - fnirs_mean) / fnirs_std).astype(np.float32)
    return eeg_out, fnirs_out


def _features_for_trial(
    subject_dir: Path,
    video_key: str,
    n_labels: int,
    fnirs_types: Sequence[int],
    baseline_correction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mats = _load_subject_mats(subject_dir, baseline_correction)
    return _features_for_trial_from_mats(
        mats,
        video_key,
        n_labels,
        fnirs_types=fnirs_types,
        baseline_correction=baseline_correction,
    )


def _load_subject_mats(subject_dir: Path, baseline_correction: bool) -> dict[str, dict[str, np.ndarray]]:
    mats = {
        "eeg_videos": _load_mat(subject_dir / "EEG_videos.mat"),
        "fnirs_videos": _load_mat(subject_dir / "fNIRS_videos.mat"),
    }
    if baseline_correction:
        mats["eeg_baselines"] = _load_mat(subject_dir / "EEG_baselines.mat")
        mats["fnirs_baselines"] = _load_mat(subject_dir / "fNIRS_baselines.mat")
    return mats


def _features_for_trial_from_mats(
    mats: dict[str, dict[str, np.ndarray]],
    video_key: str,
    n_labels: int,
    fnirs_types: Sequence[int],
    baseline_correction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    eeg_videos = mats["eeg_videos"]
    fnirs_videos = mats["fnirs_videos"]
    if video_key not in eeg_videos or video_key not in fnirs_videos:
        raise ValueError(f"Missing {video_key} in subject matrices")

    eeg = np.asarray(eeg_videos[video_key], dtype=np.float32)
    fnirs = np.asarray(fnirs_videos[video_key], dtype=np.float32)
    if baseline_correction:
        eeg = _subtract_eeg_baseline(eeg, mats["eeg_baselines"].get(video_key))
        fnirs = _subtract_fnirs_baseline(fnirs, mats["fnirs_baselines"].get(video_key))

    return (
        _eeg_features_by_label(eeg, n_labels),
        _fnirs_features_by_label(fnirs, n_labels, fnirs_types),
    )


def _load_mat(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_mat_v5(path)


def read_mat_v5(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        header = handle.read(128)
        if len(header) != 128 or b"MATLAB 5.0 MAT-file" not in header[:116]:
            raise ValueError(f"Unsupported .mat file format: {path}")
        payload = handle.read()

    arrays: dict[str, np.ndarray] = {}
    for data_type, data in _iter_elements(payload):
        if data_type == MI_COMPRESSED:
            for inner_type, inner_data in _iter_elements(zlib.decompress(data)):
                if inner_type == MI_MATRIX:
                    name, array = _parse_matrix(inner_data)
                    arrays[name] = array
        elif data_type == MI_MATRIX:
            name, array = _parse_matrix(data)
            arrays[name] = array
    return arrays


def _iter_elements(buffer: bytes):
    offset = 0
    size = len(buffer)
    while offset + 8 <= size:
        data_type, data, offset = _read_element(buffer, offset)
        if data_type == 0 and len(data) == 0:
            break
        yield data_type, data


def _read_element(buffer: bytes, offset: int) -> tuple[int, bytes, int]:
    raw = struct.unpack_from("<I", buffer, offset)[0]
    small_nbytes = raw >> 16
    if small_nbytes:
        data_type = raw & 0xFFFF
        data_start = offset + 4
        data_end = data_start + small_nbytes
        return data_type, buffer[data_start:data_end], offset + 8

    data_type, nbytes = struct.unpack_from("<II", buffer, offset)
    data_start = offset + 8
    data_end = data_start + nbytes
    next_offset = data_end + ((8 - (nbytes % 8)) % 8)
    return data_type, buffer[data_start:data_end], next_offset


def _parse_matrix(data: bytes) -> tuple[str, np.ndarray]:
    offset = 0
    _, _, offset = _read_element(data, offset)  # array flags

    dim_type, dim_data, offset = _read_element(data, offset)
    if dim_type not in (MI_INT32, MI_UINT32):
        raise ValueError("Unsupported MATLAB dimension element")
    dims = np.frombuffer(dim_data, dtype=MI_TO_DTYPE[dim_type]).astype(np.int64)

    name_type, name_data, offset = _read_element(data, offset)
    if name_type not in (MI_INT8, MI_UINT8):
        raise ValueError("Unsupported MATLAB variable name element")
    name = bytes(name_data).decode("utf-8").rstrip("\x00")

    real_type, real_data, _ = _read_element(data, offset)
    if real_type not in MI_TO_DTYPE:
        raise ValueError(f"Unsupported MATLAB numeric type: {real_type}")
    dtype = MI_TO_DTYPE[real_type]
    array = np.frombuffer(real_data, dtype=dtype)
    if dims.size:
        array = array.reshape(tuple(int(dim) for dim in dims), order="F")
    return name, array


def _video_keys(mat: dict[str, np.ndarray]) -> list[str]:
    keys = [key for key in mat if re.fullmatch(r"video_\d+", key)]
    return sorted(keys, key=lambda item: int(item.split("_", 1)[1]))


def _label_matrix(value: np.ndarray) -> np.ndarray:
    label = np.asarray(value, dtype=np.float32)
    if label.ndim != 2:
        raise ValueError(f"Expected label matrix with 2 dims, got shape {label.shape}")
    if label.shape[0] != 2 and label.shape[1] == 2:
        label = label.T
    if label.shape[0] != 2:
        raise ValueError(f"Expected label shape [2, time], got {label.shape}")
    return label


def _subtract_eeg_baseline(eeg: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    if baseline is None:
        return eeg
    base = np.asarray(baseline, dtype=np.float32)
    if base.ndim == 2 and base.shape[0] == eeg.shape[0]:
        return eeg - base.mean(axis=1, keepdims=True)
    return eeg


def _subtract_fnirs_baseline(fnirs: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    if baseline is None:
        return fnirs
    base = np.asarray(baseline, dtype=np.float32)
    if base.ndim == 3 and base.shape[:2] == fnirs.shape[:2]:
        return fnirs - base.mean(axis=2, keepdims=True)
    return fnirs


def _eeg_features_by_label(eeg: np.ndarray, n_labels: int) -> np.ndarray:
    channels, n_samples = eeg.shape
    samples_per_label = n_samples / float(n_labels)
    rounded = int(round(samples_per_label))
    if rounded >= 2 and abs(samples_per_label - rounded) < 1e-4:
        trimmed = eeg[:, : rounded * n_labels]
        segments = trimmed.reshape(channels, n_labels, rounded).transpose(1, 0, 2)
        return _eeg_bandpower_segments(segments, sampling_rate=float(rounded))

    features = np.empty((n_labels, channels, len(EEG_BANDS)), dtype=np.float32)
    for idx in range(n_labels):
        start = int(round(idx * n_samples / n_labels))
        end = max(int(round((idx + 1) * n_samples / n_labels)), start + 1)
        features[idx] = _eeg_bandpower_segments(
            eeg[:, start:end][None, :, :],
            sampling_rate=samples_per_label,
        )[0]
    return features


def _eeg_bandpower_segments(segments: np.ndarray, sampling_rate: float) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.float32)
    segments = segments - segments.mean(axis=-1, keepdims=True)
    length = segments.shape[-1]
    if length < 2:
        return np.zeros((*segments.shape[:2], len(EEG_BANDS)), dtype=np.float32)
    window = np.hanning(length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(segments * window, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(length, d=1.0 / sampling_rate)
    valid = (freqs >= EEG_BANDS[0][0]) & (freqs <= EEG_BANDS[-1][1])
    total = spectrum[..., valid].sum(axis=-1) + 1e-12
    band_features = []
    for low, high in EEG_BANDS:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            band = np.zeros_like(total)
        else:
            band = spectrum[..., mask].sum(axis=-1) / total
        band_features.append(np.log(np.maximum(band, 1e-12)))
    return np.stack(band_features, axis=-1).astype(np.float32)


def _fnirs_features_by_label(
    fnirs: np.ndarray,
    n_labels: int,
    fnirs_types: Sequence[int],
) -> np.ndarray:
    fnirs = np.asarray(fnirs, dtype=np.float32)
    selected = fnirs[np.asarray(fnirs_types, dtype=np.int64)]
    _, channels, n_samples = selected.shape
    feature_dim = len(fnirs_types) * 3
    features = np.empty((n_labels, channels, feature_dim), dtype=np.float32)
    for idx in range(n_labels):
        start = int(round(idx * n_samples / n_labels))
        end = max(int(round((idx + 1) * n_samples / n_labels)), start + 1)
        segment = selected[:, :, start:end]
        mean = segment.mean(axis=-1).T
        std = segment.std(axis=-1).T
        if segment.shape[-1] >= 2:
            slope = ((segment[:, :, -1] - segment[:, :, 0]) / (segment.shape[-1] - 1)).T
        else:
            slope = np.zeros((channels, len(fnirs_types)), dtype=np.float32)
        features[idx] = np.concatenate([mean, std, slope], axis=1)
    return features
