from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = ("sample_id", "valence", "arousal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a MER-PS predictions.csv file.")
    parser.add_argument("--sample-ids", required=True, help="Path to input sample_ids.csv.")
    parser.add_argument("--predictions", required=True, help="Path to output predictions.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = read_sample_ids(Path(args.sample_ids))
    predictions = read_predictions(Path(args.predictions))

    expected_set = set(expected)
    predicted_set = set(predictions)
    missing = expected_set - predicted_set
    extra = predicted_set - expected_set
    if missing:
        raise SystemExit(f"Missing {len(missing)} sample_id values, e.g. {sorted(missing)[:5]}")
    if extra:
        raise SystemExit(f"Found {len(extra)} unexpected sample_id values, e.g. {sorted(extra)[:5]}")
    if len(predictions) != len(expected):
        raise SystemExit("Duplicate sample_id values detected in predictions.csv")

    print(
        f"OK: {len(predictions)} predictions, columns={','.join(REQUIRED_COLUMNS)}, "
        "integer values in [1, 255]."
    )


def read_sample_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise SystemExit("sample_ids.csv must contain a sample_id column")
        sample_ids = [row["sample_id"].strip() for row in reader]
    if not sample_ids:
        raise SystemExit("sample_ids.csv is empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise SystemExit("sample_ids.csv contains duplicate sample_id values")
    return sample_ids


def read_predictions(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise SystemExit(f"predictions.csv missing required columns: {missing_columns}")

        predictions: dict[str, tuple[int, int]] = {}
        for line_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"].strip()
            if sample_id in predictions:
                raise SystemExit(f"Duplicate sample_id at line {line_number}: {sample_id}")
            predictions[sample_id] = (
                parse_label(row["valence"], "valence", line_number),
                parse_label(row["arousal"], "arousal", line_number),
            )
    if not predictions:
        raise SystemExit("predictions.csv is empty")
    return predictions


def parse_label(value: str, column: str, line_number: int) -> int:
    try:
        parsed_float = float(value)
    except ValueError as exc:
        raise SystemExit(f"{column} at line {line_number} is not numeric: {value}") from exc
    parsed_int = int(parsed_float)
    if parsed_float != parsed_int:
        raise SystemExit(f"{column} at line {line_number} is not an integer: {value}")
    if parsed_int < 1 or parsed_int > 255:
        raise SystemExit(f"{column} at line {line_number} outside [1, 255]: {value}")
    return parsed_int


if __name__ == "__main__":
    main()
