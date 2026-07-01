from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose dimwise MAE results from aggregate runs.")
    parser.add_argument(
        "--inputs",
        default=(
            "experiments/results/iteration_102_oof_prior_stacking.json,"
            "experiments/results/iteration_103_oof_hgb_tuner.json"
        ),
    )
    parser.add_argument("--output", default="experiments/results/iteration_104_dimwise_oof_meta.json")
    parser.add_argument("--top-n", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path_text in [item.strip() for item in args.inputs.split(",") if item.strip()]:
        path = Path(path_text)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload["aggregate_results"]:
            copied = dict(row)
            copied["source_file"] = str(path)
            rows.append(copied)

    unique_rows = dedupe(rows)
    valence_rows = sorted(unique_rows, key=lambda item: float(item["valence_mae"]))[: args.top_n]
    arousal_rows = sorted(unique_rows, key=lambda item: float(item["arousal_mae"]))[: args.top_n]
    combinations = []
    for v_row in valence_rows:
        for a_row in arousal_rows:
            overall = (float(v_row["valence_mae"]) + float(a_row["arousal_mae"])) / 2.0
            combinations.append(
                {
                    "method": f"DimwiseOOFMeta_V[{v_row['method']}]__A[{a_row['method']}]",
                    "overall_mae": round(overall, 4),
                    "valence_mae": round(float(v_row["valence_mae"]), 4),
                    "arousal_mae": round(float(a_row["arousal_mae"]), 4),
                    "overall_mse": None,
                    "valence_source_file": v_row["source_file"],
                    "arousal_source_file": a_row["source_file"],
                    "notes": (
                        "Metric-composed dimwise model selection. Valid for MAE comparison because "
                        "submission has independent valence and arousal columns; regenerate the two "
                        "columns from their source models for an actual prediction file."
                    ),
                }
            )
    combinations = sorted(combinations, key=lambda item: float(item["overall_mae"]))
    output = {
        "method": "Dimwise OOF meta-calibrator composition",
        "inputs": [item.strip() for item in args.inputs.split(",") if item.strip()],
        "top_valence_sources": valence_rows,
        "top_arousal_sources": arousal_rows,
        "dimwise_results": combinations,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def dedupe(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_method: dict[str, dict[str, object]] = {}
    for row in rows:
        method = str(row["method"])
        if method not in best_by_method:
            best_by_method[method] = row
            continue
        if float(row["overall_mae"]) < float(best_by_method[method]["overall_mae"]):
            best_by_method[method] = row
    return list(best_by_method.values())


if __name__ == "__main__":
    main()
