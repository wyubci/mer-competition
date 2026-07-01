from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = {
        "data_preprocessing": data_rows(),
        "ccmi_fusion": ccmi_rows(),
        "output_head": output_head_rows(),
        "combined_recommendation": combined_rows(),
    }
    output_json = ROOT / "experiments" / "results" / "three_module_optimization_summary.json"
    output_md = ROOT / "THREE_MODULE_OPTIMIZATION_SUMMARY.md"
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def data_rows() -> list[dict[str, object]]:
    return [
        row(
            "D1: 3 fNIRS types + baseline mean subtraction",
            "Baseline subtraction removes pre-video offset; only HbO/HbR/HbT are used.",
            "Full 24-subject CV",
            "iteration_263_276_ccmi_neurovascular.json",
            28.7176,
            26.9206,
            30.5146,
            "Strong reference for CCMI.",
            recommended=False,
        ),
        row(
            "D2: 6 fNIRS types + baseline mean subtraction",
            "Adds Abs 780/805/830; extra raw absorption evidence is controlled by CCMI gates.",
            "Full 24-subject CV",
            "iteration_293_306_ccmi_fnirs_all6.json",
            28.7145,
            26.9144,
            30.5146,
            "Best data preprocessing choice.",
            recommended=True,
        ),
        row(
            "D3: 6 fNIRS types + no baseline subtraction",
            "Keeps slow drift and absolute offset; tests whether baseline mean contains emotion signal.",
            "Full 24-subject CV",
            "iteration_307_320_ccmi_fnirs_all6_nobase.json",
            28.7462,
            26.9738,
            30.5186,
            "Does not generalize; falls back to 098.",
            recommended=False,
        ),
        row(
            "D4: 6 fNIRS types + trial z-score",
            "Removes trial-level amplitude and keeps within-video shape only.",
            "4-subject smoke",
            "smoke_ccmi_fnirs_all6_trialz.json",
            28.3478,
            25.5020,
            31.1936,
            "Bad smoke result; do not run full CV.",
            recommended=False,
        ),
        row(
            "D5: 6 fNIRS types + subject z-score",
            "Removes subject-level amplitude shift; tests cross-subject scale drift.",
            "4-subject smoke",
            "smoke_ccmi_fnirs_all6_subjectz.json",
            28.3259,
            25.4582,
            31.1936,
            "Bad smoke result; subject amplitude has useful signal.",
            recommended=False,
        ),
    ]


def ccmi_rows() -> list[dict[str, object]]:
    return [
        row(
            "C1: OOF agreement weighted",
            "Weights EEG and fNIRS residual experts by OOF MSE and requires sign agreement.",
            "Full 24-subject CV",
            "iteration_240_246_neurovascular_oof_gate.json",
            28.7352,
            26.9518,
            30.5186,
            "First stable multimodal residual improvement.",
            recommended=False,
        ),
        row(
            "C2: MinMagnitudeAgreement",
            "Uses sign agreement and min(|EEG|, |fNIRS|) to avoid single-modality overreach.",
            "Full 24-subject CV",
            "iteration_247_262_neurovascular_fusion_v2.json",
            28.7297,
            26.9408,
            30.5186,
            "Simple intersection beats attention-style fusion.",
            recommended=False,
        ),
        row(
            "C3: CCMI MinOverlap",
            "Formalizes C2 as conservative cross-modal intersection with stronger scale/clip range.",
            "Full 24-subject CV",
            "iteration_293_306_ccmi_fnirs_all6.json",
            28.7186,
            26.9226,
            30.5146,
            "Good, but slightly weaker than slope-gated CCMI.",
            recommended=False,
        ),
        row(
            "C4: CCMI HRFDelayedFNIRS",
            "Confirms fast EEG residual with delayed fNIRS residual to match hemodynamic lag.",
            "Full 24-subject CV",
            "iteration_293_306_ccmi_fnirs_all6.json",
            28.7178,
            26.9211,
            30.5146,
            "HRF lag helps, but less than prior-slope gate.",
            recommended=False,
        ),
        row(
            "C5: CCMI PriorSlopeGate",
            "Learns when signal residual is helpful by prior trajectory slope bucket.",
            "Full 24-subject CV",
            "iteration_293_306_ccmi_fnirs_all6.json",
            28.7145,
            26.9144,
            30.5146,
            "Best EEG-fNIRS fusion module.",
            recommended=True,
        ),
    ]


def output_head_rows() -> list[dict[str, object]]:
    return [
        row(
            "H1: 200 manual dimwise fusion",
            "Valence uses risk expert; arousal uses conformal median band.",
            "Full 24-subject CV",
            "iteration_221_228_bcrf_module_seed2026.json",
            28.6912,
            26.9046,
            30.4777,
            "Strong output-head baseline.",
            recommended=False,
        ),
        row(
            "H2: 218 SCRF",
            "Applies sign-consistent hierarchical residual field only on valence.",
            "Full 24-subject CV",
            "iteration_221_228_bcrf_module_seed2026.json",
            28.6869,
            26.8961,
            30.4777,
            "Best interpretable output calibration.",
            recommended=False,
        ),
        row(
            "H3: 222 BCRF on SCRF",
            "Adds Bayesian credible residual field to SCRF with confidence scaling.",
            "Full 24-subject CV",
            "iteration_221_228_bcrf_module_seed2026.json",
            28.6868,
            26.8958,
            30.4777,
            "Current global best, but tiny gain over SCRF.",
            recommended=True,
        ),
        row(
            "H4: 224 BCRF brake disagreement",
            "Uses SCRF correction only when BCRF and SCRF agree.",
            "Full 24-subject CV",
            "iteration_221_228_bcrf_module_seed2026.json",
            28.6880,
            26.8983,
            30.4777,
            "Safer than raw BCRF but not best.",
            recommended=False,
        ),
        row(
            "H5: arousal residual probe",
            "Tests whether output residual fields should modify arousal.",
            "Full 24-subject CV",
            "iteration_221_228_bcrf_module_seed2026.json",
            28.6912,
            26.9046,
            30.4778,
            "Arousal correction is not useful; keep arousal conservative.",
            recommended=False,
        ),
    ]


def combined_rows() -> list[dict[str, object]]:
    return [
        {
            "component": "Best signal-only framework",
            "choice": "D2 + C5 over 098",
            "verified_overall_mae": 28.7145,
            "meaning": "Best verified EEG-fNIRS path: 6 fNIRS types, baseline subtraction, CCMI PriorSlopeGate.",
        },
        {
            "component": "Best output-head framework",
            "choice": "H3: 222 BCRF on SCRF",
            "verified_overall_mae": 28.6868,
            "meaning": "Best verified non-signal output calibration.",
        },
        {
            "component": "Target combined framework",
            "choice": "H3 + small CCMI residual",
            "verified_overall_mae": None,
            "meaning": "Not yet verified because sample-level 222 cache is required; previous direct overlay timed out.",
        },
    ]


def row(
    name: str,
    reason: str,
    validation: str,
    source: str,
    overall: float,
    valence: float,
    arousal: float,
    conclusion: str,
    recommended: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "reason": reason,
        "validation": validation,
        "source": source,
        "overall_mae": overall,
        "valence_mae": valence,
        "arousal_mae": arousal,
        "conclusion": conclusion,
        "recommended": recommended,
    }


def render_markdown(rows: dict[str, object]) -> str:
    parts = ["# Three Module Optimization Summary", ""]
    for section in ["data_preprocessing", "ccmi_fusion", "output_head"]:
        parts.extend([f"## {section}", "", "| Candidate | Overall | Valence | Arousal | Validation | Recommended | Conclusion |", "| --- | ---: | ---: | ---: | --- | --- | --- |"])
        for item in rows[section]:
            parts.append(
                f"| {item['name']} | {item['overall_mae']:.4f} | {item['valence_mae']:.4f} | "
                f"{item['arousal_mae']:.4f} | {item['validation']} | {item['recommended']} | {item['conclusion']} |"
            )
        parts.append("")
    parts.extend(["## combined_recommendation", ""])
    for item in rows["combined_recommendation"]:
        parts.append(f"- {item['component']}: {item['choice']} -> {item['verified_overall_mae']} ({item['meaning']})")
    parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
