# Three Module Optimization Summary

## data_preprocessing

| Candidate | Overall | Valence | Arousal | Validation | Recommended | Conclusion |
| --- | ---: | ---: | ---: | --- | --- | --- |
| D1: 3 fNIRS types + baseline mean subtraction | 28.7176 | 26.9206 | 30.5146 | Full 24-subject CV | False | Strong reference for CCMI. |
| D2: 6 fNIRS types + baseline mean subtraction | 28.7145 | 26.9144 | 30.5146 | Full 24-subject CV | True | Best data preprocessing choice. |
| D3: 6 fNIRS types + no baseline subtraction | 28.7462 | 26.9738 | 30.5186 | Full 24-subject CV | False | Does not generalize; falls back to 098. |
| D4: 6 fNIRS types + trial z-score | 28.3478 | 25.5020 | 31.1936 | 4-subject smoke | False | Bad smoke result; do not run full CV. |
| D5: 6 fNIRS types + subject z-score | 28.3259 | 25.4582 | 31.1936 | 4-subject smoke | False | Bad smoke result; subject amplitude has useful signal. |

## ccmi_fusion

| Candidate | Overall | Valence | Arousal | Validation | Recommended | Conclusion |
| --- | ---: | ---: | ---: | --- | --- | --- |
| C1: OOF agreement weighted | 28.7352 | 26.9518 | 30.5186 | Full 24-subject CV | False | First stable multimodal residual improvement. |
| C2: MinMagnitudeAgreement | 28.7297 | 26.9408 | 30.5186 | Full 24-subject CV | False | Simple intersection beats attention-style fusion. |
| C3: CCMI MinOverlap | 28.7186 | 26.9226 | 30.5146 | Full 24-subject CV | False | Good, but slightly weaker than slope-gated CCMI. |
| C4: CCMI HRFDelayedFNIRS | 28.7178 | 26.9211 | 30.5146 | Full 24-subject CV | False | HRF lag helps, but less than prior-slope gate. |
| C5: CCMI PriorSlopeGate | 28.7145 | 26.9144 | 30.5146 | Full 24-subject CV | True | Best EEG-fNIRS fusion module. |

## output_head

| Candidate | Overall | Valence | Arousal | Validation | Recommended | Conclusion |
| --- | ---: | ---: | ---: | --- | --- | --- |
| H1: 200 manual dimwise fusion | 28.6912 | 26.9046 | 30.4777 | Full 24-subject CV | False | Strong output-head baseline. |
| H2: 218 SCRF | 28.6869 | 26.8961 | 30.4777 | Full 24-subject CV | False | Best interpretable output calibration. |
| H3: 222 BCRF on SCRF | 28.6868 | 26.8958 | 30.4777 | Full 24-subject CV | True | Current global best, but tiny gain over SCRF. |
| H4: 224 BCRF brake disagreement | 28.6880 | 26.8983 | 30.4777 | Full 24-subject CV | False | Safer than raw BCRF but not best. |
| H5: arousal residual probe | 28.6912 | 26.9046 | 30.4778 | Full 24-subject CV | False | Arousal correction is not useful; keep arousal conservative. |

## combined_recommendation

- Best signal-only framework: D2 + C5 over 098 -> 28.7145 (Best verified EEG-fNIRS path: 6 fNIRS types, baseline subtraction, CCMI PriorSlopeGate.)
- Best output-head framework: H3: 222 BCRF on SCRF -> 28.6868 (Best verified non-signal output calibration.)
- Target combined framework: H3 + small CCMI residual -> None (Not yet verified because sample-level 222 cache is required; previous direct overlay timed out.)
