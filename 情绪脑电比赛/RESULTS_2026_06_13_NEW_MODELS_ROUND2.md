# MER-PS New Model Round 2

Date: 2026-06-13

This round tested additional lightweight recent time-series modules and an arousal-specific residual branch. The motivation was to avoid simply making the model larger: MER-PS has only 24 public subjects, so compact modules with strong inductive bias are more realistic than large foundation backbones.

## Protocol

```text
train subjects: test_1-test_20
val subjects:   test_21-test_24
train trials:   300
val trials:     60
train samples:  30720
val samples:    6144
metric:         Overall MAE on raw [1,255] labels
prior:          VideoTimeMean
```

## Tested Ideas

| Iteration | Idea | Adapted Module | Best Result | Judgment |
| --- | --- | --- | ---: | --- |
| 043 | TimeMixer, ICLR 2024 | trend/seasonal decomposition + multiscale temporal mixing | 29.8091 | Useful single model, no ensemble gain |
| 044 | FITS, ICLR 2024 spotlight | low-frequency FFT residual mixer | 29.8437 | Too smooth for residual decoding |
| 045 | Robust iTransformer | iTransformer-lite + modality dropout 0.10 | 29.8293 | Dropout weakens the useful aggressive residual |
| 046 | New temporal ensemble | old best + iTransformer + TimeMixer + Fourier + moddrop iTransformer | 29.6348 | No gain over refined 29.6336 |
| 047 | Arousal branch | iTransformer-lite trained on arousal residual | 29.9460 | Arousal residual is not reliable yet |

References:

- TimeMixer: https://openreview.net/forum?id=7oLshfEIC2
- FITS: https://openreview.net/forum?id=bWcnvZ3qMb

## Current Best Still Holds

```text
WeightedEnsemble_moddrop010_seed123_0.72+itransformer_hybrid_159_0.28_scale8.20_smooth9
Overall MAE = 29.6336
Valence MAE = 27.8775
Arousal MAE = 31.3897
```

The broader ensemble in iteration 046 reached `29.6348`, close but slightly worse.

## Findings

1. TimeMixer-lite is a good fallback temporal block. It beats PatchTST and TimesNet in this setup, but it is not as complementary as iTransformer.
2. FITS/Fourier-lite is too conservative for valence residuals. Keeping low frequencies is sensible for emotion trajectories, but the useful residual signal also needs sharper local deviations.
3. Modality dropout helps gated-SSM robustness but hurts iTransformer. The iTransformer branch works because it is a high-variance complementary residual; making it conservative removes that value.
4. Arousal remains the hardest dimension. The arousal-specific iTransformer branch made arousal worse than simply keeping the VideoTimeMean/smoothed prior.

## Next Direction

The next useful innovation should not be another larger temporal block. Better targets:

1. confidence-calibrated residual scaling, learned per sample or per trial;
2. separate valence and arousal treatment, with arousal allowed to abstain when confidence is low;
3. masked temporal pretraining for iTransformer, because its best epoch arrives very early;
4. subject-invariant adversarial/domain normalization to reduce validation-subject mismatch.
