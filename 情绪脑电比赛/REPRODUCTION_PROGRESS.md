# MER-PS Reproduction Progress

This file tracks paper-inspired modules tested on MER-PS. These are module-level reproductions on the competition data, not full official-code reproductions of each paper.

Validation protocol:

```text
train subjects: test_1-test_20
val subjects:   test_21-test_24
metric:         raw [1,255] Overall MAE
```

## Tested Modules

| # | Paper / Direction | Reproduced Module | Best Local Result | Judgment |
| ---: | --- | --- | ---: | --- |
| 1 | Official ASAC baseline | ASAC-style graph + cross-modal attention | 47.0087 | submission interface only, weak baseline |
| 2 | Strong temporal prior | VideoTimeMean prior | 29.9574 | must keep |
| 3 | Residual learning | valence-only residual over prior | ~29.8-29.9 | useful |
| 4 | Residual initialization | zero-init residual head | stable training | must keep |
| 5 | Mamba | pure PyTorch selective scan block | usable, not best | heavier than needed |
| 6 | Mamba-lite / gated SSM | gated state-space temporal block | core single-model path | useful |
| 7 | MSGM / multiscale EEG | scale-gated `1/5/9s` features | 29.7984 | useful |
| 8 | Dynamic graph EEG | functional connectivity graph | 29.9582 | too noisy alone |
| 9 | Hybrid graph | static graph + functional graph gate | 29.8088 | useful ensemble member |
| 10 | Sparse dynamic graph | sparse top-k functional graph | unfinished/unstable | postpone |
| 11 | ModernTCN / ConvMixer | depthwise temporal conv mixer | 29.8545 | usable, worse than gated SSM |
| 12 | Cross-modal attention | ASAC-style EEG-fNIRS attention fusion | ~29.94 | worse than simple pooling |
| 13 | Domain normalization / RevIN-like | subject/trial feature normalization | 29.8622 | stabilizes, not best |
| 14 | Modality dropout | randomly drop EEG/fNIRS during training | 4-fold more robust | useful for robust submission |
| 15 | Checkpoint ensemble | residual ensemble + scale/smooth calibration | **29.7369** | current best |
| 16 | AffectGPT | LoRA-stat semantic teacher distillation | 29.8815 | useful regularizer, not ensemble-best |
| 17 | AffectGPT-only teacher | adapter-stat teacher only | 29.8829 | similar to semantic teacher |
| 18 | EEG-DisGCMAE / graph MAE | masked EEG/fNIRS graph reconstruction pretrain | 29.7995 single, 29.7477 ensemble | useful but not new best |
| 19 | PatchTST / ICLR time-series | patch token Transformer temporal block | 29.8382 | stable, worse than gated SSM |
| 20 | TimesNet / ICLR time-series | FFT period discovery + 2D temporal block | 29.8394 single, no ensemble gain | stable, not better than gated SSM |
| 21 | Multimodal reliability gate | learned EEG/fNIRS softmax modality gate | 29.9460 | too weak; collapses toward prior |
| 22 | LGGNet-style local-global graph | local-region + global graph readout for EEG/fNIRS | 29.8214 | useful single-model module |
| 23 | iTransformer / ICLR time-series | inverted variable attention over latent graph features | 29.7457 single, **29.6336 ensemble** | best new module |
| 24 | Neuro-HGLN-style graph + iTransformer | local-global readout plus iTransformer-lite | 29.8168 | useful, but combined model overfits |
| 25 | TimeMixer / ICLR time-series | trend-seasonal decomposition + multiscale temporal mixing | 29.8091 | useful fallback, no ensemble gain |
| 26 | FITS / ICLR time-series | low-frequency Fourier residual mixer | 29.8437 | too smooth for residual decoding |
| 27 | Robust iTransformer | iTransformer-lite + modality dropout 0.10 | 29.8293 | dropout weakens useful residual variance |
| 28 | New temporal ensemble | iTransformer + TimeMixer + Fourier + moddrop variants | 29.6348 | close, but not above 29.6336 |
| 29 | Arousal residual branch | iTransformer-lite trained on arousal residual | 29.9460 | arousal residual not reliable yet |

## Current Best

```text
WeightedEnsemble_moddrop010_seed123_0.72+itransformer_hybrid_159_0.28_scale8.20_smooth9
Overall MAE = 29.6336
Valence MAE = 27.8775
Arousal MAE = 31.3897
```

## Useful Modules To Keep

1. VideoTimeMean prior.
2. Valence-only residual learning.
3. Zero-init residual head.
4. Scale-gated multiscale features.
5. Gated SSM temporal block.
6. Hybrid graph as an ensemble member.
7. Modality dropout for robust variants.
8. Checkpoint residual ensemble + scale/smooth calibration.
9. Masked graph pretraining as a candidate initialization module.
10. AffectGPT teacher as a weak regularizer or confidence teacher.
11. iTransformer-lite as a separate complementary ensemble branch.
12. Local-global graph readout as a controlled single-model regularizer.
13. TimeMixer-lite as a secondary temporal fallback.

## Modules Not Worth Prioritizing Now

1. Pure functional dynamic graph.
2. More complex cross-modal attention.
3. Larger Mamba/Transformer backbones.
4. Direct AffectGPT fine-tuning inside the submission model.
5. ConvMixer as the main temporal block.
6. Plain EEG/fNIRS softmax modality gate.
7. Stacking local-global graph and iTransformer in one larger model without stronger regularization.
8. FITS/Fourier-only residual branch.
9. Modality dropout on iTransformer branch.
10. Direct arousal residual without confidence gating.

## Next Best Reproduction Targets

1. Masked temporal reconstruction: mask time spans instead of channels.
2. Adaptive residual scale/confidence head: learn when to trust physiological residuals.
3. Adaptive/moddrop branch masked pretraining: current pretraining only tested hybrid branch.
4. Domain-adversarial subject-invariant encoder if time permits.
5. Modality dropout and masked temporal pretraining for the iTransformer branch.
6. Learned per-sample residual confidence/scale head to replace global scale tuning.
7. Arousal abstention/confidence head: predict arousal residual only when the model is confident.
