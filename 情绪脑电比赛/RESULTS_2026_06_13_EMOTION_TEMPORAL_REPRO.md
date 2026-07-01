# MER-PS Emotion / Time-Series Module Reproduction

Date: 2026-06-13

This report records module-level reproductions of recent EEG emotion-recognition and time-series papers on the local MER-PS validation split. These are not full official-code reproductions; each paper idea was adapted to the existing residual-regression pipeline so the comparison stays fair.

## Protocol

```text
train subjects: test_1-test_20
val subjects:   test_21-test_24
train trials:   300
val trials:     60
train samples:  30720
val samples:    6144
metric:         Overall MAE on raw [1,255] valence/arousal labels
prior:          VideoTimeMean
main target:    valence residual; arousal is still prior/smoothed prior
```

## Paper Ideas Tested

| Direction | Paper / Source | Adapted Module | Result | Judgment |
| --- | --- | --- | ---: | --- |
| Local-global EEG graph | LGGNet, IEEE TNNLS / arXiv 2105.02786 | EEG/fNIRS local-region pooling + global summary readout | 29.8214 | Useful single-model module, but not ensemble best |
| Hierarchical graph + Transformer | Neuro-HGLN, arXiv 2601.10525 | local-global readout + iTransformer-lite temporal encoder | 29.8168 | Better than local-global alone, but overfits more than pure iTransformer |
| Inverted variable attention | iTransformer, ICLR 2024 spotlight | variable-as-token attention over latent graph features | 29.7457 single, 29.6336 ensemble | Most useful new module |
| Patch time-series Transformer | PatchTST, ICLR 2023 | patch temporal Transformer over latent sequence | 29.8382 | Stable, not best |
| 2D temporal variation | TimesNet, ICLR 2023 | FFT period discovery + lightweight 2D temporal block | 29.8394 | Stable, no ensemble gain |
| Multimodal reliability gate | multimodal fusion family | EEG/fNIRS learned softmax gate | 29.9460 | Too weak; collapses toward prior |

References:

- LGGNet: https://arxiv.org/abs/2105.02786
- Neuro-HGLN: https://arxiv.org/abs/2601.10525
- iTransformer: https://openreview.net/forum?id=JePfAI8fah
- PatchTST: https://openreview.net/forum?id=Jbdc0vTOcol
- TimesNet: https://openreview.net/forum?id=ju_Uqw384Oq

## New Experiments

| Iteration | Model | Checkpoint | Best Local Result |
| --- | --- | --- | ---: |
| 037 | `modal_gate + hybrid_functional + gated_ssm` | `experiments/checkpoints/graph_mamba/modal_gate_hybrid_159.pt` | 29.9460 |
| 038 | `local_global + hybrid_functional + gated_ssm` | `experiments/checkpoints/graph_mamba/local_global_hybrid_159.pt` | 29.8214 |
| 039 | `pool + hybrid_functional + itransformer` | `experiments/checkpoints/graph_mamba/itransformer_hybrid_159.pt` | 29.7457 |
| 040 | old top checkpoints + iTransformer/local-global ensemble | `experiments/results/iteration_040_emotion_temporal_ensemble.json` | 29.6348 |
| 041 | refined `moddrop010_seed123 + itransformer` ensemble | `experiments/results/iteration_041_itransformer_refined_ensemble.json` | **29.6336** |
| 042 | `local_global + hybrid_functional + itransformer` | `experiments/checkpoints/graph_mamba/local_global_itransformer_159.pt` | 29.8168 |

## Current Best

```text
WeightedEnsemble_moddrop010_seed123_0.72+itransformer_hybrid_159_0.28_scale8.20_smooth9
Overall MAE = 29.6336
Valence MAE = 27.8775
Arousal MAE = 31.3897
Overall MSE = 1493.1781
```

This improves the previous best:

```text
Ensemble_moddrop010_seed123+hybrid_functional_graph_scale6.00_smooth7
Overall MAE = 29.7369
```

Absolute gain: `0.1033` Overall MAE.

## What Worked

1. `iTransformer-lite` is the strongest new module. It overfits quickly, but the early-stopped checkpoint has residual directions that are highly complementary to the modality-dropout branch.
2. Weighted residual ensembles matter more than bigger single models. The best mix is about `72% moddrop010_seed123 + 28% iTransformer`.
3. Smoothing remains important. Best iTransformer ensemble uses `smooth9`; old best used `smooth7`.
4. Local-global graph readout is useful as a single-model regularizer, but combining it with iTransformer inside one model did not beat using iTransformer as a separate ensemble branch.

## What Did Not Work Well

1. A plain EEG/fNIRS softmax modality gate underfits and collapses toward the VideoTimeMean prior.
2. PatchTST and TimesNet are robust but not competitive with the gated SSM or iTransformer branches here.
3. Stacking local-global graph and iTransformer in the same model adds capacity and overfits by epoch 5.

## Next Iteration

1. Train `itransformer_hybrid_159` with stronger early stopping seeds because epoch 1 was already best.
2. Do not prioritize modality dropout on the iTransformer branch; round 2 showed it weakens the complementary residual.
3. Add a learned confidence/residual-scale head so scale `8.20` is learned per sample instead of tuned globally.
4. Test temporal masked pretraining for iTransformer, not only node masked graph reconstruction.

Round 2 follow-up: `RESULTS_2026_06_13_NEW_MODELS_ROUND2.md`.
