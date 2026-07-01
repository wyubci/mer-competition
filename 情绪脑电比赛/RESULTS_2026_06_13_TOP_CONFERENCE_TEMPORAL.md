# 2026-06-13 Top-Conference Temporal Module Reproduction

This run tested two top-conference time-series ideas on MER-PS:

- PatchTST-style patch token Transformer.
- TimesNet-style FFT period discovery + 2D temporal variation convolution.

These are module-level reproductions inside our existing MER-PS graph residual framework, not full official-code reproductions.

Validation protocol:

```text
train subjects: test_1-test_20
val subjects:   test_21-test_24
metric:         raw [1,255] Overall MAE
```

## Implemented Modules

| Module | Paper Idea | Local Implementation |
| --- | --- | --- |
| `patch_tst` | PatchTST: patch tokenization + Transformer over patches | `PatchTSTLiteBlock` over latent graph sequence |
| `timesnet` | TimesNet: FFT dominant periods + 2D temporal variation | `TimesNetLiteBlock` over latent graph sequence |

Code:

```text
emotion_merps/graph_mamba.py
tools/train_graph_mamba.py
```

## Results

| Method | Raw Overall MAE | Best Postprocessed Overall MAE | Valence MAE | Arousal MAE | Result File |
| --- | ---: | ---: | ---: | ---: | --- |
| gated SSM + hybrid graph baseline | 29.8735 | 29.8088 | 28.2284 | 31.3892 | `iteration_019_hybrid_functional_graph.json` |
| PatchTST-lite + hybrid graph | 29.8840 | 29.8382 | 28.2872 | 31.3892 | `iteration_034_patchtst_hybrid_159.json` |
| TimesNet-lite + hybrid graph | 29.8739 | 29.8394 | 28.2897 | 31.3892 | `iteration_035_timesnet_hybrid_159.json` |
| top2 + TimesNet ensemble | 29.7369 | 29.7369 | 28.0866 | 31.3872 | `iteration_036_timesnet_top2_ensemble.json` |

Current best remains:

```text
Ensemble_moddrop010_seed123+hybrid_functional_graph_scale6.00_smooth7
Overall MAE = 29.7369
```

## Judgment

PatchTST-lite and TimesNet-lite are both runnable and stable, but neither beats the gated SSM residual head.

What worked:

- TimesNet-lite raw MAE is close to hybrid gated SSM raw MAE.
- Patch/period modules provide alternative residual shapes and can be kept as ablation evidence.

What did not work:

- After residual scale/smoothing, both are worse than gated SSM.
- Adding TimesNet to the current top2 ensemble does not improve the best score.

Interpretation:

MER-PS public data has only 24 subjects and 360 trials. Patch attention and FFT-period 2D convolution add temporal freedom that is useful in large time-series benchmarks, but here the strongest signal still comes from:

```text
VideoTimeMean prior
+ small valence residual
+ gated SSM smoothing
+ conservative calibration
```

## Reproduction Commands

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 5 --target-mode valence --input-modality both --fusion-mode pool --graph-encoder hybrid_functional --temporal-block patch_tst --scale-fusion gated --multiscale-windows 1,5,9 --graph-hidden 32 --d-model 128 --mamba-layers 2 --batch-size 8 --lr 0.001 --dropout 0.15 --output experiments/results/iteration_034_patchtst_hybrid_159.json --checkpoint experiments/checkpoints/graph_mamba/patchtst_hybrid_159.pt --seed 42
```

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 5 --target-mode valence --input-modality both --fusion-mode pool --graph-encoder hybrid_functional --temporal-block timesnet --scale-fusion gated --multiscale-windows 1,5,9 --graph-hidden 32 --d-model 128 --mamba-layers 2 --batch-size 8 --lr 0.001 --dropout 0.15 --output experiments/results/iteration_035_timesnet_hybrid_159.json --checkpoint experiments/checkpoints/graph_mamba/timesnet_hybrid_159.pt --seed 42
```
