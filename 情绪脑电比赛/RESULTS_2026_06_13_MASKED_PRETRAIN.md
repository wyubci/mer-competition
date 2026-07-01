# 2026-06-13 Masked Graph Pretraining Reproduction

## Goal

Reproduce a masked graph modeling module inspired by EEG-DisGCMAE / graph masked autoencoder work, then test whether it improves MER-PS valence residual prediction.

The validation protocol is unchanged:

```text
train subjects: test_1-test_20
val subjects:   test_21-test_24
metric:         raw [1,255] Overall MAE
```

## Implemented Module

Added masked EEG/fNIRS graph reconstruction:

```text
EEG/fNIRS multiscale features
  -> randomly mask channels
  -> graph encoder
  -> reconstruct masked channel features
  -> save graph encoder + scale gate
  -> fine-tune Graph-Mamba residual model
```

Code:

| File | Purpose |
| --- | --- |
| `tools/pretrain_masked_graph.py` | masked graph reconstruction pretraining |
| `tools/train_graph_mamba.py` | added `--pretrained-graph` loading |

Pretraining config:

```text
graph_encoder = hybrid_functional
multiscale_windows = 1,5,9
scale_fusion = gated
mask_ratio = 0.35
epochs = 5
```

## Pretraining Result

The reconstruction objective learned normally:

| Epoch | Train Loss | Val Loss | Val EEG Loss | Val fNIRS Loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0778 | 1.4526 | 0.4168 | 1.0358 |
| 3 | 0.9056 | 1.2699 | 0.3127 | 0.9571 |
| 5 | 0.7632 | 1.0506 | 0.2196 | 0.8310 |

Checkpoint:

```text
experiments/checkpoints/pretrain/masked_graph_hybrid_159.pt
```

## Fine-Tuning Result

| Method | Overall MAE | Valence MAE | Arousal MAE | Result File |
| --- | ---: | ---: | ---: | --- |
| Hybrid functional graph, no pretraining | 29.8088 | 28.2284 | 31.3892 | `iteration_019_hybrid_functional_graph.json` |
| Masked pretrain + hybrid fine-tune | **29.7995** | **28.2097** | 31.3892 | `iteration_032_masked_pretrain_hybrid_finetune.json` |
| Current best top2 ensemble | **29.7369** | **28.0866** | **31.3872** | `iteration_025_ensemble_top2_scale_expand.json` |
| Moddrop010 + masked-pretrain hybrid ensemble | 29.7477 | 28.1082 | 31.3872 | `iteration_033_masked_pretrain_hybrid_top2_ensemble.json` |

## Judgment

Masked graph pretraining is useful, but not yet a new best.

Useful part:

- It improves the hybrid single model from 29.8088 to 29.7995.
- It confirms that self-supervised physiological reconstruction can help the encoder.

Not useful enough yet:

- It does not beat the current top2 ensemble.
- When replacing the original hybrid checkpoint in the ensemble, best MAE becomes 29.7477, slightly worse than 29.7369.

Next experiments worth doing:

1. Pretrain the adaptive/moddrop branch as well, then ensemble two pretrained branches.
2. Change the self-supervised task from channel reconstruction to temporal masked reconstruction.
3. Use the pretrained encoder only as initialization for the first epoch, then apply stronger regularization or lower learning rate.
4. Combine masked pretraining with AffectGPT teacher confidence, not direct hidden-state distillation.

## Reproduction Commands

```bash
F:/anaconda/envs/eegpt/python.exe tools/pretrain_masked_graph.py --epochs 5 --batch-size 8 --graph-encoder hybrid_functional --multiscale-windows 1,5,9 --scale-fusion gated --graph-hidden 32 --d-model 128 --cheb-order 2 --temporal-block gated_ssm --dropout 0.15 --mask-ratio 0.35 --output experiments/checkpoints/pretrain/masked_graph_hybrid_159.pt --summary-output experiments/results/iteration_031_masked_graph_hybrid_pretrain.json --seed 42
```

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 5 --target-mode valence --input-modality both --fusion-mode pool --graph-encoder hybrid_functional --temporal-block gated_ssm --scale-fusion gated --multiscale-windows 1,5,9 --graph-hidden 32 --d-model 128 --mamba-layers 2 --batch-size 8 --lr 0.001 --dropout 0.15 --pretrained-graph experiments/checkpoints/pretrain/masked_graph_hybrid_159.pt --output experiments/results/iteration_032_masked_pretrain_hybrid_finetune.json --checkpoint experiments/checkpoints/graph_mamba/masked_pretrain_hybrid_159.pt --seed 42
```
