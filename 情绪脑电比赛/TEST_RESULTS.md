# MER-PS 实验测试与结果

更新时间：2026-06-10

本文件记录当前项目在公开训练验证集上的首轮算法迭代。MER-PS 是连续价-唤醒回归任务，所以这里的“准确率”用官方主指标替代：`overall MAE`，即 valence 与 arousal 在原始 `[1, 255]` 尺度上的平均绝对误差。分数越低越好。

## 数据与切分

本轮使用本地数据：

```text
data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval
```

验证范式严格采用 subject-disjoint：

| 集合 | 被试 | 样本数 |
| --- | --- | ---: |
| Train | `test_1` - `test_20` | 30,720 |
| Validation | `test_21` - `test_24` | 6,144 |

说明：

- 每个样本对应一个 1 Hz annotation sample。
- 本地检查发现 `test_1` 的 EEG 视频信号为 `label_time * 1000`，即当前本地文件保留 1000 Hz EEG；fNIRS 为 47.62 Hz。
- 所有结果均在隐藏测试集不可见的前提下，仅使用公开 train/validation 数据。

## 特征缓存

信号模型使用官方 starting kit 的 ASAC 特征抽取逻辑，并缓存到：

```text
experiments/features/asac_features_20_4.npz
```

特征摘要：

| 项目 | 数值 |
| --- | ---: |
| 总样本 | 36,864 |
| 特征维度 | 779 |
| EEG 特征 | `36864 x 64 x 5` |
| fNIRS 特征 | `36864 x 51 x 9` |

EEG 特征为每秒窗口的 5 个频带 bandpower。fNIRS 使用 HbO/HbR/HbT 三类信号，每类提取 mean/std/slope。

## 已测试方法

| 方法 | 说明 |
| --- | --- |
| `Center128` | 恒定预测 `(128,128)` |
| `TrainMean` | 训练被试全局均值 |
| `VideoTimeMean` | 按视频编号和秒级 timestamp 统计训练被试平均情绪轨迹 |
| `RidgeSignal` | EEG/fNIRS ASAC 特征展平后做多输出 Ridge 回归 |
| `ResidualRidge` | `VideoTimeMean` 作为先验，Ridge 预测 EEG/fNIRS 个体残差 |
| `MLPFeatureRaw` | 小型 MLP 直接从 EEG/fNIRS 特征预测 VA |
| `MLPFeatureResidual` | 小型 MLP 预测 `VideoTimeMean` 残差 |
| `MLPFeatureResidual_valence_only` | 只用 MLP residual 修正 valence，arousal 保持 `VideoTimeMean` |
| `GraphMambaResidual_gated_ssm` | EEG/fNIRS 图编码器 + 轻量 gated SSM 时序残差 |
| `GraphMambaResidual_full_mamba` | EEG/fNIRS 图编码器 + 纯 PyTorch selective-scan Mamba 时序残差 |

## 结果

## 重要纠错：结果必须分两类看

上一版表格把 `VideoTimeMean`、`MLPFeatureResidual`、`GraphMambaResidual` 和官方 ASAC demo 放在同一张表里，容易造成误解。这里必须明确：

- **官方 ASAC demo 是信号模型**：输入 EEG/fNIRS 特征，预测 VA。
- **`VideoTimeMean` 是标签先验模型**：使用训练被试的 `video_id + timestamp` 平均标签轨迹，不使用 EEG/fNIRS。
- **`MLPFeatureResidual` / `GraphMambaResidual` 是先验 + 信号残差模型**：先用 `VideoTimeMean` 给出主轨迹，再用 EEG/fNIRS 特征修正 residual。

所以 `29.x MAE` 主要来自视频时间先验，不应解释成纯 EEG/fNIRS 模型已经远超官方 demo。公平比较官方 demo 时，应看“信号模型”表。

### 信号模型结果

这些方法不使用 `VideoTimeMean` 标签轨迹先验。

| 方法 | 参数量 | Overall MAE | Valence MAE | Arousal MAE | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Center128` | 0 | 46.5051 | 47.9976 | 45.0127 | 无信号，常数中心 |
| `Official ASAC demo best_model.pt` | 21,783 | 47.0087 | 49.2285 | 44.7890 | 官方提交 demo checkpoint |
| `MLPFeatureRaw_best_smooth9` | 267,544 | 47.2992 | 48.8159 | 45.7824 | EEG/fNIRS ASAC 特征 + MLP |
| `RidgeSignal_a1000_smooth9` | 约 1,560 | 49.3743 | 53.2800 | 45.4686 | EEG/fNIRS ASAC 特征 + Ridge |

结论：目前纯信号模型没有明显超过 `Center128`，官方 demo 也只是接口 baseline，不是强成绩 baseline。

### 先验与残差模型结果

这些方法使用 `VideoTimeMean` 作为主预测轨迹，因此不能和官方纯信号 demo 直接比较。

| 排名 | 方法 | Overall MAE | Valence MAE | Arousal MAE | Overall MSE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `GraphMambaResidual_scalegated_MSGM_1-5-9_scale2.75_smooth5` | **29.7984** | **28.2076** | 31.3892 | 1508.4087 |
| 2 | `GraphMambaResidual_scalegated_MSGM_1-3-5_scale2.50_smooth5` | 29.8007 | 28.2121 | 31.3892 | 1509.7169 |
| 3 | `GraphMambaResidual_scalegated_MSGM_1-3-5-9_scale2.50_smooth5` | 29.8048 | 28.2204 | 31.3892 | 1509.2465 |
| 4 | `GraphMambaResidual_valence_scale2.25_smooth5` | 29.8214 | 28.2536 | 31.3892 | 1511.8842 |
| 5 | `GraphMambaResidual_valence_scale2.50_smooth5` | 29.8226 | 28.2560 | 31.3892 | 1511.2919 |
| 6 | `GraphMambaResidual_valence_scale2.00_smooth5` | 29.8231 | 28.2571 | 31.3892 | 1512.6674 |
| 7 | `GraphMambaResidual_targetscale2.25_smooth5` | 29.8308 | 28.2724 | 31.3892 | 1512.4111 |
| 8 | `GraphMambaResidual_valence_smooth5_zero_init` | 29.8617 | 28.3342 | 31.3892 | 1517.7076 |
| 9 | `GraphMambaResidual_trial_norm` | 29.8622 | 28.3353 | 31.3892 | 1515.2206 |
| 10 | `GraphMambaResidual_subject_norm` | 29.8627 | 28.3362 | 31.3892 | 1514.9961 |
| 11 | `GraphMambaResidual_MSGMlite_concat_1-3-5` | 29.8791 | 28.3690 | 31.3892 | 1516.8676 |
| 12 | `MLPFeatureResidual_valence_only_smooth9` | 29.9217 | 28.4327 | 31.4107 | 1516.2280 |
| 13 | `GraphMambaResidual_full_mamba_small_smooth5` | 29.9223 | 28.4555 | 31.3892 | 1522.0717 |
| 14 | `MLPFeatureResidual_valence_only_smooth5` | 29.9382 | 28.4658 | 31.4107 | 1519.1461 |
| 15 | `VideoTimeMean_smooth5` | 29.9461 | 28.5030 | 31.3892 | 1525.8004 |
| 16 | `VideoTimeMean` | 29.9574 | 28.5041 | 31.4107 | 1529.1310 |
| 17 | `MLPFeatureResidual_valence_only_smooth3` | 29.9606 | 28.5106 | 31.4107 | 1521.6929 |
| 18 | `MLPFeatureResidual_best_smooth9` | 30.3101 | 28.4327 | 32.1875 | 1563.0615 |
| 19 | `Blend_RidgeSignal_a1000_VideoTimeMean_0.25` | 32.2043 | 31.4202 | 32.9885 | 1651.4058 |
| 20 | `ResidualRidge_a1000_smooth9` | 33.7939 | 29.8060 | 37.7819 | 1852.9335 |
| 21 | `TrainMean` | 48.6331 | 50.7569 | 46.5094 | 3791.3252 |

完整 JSON 结果：

```text
experiments/results/iteration_001.json
experiments/results/iteration_001_mlp_dim60.json
experiments/results/iteration_002_graph_mamba_valence_zeroinit.json
experiments/results/smoke_graph_mamba_full.json
experiments/results/iteration_003_full_mamba_mid.json
experiments/results/iteration_003_full_mamba_small_lr.json
experiments/results/iteration_004_gated_ssm_scale_extended2.json
experiments/results/iteration_005_msgm_lite_135.json
experiments/results/iteration_005_msgm_lite_159.json
experiments/results/iteration_006_gated_ssm_finescale_seed42.json
experiments/results/iteration_007_subject_norm.json
experiments/results/iteration_007_trial_norm.json
experiments/results/iteration_008_signed_graph.json
experiments/results/iteration_009_target_scale225.json
experiments/results/iteration_009_target_scale175.json
experiments/results/iteration_010_scalegated_msgm_135.json
experiments/results/iteration_010_scalegated_msgm_159.json
experiments/results/iteration_010_scalegated_msgm_1359.json
experiments/results/official_demo_eval.json
```

### 论文启发复现实验

| 方向 | 本地复现方式 | 最好结果 | 结论 |
| --- | --- | ---: | --- |
| Mamba selective scan | `MambaSelectiveScanBlock` 纯 PyTorch 完整块 | 29.9223 | 能跑，但小数据上不如更强约束的 gated SSM |
| MSGM/miMamba 多尺度 concat | `--multiscale-windows 1,3,5` 与 `1,5,9` | 29.8791 | 简单 concat 多尺度会放大噪声 |
| MSGM/miMamba 多尺度 gate | `--scale-fusion gated --multiscale-windows 1,5,9` | **29.7984** | scale gate 有效，是当前最优创新点 |
| 跨被试域归一化 | `--feature-norm subject/trial` | 29.8622 | 有一点稳定化，但未超过单尺度最优 |
| balanced signed graph | `--graph-encoder signed` 正/负邻接 | 未产生有效残差收益 | 自由度太高，当前公开集下更容易学噪声 |
| residual target scaling | `--residual-target-scale 1.75/2.25` | 29.8308 | 训练期放大不如验证后 residual scale 稳定 |
| residual scale sweep | scale `2.00/2.25/2.50/2.75` | **29.8214** | 当前最有效，说明模型学到了方向但幅度偏保守 |

## 数据量与参数量分析

公开集规模：

| 项目 | 数值 |
| --- | ---: |
| 总被试 | 24 |
| 训练被试 | 20 |
| 验证被试 | 4 |
| 总 trial | 360 |
| 训练 trial | 300 |
| 验证 trial | 60 |
| 总秒级样本 | 36,864 |
| 训练秒级样本 | 30,720 |
| 验证秒级样本 | 6,144 |
| 每个被试样本数 | 1,536 |
| trial 长度范围 | 60 - 170 秒 |
| 平均 trial 长度 | 102.4 秒 |

注意：30,720 个训练样本不是独立样本。同一 subject、同一 video 内标签和生理信号高度自相关，因此更保守的有效样本单位应看作 20 个训练被试或 300 个训练 trial。这个数据规模不支持从零训练高参数 Mamba/Transformer/foundation model。

当前模型参数量：

| 模型 | 参数量 | 观察 |
| --- | ---: | --- |
| Official ASAC demo, hidden=16 | 21,783 | 能跑通提交，但效果接近中心预测 |
| Graph-Mamba gated SSM small, `g16,d64,l1` | 58,631 | 更适合作为下一步低风险版本 |
| ASAC, hidden=64 | 201,735 | 参数适中，可作为强学生模型 |
| MLP 779->256 | 267,544 | 很快过拟合，epoch 1 最好 |
| Graph-Mamba full Mamba, `g32,d128,l2` | 275,143 | 纯 PyTorch selective scan，可运行但本轮效果不如 gated SSM |
| Graph-Mamba gated SSM, `g32,d128,l2` | 376,007 | zero-init 后有效，但仍在前几 epoch 过拟合 |

建议参数预算：

- 第一阶段可提交模型：`20k - 100k` trainable parameters。
- 第二阶段 Graph-Mamba 残差模型：优先控制在 `50k - 200k`。
- 若使用 AffectGPT/EEG 大模型/fNIRS 大模型，只建议冻结教师，训练 adapter/gate/head，trainable parameters 尽量 `< 300k`。
- 不建议在 20 个训练被试上从零训练百万级以上模型；除非做强预训练、冻结主体、LoRA/adapter 或多折验证确认稳定。

## 结论

1. `VideoTimeMean` 是非常强的公开集 baseline。  
   仅用训练被试的同视频同秒平均轨迹，overall MAE 就达到 29.9574，显著强于 `Center128` 的 46.5051。

2. 直接信号回归目前不可靠。  
   `RidgeSignal` 和 `MLPFeatureRaw` 都没有打过视频轨迹先验，说明当前手工 EEG/fNIRS 特征跨被试泛化弱，直接从生理信号预测完整 VA 轨迹容易过拟合或学到偏移。

3. 残差建模是正确方向。  
   `MLPFeatureResidual` 对 valence 有帮助，但会伤害 arousal。因此当前最佳做法是：arousal 使用 `VideoTimeMean`，valence 用图时序残差修正。

4. Graph-Mamba + 图结构有增益，但必须小心初始化和参数量。  
   默认 Graph-Mamba 随机 residual head 时 epoch 1 后迅速过拟合；zero-init residual head 后，初始模型等价于 `VideoTimeMean`，再学习小残差。当前最佳是 scale-gated MSGM-lite + gated SSM，把 valence residual 放大到 `2.75` 后再做 `smooth5`，overall MAE 到 29.7984。

5. 官方 demo 是接口 baseline，不是强成绩 baseline。  
   官方 ASAC demo checkpoint 在本地 `test_21-test_24` 上 overall MAE 47.0087，略差于 `Center128` 的 46.5051，主要说明提交格式和模型加载流程正确。它没有使用 `VideoTimeMean` 这类标签轨迹先验。

6. 当前提升仍然不大，但方向更清楚。  
   最佳 Graph-Mamba 方法比 `VideoTimeMean` 提升约 0.1590 MAE，valence 从 28.5041 降到 28.2076，arousal 因 smoothing 从 31.4107 降到 31.3892。这个提升还需要多折验证确认，但说明“图结构 + 多尺度 gate + 快速时序模型 + 强先验残差”比逐秒 MLP 更合理。

7. 完整 Mamba 已实现并测试，但当前不是最优。  
   `mamba-ssm` 官方包在本机 Windows + CUDA 环境没有可用的 `causal-conv1d` 二进制包，因此项目中实现了一个纯 PyTorch `MambaSelectiveScanBlock`，包含 input projection、causal depthwise conv、输入依赖的 `dt/B/C`、对角状态矩阵 `A`、skip `D` 和 selective scan。它能正常训练，但本轮最佳只有 29.9223，弱于轻量 gated SSM 的 29.8226。

8. 论文思路不能机械照搬。  
   这轮复现了 MSGM-style 多尺度窗口、balanced signed graph、subject/trial 无监督归一化、训练期 residual target scaling。多尺度 concat 不好，但改成 scale gate 后有效；signed graph 和 subject norm 没有超过当前最优。原因大概率是 MER-PS 公开集有效样本太少，复杂自由度会先放大噪声。

## 问题分析

- 验证集被试只有 4 人，单一 20/4 切分可能对模型选择不稳定。
- `VideoTimeMean` 使用了视频编号和时间位置，隐含强刺激先验；如果隐藏测试视频结构相同，这个先验可能非常重要。
- EEG/fNIRS 特征仍偏浅：EEG 只用 bandpower，fNIRS 只用 mean/std/slope，没有建模更长上下文、跨通道拓扑和个体基线差异。
- MLP residual 在 epoch 1 最好，之后验证变差，说明小模型也很快过拟合。
- Arousal 维度目前不适合让信号残差修正，至少在本轮特征和模型下会变差。

## 下一步迭代

1. 做 5-fold subject-disjoint 验证  
   当前只用了 `test_21` - `test_24` 验证。下一步应做 grouped k-fold，确认 `MLPFeatureResidual_valence_only` 的微小收益是否稳定。

2. 建立强提交 baseline  
   提交模型先采用 `VideoTimeMean` + 可选 valence residual 的形式，写成符合 Codabench `model.py` 接口的轻量模型。

3. 改造信号模型为时序残差模型  
   不再逐秒独立预测，而是使用 Graph-Mamba/TCN/GRU 对每个 trial 的连续序列建模，输出 residual trajectory。完整 Mamba 已能运行，下一步优先做多折验证和更强正则，而不是继续盲目增大模型。

4. 分维度建模  
   当前 valence 和 arousal 行为不同。后续应分别建模、分别选择是否使用生理残差，并分别调 smoothing。

5. 引入多教师蒸馏  
   使用已下载的 AffectGPT checkpoint 生成情绪语义先验；同时把 EEG/fNIRS 编码器作为生理 teacher，训练一个学生模型预测相对 `VideoTimeMean` 的个体残差。

6. 更强特征  
   增加 EEG 时频图、DE/PSD 多频带统计、通道区域聚合、fNIRS HbO/HbR 滞后窗口、trial-level baseline 差异和前后文窗口。

## 复现实验命令

标签先验：

```bash
F:/anaconda/envs/eegpt/python.exe tools/run_iteration_experiments.py --skip-signal --output experiments/results/iteration_001_label_only.json
```

信号 Ridge 与 residual Ridge：

```bash
F:/anaconda/envs/eegpt/python.exe tools/run_iteration_experiments.py --output experiments/results/iteration_001.json
```

MLP residual：

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_feature_mlp.py --epochs 60 --output experiments/results/iteration_001_mlp_dim60.json
```

Graph-Mamba gated SSM 残差：

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 1 --target-mode valence --temporal-block gated_ssm --graph-hidden 32 --d-model 128 --mamba-layers 2 --batch-size 8 --lr 0.001 --dropout 0.15 --output experiments/results/iteration_004_gated_ssm_scale_extended2.json --checkpoint experiments/checkpoints/graph_mamba/gated_ssm_scale_extended2.pt
```

Graph-Mamba full Mamba 残差：

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 1 --target-mode valence --temporal-block mamba --graph-hidden 16 --d-model 64 --mamba-layers 1 --batch-size 8 --lr 0.001 --dropout 0.15 --output experiments/results/smoke_graph_mamba_full.json --checkpoint experiments/checkpoints/graph_mamba/full_mamba_smoke.pt
```

## 2026-06-11 迭代补充：模态消融、稳健性和后处理

本轮继续沿用严格 subject-disjoint 范式。单折验证仍为 `test_1-test_20` 训练、`test_21-test_24` 验证；另外补充 4 折验证，每折 6 个 subject 作为验证集。

### 已确认有用的部分

| 组件 | 结论 |
| --- | --- |
| `VideoTimeMean` 标签轨迹先验 | 仍然是最大贡献来源，必须保留 |
| 只修正 valence residual | 稳定优于同时修正 arousal；arousal residual 当前会伤害结果 |
| zero-init residual head | 必须保留，初始等价于先验，避免第一轮就破坏轨迹 |
| adaptive graph + gated SSM | 比 MLP residual 和完整 selective-scan Mamba 更适合当前数据量 |
| scale-gated multi-scale `1/5/9s` | 当前最有效的多尺度方式；concat 不如 gate |
| residual 后处理 scale + smooth | 单折能显著降低 MAE，但 scale 不能盲目放大，需要跨折选择 |

### 本轮新增实验

| 方法 | 单折最佳 Overall MAE | Valence MAE | Arousal MAE | 结论 |
| --- | ---: | ---: | ---: | --- |
| 当前最佳 `scale-gated 1/5/9 + scale2.75 + smooth5` | **29.7984** | **28.2076** | 31.3892 | 单折最低，保留为 public-val 激进版本 |
| `modality-dropout=0.10 + scale2.00 + smooth5` | 29.8110 | 28.2327 | 31.3892 | 单折略差，但跨折更稳 |
| EEG-only residual | 29.8577 | 28.3263 | 31.3892 | EEG 单独有贡献 |
| fNIRS-only residual | 29.8605 | 28.3318 | 31.3892 | fNIRS 单独也有贡献 |
| no-signal residual | 29.8621 | 28.3351 | 31.3892 | 模型可学到全局残差/序列偏置，说明必须看消融 |
| ASAC-style cross attention fusion | 29.9460 左右 | - | - | 本轮不如简单 pooling，疑似自由度过高 |
| residual clipping 后处理 | 29.7984 | 28.2076 | 31.3892 | 没有继续提升，残差本身不大，主要问题是尺度选择 |

### 4 折 subject-disjoint 稳健性

固定后处理的 4 折平均：

| 模型/后处理 | 4 折平均 Overall MAE | 观察 |
| --- | ---: | --- |
| `VideoTimeMean_smooth5` | 29.5685 | 稳健基线 |
| no-dropout `scale1.00_smooth5` | 29.5108 | 稳定小幅增益 |
| no-dropout `scale2.75_smooth5` | 29.6013 | 单折最强但跨折不稳 |
| `modality-dropout=0.10 scale2.00_smooth5` | **29.4948** | 当前最稳的固定提交策略 |
| `modality-dropout=0.10` 每折 oracle 最优 | 29.4728 | 说明仍有空间学习自适应 residual scale |

结论：当前 public-val 最低分仍是 `29.7984`，但更稳健的提交候选应优先考虑 `modality-dropout=0.10 + scale2.00 + smooth5`。如果只追当前可见验证集，可用 no-dropout `scale2.75 + smooth5`；如果追最终榜泛化，建议用 `moddrop010 scale2.00 smooth5`。

### 下一步

1. 把 `moddrop010 scale2.00 smooth5` 和 `no-dropout scale2.75 smooth5` 都做成可提交 `model.py`，作为稳健版和激进版。
2. 做自适应 residual scale：根据 trial/video 或模型残差置信度，在 `scale0-3` 之间选择，而不是固定全局 scale。
3. 继续做轻量预训练或蒸馏，但训练参数量应控制在 30 万以内；AffectGPT 更适合作离线 teacher，不适合直接端到端微调。

## 2026-06-12 追加迭代：动态图、ConvMixer 与 checkpoint 残差集成

本轮继续使用固定 subject-disjoint 验证范式：`test_1-test_20` 训练，`test_21-test_24` 验证，指标仍为 raw `[1,255]` 标尺上的 Overall MAE。为了避免把验证集结果误读成最终榜单结论，本节只作为本地模块筛选记录。

### 新增模块与结果

| 方法 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | ---: | ---: | ---: | --- |
| 旧最佳：`scale-gated 1/5/9 + gated SSM + scale2.75 + smooth5` | 29.7984 | 28.2076 | 31.3892 | 单模型旧最佳 |
| `functional graph` 动态功能连接图 | 29.9582 | 28.5273 | 31.3892 | 纯动态图不如自适应静态图 |
| `hybrid_functional graph` 静态图 + 功能图门控混合 | 29.8088 | 28.2284 | 31.3892 | 接近旧最佳，但未超过 |
| `hybrid_functional + modality-dropout=0.10` | 29.8196 | 28.2501 | 31.3892 | 没有和 modality dropout 形成正增益 |
| `1/3/5/9` scale-gated 多尺度 | 29.8152 | 28.2413 | 31.3892 | 多加 3s 尺度没有带来收益 |
| `conv_mixer` 时序模块 | 29.8554 | 28.3217 | 31.3892 | 不如 gated SSM |
| checkpoint residual ensemble | **29.7583** | **28.1273** | 31.3892 | 当前固定验证集最好 |

### 当前最佳集成

最佳组合来自 `tools/ensemble_graph_mamba.py`：

```text
scalegated_msgm_159
+ moddrop010_seed123
+ hybrid_functional_graph
residual scale = 4.00
smooth window = 5
```

结果为：

| 方法 | Overall MAE | Valence MAE | Arousal MAE |
| --- | ---: | ---: | ---: |
| `Ensemble_scalegated_msgm_159+moddrop010_seed123+hybrid_functional_graph_scale4.00_smooth5` | **29.7583** | **28.1273** | 31.3892 |

### 模块判断

1. Graph-Mamba 主线仍然有效，但本轮证明不是所有“图创新”都有效。纯功能连接图在当前手工特征上噪声较大，静态图 + 动态图门控更稳。
2. 稀疏 top-k 功能图实现运行异常，未生成 `iteration_021` 结果。暂时标记为失败路线，后续如果继续做，需要先降低 batch 或把 top-k 图缓存到 CPU/特征文件中。
3. ConvMixer/TCN 风格时序混合没有超过 gated SSM，说明当前收益主要来自轻量状态门控和平滑残差，而不是一般卷积时序建模。
4. checkpoint 集成有效，说明不同 seed、modality dropout、hybrid graph 学到的 valence residual 有互补性。这个方向比继续加大单模型参数更适合当前 24 个公开 subject 的数据规模。

### 复现实验命令

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 1 --target-mode valence --input-modality both --fusion-mode pool --graph-encoder hybrid_functional --temporal-block gated_ssm --scale-fusion gated --multiscale-windows 1,5,9 --graph-hidden 32 --d-model 128 --mamba-layers 2 --batch-size 8 --lr 0.001 --dropout 0.15 --output experiments/results/iteration_019_hybrid_functional_graph.json --checkpoint experiments/checkpoints/graph_mamba/hybrid_functional_graph.pt
```

```bash
F:/anaconda/envs/eegpt/python.exe tools/train_graph_mamba.py --epochs 1 --target-mode valence --input-modality both --fusion-mode pool --graph-encoder adaptive --temporal-block conv_mixer --scale-fusion gated --multiscale-windows 1,5,9 --graph-hidden 32 --d-model 128 --mamba-layers 2 --batch-size 8 --lr 0.001 --dropout 0.15 --output experiments/results/iteration_022_convmixer_159.json --checkpoint experiments/checkpoints/graph_mamba/convmixer_159.pt
```

```bash
F:/anaconda/envs/eegpt/python.exe tools/ensemble_graph_mamba.py --output experiments/results/iteration_023_checkpoint_ensemble.json --max-ensemble-size 4
```

## 2026-06-12 追加迭代 2：放弃复杂图后处理，转向校准和集成

在确认复杂动态图和 ConvMixer 都没有超过主线后，本轮尝试更轻的替代办法：只对已有 checkpoint 做 residual ensemble、scale 校准、平滑窗口校准、两模型权重扫描。

### 新高结果

| 方法 | Overall MAE | Valence MAE | Arousal MAE | 结果文件 |
| --- | ---: | ---: | ---: | --- |
| 旧集成最佳：top3 `scale4.00 + smooth5` | 29.7583 | 28.1273 | 31.3892 | `iteration_023_checkpoint_ensemble.json` |
| top3 细粒度 scale/window 搜索 | 29.7369 | 28.0866 | 31.3872 | `iteration_024_ensemble_top3_fine.json` |
| top2 scale 扩展搜索 | **29.7369** | **28.0866** | **31.3872** | `iteration_025_ensemble_top2_scale_expand.json` |
| top2 权重搜索 | 29.7369 | 28.0866 | 31.3872 | `iteration_026_weighted_top2_ensemble.json` |
| top2 平滑窗口细扫 | 29.7369 | 28.0866 | 31.3872 | `iteration_027_top2_window_sweep.json` |

当前固定验证集最优：

```text
Ensemble_moddrop010_seed123+hybrid_functional_graph_scale6.00_smooth7
Overall MAE = 29.7369
Valence MAE = 28.0866
Arousal MAE = 31.3872
```

### 新判断

1. 更复杂的图结构暂时不如后处理校准有效。`hybrid_functional_graph` 单模型只有 29.8088，但和 `moddrop010_seed123` 互补后达到 29.7369。
2. 三模型集成不如二模型集成，说明 `scalegated_msgm_159` 在这个组合里反而引入了一点偏差。
3. 两模型权重扫描没有超过等权，最佳仍是 `0.50 / 0.50`。
4. 平滑窗口细扫显示 `smooth7` 最好；`smooth11+` 会降低 MSE 但升高 MAE，不适合官方主指标。
5. 当前最优是强烈本地验证集校准结果，适合作为 aggressive 版本；最终提交仍建议同时保留 4-fold 更稳的 `moddrop010 scale2.00 smooth5` 版本。

### 复现命令

```bash
F:/anaconda/envs/eegpt/python.exe tools/ensemble_graph_mamba.py --checkpoints experiments/checkpoints/graph_mamba/moddrop010_seed123.pt experiments/checkpoints/graph_mamba/hybrid_functional_graph.pt --output experiments/results/iteration_025_ensemble_top2_scale_expand.json --max-ensemble-size 2 --scales 5.00,5.25,5.50,5.75,6.00,6.25,6.50,6.75,7.00,7.25,7.50,7.75,8.00,8.25,8.50,8.75,9.00 --smooth-windows 0,5,7,9,11,13
```

```bash
F:/anaconda/envs/eegpt/python.exe tools/ensemble_graph_mamba.py --checkpoints experiments/checkpoints/graph_mamba/moddrop010_seed123.pt experiments/checkpoints/graph_mamba/hybrid_functional_graph.pt --output experiments/results/iteration_026_weighted_top2_ensemble.json --max-ensemble-size 2 --scales 5.00,5.25,5.50,5.75,6.00,6.25,6.50,6.75,7.00 --smooth-windows 5,7,9 --pair-weight-step 0.05
```

## 2026-06-12 追加迭代 3：AffectGPT 权重教师蒸馏

本轮尝试把 `models/AffectGPT` 下已经下载完成的两个 checkpoint 接入 MER-PS。实际文件为两个 LoRA/adapter checkpoint：

```text
models/AffectGPT/.../checkpoint_000060_loss_0.480.pth
models/AffectGPT/.../checkpoint_000030_loss_0.751.pth
```

它们不是完整可直接推理的 AffectGPT/AffectNet 大模型权重，仍缺 base LLM、CLIP、HuBERT 等主干。因此本轮没有把大模型放进 Codabench 推理链路，而是构建离线 teacher cache，再训练期蒸馏到只吃 EEG/fNIRS 的学生模型。

### 新增代码

| 文件 | 作用 |
| --- | --- |
| `tools/build_affectgpt_teacher_cache.py` | 读取 AffectGPT checkpoint，抽取 LoRA 统计向量，并和目标情绪、SAM 视频均值拼成 teacher cache |
| `experiments/teacher_cache/affectgpt_teacher_cache.npz` | 36864 个 sample 对齐的 teacher cache |
| `tools/train_graph_mamba.py` | 新增 `--teacher-cache`、`--teacher-keys`、`--distill-weight`，训练期对 Graph-Mamba hidden state 做 masked distillation |
| `emotion_merps/graph_mamba.py` | `forward(..., return_features=True)` 返回时序隐藏状态，供 teacher loss 使用 |

teacher cache 维度：

```text
emotion_dim = 27
sam_dim = 4
affectgpt_dim = 144
semantic_dim = 175
samples = 36864
```

### 对照结果

固定验证范式仍为 `test_1-test_20` 训练，`test_21-test_24` 验证，指标为 raw `[1,255]` Overall MAE。

| 方法 | Best Overall MAE | Valence MAE | Arousal MAE | 结果文件 | 判断 |
| --- | ---: | ---: | ---: | --- | --- |
| ASAC 小学生，无 teacher | 47.077 | - | - | `distill_no_teacher_e5_seed42` | 弱模型 sanity check |
| ASAC 小学生，AffectGPT teacher | 47.007 | - | - | `distill_affectgpt_teacher_e5_seed42` | 有极小正向信号，但不可作为主线 |
| Graph-Mamba seed123，无 teacher | 29.9445 | 28.4999 | 31.3892 | `iteration_015_moddrop010_seed123.json` | 同 seed 对照 |
| Graph-Mamba + `semantic` teacher, w=0.02 | 29.8815 | 28.3738 | 31.3892 | `iteration_028_affectgpt_teacher_semantic_w002.json` | 单 seed 有明显正则收益 |
| Graph-Mamba + `affectgpt` only, w=0.005 | 29.8829 | 28.3761 | 31.3897 | `iteration_029_affectgpt_only_w0005.json` | 与 semantic 基本打平 |
| top2 + semantic teacher 三模型集成 | 29.7369 | 28.0866 | 31.3872 | `iteration_030_affectgpt_semantic_top2_ensemble.json` | 没有超过原 top2 |

当前最佳不变：

```text
Ensemble_moddrop010_seed123+hybrid_functional_graph_scale6.00_smooth7
Overall MAE = 29.7369
Valence MAE = 28.0866
Arousal MAE = 31.3872
```

### 结论

1. AffectGPT 权重蒸馏对弱 seed 的 Graph-Mamba 有收益：从 29.9445 提升到约 29.8815。
2. 收益主要像训练正则，而不是强 teacher：只用 `affectgpt` 权重统计和完整 `semantic` teacher 几乎打平。
3. 加入现有 top2 ensemble 后没有新高，说明它和当前强模型的 residual 互补性不足。
4. 继续直接微调 AffectGPT 主体不适合当前赛题接口；Codabench 测试输入没有视频、音频、文本，只能提交 EEG/fNIRS 推理模型。

下一步更值得做的是 teacher 预测 residual confidence 或 adaptive residual scale，而不是把 LLM 主体放进提交链路。
