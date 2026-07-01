# MER-PS 模型文献筛选与复现记录

更新时间：2026-06-10

本文记录面向 MER-PS 的下一阶段模型选择。结论先行：当前数据只有 24 个公开被试、360 个 trial，不适合从零训练大参数 foundation model。最值得保留的方向是“强视频时间先验 + EEG/fNIRS 图时序残差 + 轻量蒸馏/正则”，而不是直接把大模型塞进提交包。

## 已核验论文与可借鉴模块

| 论文/方向 | 关键思想 | 对 MER-PS 的可用部分 | 本地处理 |
| --- | --- | --- | --- |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | selective SSM，线性复杂度长序列建模 | 替代 Transformer/GRU 做 trial 内时序建模 | 已实现纯 PyTorch `MambaSelectiveScanBlock` |
| MSGM: Multi-Scale Spatiotemporal Graph Mamba for EEG Emotion Recognition | 多窗口时序分割、全局-局部图、Mamba 融合 | 多尺度时序、图结构、轻量时序头 | 已复现 concat 与 scale-gated MSGM-lite |
| EEG-DisGCMAE | graph contrastive masked autoencoder + topology distillation | 用无标签 EEG/fNIRS 做图预训练，再蒸馏到轻量学生 | 尚未训练预训练阶段，适合作为下一步 |
| AffectGPT | 多模态情感理解大模型/情绪语义教师 | 离线生成情绪语义先验或 soft label | 已下载 2 个 checkpoint，剩余权重后台下载中 |
| 跨被试/域适配思路 | 减小 subject distribution shift | subject/trial test-time normalization、domain-invariant head | 已测试 subject/trial 无监督归一化 |

## 本地复现实验结论

| 复现项 | 命令参数 | 最优 Overall MAE | 结论 |
| --- | --- | ---: | --- |
| Scale-gated MSGM-lite + Graph-Mamba gated SSM | `--scale-fusion gated --multiscale-windows 1,5,9` | **29.7984** | 当前最强，保留 |
| 原始单尺度 Graph-Mamba gated SSM | `--temporal-block gated_ssm` | 29.8214 | 稳定基线 |
| 完整 Mamba selective scan | `--temporal-block mamba` | 29.9223 | 结构更完整，但此数据规模上不如 gated SSM |
| MSGM-lite 多尺度 concat | `--multiscale-windows 1,3,5` | 29.8791 | 简单拼接多尺度不够 |
| MSGM-lite 多尺度 gate | `--scale-fusion gated --multiscale-windows 1,5,9` | **29.7984** | gate 控噪有效 |
| subject/trial 归一化 | `--feature-norm subject/trial` | 29.8622 | 有稳定化迹象，但没有超过主线 |
| signed graph | `--graph-encoder signed` | 未产生有效残差收益 | 正负图自由度太高，当前容易过拟合 |
| residual target scaling | `--residual-target-scale 1.75/2.25` | 29.8308 | 训练期放大不如后处理 scale 稳 |

## 推荐创新模型

建议命名为 **Prior-Guided Graph Mamba Distillation, PG-GMD**：

```text
VideoTimeMean prior
      |
      +--> raw VA coarse trajectory

EEG ASAC features ----> EEG graph encoder ----\
                                                gated fusion -> gated SSM/Mamba -> valence residual
fNIRS ASAC features --> fNIRS graph encoder ---/

AffectGPT / label teacher / EEG graph teacher
      |
      +--> offline teacher embedding / soft residual target
```

核心设计：

- 主预测继续使用 `VideoTimeMean`，因为它在公开 subject-disjoint 验证上是强先验。
- 生理信号只预测 residual，且先只修正 valence；arousal 暂时保持先验 + smoothing。
- 图编码器先保持 adaptive positive graph，不急着用 signed graph。
- 时序头以 gated SSM 为主，Mamba selective scan 保留为可选 ablation。
- 多尺度不再直接 concat，而是学习 `scale gate`：`1s/5s/9s` 多尺度先融合回原始特征维度，再送入同一图编码器。当前 `1/5/9s` 好于 `1/3/5s` 和 `1/3/5/9s`。
- 蒸馏只在训练期使用，提交包只带学生模型和 checkpoint。

## 下一步最值得做的三件事

1. **5-fold subject-disjoint 验证**  
   当前 20/4 split 只有 4 个验证被试，微小提升可能不稳。先确认 `29.8214` 是否在多折上仍成立。

2. **把 scale-gated MSGM-lite 做成可提交模型**  
   目前训练脚本已验证有效，下一步要把 `--scale-fusion gated --multiscale-windows 1,5,9` 的推理逻辑接进 Codabench `model.py`。

3. **做 EEG/fNIRS masked graph pretraining**  
   参考 EEG-DisGCMAE，用公开训练信号做 masked reconstruction/contrastive pretrain，再 fine-tune residual。这个比直接调用 AffectGPT 更贴近生理信号。

## 参考来源

- Mamba: https://arxiv.org/abs/2312.00752
- MSGM: https://arxiv.org/abs/2507.15914
- EEG-DisGCMAE: https://arxiv.org/abs/2411.19230
- AffectGPT: https://arxiv.org/abs/2501.16566
- MER Challenge: https://zeroqiaoba.github.io/MER-Challenge/
- MER-PS trainval: https://huggingface.co/datasets/MER-PS/MER-PS-trainval

## 2026-06-11 追加判断

今天的实验证明，MER-PS 当前阶段不应该继续单纯堆更复杂的跨模态模块。ASAC-style cross attention 在单折验证中没有超过 pooling，完整 Mamba selective scan 也没有超过轻量 gated SSM；真正有效的是强先验、轻量 residual、图结构、多尺度 gate、zero-init 和稳健后处理。

新的推荐模型路线调整为：

```text
VideoTimeMean prior
  + scale-gated EEG/fNIRS graph encoder
  + gated SSM valence residual
  + modality dropout during training
  + conservative fixed residual scale for final submission
```

单折最低仍是 no-dropout `scale2.75_smooth5`，Overall MAE 为 `29.7984`；但 4 折 subject-disjoint 平均显示 `modality-dropout=0.10 + scale2.00_smooth5` 更稳，平均 `29.4948`，优于 `VideoTimeMean_smooth5` 的 `29.5685`。

因此下一步创新重点不应是直接接 AffectGPT 大模型端到端训练，而是做轻量蒸馏/自适应尺度：

1. 用 AffectGPT 或标签轨迹模型生成离线 teacher embedding/soft residual，不训练大模型主体。
2. 学一个 residual confidence / residual scale head，让每个 trial 在 `scale0-3` 间自适应选择修正强度。
3. 做 masked EEG/fNIRS graph pretraining，再把预训练图编码器蒸馏到当前 30 万参数以内的学生模型。

## 2026-06-12 追加：最新模块复现后的取舍

本轮围绕近年 EEG 情绪识别和时间序列论文中常见的三个模块做了本地复现：功能连接图、轻量 ConvMixer/TCN 时序混合、checkpoint ensemble。固定验证范式仍是 `test_1-test_20` 训练、`test_21-test_24` 验证。

| 论文/方向启发 | 本地复现模块 | 结果 | 取舍 |
| --- | --- | ---: | --- |
| 功能连接图 / dynamic graph | `FunctionalChebGraphEncoder` | 29.9582 | 纯动态相关图噪声偏大，不建议作为主干 |
| 物理/自适应图 + 功能图混合 | `HybridFunctionalChebGraphEncoder` | 29.8088 | 有效但弱于当前单模型最佳，可作为集成成员 |
| 稀疏功能连接 top-k | `SparseHybridFunctionalChebGraphEncoder` | 未完成 | 当前实现速度和稳定性不够，暂缓 |
| ModernTCN / PatchMixer 类时序混合 | `TemporalConvMixerBlock` | 29.8554 | 不如 gated SSM，说明 MER-PS 当前更需要状态门控而非普通卷积混合 |
| 竞赛常用模型集成 | checkpoint residual ensemble | **29.7583** | 当前固定验证集最佳，值得进入提交候选 |

新的经验判断：

1. 图论 + Mamba 方向仍可保留，但创新点要收敛到“小自由度、强正则、可集成”。直接引入复杂动态图容易放大 subject 噪声。
2. 对 MER-PS 当前公开规模，`VideoTimeMean prior + valence residual` 是主框架；新模块只要不能稳定改善 valence residual，就不应进入提交模型。
3. Hybrid functional graph 虽然单模型未超过旧最佳，但和 scale-gated MSGM、modality-dropout seed 模型有互补性，进入 ensemble 后贡献了当前新高。
4. 下一步更值得做的是 teacher/student residual distillation 和自适应 residual scale，而不是继续堆 Transformer/大 Mamba 参数。

当前推荐提交候选：

```text
Aggressive local-val version:
VideoTimeMean prior
+ ensemble(scalegated_msgm_159, moddrop010_seed123, hybrid_functional_graph)
+ valence residual scale 4.00
+ smooth5
Local Overall MAE = 29.7583

Robust version:
VideoTimeMean prior
+ modality-dropout=0.10 Graph-Mamba residual
+ fixed valence scale 2.00
+ smooth5
4-fold mean Overall MAE = 29.4948
```

## 2026-06-12 追加 2：如果不继续堆 Graph-Mamba，替代路线是什么

本轮尝试了不再增加主干复杂度的路线：checkpoint residual ensemble、scale 校准、平滑窗口校准、两模型权重扫描。结论是：对 MER-PS 这种小 subject 数、强视频时间先验的数据，**模型互补 + 后处理校准** 比继续增加动态图/大时序模型更有效。

| 路线 | 最佳结果 | 判断 |
| --- | ---: | --- |
| 复杂动态图单模型 | 29.8088 | 可作为互补成员，不适合作主力 |
| ConvMixer 时序替代 | 29.8545 | 不如 gated SSM |
| top3 checkpoint ensemble | 29.7583 | 有效 |
| top2 checkpoint ensemble + scale/window calibration | **29.7369** | 当前本地最佳 |
| top2 weighted ensemble | 29.7369 | 权重扫描没有超过等权 |

新的推荐策略：

```text
Do not keep increasing backbone size.
Use:
  VideoTimeMean prior
  + two complementary valence residual models
  + equal residual average
  + residual scale 6.00
  + smooth7
```

对应 checkpoint：

```text
moddrop010_seed123.pt
hybrid_functional_graph.pt
```

下一步真正值得投入的是把这个 top2 ensemble 写成 Codabench `model.py` 提交包，同时准备一个更保守的 4-fold 版本。AffectGPT/多教师蒸馏仍然可以做，但更适合离线生成 teacher residual 或 confidence，不适合直接端到端微调成大模型。

## 2026-06-12 追加 3：AffectGPT 权重该如何用于 MER-PS

本轮本地权重检查显示，`models/AffectGPT` 中可用的是两个约 634MB 的 LoRA/adapter checkpoint。checkpoint 配置指向 Qwen25、CLIP-ViT-Large、HuBERT-Large 等主干，但这些主干权重不在当前目录。因此它们不能被当作完整 AffectGPT 直接推理，也不适合直接放进 Codabench 提交包。

更合理的使用方式是训练期蒸馏：

```text
AffectGPT LoRA checkpoint
  -> LoRA statistics / emotion semantic teacher
  -> Graph-Mamba hidden-state distillation
  -> EEG/fNIRS-only student
  -> Codabench inference
```

本地结果说明：

| teacher 方式 | 作用 | 结果判断 |
| --- | --- | --- |
| `semantic = emotion + SAM + AffectGPT LoRA stats` | 给学生一个情绪语义/视频先验方向 | 同 seed 从 29.9445 提到 29.8815 |
| `affectgpt` only | 只使用 AffectGPT 权重统计 | 29.8829，几乎与 semantic 打平 |
| 加入 top2 checkpoint ensemble | 尝试利用互补性 | 没有超过 29.7369 |

这说明 AffectGPT teacher 当前提供的是轻量正则收益，而不是可直接替代生理建模的强 teacher。原因大概率有三个：

1. MER-PS 测试输入没有视频、音频、文本，AffectGPT 的主要强项无法在推理时使用。
2. 当前只有 LoRA adapter，没有完整 base model，直接微调成本和工程风险都很高。
3. 公开训练集只有 24 个 subject，参数量过大时会把 subject/video 偏差学进去，泛化风险高。

后续创新建议：

1. 用 AffectGPT teacher 预测 `residual confidence`，让模型判断哪些时间点该相信生理 residual，哪些时间点退回 VideoTimeMean prior。
2. 用 teacher 生成 trial-level adaptive scale，而不是直接约束每秒隐藏状态。
3. 继续保持学生模型小参数量。当前 Graph-Mamba 主体适合控制在几十万到一两百万参数内；大模型只作为离线 teacher。
4. 如果要写论文创新点，可以命名为 physiological-only student distilled from affective multimodal foundation adapters，重点强调“训练期多模态语义蒸馏，推理期只需生理信号”。
