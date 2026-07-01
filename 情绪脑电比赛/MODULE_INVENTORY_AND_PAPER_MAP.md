# MER-PS Module Inventory And Paper Map

当前最可信验证范式：

```text
6-fold subject-disjoint
subjects = test_1 ... test_24
fold size = 4 subjects
metric = overall MAE = mean(valence MAE, arousal MAE)
```

当前最优：

```text
AutoDimwiseConfidence_097
overall = 28.7699
valence = 26.9847
arousal = 30.5551
```

## 可改模块

| 模块 | 当前状态 | 风险 | 下一步优先级 |
| --- | --- | --- | --- |
| Video-time label prior | 已从 mean 改到 robust median + lag + smooth | 低 | 继续做多专家/状态专家 |
| Uncertainty gate | 已用 train-subject MAD 做 gate，有效 | 低 | 继续做 pattern-specific expert |
| Valence/arousal 解码 | 096/097 证明分维度有效 | 低 | 固化为提交接口 |
| Signal residual | TBCR/LRAG 跨折失败 | 中 | 只能在 097 prior 上做低自由度 gated residual |
| EEG/fNIRS 特征预处理 | 已有 779 维 ASAC 特征，含 EEG 64x5 + fNIRS 51x9 | 中 | 尝试更稳健的 residual probe、MI feature selection |
| Graph 空间编码 | 已有 adaptive / functional / hybrid graph | 高 | 数据量小，不宜继续加大网络 |
| Temporal backbone | 已有 Mamba、PatchTST、TimesNet、iTransformer、TimeMixer、Fourier | 高 | 不再盲目堆深模型 |
| Foundation/蒸馏 | AffectGPT/EEGPT/LaBraM/NeuroLM 可参考 | 高 | 优先复现小模块，不直接大规模微调 |

## 论文到模块映射

| 论文方向 | 可迁移模块 | 对 MER-PS 的判断 |
| --- | --- | --- |
| TimeMixer / TimeMixer++ | decomposable multiscale mixing；多尺度 trend/seasonal 专家 | 已在 094-097 的 long smooth + uncertainty 中起效；下一步做状态专家 |
| Pathformer / pattern-specific experts | 不同时间状态走不同专家 | 适合 MER-PS，因为 joystick 标签在稳定段和剧烈变化段噪声结构不同 |
| PatchTST | patch/channel-independent 表征 | 已有 lite block；公开数据太小，不宜继续加大 |
| iTransformer | variate-token attention | 已有 lite block；更适合作为 signal residual 小模型，不适合作主预测 |
| TimesNet | 周期/2D temporal variation | 已有 lite block；MER-PS 单 trial 较短，优势有限 |
| MSGM / miMamba / EEGMamba | 多尺度 EEG + 图 + Mamba | 已部分复现；跨折信号残差不稳，说明应先做低自由度 signal gate |
| LaBraM / EEGPT / CBraMod | EEG channel-patch/tokenizer/foundation 表征 | 参数量远大于本数据；优先尝试 frozen feature 或 linear probe |
| NeuroLM / AffectGPT | LLM/MLLM 情绪语义先验 | 更适合视频/文本语义蒸馏；MER-PS 当前 test 只有生理信号和 sample_id，不能依赖外部视频内容 |
| EEG-fNIRS graph/cross-attention fusion | 双模态图、跨模态注意力、互信息筛选 | 可尝试低自由度特征选择；高参数 cross-attention 过拟合风险高 |

## 当前经验约束

```text
1. 直接预测 y 的模型大多不如 video-time prior。
2. 097 说明标签轨迹先验非常强，signal 模型只能预测 residual。
3. 任何新增 EEG/fNIRS 模块都要满足：
   residual = y - AutoDimwiseConfidencePrior
   correction 幅度要小
   correction 要有 uncertainty gate
4. 参数量建议：
   后处理/统计模块: < 100 参数
   signal residual probe: < 1e5 参数
   neural residual model: < 3e5 参数
   不建议在 24 subjects 上微调 10M+ foundation model
```

## 下一轮复现顺序

```text
098 Pattern-Specific Prior Expert:
  复现 TimeMixer/Pathformer/Pattern-Expert 的思想。
  稳定段和动态段用不同 prior expert。

099 Strong-Prior Signal Residual Probe:
  在 097 prior 上训练低自由度 EEG/fNIRS residual。
  只允许 gated small correction。

100 Mutual-Information Feature Selection:
  复现 EEG-fNIRS fusion 文献里的互信息筛选思想。
  目标是降低 779 维特征对 360 个 trial 的过拟合。
```
