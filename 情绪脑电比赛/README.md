# MER-PS 2026 情绪脑电与近红外多模态建模

本项目用于参加 [MER Challenge 2026](https://zeroqiaoba.github.io/MER-Challenge/) 的 **MER-PS: Physiological Signal-Based Emotion** 赛道。赛题要求根据同步采集的 EEG 与 fNIRS 生理信号，预测被试观看情绪诱发视频时的连续情绪轨迹。

我们的目标分两层：

1. 先严格复现官方 starting kit，跑通 Codabench code submission。
2. 再参考 `多教师跨方式调整基础模型.pdf` 中的 MTDP 思路，把 CBraMod-MTDP 风格的 EEG foundation model 改造成面向情绪识别的 EEG-fNIRS 多模态蒸馏模型。

## 赛题定义

MER-PS 是二维连续情绪回归任务，每个 1 Hz annotation sample 需要输出：

- `valence`：情绪愉悦度
- `arousal`：情绪激活水平

标签使用原始 MER-PS 整数尺度 `[1, 255]`，其中 `(128, 128)` 表示价-唤醒平面的中心。提交预测也必须是这个尺度上的整数值。

| 项目 | 内容 |
| --- | --- |
| 输入 | EEG + fNIRS |
| 输出 | `sample_id,valence,arousal` |
| 标签频率 | 1 Hz |
| 标签尺度 | `[1, 255]` raw integer scale |
| 主指标 | overall MAE over valence and arousal |
| 辅助指标 | valence MAE, arousal MAE, overall MSE |
| 划分 | subject-disjoint |
| 提交形式 | Codabench code submission |

## 数据集

公开训练验证集为 [MER-PS/MER-PS-trainval](https://huggingface.co/datasets/MER-PS/MER-PS-trainval)，包含匿名被试 `test_1` 到 `test_24`。隐藏 leaderboard 使用额外测试被试。

EEG 原始采集频率为 1000 Hz，公开文件中已下采样为 200 Hz。fNIRS 采样率为 47.62 Hz。每个 trial 在视频开始前包含 5 秒 baseline。

建议将数据放在：

```text
data/
+-- MER_PS_codabench_public_trainval/
    +-- Metadata.csv
    +-- SAM_score.csv
    +-- PANAS_score.csv
    +-- Targeted_emotions.txt
    +-- fNIRS_coordinates.csv
    +-- fNIRS_reservations.csv
    +-- data/
    |   +-- test_1/
    |   |   +-- EEG_baselines.mat
    |   |   +-- EEG_videos.mat
    |   |   +-- fNIRS_baselines.mat
    |   |   +-- fNIRS_videos.mat
    |   +-- ...
    |       +-- test_24/
    +-- annotations/
        +-- test_1_label.mat
        +-- ...
        +-- test_24_label.mat
```

当前本机已下载的数据路径：

```text
D:/Downloads/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval
```

### 数据形状

| 模态 | 通道数 | 采样率 | `.mat` 数组形状 |
| --- | ---: | ---: | --- |
| EEG | 64 | 200 Hz | `channel x time` |
| fNIRS | 51 | 47.62 Hz | `signal_type x channel x time` |
| annotation | 2 | 1 Hz | `2 x time` |

fNIRS 的 `signal_type` 包括 HbO、HbR、HbT、Abs 780 nm、Abs 805 nm 和 Abs 830 nm。annotation 第一行为 valence，第二行为 arousal。

## 官方提交接口

官方 starting kit 位于：

```text
D:/Downloads/starting_kit/starting_kit/
```

Codabench 不接收手工生成的预测文件，而是接收一个 zip 代码包。zip 根目录必须包含：

```text
model.py
```

`model.py` 必须定义：

```python
def predict(input_dir, output_dir):
    ...
```

评测时，`input_dir` 结构为：

```text
sample_ids.csv
data/<prediction_subject_id>/
  EEG_baselines.mat
  EEG_videos.mat
  fNIRS_baselines.mat
  fNIRS_videos.mat
```

提交代码必须读取 `sample_ids.csv`，并在 `output_dir` 下直接写出：

```text
predictions.csv
```

输出 schema 必须为：

```csv
sample_id,valence,arousal
predict1_V01_T000,128,128
predict1_V01_T001,132,126
```

硬性要求：

- `predictions.csv` 必须包含 `sample_ids.csv` 中每个 `sample_id`，且每个只出现一次。
- 不能多写未知 `sample_id`。
- `sample_id` 必须原样复制。
- `valence` 和 `arousal` 必须是有限、数值、整数，且位于 `[1, 255]`。
- 评测按 `sample_id` 对齐，行顺序不重要。
- 提交包中应包含训练好的 checkpoint，平台不会自动帮参赛者训练模型。

## Starting Kit Baseline

官方 baseline 的代码结构：

```text
starting_kit/
+-- README.md
+-- train_baseline.py
+-- model.py
+-- sample_code_submission.zip
+-- asac_merps/
    +-- __init__.py
    +-- features.py
    +-- model.py
```

`sample_code_submission.zip` 是一个可提交示例，包含：

```text
model.py
best_model.pt
asac_merps/
  __init__.py
  features.py
  model.py
```

官方模型是 ASAC-style EEG-fNIRS fusion regressor，主要组成：

- EEG 分支：基于通道节点的 adaptive Chebyshev graph encoder
- fNIRS 分支：同样使用 adaptive Chebyshev graph encoder
- 跨模态融合：EEG-to-fNIRS 与 fNIRS-to-EEG multi-head attention
- 对齐损失：EEG/fNIRS global embedding 的 contrastive alignment loss
- 回归头：输出归一化 `[0, 1]`，推理时映射回 `[1, 255]`

`train_baseline.py` 默认使用 `test_1` 到 `test_20` 训练，`test_21` 到 `test_24` 验证。后续实验应保留 subject-disjoint 验证原则。

## 论文方法理解

本地 PDF：

```text
多教师跨方式调整基础模型.pdf
```

对应论文为 **Standing on the Shoulders of Giants: Rethinking EEG Foundation Model Pretraining via Multi-Teacher Distillation**。核心思想是：不要只用 masked reconstruction 预训练 EEG foundation model，而是让成熟的大模型教师帮助 EEG 模型学习更有语义的表示。

论文提出 MTDP，两阶段如下：

1. **Teacher Representation Fusion**  
   冻结多个教师模型，例如 vision foundation model DINOv3 和 time-series foundation model Chronos。对 EEG 输入做 mask，分别提取 masked/unmasked teacher representations。用一个 learnable gating network 学习每个教师的重要性，并通过 masked latent denoising objective 融合教师表示。

2. **Knowledge Distillation**  
   冻结第一阶段学到的 gate，把融合后的 teacher representation 作为目标，用 cosine similarity 等蒸馏损失训练 EEG student foundation model。论文中 student 可以是 CBraMod 风格的 EEG foundation model。

对 MER-PS 来说，我们不只是复现论文，而是要把它改成情绪识别任务：

```text
原论文: DINOv3 / Chronos teachers -> gate -> CBraMod EEG student
本项目: 情绪语义 teacher / EEG teacher / fNIRS teacher -> gate -> EEG-fNIRS emotion student
```

## 我们的改造方向

### 教师模型

1. **情绪大模型教师**
   - 参考 AffectGPT 一类多模态情绪理解模型。
   - 用于提供 video-level 或 sample-level 的情绪语义先验，例如细粒度情绪类别、valence/arousal 语言描述、情绪变化趋势。
   - 更适合离线生成软标签或语义 embedding，不建议提交时依赖在线大模型。
   - AffectGPT 官方提供了预训练 checkpoint，可从 Baidu 网盘或 Hugging Face 下载。Hugging Face 仓库为 `MERChallenge/AffectGPT`，包含多个约 665 MB 的 `.pth` checkpoint；完整仓库约 7.31 GB。
   - 注意 AffectGPT 仍依赖基础模型组件，例如 `clip-vit-large-patch14`、`chinese-hubert-large` 和 `Qwen2.5-7B-Instruct`。这些更适合放在训练环境中离线跑教师推理，不适合直接塞进 Codabench 提交包。
   - 当前已下载两份精选 checkpoint 到 `models/AffectGPT/`：
     - `emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100/checkpoint_000060_loss_0.480.pth`
     - `mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth`
   - 其余 Hugging Face checkpoint 已启动后台 BITS 下载任务 `AffectGPT-remaining-checkpoints`，下载完成后仍放在 `models/AffectGPT/`。

2. **EEG 大模型教师**
   - 复现或迁移 CBraMod / CBraMod-MTDP 思路。
   - 也可以把 EEG 片段转换为时频图，让 DINOv3/ViT 教师抽取视觉式表示。
   - 目标是让 EEG 分支不只拟合噪声，而是学习更稳定的情绪相关神经表征。

3. **fNIRS 大模型教师**
   - fNIRS 可作为低频血氧动力学时序，由 Chronos、TimesFM、PatchTST 或自训练 masked fNIRS encoder 作为教师。
   - 重点建模 HbO/HbR/HbT 的慢变化、斜率、滞后和跨通道空间模式。

### 学生模型

学生模型必须服务于比赛提交，因此建议从官方 ASAC baseline 改造，而不是从零写一个巨大模型：

```text
EEG raw/features  -> EEG encoder   \
                                  cross-modal fusion -> temporal head -> valence/arousal
fNIRS raw/features -> fNIRS encoder /
```

可逐步增强为：

- ASAC + temporal context
- ASAC + TCN/GRU/Transformer temporal head
- ASAC/Graph encoder + Mamba selective scan temporal residual head
- ASAC student + MTDP teacher fusion loss
- EEG-fNIRS dual foundation encoder + lightweight regression head

### 损失函数

训练时可以组合：

```text
L = L_regression
  + lambda_align * L_eeg_fnirs_contrastive
  + lambda_teacher * L_teacher_distill
  + lambda_semantic * L_emotion_semantic
  + lambda_smooth * L_temporal_smooth
```

其中：

- `L_regression`：对 `[valence, arousal]` 的 MSE/MAE/Huber loss
- `L_eeg_fnirs_contrastive`：保留官方 baseline 的 EEG-fNIRS 对齐约束
- `L_teacher_distill`：学生 embedding 对齐融合 teacher embedding
- `L_emotion_semantic`：对齐 AffectGPT/情绪语义教师产生的情绪 embedding 或软标签
- `L_temporal_smooth`：约束连续预测轨迹不要出现不合理跳变

## 实验路线

### Stage 0: 官方 baseline 复现

目标是确认数据、训练和提交接口完全正确。

- 官方 starting kit 已复制到 `official_starting_kit/`
- 可提交 baseline 已改造成项目包 `emotion_merps/`
- 跑通 `train_baseline.py`
- 生成 `best_model.pt`
- 本地模拟 `predict(input_dir, output_dir)`
- 检查 `predictions.csv` 的列名、sample_id 完整性和 `[1,255]` 整数范围

快速训练命令：

```bash
python train_baseline.py --epochs 50 --batch-size 128
```

调试时可以先限制样本量：

```bash
python train_baseline.py --epochs 2 --limit-train-samples 2048 --limit-val-samples 512
```

### Stage 1: 强化 ASAC baseline

- 加入前后文窗口，例如 `t-2` 到 `t+2`
- 增加 EEG bandpower、RMS、差分、通道统计
- 增加 fNIRS HbO/HbR/HbT 均值、标准差、斜率、滞后特征
- 尝试 Huber loss、MAE loss、label smoothing 和 temporal smoothing
- 使用 grouped K-fold 或 leave-one-subject-out 验证

### Stage 2: 复现 MTDP

- 实现 teacher feature cache，避免训练时重复跑大教师模型
- 实现 gating network
- 实现 masked latent denoising
- 实现 student embedding distillation
- 先在 EEG-only 上复现，再扩展到 EEG+fNIRS

### Stage 3: 情绪多模态蒸馏

- 引入 AffectGPT/情绪 MLLM 作为情绪语义教师
- 以视频 ID、目标情绪、SAM/PANAS 和连续 VA 标签构造情绪 prompt/软标签
- 用语义 embedding 或软 VA 轨迹蒸馏学生模型
- 将 fNIRS teacher 与 EEG teacher 通过 gate 融合
- 最终只提交轻量学生模型和 checkpoint

teacher cache 采用 `.npz` 格式：

```text
sample_ids: [N] string
emotion:    [N, D_emotion] float32
eeg:        [N, D_eeg] float32
fnirs:      [N, D_fnirs] float32
```

其中 `sample_ids` 必须与训练样本 ID 对齐，例如 `test_1_V01_T000`。可以先用公开标签构造一个 smoke-test teacher cache 检查训练链路：

```bash
python tools/build_label_teacher_cache.py
python train_distill.py --teacher-cache experiments/checkpoints/label_teacher_cache.npz --teacher-keys emotion
```

### Stage 4: 提交模型

最终提交包必须仍然像官方示例一样简洁：

```text
model.py
best_model.pt
emotion_merps/
  __init__.py
  features.py
  model.py
  distill.py
```

提交时不能依赖外部网络或远程 API。所有模型权重、特征标准化参数和必要模块都必须放进 zip。

## 验证策略

必须使用跨被试验证，避免 subject leakage。

推荐：

```text
Fold k:
  train subjects: 公开被试中的 23 人
  valid subjects: 剩余 1 人
```

或先使用轻量 5-fold grouped K-fold。不要把同一 subject 的不同 trial 随机拆进训练和验证。

本地 MAE：

```python
mae = np.mean(np.abs(y_pred_raw - y_true_raw))  # shape: (n_samples, 2)
```

其中 `y_pred_raw` 和 `y_true_raw` 都必须在 `[1, 255]` 原始尺度。

## 当前项目建议结构

```text
.
+-- README.md
+-- model.py
+-- train_baseline.py
+-- train_distill.py
+-- configs/
|   +-- baseline.yaml
+-- data/
|   +-- MER_PS_codabench_public_trainval/
+-- official_starting_kit/
+-- emotion_merps/
|   +-- features.py
|   +-- model.py
|   +-- distill.py
+-- 多教师跨方式调整基础模型.pdf
+-- tools/
|   +-- inspect_data.py
|   +-- build_label_teacher_cache.py
|   +-- validate_predictions.py
|   +-- make_submission.py
+-- models/
|   +-- AffectGPT/
+-- submissions/
|   +-- asac_baseline/
|   +-- emotion_mtdp/
+-- experiments/
    +-- logs/
    +-- checkpoints/
```

当前项目已经包含官方 baseline 的本地改造版。下一步优先跑通 `train_baseline.py`，再接 AffectGPT/EEG/fNIRS teacher cache。

## 提交流程

训练完成后，将 checkpoint 打包为 Codabench zip：

```bash
python tools/make_submission.py --checkpoint experiments/checkpoints/asac_baseline/best_model.pt --output submissions/asac_baseline.zip
```

本地生成预测后，可校验提交文件：

```bash
python tools/validate_predictions.py --sample-ids path/to/sample_ids.csv --predictions path/to/predictions.csv
```

## 注意事项

- 代码提交入口必须是 zip 根目录的 `model.py`。
- `predict(input_dir, output_dir)` 只能读取测试输入信号和 `sample_ids.csv`。
- 输出必须是 `<output_dir>/predictions.csv`。
- 预测值必须是 `[1, 255]` 范围内的整数。
- EEG/fNIRS/annotation 采样率不同，窗口切分要用真实时间对齐。
- baseline 可以用于 trial-level correction，但不要把 baseline 段当作视频情绪标签训练。
- AffectGPT/情绪大模型更适合作为训练期教师或先验，不应让提交代码依赖外部服务。
- 数据集为非商业科研用途，需要遵守 CC BY-NC-SA 4.0 与组织方数据使用条款。

## 外部链接

- [MER Challenge 2026](https://zeroqiaoba.github.io/MER-Challenge/)
- [MER-PS train/validation dataset](https://huggingface.co/datasets/MER-PS/MER-PS-trainval)
- [AffectGPT paper](https://arxiv.org/abs/2501.16566)
- [MTDP EEG foundation model paper](https://arxiv.org/abs/2603.04478)
- [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
