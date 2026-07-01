# 外部数据集与审稿人视角评估

## 审稿人视角结论

如果我是二区以上期刊审稿人，只看到目前 MER-PS 内部 6 折结果，我不会直接认可 SCRF-BCRF 是充分成熟的论文贡献。

主要原因：

```text
1. 提升幅度太小：
   200 -> 218: 28.6912 -> 28.6869
   218 -> 222: 28.6869 -> 28.6868
   当前更像稳定后处理，不像强方法突破。

2. 模块还没有证明真正利用 EEG/fNIRS：
   当前强基线高度依赖 video/time prior。
   审稿人会质疑模型是否主要预测刺激平均情绪轨迹，而不是从脑信号解码个体情绪。

3. 创新边界容易被认为是经验校准：
   SCRF-BCRF 本质接近 residual calibration、empirical Bayes shrinkage、uncertainty-aware correction。
   如果没有外部数据集、消融和显著性检验，创新性不够硬。

4. arousal 不动虽然合理，但也暴露模块适用范围有限：
   当前模块主要改善 valence，且幅度很小。

5. 需要证明跨数据集泛化：
   只在一个比赛数据集上成立，容易被认为是 leaderboard-specific engineering。
```

更准确的论文定位应该是：

```text
不是“大模型情绪识别”；
不是“通用 EEG-fNIRS foundation model”；
而是：
  subject-disjoint physiological affect decoding 中的
  uncertainty-aware residual calibration / credible residual correction module。
```

## 这个工作属于什么方向

推荐论文方向表述：

```text
一级方向：
  Affective Computing
  Brain-Computer Interface
  Neurophysiological Emotion Recognition

二级任务：
  EEG/fNIRS-based emotion recognition
  physiological valence-arousal regression
  cross-subject affective decoding
  multimodal neural signal fusion

方法类别：
  subject-disjoint generalization
  residual calibration
  empirical-Bayes shrinkage
  uncertainty-aware post-hoc correction
  reliable low-dimensional affective prior modeling
```

如果要投论文，建议题目不要写得太大。更可信的题目类似：

```text
Sign-Consistent Credible Residual Calibration for Cross-Subject Physiological Affect Decoding
```

## 最值得补的公开数据集

### 1. REFED / MER-PS

相关性：最高，但它和当前比赛数据高度重合，不能算真正外部验证。

特点：

```text
EEG + fNIRS
continuous valence/arousal dynamic labels
64-channel EEG
51-channel fNIRS
15 emotion-inducing videos
```

适合任务：

```text
continuous valence-arousal regression
dynamic affective BCI
EEG-fNIRS multimodal fusion
```

用途：

```text
作为主实验数据集。
不能单独支撑跨数据集泛化。
```

### 2. ENTER: EEG-NIRS dataset TYUT emotion recognition

相关性：很高，公开申请制 EEG-fNIRS 情绪数据。

公开信息：

```text
50 subjects
64-channel EEG, 1000 Hz
18-channel fNIRS, 11 Hz
60 emotional video clips
four emotion categories: sad, happy, calm, fear
```

适合任务：

```text
EEG-fNIRS emotion classification
cross-subject classification
modality fusion
```

和 MER-PS 差异：

```text
ENTER 更偏离散分类；
MER-PS 是连续 valence/arousal 轨迹回归。
```

建议用途：

```text
把 SCRF-BCRF 改成 classification calibration：
  对每个类别 logit 做 credible residual correction；
  或对 valence-like / arousal-like 二分类做后验校准。
```

### 3. FEAD: fNIRS-EEG Affective Database

相关性：高，EEG-fNIRS 情绪数据库。

公开信息：

```text
37 participants
24 affective audio-visual stimuli
EEG + fNIRS
categorical and dimensional emotion ratings
```

适合任务：

```text
EEG-fNIRS affective recognition
valence/arousal classification or regression
multimodal fusion
```

建议用途：

```text
如果能申请到数据，这是最适合验证 SCRF-BCRF 的外部数据集之一。
```

### 4. FEEL: fNIRS-EEG Emotion Dataset and Benchmark Library

相关性：高，最新公开 fNIRS-EEG 情绪 benchmark。

适合任务：

```text
EEG-fNIRS emotion recognition
benchmark comparison
cross-subject evaluation
```

建议用途：

```text
优先确认是否能直接下载。
如果可下载，它比 DEAP/DREAMER 更贴近 MER-PS。
```

### 5. DEAP

相关性：中高，经典 EEG + 外周生理 valence/arousal 数据。

公开信息：

```text
32 participants
40 one-minute music videos
EEG + peripheral physiological signals
ratings: valence, arousal, dominance, liking, familiarity
```

适合任务：

```text
EEG-based valence/arousal classification
trial-level valence/arousal regression
cross-subject affective decoding
```

和 MER-PS 差异：

```text
没有 fNIRS；
标签是 trial-level，不是 1 Hz continuous trajectory。
```

建议用途：

```text
验证 SCRF-BCRF 是否能从 continuous regression 扩展到 trial-level VA regression。
```

### 6. DREAMER

相关性：中，EEG + ECG valence/arousal/dominance。

公开信息：

```text
23 participants
18 audio-visual stimuli
14-channel EEG, 128 Hz
ECG
ratings: valence, arousal, dominance
```

适合任务：

```text
small-sample physiological affect recognition
trial-level regression/classification
```

建议用途：

```text
作为小样本泛化测试。
如果 SCRF-BCRF 在 DREAMER 上仍有效，说明它不是 MER-PS 特化。
```

### 7. AMIGOS

相关性：中，EEG/ECG/GSR 多模态情绪数据。

公开信息：

```text
40 participants
short and long emotional videos
EEG, ECG, GSR
valence/arousal/dominance annotations
individual and group settings
```

适合任务：

```text
multimodal affect recognition
personality/mood-aware emotion modeling
cross-subject evaluation
```

建议用途：

```text
验证模块在 multimodal physiological fusion 上是否稳健。
```

### 8. MAHNOB-HCI

相关性：中，经典 multimodal emotion dataset。

公开信息：

```text
EEG + physiological signals + face/video/audio/eye gaze
emotion elicitation by videos/images
valence/arousal self-report
```

适合任务：

```text
multimodal emotion recognition
valence/arousal classification
physiological affective computing
```

建议用途：

```text
作为传统 benchmark 对照，但数据预处理成本较高。
```

### 9. FACED

相关性：中，EEG-only，但 subject 数量大。

公开信息：

```text
123 subjects
32-channel EEG
28 video clips
9 emotion categories
```

适合任务：

```text
cross-subject EEG emotion classification
large-subject affective computing
```

建议用途：

```text
验证模块是否能提升大 subject 数量下的跨主体分类校准。
```

### 10. EmoEEG-MC

相关性：中高，适合跨情境泛化。

公开信息：

```text
60 participants
64-channel EEG
GSR and PPG
video-induced and imagery-induced emotion contexts
7 emotion categories
valence/arousal subjective reports
```

适合任务：

```text
cross-subject emotion decoding
cross-context emotion decoding
EEG affective computing
```

建议用途：

```text
用来证明 SCRF-BCRF 不只适合跨 subject，还适合跨 context。
```

## 推荐实验路线

优先级：

```text
第一优先：
  FEEL / FEAD / ENTER
  原因：同属 EEG-fNIRS emotion recognition，和 MER-PS 最接近。

第二优先：
  DEAP / DREAMER / AMIGOS
  原因：下载和复现实验更成熟，适合快速补外部验证。

第三优先：
  FACED / EmoEEG-MC
  原因：适合做 cross-subject 或 cross-context 泛化，但任务从 regression 变成 classification。
```

建议先跑一个最容易落地的版本：

```text
DEAP trial-level valence/arousal regression:
  base = subject-disjoint video/stimulus prior + EEG feature model
  module = SCRF-BCRF residual calibration
  metric = MAE / RMSE / CCC
  ablation = no residual, SCRF, BCRF, SCRF-BCRF
```

然后再申请或下载 EEG-fNIRS 数据：

```text
ENTER or FEAD:
  base = EEG branch + fNIRS branch + fusion classifier/regressor
  module = credible residual/logit calibration
  metric = balanced accuracy / macro-F1 / MAE
```

## 最关键的审稿补强实验

```text
1. Signal ablation:
   video/time prior only
   EEG only
   fNIRS only
   EEG + fNIRS
   EEG + fNIRS + SCRF-BCRF

2. Split robustness:
   leave-subject-out
   repeated group k-fold
   cross-stimulus split if possible

3. Module ablation:
   no shrink
   no MAD gate
   no sign consistency
   no credible z-score
   no multi-view aggregation

4. Statistical testing:
   paired bootstrap confidence interval
   per-subject paired Wilcoxon or permutation test

5. Interpretability:
   residual field heatmap over time/stimulus/value bins
   confidence distribution
   cases where correction is accepted vs rejected
```

## 最终判断

```text
当前 SCRF-BCRF：
  比赛工程：可以继续用。
  论文模块：有雏形，但证据不足。

如果补上一个外部公开数据集 + 完整消融 + 显著性检验：
  可以成为一篇 Q2/Q3 水平方法论文的核心模块。

如果能在 EEG-fNIRS 外部数据集上稳定提升，并证明 physiological signal 对 residual confidence 有贡献：
  才有机会冲更高水平期刊。
```
