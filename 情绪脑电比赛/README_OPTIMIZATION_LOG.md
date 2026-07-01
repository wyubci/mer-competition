# MER-PS 优化逻辑与实验记录

更新时间：2026-06-15

## 当前验证范式

- 数据切分：核心结果使用 `test_1-test_24` 的 6 折 subject-disjoint 交叉验证，每折 4 个 subject。
- 指标：`overall_mae = mean(abs(pred - label))`，在原始 `[1,255]` 标签尺度上计算，越低越好。
- 当前任务：主要优化 valence 的可信残差校正；arousal 使用更保守的 conformal/projection 路径，因为多轮残差修正没有稳定超过它。
- 强基线：`VideoTimeMean`，按训练集同一视频、同一秒的平均标签预测验证集。

## 当前核心公式

我们不是直接预测标签，而是做残差学习：

```text
p0(t, v) = mean_train_label(video=v, time=t)
r*(t) = (y(t) - p0(t, v)) / 127
model_input = phi(EEG, fNIRS)
model_output = r_hat(t)
prediction = clip(p0(t, v) + alpha * 127 * r_hat(t), 1, 255)
prediction_final = smooth_per_trial(prediction, window=k)
```

## 221-228 追加观察：BCRF 可信残差场模块

这一轮的目标不是继续堆 backbone，而是把 218 的 SCRF 从经验规则推进成更适合论文表达的原创模块：

```text
BCRF: Bayesian Credible Residual Field
中文：贝叶斯可信残差场
定位：在 subject-disjoint 生理情绪预测中，只对可信的系统性残差做小幅校正。
```

### 模块动机

前 200 次实验给出三个稳定事实：

```text
1. video/time prior 极强，很多大模型只是在学习这个低维先验。
2. valence 存在可修正的系统性残差，arousal 的残差修正大多是噪声。
3. 大自由度模型容易过拟合，小幅、低自由度、带 shrink 的校准更稳。
```

因此 BCRF 不直接学习一个大函数 `f(signal)`，而是学习残差是否可信：

```text
base prediction:
  y_base = y_200 or y_218

training residual:
  r_i = y_i - p_i

conditional residual views:
  video-time
  video-value
  time-value
  value-slope
  video
  time
```

对每个残差桶估计中位残差、噪声和符号一致性：

```text
m_g   = median(r_i | i in group g)
MAD_g = median(|r_i - m_g|)
n_g   = sample count

se_g = 1.4826 * MAD_g / sqrt(n_g)
z_g  = |m_g| / (se_g + 1)
s_g  = |mean(sign(r_i))|

credible_gate_g = sigmoid(z_g - tau)
noise_gate_g    = 1 / (1 + MAD_g / c)
count_gate_g    = n_g / (n_g + k)

w_g = count_gate_g * s_g * credible_gate_g * noise_gate_g
c_g = w_g * m_g
```

对一个验证样本，把多个视角的残差校正合成：

```text
mean_corr = sum(w_h * c_h) / sum(w_h)
sign_agreement = |sum(w_h * sign(c_h))| / sum(w_h)
dispersion_gate = 1 / (1 + std(c_h) / d)

confidence = count_confidence * sign_agreement * dispersion_gate
delta = strength * confidence * clip(mean_corr)

y_new = y_base + delta
```

### 和 SCRF 的区别

```text
SCRF:
  只判断两个残差场方向是否一致。
  优点是简单、稳、当前实测更有效。

BCRF:
  进一步估计残差桶的可信度。
  它不仅问“方向是否一致”，还问“这个方向是否有足够样本、低噪声、强符号一致性和统计显著性”。
```

### 本轮结果

结果文件：

```text
experiments/results/iteration_221_228_bcrf_module_seed2026.json
```

实现文件：

```text
tools/cross_fold_bcrf_module.py
```

| 编号 | 方法 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| 222 | BCRF_onSCRF | 28.6868 | 26.8958 | 30.4777 | 当前数值最优，但只比 SCRF 好 0.0001，不能过度宣称。 |
| 218 | SCRF_reference | 28.6869 | 26.8961 | 30.4777 | 实用层面的稳健最优。 |
| 228 | BCRF_SCRF_plusArousal_probe | 28.6869 | 26.8961 | 30.4778 | arousal 继续不该动。 |
| 224 | BCRF_BrakeSCRFDisagreement | 28.6880 | 26.8983 | 30.4777 | 过度保守，收益下降。 |
| 221 | BCRF_valenceOnly | 28.6896 | 26.9014 | 30.4777 | BCRF 单独弱于 SCRF。 |
| 200 | CurrentManualFusion | 28.6912 | 26.9046 | 30.4777 | 原基线。 |

诊断：

```text
SCRF valence 平均收益:
  -0.008564 MAE

BCRF valence 平均收益:
  -0.003212 MAE

BCRF mean confidence:
  valence = 0.092669
  arousal = 0.080537

BCRF mean abs delta:
  valence = 0.053440
  arousal = 0.009833
```

### 结论

```text
1. BCRF 的理论结构比 SCRF 更完整，但单独使用时过于保守。
2. SCRF 是当前更强的实用模块，BCRF 适合作为 SCRF 的可信度外壳。
3. arousal 残差场依然不可靠，最终模型应冻结 arousal 的 conformal/projection 路径。
4. 222 可以作为当前提交候选，但论文中必须诚实说明：
   BCRF 相比 SCRF 的数值提升极小，主要贡献是可解释的可信残差建模框架。
```

### 可发表模块定位

推荐把论文模块命名为：

```text
SCRF-BCRF:
Sign-Consistent Bayesian Credible Residual Field

中文：
符号一致贝叶斯可信残差场
```

核心贡献可以写成：

```text
1. 从 200 组 subject-disjoint 实验中发现：
   valence 的可迁移信息主要表现为低维系统残差，而不是高容量 backbone。

2. 提出层级残差场：
   在 video/time/value/slope 条件空间中估计低自由度残差。

3. 提出符号一致约束：
   只有多个残差视角给出同向修正时才更新，抑制 fold-specific 噪声。

4. 提出可信度估计：
   使用样本数、MAD、符号一致性、近似 z-score 同时衡量残差桶是否可信。

5. 给出维度选择结论：
   valence 可校正，arousal 不宜继续残差修正。
```

论文风险也要写清楚：

```text
1. 当前提升很小，需要 bootstrap CI、per-fold 显著性和官方榜验证。
2. BCRF 当前更像可靠性建模，不是大幅提分模块。
3. 若要投较好期刊，需要增加外部数据集或至少增加跨切分稳定性实验。
```

下一步建议不再盲目新增模型，而是围绕这个原创模块补三类证据：

```text
1. bootstrap paired CI:
   证明 200 -> 218/222 的提升是否稳定。

2. residual field visualization:
   画出 video-time/value/slope 残差场，证明模块不是黑箱调参。

3. ablation:
   去掉 count shrink、MAD shrink、sign consistency、credible gate，逐个验证贡献。
```

## 229-239 追加观察：EEG-fNIRS 神经血氧融合第一轮

这一轮专门回到用户指出的问题：数据有 EEG 和 fNIRS 两种模态，不能只做 video/time prior 或输出级拼接。

现有 ASAC/GraphMamba 代码确实有多模态结构：

```text
EEG   -> graph encoder -> EEG representation
fNIRS -> graph encoder -> fNIRS representation

fusion choices:
  pool concat
  cross-modal attention
  modality gate
  local/global pooling
```

但前面实验说明，普通 flatten / concat / residual Ridge 没有稳定超过强先验。因此这一轮做新的融合模块，而不是简单拼接：

```text
229 EEGLagRidge:
  只用 EEG 的多秒 lag 特征。

230 FNIRSSlowRidge:
  只用 fNIRS 的慢变 rolling/lag 特征。

231 EarlyConcatLagRidge:
  EEG lag + fNIRS slow feature 早期拼接。

232 NeurovascularLagProductRidge:
  EEG 经 HRF-like delay kernel 后，与当前/慢变 fNIRS 做 product/diff/abs-diff。

233 HRFKernelFusionRidge:
  显式使用 EEG -> hemodynamic response kernel。

234 CoupledSlopeFusionRidge:
  EEG 快速斜率与 fNIRS 慢速斜率耦合。

235 LowRankBilinearPCA:
  EEG/fNIRS 分别 PCA 后做低秩双线性交互。

236 PLSJointLatent:
  用 PLS 学习与 residual 最大协方差的跨模态潜变量。

237 DualModalityAgreementGate:
  EEG 和 fNIRS 分别预测 residual，只在两者方向一致时保留。

238 CoherenceConfidenceGate:
  用 EEG-HRF 与 fNIRS rolling coherence 缩放 correction。

239 ModalityVarianceWeighted:
  按训练误差给 EEG/fNIRS 两个专家加权。
```

结果文件：

```text
experiments/results/iteration_229_239_neurovascular_fusion.json
```

预计算特征缓存：

```text
experiments/features/neurovascular_precompute_baseline.npz
```

第一轮全量结果：

| 编号 | 最佳方法 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| 237 | DualModalityAgreementGate | 28.7390 | 26.9595 | 30.5186 | 第一轮最佳；方向一致比普通 concat 更有效。 |
| 236 | PLSJointLatent | 28.7434 | 26.9681 | 30.5186 | 有效但弱于 agreement gate。 |
| 235 | LowRankBilinearPCA | 28.7443 | 26.9700 | 30.5186 | 低秩交互略有效。 |
| 098 | PatternPrior reference | 28.7462 | 26.9738 | 30.5186 | 强非信号先验。 |

关键结论：

```text
1. EEG/fNIRS 信号这次确实带来微小跨折增益：
   28.7462 -> 28.7390。

2. 最有效的不是 early concat、不是 PLS、不是 HRF product，
   而是 EEG/fNIRS 两个残差专家的 sign agreement。

3. 收益几乎全部来自 valence：
   valence 26.9738 -> 26.9595；
   arousal 30.5186 不变。

4. 这说明 fNIRS/EEG 的作用不是直接重建情绪曲线，
   而是作为“残差方向是否可信”的证据。
```

## 240-246 追加观察：NOVA-Gate 嵌套 OOF 神经血氧一致性门控

第一轮发现 `DualModalityAgreementGate` 有效，但它仍然是手写规则。第二轮把它升级为更严格的嵌套 OOF 融合：

```text
NOVA-Gate:
Nested OOF Neurovascular Agreement Gate
中文：嵌套 OOF 神经血氧一致性门控
```

核心思想：

```text
对每个 outer fold：
  对训练 subject 再做 inner leave-one-subject-out。

  EEG expert:
    r_eeg = f_eeg(EEG)

  fNIRS expert:
    r_fnirs = f_fnirs(fNIRS)

  用 inner OOF prediction 估计两个专家的可靠性：
    mse_eeg
    mse_fnirs

  计算维度权重：
    w_eeg = (1 / mse_eeg) / (1 / mse_eeg + 1 / mse_fnirs)

  只在方向一致时启用：
    agree = 1[sign(r_eeg) = sign(r_fnirs)]

  幅度一致性：
    ratio = min(|r_eeg|, |r_fnirs|) / max(|r_eeg|, |r_fnirs|)

  最终 residual:
    r = agree * (0.35 + 0.65 * ratio)
        * (w_eeg * r_eeg + (1 - w_eeg) * r_fnirs)
```

这一轮模块：

```text
240 OOFLinearStack_EEG_FNIRS
241 OOFLinearStack_EEG_FNIRS_NV
242 OOFAgreementWeighted
243 OOFDisagreementAwareStack
244 OOFHelpfulGateConsensus
245 NeurovascularHelpfulGate
246 ConsensusConfidenceShrink
```

结果文件：

```text
experiments/results/iteration_240_246_neurovascular_oof_gate.json
```

第二轮全量结果：

| 编号 | 方法 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| 242 | OOFAgreementWeighted | 28.7352 | 26.9518 | 30.5186 | 当前 EEG/fNIRS 融合最佳。 |
| 246 | ConsensusConfidenceShrink | 28.7353 | 26.9520 | 30.5186 | 几乎打平 242。 |
| 244 | OOFHelpfulGateConsensus | 28.7358 | 26.9529 | 30.5186 | helpful classifier 有效但略弱。 |
| 098 | PatternPrior reference | 28.7462 | 26.9738 | 30.5186 | 对照。 |
| 245 | NeurovascularHelpfulGate | 28.7484 | 26.9782 | 30.5186 | 单独 neurovascular gate 不够稳。 |
| 240 | OOFLinearStack_EEG_FNIRS | 28.7499 | 26.9811 | 30.5186 | 线性 stacking 过拟合/不稳。 |

相对 098 的增益：

```text
overall: 28.7462 -> 28.7352, gain = 0.0110
valence: 26.9738 -> 26.9518, gain = 0.0220
arousal: 30.5186 -> 30.5186, gain = 0.0000
```

这说明 NOVA-Gate 是目前第一个明确吃到 EEG/fNIRS 多模态信号的模块，比之前的 flattened ASAC Ridge residual 更有效。

但它还不是全局最好：

```text
当前全局最好:
  222_BCRF_onSCRF
  overall = 28.6868
  valence = 26.8958
  arousal = 30.4777

当前 EEG/fNIRS 融合最好:
  242_OOFAgreementWeighted
  overall = 28.7352
  valence = 26.9518
  arousal = 30.5186
```

尝试把 NOVA-Gate 叠加到 200/218/222 当前最强基座：

```text
脚本:
  tools/cross_fold_neurovascular_overlay_current_best.py

状态:
  90 分钟内未完成，主要耗时在重建 200/218/222 当前最强基座。

结论:
  这条验证需要轻量化 current-best base 的预测缓存后再跑。
```

当前判断：

```text
1. NOVA-Gate 比 SCRF-BCRF 更像真正的 EEG-fNIRS 融合创新模块。
2. 但从比赛效果看，它暂时只能作为 signal evidence module，不能替代 222。
3. 它的论文价值在于：
   用 EEG/fNIRS 两个模态的一致性判断 residual 是否可信，
   而不是把两种模态直接 concat 后交给模型自己学。
4. 下一步应缓存 200/218/222 的 OOF 预测，再测试：
   y_final = y_222 + lambda * NOVA_Gate(EEG, fNIRS)
```

这解释了目前结果的主要现象：

- `VideoTimeMean` 已经很强，说明视频内容和时间位置解释了大量标签方差。
- 生理模型只需要学习 `y - p0` 的残差，残差信号更小、更噪，因此模型参数不能太大。
- 很多单模型看起来只提升 0.1 左右 MAE，是因为它们只在强先验剩余的难样本上起作用。
- 后处理里的 `alpha` 不是作弊，而是残差幅度校准：模型训练目标是归一化残差，但最优标签尺度残差常常需要缩放。

## 模块作用与逻辑判断

| 模块 | 作用 | 有效/无效原因 |
| --- | --- | --- |
| `VideoTimeMean` | 视频-时间标签先验 | 捕获刺激材料引起的群体平均情绪轨迹，是当前最强非生理先验。 |
| baseline correction | `signal - mean(5s baseline)` | 可去除个体静态偏移，但也可能抹掉视频前情绪/生理基线差异，必须实测。 |
| EEG bandpower | 每秒 5 个频段相对功率 | 小数据下比原始 200Hz 序列稳健，但丢掉相位、跨频耦合和细粒度瞬态。 |
| fNIRS mean/std/slope | 每秒血氧统计 | 适合低频血氧变化，但 1 秒窗口可能太短，斜率噪声大。 |
| multiscale windows | `1/5/9` 秒平滑特征拼接 | 给模型同时看瞬时、短期、较慢变化；目前稳定有效。 |
| adaptive graph | 学习通道邻接 | 对 EEG/fNIRS 空间结构有帮助，但小数据下容易学到验证集不稳的图。 |
| hybrid functional graph | 静态图 + 样本功能连接 | 比纯 adaptive 更稳，能利用通道相关性。 |
| local/global pooling | 脑区摘要 + 全局摘要 | 单模型有增益，但和 iTransformer 叠加后容易冗余/过拟合。 |
| Gated SSM | 快速局部时序混合 | 稳定，适合小数据，是目前可靠底座。 |
| iTransformer-lite | 变量维注意力 | 单模型最强之一，能学习不同隐变量之间的关系，但方差大。 |
| TimeMixer/Fourier | 趋势/频域低频归纳偏置 | 对残差任务过平滑，通常不如 Gated SSM/iTransformer。 |
| Hybrid temporal gate | SSM/iTransformer/TimeMixer/Fourier 并联门控 | 比串联堆叠合理，但仍受小数据限制，未超过旧 best。 |
| modality dropout | 训练时随机丢 EEG/fNIRS | 对 SSM 类有鲁棒化帮助；对 iTransformer 反而削弱变量关系。 |
| ensemble | 多残差模型平均 | 有效条件是残差方向互补；不是越多越好，方向冲突会变差。 |

## 已跑主切分结果

| 方法 | Overall MAE | Valence MAE | Arousal MAE | 结论 |
| --- | ---: | ---: | ---: | --- |
| 官方 ASAC demo | 47.0087 | 49.2285 | 44.7890 | 主要是提交接口参考，不是强 baseline。 |
| Center128 | 46.5051 | - | - | 常数中心基线。 |
| VideoTimeMean | 29.9574 | 28.5041 | 31.4107 | 当前强先验。 |
| GraphMambaResidual zero-init | 29.8617 | 28.3342 | 31.3892 | 生理信号开始有增益。 |
| Hybrid functional graph | 29.8088 | 28.2284 | 31.3892 | 通道功能连接有效。 |
| Masked graph pretrain | 29.7995 | 28.2097 | 31.3892 | 预训练略有帮助。 |
| PatchTST-lite | 29.8382 | 28.2872 | 31.3892 | 可用但不突出。 |
| TimesNet-lite | 29.8394 | 28.2897 | 31.3892 | 可用但不突出。 |
| TimeMixer-lite | 29.8091 | 28.2291 | 31.3892 | 接近图模型，但不是最强。 |
| Fourier/FITS-lite | 29.8437 | 28.2978 | 31.3897 | 低频过强，残差被抹平。 |
| iTransformer-lite | 29.7457 | 28.1022 | 31.3892 | 当前最强单模型之一。 |
| `pool_local_global + iTransformer` | 29.8206 | 28.2520 | 31.3892 | 结构摘要和变量注意力冗余，未提升。 |
| `SSM -> iTransformer` | 29.8368 | 28.2844 | 31.3892 | 稳但偏平滑。 |
| `TimeMixer -> iTransformer` | 29.9460 | 28.4963 | 31.3957 | 串联堆叠失败，接近退化为先验。 |
| `pool_local_global + hybrid_temporal` | 29.8271 | 28.2650 | 31.3892 | 并联门控优于串联，但未超过旧 best。 |
| `moddrop + poollocal_hybrid` pair ensemble | 29.7446 | 28.1019 | 31.3872 | 有互补，但弱于 iTransformer 组合。 |
| `moddrop + SSM->iTransformer` pair ensemble | 29.7612 | 28.1353 | 31.3872 | 互补性一般。 |
| `moddrop + poollocal_iTransformer` pair ensemble | 29.9040 | 28.4189 | 31.3892 | 残差方向污染，不能拼。 |
| 当前最好：`moddrop010_seed123 + iTransformer_hybrid_159` weighted ensemble | 29.6336 | 27.8775 | 31.3897 | 当前主切分 best。 |

## 当前判断

单纯拼接模型已经接近瓶颈。有效逻辑不是“模块越多越好”，而是：

```text
强视频时间先验
+ 小参数生理残差模型
+ 正确的通道/时间归纳偏置
+ 残差幅度校准
+ 少量互补 ensemble
```

下一轮优先级：

1. 数据预处理：baseline correction、fNIRS signal type、feature norm、多尺度窗口。
2. decoder/连接部分：残差头、低秩 residual adapter、valence/arousal 分头、残差幅度可学习校准。
3. 模型结构：只保留确实互补的模块，不继续盲目堆叠。

## 待跑实验

| 编号 | 方向 | 设置 | 目的 |
| --- | --- | --- | --- |
| 059 | 数据预处理 | fNIRS 使用 0-5 全部 signal types | 检查 Abs 波长信息是否提供额外生理线索。已完成 cache/Ridge sanity check，见下表。 |
| 060 | 数据预处理 | 关闭 baseline correction | 检查 baseline 相减是否抹掉情绪前状态信息。 |
| 061 | 数据预处理 | iTransformer + subject/trial feature norm | 检查归一化是否降低跨主体差异。 |
| 062 | decoder | 更细 residual scale/clip/smooth 局部搜索 | 检查后处理是否还能榨出稳定收益。 |

## 追加运行记录

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 059-a | 全 6 类 fNIRS cache + Ridge sanity check | `experiments/results/iteration_059_feature_allfnirs_ridge.json` | 29.9574 | 28.5041 | 31.4107 | 最优仍是 `VideoTimeMean`；线性 Ridge 使用全 fNIRS 特征明显变差，说明 Abs 波长特征直接展平会引入大量噪声，需要图/门控模型筛选。 |
| 059-b | 全 6 类 fNIRS + hybrid functional graph + Gated SSM | `experiments/results/iteration_059_allfnirs_gatedssm_159.json` | 29.8482 | 28.3073 | 31.3892 | 图模型可以筛掉部分噪声，但仍弱于默认 HbO/HbR/HbT 三类 fNIRS 的 29.8088；当前数据量下直接加入 Abs 780/805/830 会增加维度和过拟合风险。 |
| 060-a | 关闭 baseline correction cache + Ridge sanity check | `experiments/results/iteration_060_feature_nobase_ridge.json` | 29.9574 | 28.5041 | 31.4107 | 最优仍是 `VideoTimeMean`；线性残差模型比全 fNIRS 类型更好一些，但依旧不能有效利用生理特征。 |
| 060-b | 关闭 baseline + hybrid functional graph + Gated SSM | `experiments/results/iteration_060_nobase_gatedssm_159.json` | 29.8063 | 28.2234 | 31.3892 | 略好于 baseline-on 同类模型 29.8088，说明 baseline 相减不是稳定收益，可能会损失一部分个体/视频前状态信息。 |

### 059 结论

fNIRS 全类型输入的理论形式是：

```text
x_fnirs = [mean, std, slope] x [HbO, HbR, HbT, Abs780, Abs805, Abs830]
```

维度从 `51 x 9` 增到 `51 x 18`，多尺度后从 `51 x 27` 增到 `51 x 54`。在只有 20 个训练主体时，额外 Abs 特征提高了输入自由度，却没有明显提供与 `valence residual` 稳定相关的信息。后续如果继续用 Abs 信息，应该做低秩投影或门控选择，而不是直接拼接。

### 060-a 观察

关闭 baseline correction 后：

```text
baseline-on:  x = phi(signal - mean(baseline))
baseline-off: x = phi(signal)
```

Ridge sanity check 没有超过 `VideoTimeMean`，但 residual Ridge 从全 fNIRS 的 36.42 降到 34.23，说明“少加噪声”比“多加原始波长类型”更重要。是否优于 baseline-on 需要看图模型结果。

### 060-b 结论

图模型下关闭 baseline 后最好为 `scale1.00_clip5_smooth5 = 29.8063`。差异很小，但方向有意义：

```text
如果 baseline 表示纯噪声/漂移：signal - baseline 应该更好
如果 baseline 表示被试进入视频前的真实生理状态：signal - baseline 可能删掉可解释的个体状态
```

当前结果偏向第二种：baseline correction 不是必须，后续强模型值得同时保留 baseline-on 和 baseline-off 两个分支做 ensemble。
## 060-c 追加观察：无 baseline + iTransformer

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 060-c | 关闭 baseline + hybrid functional graph + iTransformer-lite | `experiments/results/iteration_060_nobase_itransformer_159.json` | 29.7772 | 28.1652 | 31.3892 | 明显好于无 baseline 的 Gated SSM 29.8063，也好于 baseline-on 的 iTransformer 29.7457? 否，仍弱于 baseline-on iTransformer；但它证明了无 baseline 分支有独立可用残差信号，适合进入 ensemble。 |

后处理最优为：

```text
y_hat = VideoTimeMean + smooth_5( clip(3.0 * r_theta(x), -5, 5) )
```

这个式子很重要。它说明模型不是在直接重建完整情绪曲线，而是在估计相对 `VideoTimeMean` 的小幅、方向性 residual。由于训练主体只有 20 个，若不限制 residual：

```text
MAE(y_base + r_theta, y) > MAE(y_base + clip(alpha * r_theta, -c, c), y)
```

通常意味着 `r_theta` 同时包含有效生理校正项和主体噪声项。`clip` 相当于给 residual 加了先验：生理信号只允许小幅修正强时间先验，不能把视频均值曲线推翻。这也是后面 decoder 设计的核心，应该做“受约束残差解码”，不是简单拼接大模型。
## 061-a 追加观察：iTransformer + subject norm

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 061-a | baseline-on cache + iTransformer-lite + `feature_norm=subject` | `experiments/results/iteration_061_itransformer_subjectnorm_159.json` | 29.8654 | 28.3415 | 31.3892 | 明显弱于默认 iTransformer 29.7457，说明主体级绝对生理偏置不能简单视为噪声。 |

公式上，subject norm 做的是：

```text
x'_s,t = (x_s,t - mean_t(x_s,t)) / std_t(x_s,t)
```

如果跨主体差异主要是设备漂移，这应该提升泛化；但结果下降，说明 `mean_t(x_s,t)` 中有一部分是情绪可解释的个体状态或反应强度。后续不应在主干里强行 subject de-mean，更适合让模型通过轻量门控自己判断哪些通道/尺度要去主体偏置。
## 061-b 追加观察：iTransformer + trial norm

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 061-b | baseline-on cache + iTransformer-lite + `feature_norm=trial` | `experiments/results/iteration_061_itransformer_trialnorm_159.json` | 29.8496 | 28.3099 | 31.3892 | 比 subject norm 略好，但仍弱于默认 iTransformer 29.7457。试次内强归一化也会洗掉视频诱发的慢变状态。 |

trial norm 做的是：

```text
x'_{s,v,t} = (x_{s,v,t} - mean_t(x_{s,v,t})) / std_t(x_{s,v,t})
```

它会突出试次内相对波动，却压低整段视频刺激导致的低频水平位移。MER-PS 标签是 1 Hz 连续轨迹，`VideoTimeMean` 已经证明低频视频时间先验很强，因此过强 trial de-mean 会与标签主结构冲突。更合理的方向是保留：

```text
x_abs = phi(signal)
x_rel = phi(signal - local_mean)
gate = sigmoid(W[x_abs, x_rel])
h = gate * h_abs + (1 - gate) * h_rel
```

也就是让模型学习“什么时候用绝对状态，什么时候用相对变化”，而不是手工只保留其中一个。
## 062 追加观察：受约束 residual ensemble

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 062 | `moddrop010_seed123`、`itransformer_hybrid_159`、`nobase_itransformer_159`、`nobase_gatedssm_159` 的加权/等权 residual ensemble，新增 residual clip 搜索 | `experiments/results/iteration_062_constrained_residual_ensemble.json` | 29.6149 | 27.8402 | 31.3897 | 刷新主切分 best。有效组合仍是 `moddrop010_seed123 + itransformer_hybrid_159`；两个 no-baseline 分支没有进入前排，说明它们的信息和主 best 不互补或噪声更大。 |

最优连接形式：

```text
r = 0.72 * r_moddrop + 0.28 * r_itransformer
y_hat = smooth_9( VideoTimeMean + clip(10.0 * r, -10, 10) )
```

为什么它有效：

```text
r_moddrop       : 训练时随机丢模态，降低对某一模态噪声的依赖，残差更保守
r_itransformer : 在 channel/feature token 维度做注意力，能找变量间的协同修正方向
clip            : 防止 10x 放大后残差覆盖 VideoTimeMean 强先验
smooth_9        : 标签是 1 Hz 连续轨迹，短时尖峰多半是模型噪声而非真实情绪跳变
```

这轮的负结果同样有价值：`nobase_itransformer` 单模型在 `clip` 后可用，但加入 ensemble 没有超过两模型组合。说明“关闭 baseline correction”不是一个稳定互补源，更可能是同一类 valence residual 的弱变体。后续重点应转向 arousal 分支和可学习 residual gate，而不是继续堆更多 valence 模型。
## 063 追加观察：无 baseline arousal 专门模型

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 063 | no-baseline cache + iTransformer-lite + `target_mode=arousal` | `experiments/results/iteration_063_nobase_itransformer_arousal_159.json` | 29.9391 | 28.5030 | 31.3753 | arousal 比 `VideoTimeMean_smooth5` 的 31.3892 小幅改善，但 valence 保持先验，因此整体不如 valence 模型。 |

这说明 arousal 不是完全不可学，但信号极弱。合理连接方式不是把 arousal 模型和 valence 模型整体平均，而是分维度融合：

```text
y_valence = prior_valence + g_v * r_valence_model
y_arousal = prior_arousal + g_a * r_arousal_model
g_v != g_a
```

原因是两个维度的可学习性不同：valence residual 可以被放大到 `scale=10` 后再 clip；arousal residual 只在 `scale=0.25` 附近有微弱收益，放大后迅速变差。后续应使用 dimension-wise residual gate，而不是一个共享 ensemble 权重。
## 064 追加观察：dimension-wise residual fusion

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 064-a | 分维度融合大搜索 | `experiments/results/iteration_064_dimwise_fusion.json` | - | - | - | 搜索空间过大，后处理重复平滑导致 20 分钟超时；需要缓存平滑结果或缩小搜索。 |
| 064-b | 分维度融合窄搜索：valence 使用当前 best 连接，arousal 小 scale 搜索 | `experiments/results/iteration_064_dimwise_fusion_narrow.json` | 29.6023 | 27.8402 | 31.3645 | 刷新 best。说明 arousal 小残差可以和 valence 强残差共存，但必须分维度控制。 |

最优公式：

```text
r_v = 0.72 * r_moddrop010_seed123 + 0.28 * r_itransformer_hybrid
y_v = smooth_9( prior_v + clip(10.0 * r_v, -10, 10) )

r_a = r_nobase_itransformer_arousal
y_a = smooth_9( prior_a + 0.25 * r_a )
```

这比共享权重 ensemble 更合理，因为两个维度的信噪比不同：

```text
valence:  residual direction stronger, needs large scale + clip
arousal:  residual direction weak, only tiny scale helps
```

所以后续新模型不应该只有一个 shared regression head。更合理的 decoder 是：

```text
h = encoder(eeg, fnirs)
r_v = head_v(h)
r_a = head_a(h)
alpha_v, clip_v = decoder_prior_v(h)
alpha_a, clip_a = decoder_prior_a(h)
y = prior + [clip(alpha_v * r_v), clip(alpha_a * r_a)]
```

其中 `alpha_v` 和 `alpha_a` 不能共享。这个结果也解释了之前很多拼接模型无效：它们增强了 valence residual，但同时把 arousal 噪声一起放大，overall 被抵消。
## 065 追加观察：arousal residual scale 细扫

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 065 | 固定 valence 最优连接，细扫 arousal scale `0.15-0.40` | `experiments/results/iteration_065_dimwise_arousal_fine.json` | 29.6009 | 27.8402 | 31.3615 | 当前主切分 best。arousal 小残差确实有用，但收益上限很低。 |

当前最优：

```text
r_v = 0.72 * r_moddrop010_seed123 + 0.28 * r_itransformer_hybrid
y_v = smooth_9( prior_v + clip(10.0 * r_v, -10, 10) )

r_a = r_nobase_itransformer_arousal
y_a = smooth_9( prior_a + 0.35 * r_a )

Overall MAE = (MAE_v + MAE_a) / 2
            = (27.8402 + 31.3615) / 2
            = 29.6009
```

这轮最终确认了两个设计原则：

1. `VideoTimeMean` 是强先验，生理模型应该预测 residual，不应该直接预测完整标签。
2. valence 和 arousal 的 residual 强度不同，必须分头校准；共享 scale/clip 会让 arousal 噪声抵消 valence 收益。

下一步最值得做的是把目前的手工公式做成可学习 decoder：

```text
alpha_v = softplus(w_v^T h + b_v)
alpha_a = softplus(w_a^T h + b_a)
r_v, r_a = head_v(h), head_a(h)
y_v = prior_v + clip(alpha_v * r_v, -c_v, c_v)
y_a = prior_a + clip(alpha_a * r_a, -c_a, c_a)
```

但参数量必须小。当前训练主体只有 20 个、训练 trial 300 个，适合的是 2e4 到 2e5 级别的 residual decoder，而不是千万级大模型端到端微调。大模型/AffectGPT 更适合做 teacher prior 或文本/视频情绪先验蒸馏，不适合直接作为 EEG/fNIRS 主干。
## 066 追加观察：DPRG 分维度先验残差门控

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 066 | DPRG：分别搜索 valence/arousal 的 residual 权重、scale、clip、smooth；arousal 使用 3 个弱教师残差 | `experiments/results/iteration_066_dprg_search.json` | 29.5921 | 27.8229 | 31.3613 | 当前 best。数学上利用 `overall_mae=(mae_v+mae_a)/2` 将搜索拆成两个独立子问题，避免 valence/arousal 参数互相污染。 |

DPRG 的目标函数：

```text
prior_d(t) = E_train[y_d | video, t]
r_{k,d}(t) = f_{k,d}(x_t) - prior_d(t)

min over theta_v, theta_a:
  L = 1/2 * MAE(y_v, prior_v + g_v(r_v; theta_v))
    + 1/2 * MAE(y_a, prior_a + g_a(r_a; theta_a))

因为 theta_v 和 theta_a 不共享，所以：
  argmin L = {argmin L_v, argmin L_a}
```

最优配置：

```text
valence:
  r_v = 0.725 * r_moddrop010_seed123 + 0.275 * r_itransformer_hybrid
  y_v = smooth_9(prior_v + clip(11.5 * r_v, -10, 10))
  MAE_v = 27.8229

arousal:
  r_a = 0.3 * r_nobase_itransformer_arousal + 0.7 * r_scalegated_msgm_arousal
  y_a = smooth_9(prior_a + 0.2 * r_a)
  MAE_a = 31.3613

overall = (27.8229 + 31.3613) / 2 = 29.5921
```

这轮把“多教师跨模态调整”的思想落成了一个很小的可靠性门控：每个 teacher/student 残差不是直接平均，而是按维度学习可信度。它借鉴但不照搬大模型蒸馏，因为 MER-PS 当前训练主体只有 20 个，端到端训练大参数 teacher-student 容易过拟合。

和近期论文模块的对应关系：

| 论文/模块 | 可借鉴点 | 在当前数据上的落地方式 |
| --- | --- | --- |
| MSGM / multi-scale graph Mamba | 多窗口时序 + 图结构 + Mamba 线性复杂度 | 已用多尺度 `1,5,9`、hybrid functional graph、scale-gated 分支；继续扩大主干收益有限。 |
| iTransformer | 把变量/通道当 token 建模跨变量依赖 | 当前 valence 最强残差来源之一，保留为 teacher/residual expert。 |
| Mambaformer / hybrid SSM-attention | SSM 建模长程平滑，attention 建模变量交互 | 当前实验显示串联堆叠容易过平滑，适合做并联 expert 而不是更深主干。 |
| OMCRD/contrastive distillation | 多学生/多层对比蒸馏迁移跨主体知识 | 可以作为下一步训练损失，但输出层仍需 DPRG 分维度校准。 |

下一步最优先不是更大模型，而是把 DPRG 固化进最终提交推理：

```text
1. 使用 trainval 训练/保存各 residual experts
2. 在内部 subject-disjoint validation 上学习 DPRG 标量
3. test 推理时只加载 experts + DPRG 标量
4. 输出 raw [1,255] integer predictions
```
## 067 追加观察：TBCR 试次级余弦基残差回归

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 067-a | TBCR：baseline-on cache，trial mean/std 特征预测低频 residual cosine coefficients | `experiments/results/iteration_067_tbcr_meanstd.json` | 29.7319 | 28.0241 | 31.4396 | 不是新 best，但证明 trial 级低频残差可以从整段生理特征中预测；arousal 不稳是主要短板。 |

TBCR 的新假设是把逐秒预测：

```text
y(t) = f(x_t)
```

改成低维轨迹预测：

```text
y_d(t) = prior_d(video,t) + sum_{k=0}^{K-1} c_{d,k} cos(pi * (t + 0.5) * k / T)
c = Ridge(Phi_trial)
```

这里 `Phi_trial` 是整段 EEG/fNIRS 的 mean/std 聚合特征。这个方法和之前所有 sample-wise residual 模型都不同：它强制残差在 trial 内低频平滑，显著降低输出自由度。结果没有超过 DPRG，说明目前最好信息仍来自每秒局部 residual expert；但 TBCR 可以作为后续 ensemble 的“低频专家”，尤其用于修正 trial-level offset。
## 068 追加观察：no-baseline TBCR

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 068 | TBCR：no-baseline cache，trial mean/std 特征预测低频 residual cosine coefficients | `experiments/results/iteration_068_tbcr_nobase_meanstd.json` | 29.8516 | 28.3232 | 31.3800 | 明显弱于 baseline-on TBCR 29.7319。试次级低频模型更需要去掉 baseline 漂移。 |

这个结果补充了预处理判断：

```text
sample-wise residual expert:
  no-baseline 有时可用，因为每秒局部变化保留了个体初始状态

trial-basis coefficient regression:
  baseline-on 更好，因为整段低频系数容易被主体漂移污染
```

所以后续如果做双分支预处理，不应简单选一个版本，而应按专家类型决定：

```text
local residual expert      -> 可保留 no-baseline 分支
low-frequency trial expert -> 优先 baseline-subtracted 分支
```
## 069 追加观察：L-DPRG 残差时间滞后搜索

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 069 | L-DPRG：在 DPRG 残差上按 trial 搜索 valence/arousal 独立 lag `[-12,12]` | `experiments/results/iteration_069_lagged_dprg.json` | 29.5823 | 27.8036 | 31.3610 | 当前 best。valence 最优 lag=-8，arousal lag 几乎不敏感。 |

L-DPRG 的假设：

```text
y_d(t) = prior_d(t) + g_d(r_d(t - tau_d))
```

脚本定义：

```text
tau > 0: 使用更早的 residual r(t - tau)
tau < 0: 使用更晚的 residual r(t - tau)
```

最优 `tau_v=-8`，说明当前 valence residual 与标签存在约 8 秒相位差；把 residual 往前对齐后，valence MAE 从 DPRG 的 27.8229 降到 27.8036。arousal lag 的 top 区间非常平，说明 arousal residual 太弱，时间对齐收益被噪声淹没。

这给出一个新的建模方向：以后主模型不应该只看 `x_t`，而应该显式建模滞后窗口：

```text
r_v(t) = sum_{tau=-K}^{K} a_tau * f_v(x_{t-tau})
sum_tau a_tau = 1
a_tau = softmax(q^T h_tau)
```

也就是可学习 temporal alignment / lag attention。它比继续加深 Mamba 更有针对性，因为目前瓶颈不是模型容量，而是生理信号与主观标注的时间对齐。
## 070 追加观察：TLRC 原始特征滞后 Ridge

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 070 | TLRC：直接拼接原始 EEG/fNIRS 特征的多 lag 窗口，用 Ridge 预测 valence residual | `experiments/results/iteration_070_tlrc_valence.json` | 29.9350 | 28.4807 | 31.3892 | 没有超过强模型，只略好于 `VideoTimeMean_smooth5`。raw lag 特征线性模型无法稳定解码 residual。 |

TLRC 公式：

```text
r_v(t) = W [x(t+12), x(t+8), x(t+4), x(t), x(t-4)] + b
y_v(t) = prior_v(t) + clip(scale * r_v(t))
```

这个实验是 L-DPRG 的反向验证：L-DPRG 说明“已经学出来的 residual”存在时间错位；TLRC 说明“原始特征 + 线性 lag”还不足以学出强 residual。结论是：

```text
时间滞后确实重要
但滞后模块应该放在非线性 expert / residual 之后
而不是直接对原始高维特征做线性滞后回归
```

后续如果做可学习 lag attention，应该使用 encoder hidden states 或 expert residuals：

```text
good: r(t) = sum_tau a_tau * expert(x_{t-tau})
weak: r(t) = W concat_tau x_{t-tau}
```
## 071 追加观察：PA-BS 先验感知双边平滑

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 071 | PA-BS：在 L-DPRG 输出后按 trial 做 prior-aware bilateral smoothing | `experiments/results/iteration_071_prior_aware_smoothing.json` | 29.5553 | 27.7620 | 31.3485 | 当前 best。说明后处理不是简单平滑，而要避免跨越情绪先验快速变化段。 |

PA-BS 的核心不是继续堆模型，而是把解码后的轨迹当作带噪观测：

```text
z_d(t) = prior_d(video,t) + residual_correction_d(t)
y_d(t) = sum_j w_ij z_d(j) / sum_j w_ij

w_ij =
  exp(-(i-j)^2 / (2 * sigma_t^2))
  * exp(-(prior_d(i)-prior_d(j))^2 / (2 * sigma_p^2))
```

普通 moving average 只看时间距离，因此窗口变大后会把情绪转折抹平；这一点在 `MovingAverage_window17 = 29.6637` 上很明显。PA-BS 加了先验轨迹距离后，`window=17, sigma_prior=10` 反而成为最优，说明当前预测误差中有一部分来自局部噪声，而不是模型必须逐秒自由抖动。

这轮给出的建模经验：

```text
有用：
  residual expert + lag alignment + prior-aware smoothing

无效或偏弱：
  直接扩大普通平滑窗口
  直接对原始高维特征做线性 lag 回归
  试图用大模型容量硬吃小数据
```

当前最优配置：

```text
L-DPRG:
  valence lag = -8
  arousal lag = -12
  valence scale = 11.5
  arousal scale = 0.2

PA-BS:
  window = 17
  sigma_prior = 10

MAE:
  overall = 29.5553
  valence = 27.7620
  arousal = 31.3485
```
## 072 追加观察：CDG 共识驱动 residual 门控

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 072 | CDG + PA-BS：用多个 residual expert 的分歧程度动态控制修正幅度 | `experiments/results/iteration_072_consensus_gate.json` | 29.5328 | 27.7183 | 31.3474 | 当前 best。说明“专家是否一致”本身是有效置信信号。 |

CDG 的动机是避免盲目放大生理 residual：

```text
base(t) = sum_i w_i r_i(t)
disagreement(t) = mean_pairwise_abs(r_i(t) - r_j(t))
confidence(t) = exp(-disagreement(t) / sigma)
gate(t) = min_gate + (max_gate - min_gate) * confidence(t)
y(t) = prior(t) + scale * gate(t) * base(t)
```

最优参数：

```text
valence:
  sigma_multiplier = 4
  min_gate = 0
  max_gate = 1.5
  sign_penalty = 1
  confidence_mean = 0.7578
  gate_mean = 1.1367
  MAE_v = 27.7183

arousal:
  sigma_multiplier = 2
  min_gate = 0.5
  max_gate = 1.5
  sign_penalty = 0.25
  confidence_mean = 0.6068
  gate_mean = 1.1068
  MAE_a = 31.3474

overall = 29.5328
```

这轮最关键的结论：valence 的 residual 不是纯噪声，过去 DPRG 的固定 scale 偏保守；当 `moddrop010_seed123` 与 `iTransformer_hybrid` 比较一致时，应该允许更强的修正。arousal 仍然很难，CDG 只从 31.3485 推到 31.3474，说明当前 arousal 的主要瓶颈不是解码后端，而是可用生理表征/标签噪声。

对后续框架的启发：

```text
encoder expert 负责产生候选 residual
lag alignment 负责解决标注相位差
consensus gate 负责判断 residual 是否可信
prior-aware smoothing 负责去掉局部噪声且保留情绪转折
```
## 073 追加观察：SCDG 斜率状态条件门控

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 073 | SCDG + PA-BS：根据 prior 轨迹斜率把每秒分成 stable/rising/falling，再分别缩放 residual | `experiments/results/iteration_073_slope_conditioned_gate.json` | 29.4954 | 27.6844 | 31.3065 | 当前 best。说明情绪变化状态是有效的解码条件。 |

SCDG 在 072 的 CDG 上继续加一个状态门：

```text
slope(t) = d prior(t) / dt

state(t) =
  rising   if slope(t) > threshold
  falling  if slope(t) < -threshold
  stable   otherwise

y(t) = prior(t)
     + scale * CDG_residual(t) * state_multiplier[state(t)]
```

最优参数：

```text
valence:
  slope_quantile = 50
  threshold = 0.625
  stable_multiplier = 1.25
  rising_multiplier = 1.10
  falling_multiplier = 0.75
  MAE_v = 27.6844

arousal:
  slope_quantile = 50
  threshold = 0.7000
  stable_multiplier = 0.75
  rising_multiplier = 0.75
  falling_multiplier = 1.50
  MAE_a = 31.3065

overall = 29.4954
```

这轮结论很有价值：模型不是在所有时间点都同样可信。valence 在稳定段可以更相信 residual，在 falling 段要明显收缩；arousal 则相反，只有 falling 段的 residual 有明显贡献，stable/rising 段更像噪声。这个模式解释了为什么前面单纯加大模型、加图、加 Mamba 收益有限：它们默认同一个解码规则覆盖所有情绪状态，而 MER-PS 的标注和生理反应明显是状态相关的。

后续值得继续尝试：

```text
state-conditioned lag:
  stable/rising/falling 分别学习不同 lag

state-conditioned expert:
  不同状态选择不同 teacher/expert 权重

state-aware training loss:
  falling/rising/stable 分段加权，而不是全局 MAE/MSE
```
## 074 追加观察：SCL 状态条件 lag alignment

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 074 | SCL + SCDG + PA-BS：stable/rising/falling 分别搜索 lag | `experiments/results/iteration_074_state_conditioned_lag.json` | 29.4356 | 27.5660 | 31.3052 | 当前 best。说明固定 lag 是明显瓶颈，尤其是 valence。 |

SCL 把 073 的状态缩放继续推广到状态条件时间对齐：

```text
state(t) in {stable, rising, falling}

r_state(t) =
  CDG_residual(t - tau_stable)   if stable
  CDG_residual(t - tau_rising)   if rising
  CDG_residual(t - tau_falling)  if falling

y(t) = prior(t) + scale * state_multiplier[state(t)] * r_state(t)
```

最优参数：

```text
valence:
  slope_quantile = 45
  threshold = 0.5250
  stable_lag = -2
  rising_lag = 0
  falling_lag = -14
  stable_multiplier = 1.25
  rising_multiplier = 1.10
  falling_multiplier = 0.75
  MAE_v = 27.5660

arousal:
  slope_quantile = 50
  threshold = 0.7000
  stable_lag = -16
  rising_lag = -12
  falling_lag = -10
  stable_multiplier = 0.75
  rising_multiplier = 0.75
  falling_multiplier = 1.50
  MAE_a = 31.3052

overall = 29.4356
```

这轮最重要的算法结论：

```text
valence:
  rising 几乎不需要滞后，lag=0
  stable 只需要轻微对齐，lag=-2
  falling 需要很强的未来 residual 对齐，lag=-14

arousal:
  lag 搜索收益很小，状态缩放比状态 lag 更重要
```

所以后续真正值得进入模型结构的创新不是“再堆 Mamba 层”，而是一个可学习的状态条件 temporal alignment 模块：

```text
h_t = encoder(EEG, fNIRS)
s_t = state_detector(prior_slope or hidden_slope)
a_{t,tau} = softmax(q_s^T h_{t-tau})
r_t = sum_tau a_{t,tau} expert(h_{t-tau})
y_t = prior_t + gate_s * r_t
```

这也解释了为什么当前后处理能大幅提升：它补上的是标注-生理相位错位，而不是单纯平滑噪声。
## 075 追加观察：SCEW 状态条件 expert 权重

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 075 | SCEW + SCL + PA-BS：在 stable/rising/falling 三个状态内分别搜索 expert residual 权重 | `experiments/results/iteration_075_state_expert_weights.json` | 29.3467 | 27.4159 | 31.2776 | 当前 best。说明不同情绪状态不仅 lag 不同，适合的 teacher/expert 也不同。 |

SCEW 在 074 的状态条件 lag 基础上继续加状态条件 expert 选择：

```text
r_state(t) =
  gate_state(t) *
  (w_state * expert_1(t - tau_state)
   + (1 - w_state) * expert_2(t - tau_state))

y(t) = prior(t) + scale * state_multiplier[state(t)] * r_state(t)
```

最优参数：

```text
valence:
  expert_1 = moddrop010_seed123
  expert_2 = itransformer_hybrid_159
  stable_first_expert_weight = 0.75
  rising_first_expert_weight = 0.10
  falling_first_expert_weight = 0.90
  MAE_v = 27.4159

arousal:
  expert_1 = nobase_itransformer_arousal_159
  expert_2 = scalegated_msgm_arousal
  stable_first_expert_weight = 1.00
  rising_first_expert_weight = 1.00
  falling_first_expert_weight = 0.00
  MAE_a = 31.2776

overall = 29.3467
```

这轮比 074 又降低了 `0.0889` overall MAE，说明之前固定 DPRG 权重仍然把不同状态混在一起了。解释上也很清楚：

```text
valence:
  stable 更适合 moddrop/graph-style expert
  rising 更适合 iTransformer expert
  falling 更适合 moddrop/graph-style expert

arousal:
  stable/rising 更适合 no-baseline iTransformer arousal
  falling 更适合 scale-gated MSGM arousal
```

这说明后续如果做真正模型结构，应该是状态路由，而不是普通 ensemble：

```text
state_router(t) -> {state, lag, expert_weight, residual_gate}
```

注意：075 使用了两阶段搜索，先用未平滑 MAE 快筛，再对候选做 PA-BS 精排。因此它适合指导方向，但最终提交前仍要做 subject-disjoint 多折验证，避免把验证集状态阈值和权重调得过细。
## 076 追加观察：SCEW 状态 expert 权重细搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 076 | SCEW fine：围绕 075 最优状态权重做局部细网格搜索 | `experiments/results/iteration_076_state_expert_weights_fine.json` | 29.3455 | 27.4134 | 31.2776 | 当前 best，但只比 075 小幅提升。说明状态 expert 路由是主要收益，继续细调权重收益很小。 |

细搜后的最优参数：

```text
valence:
  stable_first_expert_weight = 0.750
  rising_first_expert_weight = 0.175
  falling_first_expert_weight = 0.925
  MAE_v = 27.4134

arousal:
  stable_first_expert_weight = 1.000
  rising_first_expert_weight = 1.000
  falling_first_expert_weight = 0.000
  MAE_a = 31.2776

overall = 29.3455
```

076 的提升只有 `0.0012` overall MAE，所以后续不应该继续在 expert 权重小数点上耗太多时间。更有价值的方向是：

```text
1. 在当前 SCEW residual 上重新搜索 PA-BS 平滑参数
2. 做 state-conditioned smoothing，不同状态使用不同平滑强度
3. 用多折 subject-disjoint 验证检查 073-076 是否过拟合
```
## 077 追加观察：SCEW 后的 PA-BS 平滑参数重搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 077 | SCEW + PA-BS smoothing search：固定 076 的状态路由，重新搜索每个维度的平滑窗口和 prior sigma | `experiments/results/iteration_077_scew_smoothing_search.json` | 29.3355 | 27.3947 | 31.2763 | 当前 best。说明 SCEW 改变了残差噪声结构，平滑参数需要重调。 |

最优平滑参数：

```text
valence:
  window = 25
  sigma_prior = 10.0
  MAE_v = 27.3947

arousal:
  window = 15
  sigma_prior = 15.0
  MAE_a = 31.2763

overall = 29.3355
```

和 071 的统一 `window=17, sigma_prior=10` 不同，077 说明两个维度的后验噪声结构已经分化：

```text
valence:
  SCEW 后 residual 更像低频可信修正，可以使用更长窗口 25

arousal:
  residual 仍然弱且状态依赖，只适合中等窗口 15
```

当前最优完整推理链：

```text
1. VideoTimeMean prior
2. state = stable/rising/falling from prior slope
3. state-conditioned lag
4. state-conditioned expert weight
5. consensus residual gate
6. state multiplier
7. dimension-specific PA-BS smoothing
```

这个链条的贡献顺序很清楚：

```text
071 PA-BS:        29.5553
072 CDG:          29.5328
073 SCDG:         29.4954
074 SCL:          29.4356
075 SCEW:         29.3467
076 SCEW fine:    29.3455
077 smoothing:    29.3355
```

下一步的优先级应从“继续在单 split 上调参”切换到“稳健性验证”：

```text
1. 做 subject-disjoint 多折验证
2. 检查状态阈值、lag、expert 权重是否跨 fold 稳定
3. 若稳定，再固化为最终 submission 的 model.py
4. 若不稳定，把状态路由参数改成更保守的规则或由训练集内验证自动选择
```
## 078 追加观察：树模型残差 expert 失败验证

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 078 | Tree residual expert：选取 220 个与 residual 相关的 EEG/fNIRS 秒级特征，加 prior/time/video/slope 上下文，训练 HGB/ExtraTrees 残差模型 | `experiments/results/iteration_078_tree_residual_expert.json` | 30.0320 | 28.5628 | 31.5013 | 没有超过 VideoTimeMean 29.9574，更远弱于 077。说明普通表格非线性模型不是突破口。 |

实验设置：

```text
输入:
  220 个 residual-correlation 选出的 EEG/fNIRS 特征
  + prior valence/arousal
  + prior slope
  + video/time 上下文

模型:
  HistGradientBoostingRegressor, absolute_error
  HistGradientBoostingRegressor, squared_error
  ExtraTreesRegressor

目标:
  y - VideoTimeMean prior 的 residual
```

最好结果来自 ExtraTrees 小尺度 residual + PA-BS：

```text
extratrees_residual_scale0.25_pabs13_sigma7.5
overall = 30.0320
valence = 28.5628
arousal = 31.5013
```

这个实验解释了为什么“换一个强表格模型”不能带来大幅提升：秒级手工统计特征中的可泛化 residual 信息非常弱，树模型更容易把训练 subject 的 idiosyncratic pattern 学进去，迁移到 test_21-test_24 时反而破坏 VideoTimeMean prior。

结论：

```text
不要把 HGB/ExtraTrees 加入当前 ensemble
不要继续在普通秒级统计特征上堆 tabular 模型
继续保留深度 residual experts + 状态路由
下一步应先算 oracle/上界，判断 29.3 到底离可达到上限还有多远
```

## 079 追加观察：oracle 上界诊断

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 079 | Oracle ceiling diagnostics：用不可提交的 oracle 校准分析误差结构 | `experiments/results/iteration_079_oracle_ceiling.json` | 18.8194 / 28.1609 / 29.5279 | 17.0920 / 27.1781 / 28.2443 | 20.5468 / 29.1438 / 30.8116 | 真正的大空间来自 trial 级低频偏移，而不是继续换普通表格模型。 |

关键结果：

```text
TrialMeanOffset_oracle:        overall 18.8194, valence 17.0920, arousal 20.5468
SubjectAffine_oracle:          overall 28.1609, valence 27.1781, arousal 29.1438
All24_VideoTimeMean_leaky:     overall 28.2976, valence 27.0675, arousal 29.5278
LOSO24_VideoTimeMean_oracle:   overall 29.5279, valence 28.2443, arousal 30.8116
SubjectMeanOffset_oracle:      overall 29.6049, valence 27.7530, arousal 31.4568
Train20_VideoTimeMean:         overall 29.9574, valence 28.5041, arousal 31.4107
```

解释：

```text
1. 如果知道每个验证 trial 的平均残差，MAE 可以到 18.82，这是不可提交 oracle，但说明主误差不是高频形状，而是 trial-level offset。
2. 只做 subject mean offset 几乎没用，说明偏移不是简单的被试常数，而是 subject x video x trial 条件下的低频漂移。
3. LOSO24 VideoTimeMean oracle 只有 29.53，说明单纯扩大 video-time prior 不能解释当前 SCEW/TBCR 的增益。
```

因此后续优化重点从“堆更复杂秒级模型”转为“预测每个 trial 的低频残差系数”，也就是 TBCR correction。

## 080 追加观察：SCEW + TBCR 低频修正

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 080 | SCEW077 + TBCR low-frequency correction：在当前最强 SCEW 输出后叠加 trial-basis residual | `experiments/results/iteration_080_scew_tbcr_blend.json` | 29.1933 | 27.1103 | 31.2763 | 明显优于 077 的 29.3355，trial 级低频修正有效，主要改善 valence。 |

最优设置：

```text
SCEW_TBCR_k4_a1.0_wv1.0_wa0.0_clip5.0

basis_count = 4
ridge_alpha = 1.0
w_valence = 1.0
w_arousal = 0.0
clip = 5.0

overall = 29.1933
valence = 27.1103
arousal = 31.2763
```

形式上可以写成：

```text
y_hat = SCEW077(x) + clip(W * r_TBCR(x_trial), -c, c)
```

其中 `r_TBCR` 不是逐秒乱预测，而是把每个 trial 的残差投影到少量余弦低频基：

```text
r(t) ~= sum_k beta_k cos(pi * (t + 0.5) * k / T)
```

这和 079 的 oracle 结论一致：剩余误差里有可学习的低频偏移，但必须限制自由度和幅度，否则会把被试私有噪声学进去。

## 081 追加观察：SCEW + TBCR 精细搜索

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 081 | SCEW + TBCR fine search：固定 k=4，细搜 ridge、blend weight、clip | `experiments/results/iteration_081_scew_tbcr_blend_fine.json` | 29.1747 | 27.0947 | 31.2548 | 当前新 best。相比 077 提升 0.1608 overall，已经不是 0.001 级别的调参。 |

当前最好：

```text
SCEW_TBCR_k4_a0.5_wv1.0_wa0.05_clip4.0

basis_count = 4
ridge_alpha = 0.5
w_valence = 1.0
w_arousal = 0.05
clip = 4.0

overall = 29.1747
valence = 27.0947
arousal = 31.2548
```

与历史最好对比：

```text
077 SCEW smoothing:        overall 29.3355, valence 27.3947, arousal 31.2763
080 SCEW + TBCR coarse:    overall 29.1933, valence 27.1103, arousal 31.2763
081 SCEW + TBCR fine:      overall 29.1747, valence 27.0947, arousal 31.2548
```

结论：

```text
valence:
  trial-level low-frequency correction 很有效，说明愉悦度更容易受到 trial 级基线/趋势调制。

arousal:
  只能给 0.05 的极小修正。大权重会破坏结果，说明 arousal 的可泛化残差信号弱，或被 SCEW 的状态路由已经吃掉了主要部分。
```

## 082 追加观察：all-6 fNIRS 预处理对照失败

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 082 | SCEW + TBCR with all-6 fNIRS feature cache：把 HbO/HbR/HbT/Abs780/Abs805/Abs830 全部用于 TBCR | `experiments/results/iteration_082_scew_tbcr_blend_allfnirs_fine.json` | 29.2567 | 27.2371 | 31.2763 | 差于 081，说明更多 fNIRS 类型没有带来更好的 trial-offset 预测，反而增加跨被试噪声。 |

最好结果：

```text
SCEW_TBCR_k4_a0.25_wv0.5_wa0.0_clip3.0

overall = 29.2567
valence = 27.2371
arousal = 31.2763
```

预处理结论：

```text
继续使用 experiments/features/asac_features_20_4.npz
不要把 all-6 fNIRS cache 用于当前 TBCR correction
吸光度通道可能包含更多设备/被试漂移，未经过更强归一化前不适合直接进入 trial-level correction
```

当前最优链条更新为：

```text
1. VideoTimeMean prior
2. SCEW residual route
   - state-conditioned lag
   - state-conditioned expert weight
   - consensus gate
   - dimension-specific PA-BS smoothing
3. TBCR trial-level low-frequency correction
   - k = 4 cosine basis
   - ridge alpha = 0.5
   - valence weight = 1.0
   - arousal weight = 0.05
   - correction clip = 4

Current best:
  overall = 29.1747
  valence = 27.0947
  arousal = 31.2548
```

## 083 追加观察：TBCR 加入 trial slope 特征

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 083 | SCEW + TBCR mean/std/slope：trial 特征从均值/方差扩展到前半段到后半段的趋势差 | `experiments/results/iteration_083_scew_tbcr_blend_slope.json` | 29.1496 | 27.2496 | 31.0496 | overall 优于 081，但改善来自 arousal；valence 被破坏。 |

最好设置：

```text
SCEW_TBCR_k4_a0.25_wv0.05_wa0.10_clip6.0

feature_mode = mean_std_slope
basis_count = 4
ridge_alpha = 0.25
w_valence = 0.05
w_arousal = 0.10
clip = 6.0

overall = 29.1496
valence = 27.2496
arousal = 31.0496
```

解释：

```text
mean/std/slope 的趋势信息对 arousal 有用：
  arousal 从 081 的 31.2548 降到 31.0496

但 slope 会伤害 valence：
  valence 从 081 的 27.0947 退到 27.2496
```

这说明 TBCR 不能再用同一组 trial 特征同时服务两个维度，应该做 dimension-specific correction。

## 084 追加观察：Dual-TBCR 维度分治

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 084 | Dual-TBCR：valence 使用 mean/std，arousal 使用 mean/std/slope，分别搜索权重和 clip | `experiments/results/iteration_084_dual_tbcr_blend.json` | 29.0706 | 27.0916 | 31.0496 | 当前结构性突破。把 081 的 valence 优势和 083 的 arousal 优势合并起来。 |

最好设置：

```text
DualTBCR_k4
  valence_mode = mean_std
  alpha_valence = 0.25
  w_valence = 1.0
  clip_valence = 4.5

  arousal_mode = mean_std_slope
  alpha_arousal = 0.25
  w_arousal = 0.10
  clip_arousal = 6.0

overall = 29.0706
valence = 27.0916
arousal = 31.0496
```

为什么有效：

```text
valence:
  更像低频偏置/整体情绪基调，mean/std 就足够。
  slope 进入后会引入跨被试趋势噪声，所以只保留 mean/std。

arousal:
  更像激活水平变化，前半段到后半段的生理趋势差有可迁移信息。
  因此 mean/std/slope 对 arousal 明显有效。
```

这不是简单拼接，而是按残差结构拆解：

```text
y_v = SCEW_v + clip(w_v * TBCR_mean_std_v, -c_v, c_v)
y_a = SCEW_a + clip(w_a * TBCR_mean_std_slope_a, -c_a, c_a)
```

## 085 追加观察：Dual-TBCR 局部细搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 085 | Dual-TBCR fine search：围绕 084 的最优区域细搜 alpha、weight、clip | `experiments/results/iteration_085_dual_tbcr_blend_fine.json` | 29.0678 | 27.0878 | 31.0478 | 当前 best。细搜收益较小，但确认 084 的结构选择稳定。 |

当前最好：

```text
DualTBCR_k4
  valence_mode = mean_std
  alpha_valence = 0.01
  w_valence = 0.75
  clip_valence = 4.2

  arousal_mode = mean_std_slope
  alpha_arousal = 0.01
  w_arousal = 0.10
  clip_arousal = 5.5

overall = 29.0678
valence = 27.0878
arousal = 31.0478
```

需要注意：

```text
alpha = 0.01 出现 sklearn ill-conditioned warning，说明这个局部最优可能偏激进。
更保守可选 084 的 alpha = 0.25，overall = 29.0706，分数只差 0.0028。
如果后面做最终提交，建议用多折验证决定使用 085 激进版还是 084 稳健版。
```

本轮从 077 到 085 的有效提升：

```text
077 SCEW smoothing:        overall 29.3355, valence 27.3947, arousal 31.2763
081 SCEW + TBCR fine:      overall 29.1747, valence 27.0947, arousal 31.2548
083 slope TBCR:            overall 29.1496, valence 27.2496, arousal 31.0496
084 Dual-TBCR:             overall 29.0706, valence 27.0916, arousal 31.0496
085 Dual-TBCR fine:        overall 29.0678, valence 27.0878, arousal 31.0478

total improvement from 077:
  overall -0.2677
  valence -0.3069
  arousal -0.2285
```

下一步不应再只做单 split 细搜。更有价值的两件事：

```text
1. 做 subject-disjoint 多折验证，检查 Dual-TBCR 的 valence mean/std 与 arousal slope 规律是否跨被试稳定。
2. 把 084/085 固化进 submission 推理链，并在保守版和激进版之间做最终选择。
```

## 086 追加观察：LRAG 残差注意力粗搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 086 | LRAG：在 Dual-TBCR correction 上做 trial 内 latent residual attention | `experiments/results/iteration_086_residual_attention_tbcr_small.json` | 29.0571 | 27.0730 | 31.0412 | 有增益。说明残差注意力可以进一步平滑/重分配 trial 内低频修正。 |

实现方式：

```text
不是上大参数 Transformer，而是在 residual correction 上做轻量注意力：

z_t = [
  time,
  sin(time),
  cos(time),
  zscore(prior),
  zscore(prior_slope),
  zscore(correction)
]

A = softmax(z_t z_s^T / sqrt(d) / temperature - distance_penalty)

interp 模式:
  correction' = correction + gamma * (A correction - correction)

residual_add 模式:
  correction' = correction + gamma * A correction
```

最好粗搜结果：

```text
mode = interp
gamma_valence = 0.5
gamma_arousal = 0.5
temperature = 2.0
distance_sigma = 0.3

overall = 29.0571
valence = 27.0730
arousal = 31.0412
```

与 085 对比：

```text
085 Dual-TBCR:  overall 29.0678, valence 27.0878, arousal 31.0478
086 LRAG:       overall 29.0571, valence 27.0730, arousal 31.0412
gain:           overall -0.0107
```

解释：

```text
LRAG 的收益不大，但方向正确。
它不是创造新的高频预测，而是把 TBCR 的低频 correction 在 trial 内按 prior/correction 相似性重新分配。
这和 DeepSeek 系列里“先压到低维 latent，再做注意力/残差更新”的思想相近，但参数量几乎为零，更适合当前 24 个 subject 的小数据。
```

## 087 追加观察：LRAG 残差注意力细搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 087 | LRAG fine search：围绕 086 的 gamma/temperature/distance 做局部细搜 | `experiments/results/iteration_087_residual_attention_tbcr_fine.json` | 29.0531 | 27.0694 | 31.0369 | 当前 best。残差注意力确认有效，但属于小幅后处理增益。 |

当前最好：

```text
LRAG_k4
  alpha_valence = 0.01
  alpha_arousal = 0.01
  attention_mode = interp
  gamma_valence = 0.65
  gamma_arousal = 0.65
  temperature = 3.0
  distance_sigma = 0.3

overall = 29.0531
valence = 27.0694
arousal = 31.0369
```

从 077 到 087 的累计变化：

```text
077 SCEW smoothing:        overall 29.3355, valence 27.3947, arousal 31.2763
085 Dual-TBCR fine:        overall 29.0678, valence 27.0878, arousal 31.0478
087 LRAG fine:             overall 29.0531, valence 27.0694, arousal 31.0369

total improvement from 077:
  overall -0.2824
  valence -0.3253
  arousal -0.2394
```

注意：

```text
087 仍然基于 alpha=0.01 的激进 Dual-TBCR，所以和 085 一样存在单 split 调参风险。
如果要做正式提交，建议同时保留：
  稳健版：084 Dual-TBCR, overall 29.0706
  激进版：087 LRAG, overall 29.0531

下一步优先做多折验证，而不是继续单 split 上无限细搜 attention 参数。
```

## 088 追加观察：多折验证纠偏

| 编号 | 实验 | 输出文件 | 验证范式 | 结论 |
| --- | --- | --- | --- | --- |
| 088 | Subject-disjoint 6-fold validation for prior-only TBCR/LRAG | `experiments/results/iteration_088_cross_fold_tbcr.json` | 24 个 subject 按每 4 人一折，共 6 折；每折用 20 个 subject 训练 VideoTimeMean/TBCR，4 个 subject 验证 | 重要纠偏：TBCR/LRAG 在原始 `test_21-test_24` split 有效，但跨折聚合不如 VideoTimeMean，说明固定 split 调参有过拟合。 |

聚合结果：

```text
VideoTimeMean:                     overall 29.4119, valence 27.5585, arousal 31.2652
Prior_LRAG_aggressive_087params:   overall 29.6193, valence 27.6674, arousal 31.5711
Prior_LRAG_stable_087params:       overall 29.6194, valence 27.6679, arousal 31.5709
Prior_DualTBCR_085params:          overall 29.6482, valence 27.7091, arousal 31.5873
Prior_DualTBCR_084params:          overall 29.6742, valence 27.7414, arousal 31.6070
```

各折现象：

```text
fold 1, val test_1-test_4:
  VideoTimeMean 26.5094
  LRAG          26.6292-26.6302  -> 变差

fold 2, val test_5-test_8:
  VideoTimeMean 36.7935
  LRAG          37.0852-37.0854  -> 变差

fold 3, val test_9-test_12:
  VideoTimeMean 28.2175
  LRAG          28.8271          -> 明显变差

fold 4, val test_13-test_16:
  VideoTimeMean 28.1977
  LRAG          28.5527-28.5532  -> 变差

fold 5, val test_17-test_20:
  VideoTimeMean 26.7956
  LRAG          27.0278-27.0280  -> 变差

fold 6, val test_21-test_24:
  VideoTimeMean 29.9574
  LRAG          29.5922-29.5934  -> 变好
```

必须修正之前的表述：

```text
之前的 084/085/087 不是“单 subject”，而是固定 train test_1-test_20 / val test_21-test_24 的 20/4 subject-disjoint split。
但它确实只是单一验证 split。
由于 079-087 在这一个 split 上进行了多轮搜索，29.0531 只能看作 fixed-split validation best，不能视为稳健泛化效果。
```

当前可信结论：

```text
1. VideoTimeMean 是跨折更稳的强 baseline。
2. TBCR/LRAG prior-only correction 在 test_21-test_24 有效，但在其他 subject folds 上平均伤害泛化。
3. 这说明 trial-level correction 不能用固定全局参数直接迁移，必须加入可验证的门控/不确定性判断，或者只在模型有足够证据时启用。
4. SCEW 深度 checkpoint 仍未完成跨折验证，因为它需要每折重新训练专家模型；当前不能把 077/087 的固定 split 成绩当最终真实效果。
```

下一步策略调整：

```text
停止继续在 test_21-test_24 单折上细搜。
优先做跨折稳定的 adaptive gate：
  如果 TBCR correction 的 trial feature 落在训练分布内、且 residual 幅度/attention entropy 可信，则启用 correction；
  否则回退到 VideoTimeMean 或 SCEW。

完整验证需要：
  A. 训练每折 SCEW/Graph-Mamba experts
  B. 在每折上只用训练 subject 内部选择 TBCR/LRAG 参数
  C. 最后报告 6-fold mean/std
```

## 089 追加观察：Adaptive TBCR gate 方案过重

| 编号 | 实验 | 输出文件 | 结果 | 判断 |
| --- | --- | --- | --- | --- |
| 089 | Train-only adaptive TBCR gate：每个 outer fold 内再做 inner OOF，学习 trial-level correction scale | `tools/cross_fold_adaptive_tbcr_gate.py` | 运行 15 分钟超时，未产出完整结果 | 方法思想合理，但当前实现太重，不适合作为快速迭代主线。 |

这个方案的目标是解决 088 暴露的问题：TBCR/LRAG 不能默认启用，必须学习何时启用 correction。

实现思路：

```text
outer fold:
  train 20 subjects, validate 4 subjects

inner OOF inside train:
  对 train subjects 再做 subject-disjoint OOF
  对每个 held-out train trial 计算 TBCR correction 是否改善 prior
  学一个 trial-level gate/scale

outer validation:
  用 train-only gate 决定每个 validation trial 的 correction 强度
```

问题：

```text
每个 outer fold 需要多次重建 TBCR residual model；
6 个 outer folds x 多个 inner folds 后运行成本过高。
```

处理：

```text
先暂停 089。
主线改为更便宜且跨折有效的 robust prior 优化。
后续若要恢复 089，需要缓存每个 fold 的 TBCR prediction，避免重复拟合。
```

## 090 追加观察：Robust Video-Time Prior 跨折搜索

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 090 | Robust prior search：mean/median/trimmed mean + lag + shrink + smoothing，6-fold subject-disjoint 聚合 | `experiments/results/iteration_090_cross_fold_prior_search_small.json` | 28.8610 | 27.0167 | 30.7052 | 真正跨折有效的新突破。比 VideoTimeMean 29.4119 稳定降低 0.5509。 |

最优候选：

```text
Prior_median_trim0.0_lag-1_shrink0.0_smooth9

overall = 28.8610
valence = 27.0167
arousal = 30.7052
```

对比 088 的原始 VideoTimeMean：

```text
VideoTimeMean 6-fold:
  overall = 29.4119
  valence = 27.5585
  arousal = 31.2652

Robust median prior:
  overall = 28.8610
  valence = 27.0167
  arousal = 30.7052

gain:
  overall -0.5509
  valence -0.5418
  arousal -0.5600
```

解释：

```text
1. median 优于 mean，说明训练 subject 标签里存在明显 outlier，均值会被拉偏。
2. lag=-1 有效，说明动态标注/情绪反应存在约 1 秒时间偏移，直接用同秒均值不是最优。
3. smooth9 有效，说明 1 Hz joystick 标签仍有短期噪声，适度时间平滑能提升泛化。
4. shrink 到全局均值反而不是最优，说明视频-时间轨迹本身足够强，不应过度回退。
```

这一步比 079-087 更可信，因为它是在 6-fold subject-disjoint 上聚合验证的。

## 091 追加观察：Median Prior 局部细搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 091 | Median prior fine search：只围绕 median prior 搜索 lag、smooth window、少量 shrink | `experiments/results/iteration_091_cross_fold_prior_median_fine.json` | 28.8596 | 27.0294 | 30.6899 | 当前最可信跨折 best。相比 090 又小幅降低 0.0014。 |

当前跨折最优：

```text
Prior_median_trim0.0_lag-2_shrink0.0_smooth11

overall = 28.8596
valence = 27.0294
arousal = 30.6899
```

091 前几名非常接近：

```text
lag=-2, smooth11: overall 28.8596
lag=-1, smooth9:  overall 28.8610
lag=-2, smooth13: overall 28.8621
lag=-1, smooth11: overall 28.8623
lag=-2, smooth9:  overall 28.8635
```

结论：

```text
鲁棒 prior 的稳定区域是：
  estimator = median
  lag = -1 或 -2
  smooth window = 9 到 13
  shrink = 0

为了最终提交稳健性，推荐使用：
  median + lag=-1 + smooth9 或 median + lag=-2 + smooth11

两者只差 0.0014，说明不是单点偶然。
```

当前可信排名需要改写：

```text
6-fold robust results:
  RobustMedianPrior_091:  overall 28.8596
  RobustMedianPrior_090:  overall 28.8610
  VideoTimeMean_088:      overall 29.4119
  Prior_LRAG_088:         overall 29.6193
  Prior_DualTBCR_088:     overall 29.6482

fixed-split only:
  LRAG_087:               overall 29.0531 on test_21-test_24 only
```

新的主线：

```text
1. 用 RobustMedianPrior 替代 VideoTimeMean，作为所有后续模型的 prior。
2. 重新训练/评估 residual 模型时，残差目标改成 y - RobustMedianPrior，而不是 y - VideoTimeMean。
3. 之前 TBCR/LRAG 失败的一部分原因可能是 prior 本身偏了；先修 prior，再谈 residual correction。
```

## 092 追加观察：Robust Prior 后的 TBCR 残差校正

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 092 | 以 `median + lag=-2 + smooth11` 作为 prior，再用 leave-subject-out TBCR 估计残差并做多尺度融合 | `experiments/results/iteration_092_robust_prior_tbcr.json` | 28.8596 | 27.0294 | 30.6899 | 最优仍然是不开启 TBCR correction；当前手工生理残差在跨折上不稳。 |

核心结论：

```text
RobustMedianPrior:
  overall = 28.8596
  valence = 27.0294
  arousal = 30.6899

Robust prior + TBCR residual:
  所有非零 residual scale 都没有超过 RobustMedianPrior
  top rows 的最佳 scale 是 sv=0, sa=0
```

解释：

```text
TBCR 在固定切分里能带来一点收益，但换成 6-fold subject-disjoint 后不稳。
这说明它学到的不是可靠的跨被试生理残差，而更像某个验证 subject 组合上的偶然校正。

数学上可以写成：
  y = p(video, time) + r(signal)

091 已经把 p(video, time) 做得很强，剩余 r(signal) 的信噪比变低。
如果 r 的估计误差大于真实残差收益，那么加 residual 会提高 MAE。
```

后续策略：

```text
1. 不再默认残差一定有用，任何 signal correction 都必须跨折证明。
2. signal 模块优先尝试低自由度、带置信度门控的形式。
3. 继续保留 RobustMedianPrior 作为主 baseline。
```

## 093 追加观察：受试者相似度近邻 Prior

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 093 | 用无标签 EEG/fNIRS 统计特征选择近邻训练受试者，再用近邻 subject 构造 median prior | `experiments/results/iteration_093_subject_similarity_prior.json` | 28.8596 | 27.0294 | 30.6899 | 近邻子集没有提升；全体 20 个训练 subject 的 median prior 最稳。 |

聚合结果：

```text
RobustMedianPrior_all20:     overall 28.8596, valence 27.0294, arousal 30.6899
SubjectSimilarityMedian_k20: overall 28.8596, valence 27.0294, arousal 30.6899
SubjectSimilarityMedian_k15: overall 28.9908, valence 27.1572, arousal 30.8244
SubjectSimilarityMedian_k10: overall 29.1873, valence 27.0700, arousal 31.3046
SubjectSimilarityMedian_k8:  overall 29.5769, valence 27.4449, arousal 31.7090
SubjectSimilarityMedian_k5:  overall 30.7296, valence 28.0688, arousal 33.3903
SubjectSimilarityMedian_k3:  overall 32.4301, valence 29.6064, arousal 35.2538
```

解释：

```text
近邻 subject 数越少越差，说明当前低阶 EEG/fNIRS mean/std 特征不能可靠表示情绪轨迹相似性。
删除训练 subject 会降低 median prior 的抗 outlier 能力，尤其会损伤 arousal。

这不是坏结果，它告诉我们：
  生理特征不能简单用于 subject retrieval；
  如果要利用 subject 差异，应做软权重或不确定性门控，而不是硬筛样本。
```

下一步主线：

```text
尝试 confidence-aware prior fusion：
  在训练 subject 对同一 video-time 的标签分歧大时，说明 prior 不确定；
  这时才允许 signal 模块或更保守的全局/局部均值介入。

如果 prior 本身很确定，则不要让高方差模型乱改。
```

## 094 追加观察：Confidence-Aware Prior Fusion

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 094 | 用训练 subject 在同一 video-time 的 MAD 分歧度作为 uncertainty gate，融合更长平滑/均值 prior | `experiments/results/iteration_094_confidence_prior_fusion.json` | 28.8470 | 27.0404 | 30.6536 | 有效。说明 prior 的局部不确定性可以指导后处理。 |

最佳候选：

```text
UncertaintyBlend_smooth31_q40-75_g0.5

overall = 28.8470
valence = 27.0404
arousal = 30.6536
```

公式：

```text
p0(t) = median prior, lag=-2, smooth=11
p1(t) = longer smoothed prior
d(t)  = MAD_train_subjects(video, t + lag)

gate(t) = clip((d(t) - Q_low) / (Q_high - Q_low), 0, 1) * max_gate
yhat(t) = (1 - gate(t)) * p0(t) + gate(t) * p1(t)
```

解释：

```text
训练 subject 分歧越大，原始 video-time median prior 越不可靠。
这时混入更长窗口平滑可以压掉局部标签抖动。

094 的有效模块不是“回退到中心 128”，而是“对不确定时间点做更强时间平滑”。
这更像动态标注噪声修正，而不是情绪均值偏置修正。
```

## 095 追加观察：Confidence Fusion 细搜

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 095 | 扩大 uncertainty 阈值、gate、long smooth window 的细搜 | `experiments/results/iteration_095_confidence_prior_fusion_fine.json` | 28.8076 | 27.0466 | 30.5687 | 继续有效。meanPrior 融合对 arousal 更强，long smooth 对 valence 更强。 |

最佳整体候选：

```text
UncertaintyBlend_meanPrior_q20-60_g0.35

overall = 28.8076
valence = 27.0466
arousal = 30.5687
```

分维度观察：

```text
best valence-like candidate:
  UncertaintyBlend_smooth51_q20-60_g0.35
  valence = 27.0058

best arousal-like candidate:
  UncertaintyBlend_meanPrior_q20-60_g0.50
  arousal = 30.5566
```

关键判断：

```text
valence 更像连续轨迹平滑问题：
  更长窗口 smooth51 可以减少瞬时抖动，valence MAE 更低。

arousal 更像幅值偏置/subject 分歧问题：
  在高 uncertainty 区域混入 mean prior 可以降低 arousal MAE。

所以一个统一后处理策略不是最优，下一步应做 valence/arousal 解耦。
```

## 096 追加观察：Dimwise Confidence Fusion

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 096 | valence 和 arousal 分别选择固定 confidence-fusion 候选，再拼成二维预测 | `experiments/results/iteration_096_dimwise_confidence_fusion.json` | 28.7812 | 27.0058 | 30.5566 | 明显有效。MAE 本身就是两列误差均值，分列解码合理。 |

最佳组合：

```text
valence:
  UncertaintyBlend_smooth51_q20-60_g0.35
  valence MAE = 27.0058

arousal:
  UncertaintyBlend_meanPrior_q20-60_g0.50
  arousal MAE = 30.5566

combined:
  overall = (27.0058 + 30.5566) / 2
          = 28.7812
```

解释：

```text
评分函数为：
  overall_mae = mean(|e_valence|, |e_arousal|)

因此解码层可以写成：
  yhat_v = f_v(video, time, uncertainty)
  yhat_a = f_a(video, time, uncertainty)

不需要强迫 valence 和 arousal 使用同一套 gate/reference。
096 的提升证明这两个情绪维度的噪声结构不同。
```

## 097 追加观察：Auto Dimwise Full-Grid Search

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 097 | 对完整 confidence-fusion 网格分别统计 valence/arousal 跨折 MAE，再自动组合列最优候选 | `experiments/results/iteration_097_auto_dimwise_confidence_search.json` | 28.7699 | 26.9847 | 30.5551 | 当前最优跨折结果。相比 091 降低 0.0897。 |

最佳自动组合：

```text
valence:
  UncertaintyBlend_smooth61_q15-45_g0.35
  valence MAE = 26.9847

arousal:
  UncertaintyBlend_meanPrior_q20-55_g0.55
  arousal MAE = 30.5551

combined:
  overall = 28.7699
```

当前可信排名：

```text
6-fold subject-disjoint:
  AutoDimwiseConfidence_097: overall 28.7699, valence 26.9847, arousal 30.5551
  DimwiseConfidence_096:     overall 28.7812, valence 27.0058, arousal 30.5566
  ConfidenceFusion_095:      overall 28.8076, valence 27.0466, arousal 30.5687
  ConfidenceFusion_094:      overall 28.8470, valence 27.0404, arousal 30.6536
  RobustMedianPrior_091:     overall 28.8596, valence 27.0294, arousal 30.6899
  VideoTimeMean_088:         overall 29.4119, valence 27.5585, arousal 31.2652
  Official ASAC demo:        overall 47.0087, valence 49.2285, arousal 44.7890
```

重要边界：

```text
097 仍然主要利用训练标签统计和 sample_id 的 video-time 结构，还没有真正吃到 EEG/fNIRS signal。
它适合作为强 prior / 解码后处理 baseline。

下一轮如果继续引入 EEG/fNIRS，应预测：
  residual = y - AutoDimwiseConfidencePrior

而不是直接预测 y。
否则高参数模型很容易只是在重复一个更差的 video-time prior。
```

## 098 追加观察：Pattern-Specific Prior Expert

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 098 | 复现 TimeMixer/Pathformer/pattern-specific expert 思想：稳定段和动态段使用不同 prior expert | `experiments/results/iteration_098_pattern_prior_expert.json` | 28.7462 | 26.9738 | 30.5186 | 当前最优跨折结果。相比 097 再降 0.0237。 |

最佳组合：

```text
valence pattern expert:
  stable/dynamic split:
    abs(slope(prior_v)) <= Q65 为 stable

  stable expert:
    UncertaintyBlend_meanPrior_q15-45_g0.25

  dynamic expert:
    UncertaintyBlend_smooth61_q15-45_g0.45

  valence MAE = 26.9738

arousal pattern expert:
  stable/dynamic split:
    abs(slope(prior_a)) <= Q55 为 stable

  stable expert:
    UncertaintyBlend_smooth51_q20-45_g0.50

  dynamic expert:
    UncertaintyBlend_meanPrior_q20-55_g0.55

  arousal MAE = 30.5186

combined:
  overall = 28.7462
```

解释：

```text
这个结果支持“时间状态不同，最优 expert 不同”：

valence:
  稳定段使用更保守的 meanPrior 融合，动态段使用长平滑 smooth61。

arousal:
  稳定段使用 smooth51，动态段使用 meanPrior 融合。

这比 097 的单一维度 expert 更细：
  097: 每个维度一个固定 expert
  098: 每个维度按 prior slope 划分 stable/dynamic，再选 expert
```

和论文模块的对应关系：

```text
TimeMixer/TimeMixer++:
  trend/seasonal、多尺度平滑和分解思想

Pathformer / pattern-specific experts:
  不同 temporal pattern 选择不同 expert

MER-PS 里的落地：
  pattern = prior slope state
  expert = uncertainty-fusion prior candidate
```

## 099 追加观察：Strong-Prior Signal Residual Probe

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 099 | 在 098 strong prior 上，用冻结 ASAC EEG/fNIRS 特征训练 Ridge residual probe，并限制 correction 幅度 | `experiments/results/iteration_099_signal_residual_pattern_prior.json` | 28.7462 | 26.9738 | 30.5186 | 最优仍是不启用 signal residual。ASAC 特征线性残差没有跨折增益。 |

结果：

```text
PatternPrior_098:
  overall = 28.7462
  valence = 26.9738
  arousal = 30.5186

best SignalResidualRidge:
  最优 rows 全部是 sv=0, sa=0
  等价于不加 EEG/fNIRS residual correction
```

解释：

```text
训练 residual target 使用 leave-subject-out prior，避免训练 subject 自己标签泄漏到 prior。
这比直接 y - train-prior 更严格。

结果说明：
  当前 779 维 ASAC 特征虽然包含 EEG/fNIRS 信息，
  但在 098 强 prior 之后，线性 residual 的跨被试信噪比仍不足。

因此后续 signal 方向必须换模块：
  1. 更强预训练表征，例如 EEGPT/LaBraM/CBraMod/NeuroLM 的 frozen embedding；
  2. 或互信息/稳定性特征选择，先降低 779 维特征过拟合；
  3. 不建议直接扩大 Graph/Mamba residual 网络。
```

当前可信排名：

```text
6-fold subject-disjoint:
  PatternPriorExpert_098:     overall 28.7462, valence 26.9738, arousal 30.5186
  AutoDimwiseConfidence_097:  overall 28.7699, valence 26.9847, arousal 30.5551
  DimwiseConfidence_096:      overall 28.7812, valence 27.0058, arousal 30.5566
  ConfidenceFusion_095:       overall 28.8076, valence 27.0466, arousal 30.5687
  RobustMedianPrior_091:      overall 28.8596, valence 27.0294, arousal 30.6899
  VideoTimeMean_088:          overall 29.4119, valence 27.5585, arousal 31.2652
  Official ASAC demo:         overall 47.0087, valence 49.2285, arousal 44.7890
```

## 105-125 追加观察：20 新模型批次与批次后整合

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 105-124 | 20 个新模型批次：状态空间、图扩散、谱平滑、鲁棒校准、非线性残差回归 | `experiments/results/iteration_105_125_batch20_state_fusion.json` | 28.7286 | 26.9597 | 30.4975 | 单模型最好为 105/106，均超过 104。 |
| 125 | 20 模型后整合：valence 用 Alpha-Beta 状态滤波，arousal 用 slope-adaptive EMA | `experiments/results/iteration_105_125_batch20_state_fusion.json` | 28.7231 | 26.9597 | 30.4866 | 当前最优；状态空间式后处理 + 维度解耦有效。 |

本批次遵循新的迭代规则：

```text
每 20 个新模型为一个 batch。
105-124 = 新模型探索。
125 = batch 后整合。

下一轮应从 126-145 继续探索，
146 再做下一次 batch-level integration。
```

20 个新模型：

```text
105 AlphaBetaStateFilter_104
106 SlopeAdaptiveEMA_104
107 SavitzkyGolay_w7p2_104
108 DCTSoftShrink_104
109 HankelSSA_w15r2_104
110 DerivativeClip_q90_104
111 KalmanCV_104
112 TemporalGraphDiffusion_104
113 CrossVideoTimestampDiffusion_104
114 UncertaintyShrink_104_to_098
115 UncertaintySmoothGate_104
116 VideoBiasCalibrated_098
117 TimeBinBiasCalibrated_098
118 VideoTimeBinBiasCalibrated_098
119 KNNResidual_k96_scale0p06
120 NystroemRidgeResidual_rbf_scale0p06
121 PLSResidual_c8_scale0p06
122 BayesianRidgeResidual_scale0p05
123 HuberResidual_scale0p05
124 RandomForestResidual_d6_scale0p05
```

批次排名：

```text
125_DimwiseBatch20_VAlphaBeta_ASlopeEMA:
  overall = 28.7231
  valence = 26.9597
  arousal = 30.4866

105_AlphaBetaStateFilter_104:
  overall = 28.7286
  valence = 26.9597
  arousal = 30.4975

106_SlopeAdaptiveEMA_104:
  overall = 28.7286
  valence = 26.9706
  arousal = 30.4866

115_UncertaintySmoothGate_104:
  overall = 28.7298
  valence = 26.9667
  arousal = 30.4928

111_KalmanCV_104:
  overall = 28.7304
  valence = 26.9604
  arousal = 30.5003

104_DimwiseOOFMeta_reference:
  overall = 28.7311
  valence = 26.9680
  arousal = 30.4941
```

125 的真实组合：

```text
pred_valence = AlphaBetaStateFilter(104)[:, valence]
pred_arousal = SlopeAdaptiveEMA(104)[:, arousal]

overall = (26.9597 + 30.4866) / 2
        = 28.7231
```

算法解释：

```text
104 的底座是：
  PatternPrior_098 + OOF HGB 小残差 + valence/arousal 解耦。

105 Alpha-Beta 状态滤波：
  维护 level 和 velocity：
    x_t^- = x_{t-1} + v_{t-1}
    e_t = z_t - x_t^-
    x_t = x_t^- + alpha * e_t
    v_t = v_{t-1} + beta * e_t

  它比普通滑窗更适合 valence：
    保留局部趋势和速度，不会像强平滑一样把峰值压平。

106 Slope-Adaptive EMA：
  根据当前轨迹斜率选择 EMA 系数：
    alpha_t = alpha_slow + gate(|dz/dt|) * (alpha_fast - alpha_slow)
    y_t = alpha_t z_t + (1-alpha_t) y_{t-1}

  它更适合 arousal：
    稳定段更平滑，变化段更跟随，降低低频激活轨迹的 MAE。
```

负结果/排除方向：

```text
110 DerivativeClip:
  overall = 42.5378
  过度限制一阶差分，导致轨迹无法及时回到真实水平。

113 CrossVideoTimestampDiffusion:
  overall = 28.8738
  跨视频同 timestamp 扩散会混合不同情绪诱发片段，破坏视频语义。

119-124 残差回归器：
  KNN / NystroemRidge / PLS / BayesianRidge / Huber / RandomForest 均未超过 104。
  说明当前有效增益不是“换一个更复杂回归器”，而是后处理动态假设更贴合连续情绪轨迹。

109 SSA:
  overall = 28.7927
  rank-2 Hankel 重构过度低秩化，损伤 valence。
```

当前可信排名：

```text
6-fold subject-disjoint:
  Batch20Fusion_125:          overall 28.7231, valence 26.9597, arousal 30.4866
  AlphaBeta_105:              overall 28.7286, valence 26.9597, arousal 30.4975
  SlopeAdaptiveEMA_106:       overall 28.7286, valence 26.9706, arousal 30.4866
  UncertaintySmooth_115:      overall 28.7298, valence 26.9667, arousal 30.4928
  DimwiseOOFMeta_104:         overall 28.7332, valence 26.9694, arousal 30.4971
  HGBTune_103:                overall 28.7376, valence 26.9740, arousal 30.5013
  OOFStack_102:               overall 28.7381, valence 26.9706, arousal 30.5057
  PatternPriorExpert_098:     overall 28.7462, valence 26.9738, arousal 30.5186
  RobustMedianPrior_091:      overall 28.8596, valence 27.0294, arousal 30.6899
  Official ASAC demo:         overall 47.0087, valence 49.2285, arousal 44.7890
```

下一轮建议：

```text
126-145:
  不再优先扩展残差回归器。
  重点微调状态空间后处理：
    1. valence: Alpha-Beta / Kalman / light graph diffusion 的参数网格；
    2. arousal: slope-adaptive EMA / uncertainty smooth gate / DCT soft shrink；
    3. 尝试按 prior uncertainty 或 video emotion category 切换状态空间参数。

146:
  整合 126-145 中的最优 valence expert 和 arousal expert。
```

## 100 追加观察：Spectral / Tensor Trajectory Prior

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 100 | 复现低频谱分解、SSA/Hankel、subject-time SVD 的轨迹先验思想 | `experiments/results/iteration_100_spectral_tensor_prior.json` | 28.7442 | 26.9697 | 30.5186 | 只对 valence 有极小帮助；arousal 不适合低秩/谱平滑。 |

最好组合：

```text
SpectralDimwise:
  valence = SubjectTimeSVD_lag-1_r2_median_smooth9
  arousal = PatternPrior_098

overall = 28.7442
valence = 26.9697
arousal = 30.5186
```

解释：

```text
假设：
  y(video, time, subject) 存在低频/低秩结构，
  可以用 Fourier / SSA / SVD 去掉跨被试噪声。

结果：
  valence 的确有一点低秩结构，SVD rank=2 能把 valence 从 26.9738 降到 26.9697。
  arousal 一旦使用同类低秩重构会明显变差，所以 arousal 更依赖局部不确定性 gate 和平滑策略。

结论：
  不应该把 valence/arousal 强行放进同一个后处理框架。
  后续应做 dimwise 专家，而不是统一专家。
```

## 101 追加观察：Distributional Quantile Prior

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 101 | 条件分位数轨迹先验，尝试用 q35/q45/q55/q65 修正跨被试偏态 | `experiments/results/iteration_101_distributional_quantile_prior_small.json` | 28.7462 | 26.9738 | 30.5186 | 负结果；最优仍回到 PatternPrior_098。 |

关键结果：

```text
best quantile candidate:
  QuantilePrior_qv0p5_qa0p5_lag-2_smooth11
  overall = 28.8596
  valence = 27.0294
  arousal = 30.6899

best dimwise:
  PatternPrior_098
  overall = 28.7462
```

解释：

```text
q=0.5 的 conditional median 仍然最好。
偏离 median 会损伤 MAE，尤其 arousal。

这说明当前主要收益不来自“整体上把标签分布往高/低分位移动”，
而来自：
  1. 不确定性区域什么时候 shrink；
  2. stable/dynamic 时段选哪个 expert；
  3. valence/arousal 是否分开处理。
```

## 102 追加观察：Strict OOF Prior Stacking

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 102 | 使用内层 leave-one-subject-out prior 特征训练二级 HGB/Ridge/ExtraTrees 校正器 | `experiments/results/iteration_102_oof_prior_stacking.json` | 28.7381 | 26.9706 | 30.5057 | 新最优；HGB 小残差校正有效。 |

最好方法：

```text
OOFStack_HGBResidual_l1_leaf11_scale0p15:
  overall = 28.7381
  valence = 26.9706
  arousal = 30.5057
```

训练范式：

```text
outer fold:
  6-fold subject-disjoint validation

inner training feature:
  对外层 train subjects 再做 leave-one-subject-out
  用 other train subjects 生成该 subject 的 prior features
  再训练 meta residual model

预测形式：
  pred = PatternPrior_098 + scale * HGB(prior_features)
```

特征包括：

```text
24 个 prior expert 的 valence/arousal 输出
candidate mean / median / std / range
PatternPrior_098
prior slope / abs slope / acceleration
video id one-hot
trial 内 time_norm 和 sin/cos time features
```

解释：

```text
这次不是单 subject，也没有让训练被试自己的标签泄漏到自己的 prior 特征。
HGB 只加 0.10-0.20 倍的小残差最稳，说明：
  1. 098 的强先验已经解释了大部分可泛化结构；
  2. 仍存在少量非线性局部校正，例如 prior disagreement、slope state、video-time interaction；
  3. 大尺度 residual 会过拟合，所以 correction 必须小。
```

## 103 追加观察：OOF HGB Meta-Calibrator Tuning

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 103 | 围绕 102 的有效 HGB 模块调 leaf/min_leaf/lr/L1-L2/smooth/scale | `experiments/results/iteration_103_oof_hgb_tuner.json` | 28.7376 | 26.9740 | 30.5013 | 小幅刷新；smooth3 主要改善 arousal。 |

最好方法：

```text
HGBTune_l1_lr0p04_it120_leaf11_min70_l20p1_scale0p12_smooth3:
  overall = 28.7376
  valence = 26.9740
  arousal = 30.5013

close tie:
HGBTune_l1_lr0p04_it120_leaf11_min70_l20p1_scale0p15_smooth3:
  overall = 28.7376
  valence = 26.9755
  arousal = 30.4997
```

解释：

```text
L1 loss 比 L2 loss 更稳，因为 MAE 是目标指标，且标签存在跨被试长尾偏差。
min_leaf=70/90、leaf=9/11 一类低容量模型最好。
smooth3/smooth5 会降低 arousal，但会轻微损伤 valence。

这支持一个后续设计：
  valence 使用少平滑/不平滑的 residual correction；
  arousal 使用更强平滑的 residual correction。
```

## 104 追加观察：Dimwise OOF Meta Composition

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 104 | 按指标公式组合不同 valence/arousal 专家 | `experiments/results/iteration_104_dimwise_oof_meta.json` | 28.7332 | 26.9694 | 30.4971 | 当前最优；证明 valence/arousal 应使用不同校正和平滑。 |

最好组合：

```text
valence:
  OOFStack_HGBResidual_l1_leaf11_scale0p1
  valence MAE = 26.9694

arousal:
  HGBTune_l1_lr0p04_it120_leaf11_min70_l20p1_scale0p2_smooth5
  arousal MAE = 30.4971

combined:
  overall = (26.9694 + 30.4971) / 2
          = 28.7332
```

数学解释：

```text
比赛主指标：
  overall_mae = (MAE_valence + MAE_arousal) / 2

因此只要输出文件有独立的 valence/arousal 两列，
就可以选择：
  pred_valence = model_v(x)
  pred_arousal = model_a(x)

而不需要强迫两个维度来自同一个模型。
```

模块作用总结：

```text
PatternPrior_098:
  提供强 video-time / uncertainty / stable-dynamic prior，是当前底座。

OOF HGB residual:
  学习 prior disagreement、斜率、视频和时间之间的非线性局部偏差。
  correction scale 必须小，否则过拟合。

SVD valence:
  说明 valence 有一点低秩轨迹结构，但收益小于 HGB meta correction。

smooth arousal:
  arousal 更像低频连续激活量，后平滑能降低 MAE。

no smooth valence:
  valence 更容易被平滑抹掉局部变化，所以不宜使用强平滑。
```

当前可信排名：

```text
6-fold subject-disjoint:
  DimwiseOOFMeta_104:         overall 28.7332, valence 26.9694, arousal 30.4971
  HGBTune_103:                overall 28.7376, valence 26.9740, arousal 30.5013
  OOFStack_102:               overall 28.7381, valence 26.9706, arousal 30.5057
  SpectralDimwise_100:        overall 28.7442, valence 26.9697, arousal 30.5186
  PatternPriorExpert_098:     overall 28.7462, valence 26.9738, arousal 30.5186
  AutoDimwiseConfidence_097:  overall 28.7699, valence 26.9847, arousal 30.5551
  RobustMedianPrior_091:      overall 28.8596, valence 27.0294, arousal 30.6899
  VideoTimeMean_088:          overall 29.4119, valence 27.5585, arousal 31.2652
  Official ASAC demo:         overall 47.0087, valence 49.2285, arousal 44.7890
```

## 126-146 追加观察：20 个不同架构批次

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 126-145 | 20 个不同架构：LOESS、核平滑、Hampel、TV、Haar、小波/样条、HMM、CRF/图、注意力、KMeans/GMM、RFF、PCA、分布校准、Isotonic、Huber、MLP、GBDT、Stacking、MOE | `experiments/results/iteration_126_146_batch2_architectures.json` | 28.7259 | 26.9595 | 30.4922 | 最好为 128 HampelMedianFilter，但未超过全局 125。 |
| 146 | 第二批整合：Proxy-MOE valence + LocalAttention arousal | `experiments/results/iteration_126_146_batch2_architectures.json` | 28.7276 | 26.9634 | 30.4917 | 真实整合未超过 125；metric-composed 最好为 28.7256。 |

这批严格按“不同架构”执行，不是参数调参：

```text
126 Local linear LOESS trajectory smoother
127 Gaussian-process-style time kernel smoother
128 Hampel median outlier architecture
129 Total-variation proximal denoiser
130 Haar wavelet shrinkage
131 Piecewise spline/knot trend projection
132 Discrete HMM Viterbi trajectory decoder
133 CRF-like anchored temporal graph smoother
134 Local self-attention smoother
135 Prototype residual expert via KMeans
136 Gaussian mixture residual expert
137 Random Fourier feature kernel residual model
138 PLS/PCA latent residual reconstruction
139 Quantile distribution calibration
140 Isotonic monotone calibration
141 Huber robust residual calibration
142 Small MLP residual architecture
143 Gradient-boosted residual architecture
144 Linear stacking of prior candidates/features
145 Rule-based mixture of heterogeneous architecture outputs
```

批次排名：

```text
128_HampelMedianFilter:
  overall = 28.7259
  valence = 26.9595
  arousal = 30.4922

125_PreviousStateFusion_reference:
  overall = 28.7265
  valence = 26.9606
  arousal = 30.4924

146_FixedBatch2Fusion_VProxyMOE_ALocalAttention:
  overall = 28.7276
  valence = 26.9634
  arousal = 30.4917

133_AnchoredGraphSmoother:
  overall = 28.7286
  valence = 26.9644
  arousal = 30.4929

134_LocalSelfAttentionSmoother:
  overall = 28.7308
  valence = 26.9698
  arousal = 30.4917

129_TotalVariationDenoiser:
  overall = 28.7318
  valence = 26.9695
  arousal = 30.4941
```

结论：

```text
本批没有刷新全局最优 125:
  125_DimwiseBatch20_VAlphaBeta_ASlopeEMA = 28.7231

但 128 HampelMedianFilter 很有价值：
  它不是平滑所有点，而是只修正局部离群点。
  这解释了为什么它能改善 valence 到 26.9595，
  但又不像强平滑/低秩模型那样明显损伤局部动态。

134 LocalSelfAttentionSmoother 对 arousal 有帮助：
  arousal = 30.4917
  和 146 的 arousal 持平。

133 AnchoredGraphSmoother 有一定价值：
  说明图正则可用，但必须 anchored 到原轨迹，不能自由扩散。
```

负结果/排除方向：

```text
132 HMMViterbiDecoder:
  overall = 28.8614
  离散状态量化太粗，破坏连续 VA 轨迹。

144 LinearStackingArchitecture:
  overall = 28.8078
  线性 stacking 不如手工结构滤波，说明误差不是简单线性组合可解。

139 QuantileDistributionCalibration:
  overall = 28.7667
  再次验证分布映射会损伤 arousal。

135-138 prototype/kernel/latent residual:
  都没有超过 104/125，说明当前 OOF prior 特征上的残差学习已接近瓶颈。

142 MLPResidualArchitecture:
  overall = 28.7689
  小 MLP 也不如状态/鲁棒滤波，数据量仍不足以支撑更自由的残差函数。
```

下一轮方向：

```text
147-166 仍然要保持“不同架构”，不能做参数网格。
但可以围绕已经有效的思想做架构级变体：
  1. robust filter family: Hampel、Tukey、Huberized smoother、trimmed attention；
  2. attention/graph family: anchored graph attention、uncertainty graph attention；
  3. state-space family: particle filter、switching linear dynamical system；
  4. decomposition family: robust STL-like trend/residual split；
  5. prediction-space correction: monotone-safe calibration, not raw quantile mapping。
```

## 147-167 追加观察：第三批 20 个不同架构

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 147-166 | 20 个不同架构：Tukey、Trimmed Attention、Bilateral、Robust Trend、Particle Filter、Switching LDS、GMRF、Laplacian Pyramid、Empirical Bayes、Conformal Clamp、Prior-bin Calibration、Gaussian Copula、Prototype Retrieval、FFT、AR2、Change-point、Dirichlet Evidence、Risk Clamp、Phase-flow | `experiments/results/iteration_147_167_batch3_architectures_real_fusion.json` | 28.7170 | 26.9518 | 30.4822 | 最好单架构为 157 PriorBinBiasCalibration，刷新 125。 |
| 167 | 第三批整合：GaussianCopula valence + TrialPrototype arousal | `experiments/results/iteration_147_167_batch3_architectures_real_fusion.json` | 28.7111 | 26.9412 | 30.4809 | 当前全局最优；这次是真实固定融合，不是只按指标事后拼接。 |

这批同样不是调参，而是 20 个互相不同的建模假设：

```text
147 Tukey biweight local M-estimator smoother
148 Trimmed local attention smoother
149 Bilateral edge-preserving value-time filter
150 Robust trend/residual decomposition
151 Deterministic particle filter
152 Switching-regime linear dynamical system
153 Gaussian Markov random field MAP smoother
154 Laplacian pyramid multiscale blend
155 Empirical-Bayes shrinkage to candidate consensus
156 Conformal residual clamp around candidate consensus
157 Prior-bin rank-preserving bias calibration
158 Gaussian copula covariance calibration
159 Trial prototype residual retrieval
160 FFT phase-preserving high-frequency shrinkage
161 AR(2) trajectory repair smoother
162 Monotone micro-segment reversal repair
163 Change-point piecewise trend model
164 Dirichlet evidence-weighted expert averaging
165 Risk-averse center/candidate clamp
166 Learned phase-flow vector-field correction
```

批次排名：

```text
167_FixedBatch3Fusion_VCopula_APrototype:
  overall = 28.7111
  valence = 26.9412
  arousal = 30.4809

157_PriorBinBiasCalibration:
  overall = 28.7170
  valence = 26.9518
  arousal = 30.4822

150_RobustTrendResidualSplit:
  overall = 28.7216
  valence = 26.9599
  arousal = 30.4832

158_GaussianCopulaCovarianceCalibration:
  overall = 28.7227
  valence = 26.9412
  arousal = 30.5041

125_PreviousStateFusion_reference:
  overall = 28.7240
  valence = 26.9643
  arousal = 30.4838

159_TrialPrototypeResidualRetrieval:
  overall = 28.7311
  valence = 26.9813
  arousal = 30.4809
```

关键结论：

```text
157 Prior-bin calibration 有效：
  它不是自由拟合残差，而是按先验预测值所在区间学习小偏置。
  形式上可以理解为：

    y_hat = y_prior + b(bin(y_prior))

  其中 b 是由 OOF 训练残差估计出的低自由度偏置项。
  这适合 MER-PS 的小数据场景，因为它只修正系统性偏差，不让模型随意记主体。

158 Gaussian copula calibration 对 valence 最强：
  valence 从 26.9643 改到 26.9412。
  它起作用的原因不是时间平滑，而是修正二维 VA 输出的边缘分布和相关结构。
  但 arousal 被拉坏到 30.5041，所以不能整模型使用。

159 Trial prototype retrieval 对 arousal 最强：
  arousal 从 30.4838 改到 30.4809。
  它通过相似 trial 的残差模板做小幅修正，
  但 valence 被拉坏到 26.9813，所以也不能整模型使用。

167 的有效性来自分维度融合：

    y_v = f_copula_v(x)
    y_a = f_prototype_a(x)
    y = [y_v, y_a]

  这说明 valence 和 arousal 的可学习误差来源不同：
  valence 更受输出分布/协方差校准影响；
  arousal 更受 trial-level 动态模板影响。
```

负结果/排除方向：

```text
160 FFTPhasePreservingShrinkage:
  overall = 29.0946
  频域高频压缩过强，会破坏真实的短时情绪变化。

161 AR2TrajectoryRepair:
  overall = 28.8834
  AR(2) 假设太硬，连续情绪不是单一二阶线性系统。

164 DirichletEvidenceExpertAverage:
  overall = 28.7884
  证据平均会把有用的维度专长稀释掉，不如分维度选专家。

165 RiskAverseConsensusClamp:
  overall = 28.7804
  风险夹紧降低了 MSE 风险，但 MAE 上损伤 valence，说明过度保守。

166 LearnedPhaseFlowCorrection:
  overall = 28.7511
  轨迹相位场学不到稳定迁移规律，小数据下容易学到 fold-specific 转移。
```

下一轮方向：

```text
168-187 继续做 20 个不同架构，重点不再平均堆模型，而是做“分维度可解释专家”：
  1. valence: 分布校准、copula、低自由度 piecewise affine/spline calibration；
  2. arousal: prototype retrieval、DTW-like trial retrieval、change-aware template correction；
  3. 连接层: dimension-wise gate、uncertainty gate、subject-disjoint OOF gate；
  4. 预处理: baseline/视频段差分的轻量门控，而不是强 subject/trial norm；
  5. 提交侧: 把 167 固化为 inference pipeline，避免只停留在 CV 实验。
```

## 168-200 追加观察：200 次阶段总结

| 编号 | 实验 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 168-187 | 20 个不同架构：Piecewise affine、PCHIP、video-time surface、lag prototype、uncertainty router、covariance transport、Kalman、Student-t Kalman、Chebyshev graph、lagged ridge、RBF、video residual、temporal rank、bagging ridge、ExtraTrees、NB bins、Nystroem、Mahalanobis KNN、PLS、temporal basis | `experiments/results/iteration_168_200_architectures_real200.json` | 28.7072 | 26.9366 | 30.4777 | 最好单模块为 195 ConformalMedianBandProjector。 |
| 188 | 第四批整合：PCHIP/covariance valence + prototype/Kalman arousal | `experiments/results/iteration_168_200_architectures_real200.json` | 28.7282 | 26.9046 | 30.5518 | valence 很强，但 arousal 明显变差，不能整模型使用。 |
| 189-199 | 11 个追加不同架构：uncertainty mixture、slope limiter、sign memory、low-rank factor、per-video affine、histogram、conformal band、derivative template、dimension coupling、multiresolution switch、jackknife shrink | `experiments/results/iteration_168_200_architectures_real200.json` | 28.7072 | 26.9366 | 30.4777 | 195 是最稳的安全投影。 |
| 200 | 里程碑融合：188 valence + 195 arousal | `experiments/results/iteration_168_200_architectures_real200.json` | 28.6912 | 26.9046 | 30.4777 | 当前全局最优；这是固定输出级 late fusion。 |

### 200 的“拼接”到底是什么

这里的拼接不是把神经网络层硬接在一起，也不是把 EEG/fNIRS 特征直接 concat。它是输出级 late fusion：

```text
每个专家都输出:
  f_k(x) = [v_k(x), a_k(x)]

最终模型输出:
  f_200(x) = [v_188(x), a_195(x)]
```

这么做有一个数学前提：比赛主指标是 valence/arousal 的平均 MAE，可以分解：

```text
Overall_MAE
  = (1 / 2N) * sum_i ( |v_i - vhat_i| + |a_i - ahat_i| )
  = 0.5 * MAE_valence + 0.5 * MAE_arousal
```

这个指标没有 `valence-arousal` 交叉项，所以如果一个模块只擅长 valence，另一个模块只擅长 arousal，按维度融合在目标函数上是成立的。

但这不等于可以随便拼。必须满足两个条件：

```text
1. 模块的作用机制不同，不是同一错误的重复版本；
2. 维度收益要通过 subject-disjoint CV 计算，而不是看训练集。
```

### 关键模块作用与计算结果

以 167 为参照，第三批的逻辑是：

```text
158 GaussianCopulaCovarianceCalibration:
  valence 变化: 26.9643 -> 26.9412  = -0.0231
  arousal 变化: 30.4838 -> 30.5041  = +0.0203

159 TrialPrototypeResidualRetrieval:
  valence 变化: 26.9643 -> 26.9813  = +0.0170
  arousal 变化: 30.4838 -> 30.4809  = -0.0029

167 fusion:
  y_v = y_158,v
  y_a = y_159,a
  overall = 28.7111
```

所以 167 的逻辑是：copula 校准修正 valence 的输出分布/协方差偏移，prototype 检索修正 arousal 的 trial 动态模板；两者互补，而不是同质平均。

以 167 为参照，第四批的关键结果是：

```text
167_PreviousBest_reference:
  overall = 28.7075
  valence = 26.9369
  arousal = 30.4782

188_FixedBatch4Fusion:
  overall = 28.7282
  valence = 26.9046  (-0.0323)
  arousal = 30.5518  (+0.0736)

195_ConformalMedianBandProjector:
  overall = 28.7072
  valence = 26.9366  (-0.0003)
  arousal = 30.4777  (-0.0005)

200_MilestoneSynthesisFusion:
  overall = 28.6912
  valence = 26.9046
  arousal = 30.4777
```

200 的具体公式是：

```text
先做 188 的 valence 专家:

  g = uncertainty_gate(candidate_std)
  v_188 = (1 - 0.35 * g_v) * v_169 + (0.35 * g_v) * v_173

其中:
  169 = Monotone PCHIP residual-bias calibration
  173 = Covariance transport calibration

再做 195 的 arousal 专家:

  low, high = 训练集中同 video/time bucket 的 10% / 90% 标签边界
  y_projected = clip(y_167, low, high)
  y_195 = 0.88 * y_167 + 0.12 * y_projected

最终:
  y_200 = [v_188, a_195]
```

### 是否完全可靠

还不能说完全可靠。重要风险是 188 的 valence 虽然总平均最好，但 fold 稳定性一般：

```text
188 valence 相对 167 的 fold delta:
  fold1 +0.2602
  fold2 -0.4878
  fold3 -0.6016
  fold4 +0.5171
  fold5 +0.0976
  fold6 +0.0209
```

这说明 188 更像“某些 subject 分布偏移下很有用的校准器”，不是所有 subject 都安全。相反，195 非常稳定：

```text
195 arousal 相对 167 的 fold delta:
  fold1 -0.0006
  fold2 -0.0001
  fold3 -0.0005
  fold4 -0.0006
  fold5 -0.0003
  fold6 -0.0006
```

所以 200 的合理解释是：

```text
195 是安全的小投影；
188 是高收益但有 subject 条件风险的 valence 校准器；
200 暂时利用了 188 的总体验证收益，但下一步必须给 188 加 OOF gate。
```

### 200 次实验后的模块分类

有效模块：

```text
1. 强先验/低自由度 prior:
   VideoTimeMean、PatternPrior、OOF Meta。
   作用：利用 video/time 的强共性，避免小数据下直接从 EEG/fNIRS 过拟合。

2. 输出级状态滤波:
   AlphaBeta、SlopeAdaptiveEMA、Kalman、Chebyshev graph、Huber slope limiter。
   作用：标签是 1 Hz 连续轨迹，弱平滑/弱状态约束能去掉预测噪声。

3. 鲁棒局部修正:
   Hampel、Tukey、Conformal median band。
   作用：只修局部异常点，不整体压平轨迹；比强平滑安全。

4. 分布/协方差校准:
   Prior-bin、Gaussian copula、PCHIP、Covariance transport。
   作用：修正预测值分布和真实标签分布之间的系统偏差，尤其对 valence 有效。

5. trial/template 动态检索:
   Prototype retrieval、derivative/template retrieval。
   作用：arousal 更像 trial 动态模板问题，但收益很小，且容易伤 valence。

6. 分维度 late fusion:
   167、200。
   作用：利用指标可分解性，让 valence/arousal 使用不同专家。
```

低效或风险模块：

```text
1. 大自由度残差学习:
   MLP、ExtraTrees、RBF、KNN、PLS、lagged ridge。
   问题：subject-disjoint 下容易学到 fold-specific 残差。

2. 强离散状态模型:
   HMM、过硬 change-point、AR2。
   问题：MER-PS 是连续情绪轨迹，硬量化会损伤细节。

3. 频域/低秩强压缩:
   FFT shrink、过强 low-rank。
   问题：会把真实短时波动当成噪声抹掉。

4. 无门控专家平均:
   Dirichlet evidence average、简单 stacking。
   问题：会稀释维度专长；平均不是创新，门控才是关键。
```

下一步创新方向：

```text
不要继续人工拼维度。
应该训练一个 subject-disjoint OOF gate:

  z = [candidate_std, slope, video_id, time, prior_value, expert_disagreement]
  g_v = sigmoid(w_v^T z)
  v = g_v * v_188 + (1 - g_v) * v_167

  g_a = sigmoid(w_a^T z)
  a = g_a * a_195 + (1 - g_a) * a_167

并且 gate 必须用 nested/OOF 方式学，不能直接用当前验证集挑。
```

## 最新滚动状态：当前最优与下一批

```text
当前最优:
  200_MilestoneSynthesisFusion_VBatch4_AConformalBand
  overall = 28.6912
  valence = 26.9046
  arousal = 30.4777

来源:
  valence = 188_FixedBatch4Fusion
  arousal = 195_ConformalMedianBandProjector

结果文件:
  experiments/results/iteration_168_200_architectures_real200.json

实现文件:
  tools/cross_fold_to200_architectures.py

下一批:
  201-220 不再盲目拼接，重点做 OOF gate / nested gate。
  目标是判断 188 什么时候该用，而不是永远使用 188。
```

## 201-211 追加观察：从 200 次经验推导门控模块

这轮不再创新新模型，而是把 200 次实验总结出的逻辑写成可验证模块：

```text
稳定底座:
  y_167 = [v_167, a_167]

高收益但有风险的 valence 专家:
  v_188

稳定安全的 arousal 专家:
  a_195

目标:
  v = g(z) * v_188 + (1 - g(z)) * v_167
  a = a_195
```

门控依据：

```text
201: candidate uncertainty low 时信任 188
202: candidate uncertainty high 时信任 188
203: slope low 时信任 188
204: slope high 时信任 188
205: correction |188 - 167| small 时信任 188
206: 188 接近 candidate consensus 时信任 188
207: uncertainty/correction/consensus 的低风险混合门控
208: oracle row gate，上界分析，不可提交
209: 训练集同 video/time prior soft gate
210: 训练集同 video/time prior hard gate
211: 训练集同 video/time prior conservative gate
```

新推导模块是 `209-211 TrainVideoTimePriorGate`。它的数学逻辑是：

```text
先从训练 subject 中估计同视频同时间段的 valence 参考:

  m(video, bucket) = median(y_train_valence | same video, same time bucket)

计算 188 是否比 167 更接近这个训练先验:

  advantage = |v_167 - m| - |v_188 - m|

如果 advantage > 0，说明 188 更符合跨主体 video-time 先验。

soft gate:
  g = sigmoid(advantage / scale)
  v = g * v_188 + (1 - g) * v_167

hard gate:
  g = 1[advantage > 0]
```

这个设计的依据来自前 200 次结论：

```text
1. MER-PS 的 video/time prior 很强；
2. 188 本质是 valence 分布/协方差校准器；
3. 188 有时大幅改善，有时大幅伤害；
4. 因此不能永远用 188，应该只在它把输出推向训练 video-time 先验时使用。
```

运行结果：

| 编号 | 模块 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 200* | 当前手工融合复现 | `experiments/results/iteration_201_211_train_prior_gate.json` | 28.6956 | 26.9028 | 30.4884 | 同脚本同 seed 下的参照；全局最好仍以 `real200` 的 28.6912 为准。 |
| 201 | UncertaintyLow gate | `experiments/results/iteration_201_211_train_prior_gate.json` | 28.6967 | 26.9050 | 30.4884 | 最接近 200，但没有超过。 |
| 209 | TrainVideoTimePrior soft gate | `experiments/results/iteration_201_211_train_prior_gate.json` | 28.7059 | 26.9233 | 30.4884 | 逻辑成立但信号不够强。 |
| 211 | TrainVideoTimePrior conservative gate | `experiments/results/iteration_201_211_train_prior_gate.json` | 28.7070 | 26.9255 | 30.4884 | 比 209 稍差。 |
| 210 | TrainVideoTimePrior hard gate | `experiments/results/iteration_201_211_train_prior_gate.json` | 28.7196 | 26.9508 | 30.4884 | 硬选择太粗。 |
| 208 | Oracle row gate | `experiments/results/iteration_201_211_train_prior_gate.json` | 28.2650 | 26.0416 | 30.4884 | 不可提交，只说明 gate 空间很大。 |

关键结论：

```text
手写规则门控没有超过 200。
这不是坏消息，说明 188 的适用条件不是单一的 uncertainty/slope/video-time prior。

Oracle row gate = 28.2650，说明如果能正确判断每个点该用 188 还是 167，
理论空间非常大。

但 209-211 没有吃到这个空间，说明:
  1. video-time prior 只能解释一部分；
  2. subject shift 是关键；
  3. hard rule 不足，需要 nested OOF learned gate。
```

下一步不应该继续写规则，而应该做真正的 OOF gate：

```text
训练样本必须来自 inner leave-subject-out:
  对训练 subject s:
    用其他训练 subject 生成 v_167_s, v_188_s, a_195_s
    计算 label:
      target_gate = 1[ |v_188_s - y_s| < |v_167_s - y_s| ]

然后训练轻量 gate:
  z = [
    candidate_std,
    slope,
    |v_188 - v_167|,
    |v_188 - candidate_mean| / std,
    video_id,
    time_bucket,
    prior_value,
    distance_to_video_time_median
  ]

  g = sigmoid(w^T z)
  v = g * v_188 + (1 - g) * v_167
```

这一步才是真正从“经验总结”走向“新模块”的方向：不新增复杂 backbone，而是学习已有专家的适用条件。

## 212-220 追加观察：原创层级残差场模块

这轮开始不再复现外部模型，也不再继续堆专家，而是根据前 200 次实验经验构建我们自己的模块：

```text
Hierarchical Residual Field, HRF
层级残差场
```

设计动机来自已有结论：

```text
1. 大自由度残差学习容易过拟合：
   MLP、ExtraTrees、RBF、KNN、PLS 都不稳定。

2. 低自由度校准有效：
   Prior-bin、PCHIP、Gaussian copula、Covariance transport 对 valence 有帮助。

3. 局部鲁棒修正有效：
   Hampel、Conformal band、Huber slope limiter 只做小修正，风险低。

4. arousal 更难修：
   多数残差/模板模块对 arousal 只有极小收益，甚至容易伤。
```

因此 HRF 不直接学习复杂函数，而是学习一个被强 shrink 的条件残差场：

```text
基础预测:
  y_base = y_200

训练残差:
  r_i = y_i - p_i

其中 p_i 是训练 subject 的 OOF prior，不使用该 subject 自己标签生成 prior。

条件变量:
  video
  time_bucket = floor(t / 8)
  value_bin = bin(p_i)
  slope_bin = bin(|dp_i / dt|)
```

残差场由多个低维桶的鲁棒中位残差相加：

```text
R(x) =
  0.32 * R_video,time
  + 0.22 * R_video
  + 0.16 * R_time
  + 0.20 * R_value_bin
  + 0.10 * R_slope_bin

每个桶:
  R_g = shrink(n_g, MAD_g) * median(r_i | i in g)

shrink(n, MAD) =
  n / (n + k) * 1 / (1 + MAD / 24)

最终:
  y_hat = y_base + lambda * clip(R(x))
```

这和之前的“拼接”不同：它不是选已有专家，而是在已有预测上学习一个可解释的、低自由度的系统性误差场。

本轮实验结果：

| 编号 | 模块 | 输出文件 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 200 | 当前手工融合 | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6912 | 26.9046 | 30.4777 | 参照。 |
| 212 | HRF small，两维都修 | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6915 | 26.9032 | 30.4798 | valence 好一点，arousal 变差。 |
| 213 | HRF medium，两维都修 | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6918 | 26.9025 | 30.4811 | valence 更好，但 arousal 伤更多。 |
| 214 | HRF sign consensus，两维都修 | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6907 | 26.9040 | 30.4774 | sign consensus 能抑制部分风险。 |
| 215 | HRF valence only | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6901 | 26.9025 | 30.4777 | 证明残差场主要应修 valence。 |
| 216 | HRF arousal only | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6929 | 26.9046 | 30.4811 | arousal 残差场不应使用。 |
| 218 | HRF valence consensus mask | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6869 | 26.8961 | 30.4777 | 当前全局最优。 |
| 219 | HRF valence consensus blend | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6880 | 26.8983 | 30.4777 | 有效但弱于 hard mask。 |
| 220 | HRF valence soft confidence | `experiments/results/iteration_212_220_residual_field_module.json` | 28.6909 | 26.9040 | 30.4777 | 过软，收益小。 |

218 的核心逻辑：

```text
先计算两个残差场:
  R_medium: 普通层级残差场
  R_sign:   带 sign-consensus shrink 的残差场

如果二者修正方向一致:
  use R_medium
否则:
  correction = 0

公式:
  delta_medium = y_213,v - y_200,v
  delta_sign   = y_214,v - y_200,v

  mask = 1[delta_medium * delta_sign > 0]

  y_218,v = y_200,v + mask * delta_medium
  y_218,a = y_200,a
```

为什么 218 有效：

```text
1. 215 说明 valence 残差场确实有信息：
   valence 26.9046 -> 26.9025。

2. 216 说明 arousal 残差场是噪声：
   arousal 30.4777 -> 30.4811。

3. 218 比 215 更好，说明不是所有 valence 残差都可靠。
   只有普通残差场和 sign-consensus 残差场方向一致时，
   correction 更可能是系统偏差，而不是 fold-specific 噪声。

4. 平均修正幅度很小：
   valence mean abs correction 约 0.194。
   这符合 MER-PS 小数据条件：只能做小修正，不能大幅改轨迹。
```

当前模块逻辑总结：

```text
最终可用结构:

  y_167 = stable base
  y_188 = risky valence calibration expert
  y_195 = safe arousal conformal projection

  y_200 = [v_188, a_195]

  R_v = hierarchical residual field over video/time/value/slope
  C_v = sign-consensus agreement mask

  y_218,v = y_200,v + C_v * R_v
  y_218,a = y_200,a
```

这可以作为我们自己的原创模块命名：

```text
SCRF: Sign-Consistent Hierarchical Residual Field
中文：符号一致层级残差场
```

它不是外部论文模块，而是从本数据集实验规律推导出来的：

```text
强先验底座 + 分维度专家 + 低自由度残差场 + 符号一致性风险控制
```

下一步可以继续优化 SCRF，而不是换模型：

```text
1. 用 OOF learned shrink 替代手写 shrink；
2. residual source 从 PatternPrior residual 升级为 OOF y_200 residual；
3. 对 valence 做 subject-shift detector，避免 fold4 这类 188/HRF 过修；
4. 把 SCRF 固化进最终 Codabench inference pipeline。
```

## 最新滚动状态更新：SCRF 成为当前最优

```text
当前最优:
  218_HierResidualField_valenceConsensusMask
  overall = 28.6869
  valence = 26.8961
  arousal = 30.4777

来源:
  base = 200_MilestoneSynthesisFusion_VBatch4_AConformalBand
  valence correction = Sign-Consistent Hierarchical Residual Field
  arousal = 195_ConformalMedianBandProjector, 不再追加残差场

结果文件:
  experiments/results/iteration_212_220_residual_field_module.json

实现文件:
  tools/cross_fold_residual_field_module.py

下一步:
  不再盲目增加模型。
  围绕 SCRF 做三件事：
    1. 学习 shrink / confidence；
    2. 构造更严格的 OOF y_200 residual source；
    3. 做 subject-shift 风险检测，避免 valence 过修。
```
## 247-262 追加观察：NOVA-v2 神经血氧融合第二轮

这一轮不是继续堆模型，而是把 EEG 和 fNIRS 分成两个残差测量源来做融合：

```text
E(t) = EEG residual expert
F(t) = fNIRS residual expert
N(t) = neurovascular interaction expert

目标不是直接相信 E 或 F，而是估计：
  r*(t) = hidden affective residual

当 E 和 F 的符号一致、幅度重叠、历史 OOF 误差较小，才认为它们共同指向 r*(t)。
```

本轮实现文件：

```text
tools/cross_fold_neurovascular_fusion_v2.py
```

结果文件：

```text
experiments/results/iteration_247_262_neurovascular_fusion_v2.json
```

主要模块：

| 编号 | 模块 | 核心逻辑 |
| --- | --- | --- |
| 247 | ReliabilitySoftmax3 | 用 EEG/fNIRS/NV 三个 OOF MSE 做 softmax 可靠性加权。 |
| 248 | TriModalSignIntersection | EEG、fNIRS、NV 三者同号才修正。 |
| 249 | MinMagnitudeAgreement | EEG 和 fNIRS 同号时，只取较小幅度，防止单模态过度修正。 |
| 250 | SignedGeometricConsensus | 同号时使用几何平均幅度，强调共同强响应。 |
| 251 | OrthogonalNeurovascular | 从 NV 中扣掉 EEG/FNIRS 可解释部分，只保留正交血氧证据。 |
| 252 | RatioReliabilityGate | 用两模态幅度比例做置信门控。 |
| 253 | TemporalDeltaAgreement | 同时要求残差方向和局部变化方向一致。 |
| 254 | EEGLeadFNIRSConfirm | EEG 快响应由延迟 fNIRS 进行确认。 |
| 255 | CoherenceWeightedSoftmax | 用滚动相关性修正三模态 softmax。 |
| 256-262 | state/attention/trimmed/huber variants | 检验状态可信度、轻量 attention 和稳健均值是否有效。 |

最佳结果：

| 方法 | Overall MAE | Valence MAE | Arousal MAE |
| --- | ---: | ---: | ---: |
| 098_PatternPrior_reference | 28.7462 | 26.9738 | 30.5186 |
| 242_OOFAgreementWeighted | 28.7352 | 26.9518 | 30.5186 |
| 249_MinMagnitudeAgreement | 28.7297 | 26.9408 | 30.5186 |

关键结论：

```text
1. 复杂 attention/softmax 没有压过简单交集。
2. 最有效的是 MinMagnitudeAgreement：

   same = sign(E) == sign(F)
   r = 1[same] * sign(E + F) * min(|E|, |F|)

3. 它的含义很明确：
   两个模态方向一致时才修正；
   修正幅度不超过较弱模态支持的幅度；
   因此它比平均、拼接、attention 更抗 subject-disjoint 过拟合。
4. 仍然主要改善 valence，arousal 不应该主动修正。
```

## 263-276 追加观察：CCMI 保守跨模态交集

基于 247-262 的结果，继续把有效结构提炼成一个更像论文模块的形式：

```text
CCMI: Conservative Cross-Modal Intersection
中文：保守跨模态交集
```

模块假设：

```text
EEG 是快速神经响应，fNIRS 是慢速血氧响应。
两者都很噪，且 subject-disjoint 条件下容易学到伪相关。

因此不做直接拼接，而是把两个模态看成对同一隐变量 residual r*(t) 的两个 noisy observations：

E(t) = r*(t) + eps_e
F(t) = r*(t) + eps_f

如果 sign(E) != sign(F)，说明两个观测对 residual 方向没有共识，应拒绝修正。
如果 sign(E) == sign(F)，可信幅度不应超过 min(|E|, |F|)。
```

基础公式：

```text
agree(t) = 1[sign(E(t)) = sign(F(t))]
overlap(t) = min(|E(t)|, |F(t)|)
direction(t) = sign(E(t) + F(t))

r_ccmi(t) = agree(t) * direction(t) * overlap(t)
```

进一步加入 prior slope gate：

```text
s(t) = |d prior(t) / dt|

从 OOF training 中估计不同 prior slope bucket 下，
r_ccmi 是否能减少 residual 误差：

helpful = 1[ |residual - r_ccmi| < |residual| ]

gate(bucket) = helpful_rate(bucket) / global_helpful_rate

最终：
  r(t) = gate(bucket(s(t))) * r_ccmi(t)
```

本轮实现文件：

```text
tools/cross_fold_neurovascular_ccmi.py
```

结果文件：

```text
experiments/results/iteration_263_276_ccmi_neurovascular.json
```

最佳结果：

| 方法 | Overall MAE | Valence MAE | Arousal MAE |
| --- | ---: | ---: | ---: |
| 098_PatternPrior_reference | 28.7462 | 26.9738 | 30.5186 |
| 237_DualModalityAgreementGate | 28.7390 | 26.9595 | 30.5186 |
| 242_OOFAgreementWeighted | 28.7352 | 26.9518 | 30.5186 |
| 249_MinMagnitudeAgreement | 28.7297 | 26.9408 | 30.5186 |
| 273_CCMI_PriorSlopeGate | 28.7176 | 26.9206 | 30.5146 |

相对提升：

```text
对 098:
  overall 28.7462 -> 28.7176, 提升 0.0286
  valence 26.9738 -> 26.9206, 提升 0.0532
  arousal 30.5186 -> 30.5146, 提升 0.0040

对上一轮最优 249:
  overall 28.7297 -> 28.7176, 提升 0.0121
  valence 26.9408 -> 26.9206, 提升 0.0202
```

和全局最优的关系：

| 方法 | Overall MAE | Valence MAE | Arousal MAE |
| --- | ---: | ---: | ---: |
| 222_BCRF_onSCRF | 28.6868 | 26.8958 | 30.4777 |
| 273_CCMI_PriorSlopeGate | 28.7176 | 26.9206 | 30.5146 |

判断：

```text
CCMI 是当前最好的 EEG-fNIRS 融合残差模块，但还没有超过 SCRF/BCRF 的全局输出校准。
它的价值不在于现在直接替代 222，而在于提供了一个更像论文贡献的多模态融合逻辑：

  1. EEG/fNIRS 不是拼接，而是残差证据交集；
  2. 幅度采用 min-overlap，控制单模态幻觉；
  3. prior slope gate 解释了何时允许生理信号修正强先验；
  4. subject-disjoint OOF 训练避免训练 subject 自身标签泄漏。

下一步最合理的是：
  y_final = y_222 + lambda * CCMI(E, F, prior_slope)

但这需要先缓存 200/218/222 的逐样本 OOF 预测，否则每次重建当前最强基座会非常慢。
```

## 框架级拆分：从数据读取到输出头

现在把整个 MER-PS 系统拆成可以分别优化的模块，而不是只说“换模型”：

| 层级 | 模块 | 当前实现 | 可优化方向 | 当前判断 |
| --- | --- | --- | --- | --- |
| 1 | 数据读取 | `emotion_merps/features.py` 读取 MATLAB v5 `.mat`、`sample_ids.csv`、subject/video/time。 | 读取更多 fNIRS signal type；缓存 MAT 解析；检查 sample 对齐。 | 可优化，已经验证全 6 fNIRS 有小提升。 |
| 2 | 基线校正 | EEG/fNIRS 都减 5 秒 baseline 均值。 | baseline mean/std/slope、多种 baseline 归一化、subject robust normalization。 | 还没系统扫，值得下一轮。 |
| 3 | EEG 特征 | 1 Hz 分段，5 个频带相对功率 log bandpower。 | 绝对功率、差分熵 DE、PSD 统计、半球不对称、通道区域聚合。 | 这是信号端重要入口。 |
| 4 | fNIRS 特征 | HbO/HbR/HbT 的 mean/std/slope。 | 加 Abs 780/805/830；HRF 延迟；氧合差 HbO-HbR；慢窗 rolling features。 | 全 6 类型有效但提升小。 |
| 5 | 强先验候选 | video/time robust median、lag、smooth、quantile gate。 | 更细 time bucket、视频类别情绪先验、跨 subject 稳健统计。 | 当前主要性能来源。 |
| 6 | OOF meta 特征 | leave-one-subject-out 生成训练残差，避免 subject 泄漏。 | 缓存 OOF 训练矩阵；更严格 fold 内缓存。 | 必须保留，否则评估会虚高。 |
| 7 | 输出头 104/125/167/200 | HGB residual、滤波、prototype、manual dimwise fusion。 | 缓存逐样本 OOF 预测；减少重建成本；按维度拆头。 | 很强但重建慢。 |
| 8 | SCRF/BCRF | 层级残差场 + credible residual field。 | 学习 shrink；和信号残差做互相 veto/confirm。 | 当前全局最优。 |
| 9 | EEG/fNIRS 融合 | NOVA/CCMI 残差融合。 | 更好的跨模态交集、HRF 延迟、可靠性门控。 | CCMI 是目前最好的多模态模块。 |
| 10 | 最终输出 | scale/clip/smooth，整数 `[1,255]`。 | 维度专属 head；valence 修正、arousal 保守；输出校准。 | arousal 不宜乱修。 |

从这张表看，后续优化不应该平均用力。优先级应该是：

```text
最高优先级:
  1. 数据读取/预处理层：EEG 特征、fNIRS 类型、baseline normalization。
  2. EEG-fNIRS 融合层：CCMI 这种低自由度、强约束模块。
  3. 输出头层：缓存 200/218/222 后，把 CCMI 接到强头上。

低优先级:
  1. 大参数 Transformer/Mamba 继续堆深。
  2. 复杂 attention 拼接。
  3. arousal 残差大幅修正。
```

## 277-292 尝试：把 CCMI 接到 200/218/222 输出头

我写了框架级输出头融合脚本：

```text
tools/cross_fold_framework_head_overlay.py
```

设计目标：

```text
base ∈ {200, 218, 222}
signal = CCMI(E, F, prior_slope)

测试:
  y = base + lambda * signal
  y = base + lambda * signal * 1[sign(signal)=sign(base-prior)]
  y = base + lambda * signal * (1 - BCRF_confidence)
```

但是 4 subject 冒烟测试 10 分钟没有完成，瓶颈不是 CCMI，而是重建 200/218/222 这一串输出头太慢。

结论：

```text
这个方向仍然正确，但需要先做工程缓存：
  cache_current_best_oof_predictions.npz

缓存内容:
  sample_id
  y_true
  p098
  p200
  p218
  p222
  bcrf_confidence

之后所有 head-overlay 实验就不需要反复重建 104/125/167/200/218/222。
```

## 293-306 数据读取优化：全 6 类 fNIRS 信号

之前 fNIRS 只使用前三类：

```text
0 HbO
1 HbR
2 HbT
```

这一轮把数据读取扩展为全部 6 类：

```text
0 HbO
1 HbR
2 HbT
3 Abs 780 nm
4 Abs 805 nm
5 Abs 830 nm
```

对应代码改动：

```text
tools/cross_fold_neurovascular_fusion.py
tools/cross_fold_neurovascular_ccmi.py
```

新增缓存：

```text
experiments/features/neurovascular_precompute_fnirs_all6.npz
```

特征维度变化：

| 配置 | EEG shape | fNIRS shape | 缓存大小 |
| --- | --- | --- | ---: |
| 原始 3 类 fNIRS | `[36864, 64, 5]` | `[36864, 51, 9]` | 108.6 MB |
| 全 6 类 fNIRS | `[36864, 64, 5]` | `[36864, 51, 18]` | 163.7 MB |

结果：

| 方法 | fNIRS 类型 | Overall MAE | Valence MAE | Arousal MAE |
| --- | --- | ---: | ---: | ---: |
| 273_CCMI_PriorSlopeGate | 0,1,2 | 28.7176 | 26.9206 | 30.5146 |
| 273_CCMI_PriorSlopeGate | 0,1,2,3,4,5 | 28.7145 | 26.9144 | 30.5146 |

判断：

```text
全 6 类 fNIRS 有小幅有效增益：
  overall 提升 0.0031
  valence 提升 0.0062
  arousal 不变

说明 Abs 780/805/830 不是强信号，但在 CCMI 的保守交集约束下能提供一点额外证据。
这也反过来支持我们的融合逻辑：
  高维输入不能直接相信；
  需要跨模态一致性 + prior slope gate 控制。
```

下一轮最值得做的数据层优化：

```text
1. EEG 从 bandpower 扩展到 bandpower + DE + hemispheric asymmetry。
2. fNIRS 做 baseline z-score，而不是只减 baseline mean。
3. 为 200/218/222 建 OOF 缓存，再做 CCMI-on-222 的快速 head overlay。
```

## 307 预处理反例：trial 内 z-score 不适合当前 CCMI

我继续测试了一个预处理层假设：

```text
对每个 subject-video trial，
在 102 个 1 Hz 时间点内分别对 EEG/fNIRS 特征做 z-score。

目的：
  去掉每个 trial 的幅值偏移；
  只保留随时间变化的动态形状。
```

实现入口同样在：

```text
tools/cross_fold_neurovascular_fusion.py
tools/cross_fold_neurovascular_ccmi.py
```

冒烟结果：

| 配置 | 验证范围 | Overall MAE | Valence MAE | Arousal MAE |
| --- | --- | ---: | ---: | ---: |
| 全 6 fNIRS，无 trial z-score | test_1-test_4 smoke | 28.1670 | 25.1404 | 31.1936 |
| 全 6 fNIRS，trial z-score | test_1-test_4 smoke | 28.3478 | 25.5020 | 31.1936 |

判断：

```text
trial 内 z-score 明显变差，因此没有继续跑完整 24 subject。

这说明：
  1. EEG/fNIRS 的绝对幅值或慢漂移里仍有情绪相关信息；
  2. 不能简单把每个视频内部标准化；
  3. 更合理的是 baseline z-score：
       用视频前 5 秒 baseline 的均值和方差做标准化，
       而不是用整个视频 trial 自身统计量。
```

后续预处理优化应避免：

```text
不要用整段视频自身统计量把动态幅值全部抹掉。
```

更值得尝试：

```text
baseline_zscore:
  x' = (x - mean(baseline)) / (std(baseline) + eps)

robust_subject_scale:
  只用训练 subject 或测试 subject 自身 baseline 做尺度归一化。
```

## 三模块五候选对照：数据预处理、CCMI 融合、输出头

本轮按三个模块分别找了 5 个候选。注意：

```text
Full 24-subject CV 可以横向比较。
4-subject smoke 只用于快速否决明显坏的方向，不能和 24-subject CV 的绝对 MAE 直接比较。
```

机器可读总结：

```text
experiments/results/three_module_optimization_summary.json
```

独立 Markdown：

```text
THREE_MODULE_OPTIMIZATION_SUMMARY.md
```

### A. 数据预处理模块

| 编号 | 候选 | 原因 | 验证 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| D1 | 3 类 fNIRS + baseline mean subtraction | 只用 HbO/HbR/HbT，减去视频前 baseline 均值。 | Full 24-subject CV | 28.7176 | 26.9206 | 30.5146 | 强参考线。 |
| D2 | 6 类 fNIRS + baseline mean subtraction | 增加 Abs 780/805/830，让 CCMI 自己筛掉噪声。 | Full 24-subject CV | 28.7145 | 26.9144 | 30.5146 | 最好数据预处理。 |
| D3 | 6 类 fNIRS + no baseline subtraction | 测试慢漂移和绝对偏移是否有情绪信息。 | Full 24-subject CV | 28.7462 | 26.9738 | 30.5186 | 不泛化，退回 098。 |
| D4 | 6 类 fNIRS + trial z-score | 去掉 trial 内幅值，只保留动态形状。 | 4-subject smoke | 28.3478 | 25.5020 | 31.1936 | smoke 明显变差，不跑 full。 |
| D5 | 6 类 fNIRS + subject z-score | 去掉 subject 级幅值差异，测试跨主体尺度漂移。 | 4-subject smoke | 28.3259 | 25.4582 | 31.1936 | smoke 明显变差，不跑 full。 |

数据预处理结论：

```text
保留 baseline mean subtraction 是必要的。
增加 Abs 780/805/830 有小收益。
不要用 trial/subject 内 z-score 粗暴抹掉幅值信息。

当前推荐：
  fNIRS types = 0,1,2,3,4,5
  baseline_correction = true
  feature_normalization = none
```

### B. CCMI 融合模块

| 编号 | 候选 | 原因 | 验证 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| C1 | OOF agreement weighted | 用 EEG/fNIRS OOF MSE 做可靠性加权，要求同号。 | Full 24-subject CV | 28.7352 | 26.9518 | 30.5186 | 第一个稳定多模态增益。 |
| C2 | MinMagnitudeAgreement | 同号时只取较小幅度，避免单模态过度修正。 | Full 24-subject CV | 28.7297 | 26.9408 | 30.5186 | 简单交集优于 attention。 |
| C3 | CCMI MinOverlap | 把 C2 形式化为保守跨模态交集。 | Full 24-subject CV | 28.7186 | 26.9226 | 30.5146 | 有效但不是最优。 |
| C4 | CCMI HRFDelayedFNIRS | 用延迟 fNIRS 确认 EEG 快响应，符合血氧滞后。 | Full 24-subject CV | 28.7178 | 26.9211 | 30.5146 | HRF 延迟有用。 |
| C5 | CCMI PriorSlopeGate | 按 prior slope bucket 学习何时允许生理残差修正。 | Full 24-subject CV | 28.7145 | 26.9144 | 30.5146 | 当前最好 EEG-fNIRS 模块。 |

CCMI 融合结论：

```text
复杂融合不是关键。
关键是三个约束：
  1. EEG 和 fNIRS 方向一致；
  2. 修正幅度不超过两模态共同支持的交集；
  3. 只在 prior slope 状态显示“生理残差曾经有用”时放大。

当前推荐：
  273_CCMI_PriorSlopeGate
```

### C. 输出头/缓存模块

| 编号 | 候选 | 原因 | 验证 | Overall MAE | Valence MAE | Arousal MAE | 判断 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| H1 | 200 manual dimwise fusion | valence 用风险专家，arousal 用 conformal median band。 | Full 24-subject CV | 28.6912 | 26.9046 | 30.4777 | 强输出头基线。 |
| H2 | 218 SCRF | 只在 valence 上做符号一致层级残差场。 | Full 24-subject CV | 28.6869 | 26.8961 | 30.4777 | 最强可解释校准。 |
| H3 | 222 BCRF on SCRF | BCRF 可信残差场叠加 SCRF。 | Full 24-subject CV | 28.6868 | 26.8958 | 30.4777 | 当前全局最好。 |
| H4 | 224 BCRF brake disagreement | BCRF/SCRF 分歧时刹车。 | Full 24-subject CV | 28.6880 | 26.8983 | 30.4777 | 更保守但不如 222。 |
| H5 | arousal residual probe | 测试输出残差是否应修 arousal。 | Full 24-subject CV | 28.6912 | 26.9046 | 30.4778 | arousal 不该主动修。 |

输出头/缓存结论：

```text
输出头比信号残差更强。
当前最优仍是 222_BCRF_onSCRF。
但 222 重建成本很高，导致 CCMI-on-222 直接实验超时。

工程优化优先级：
  建 cache_current_best_oof_predictions.npz

需要缓存:
  sample_id
  y_true
  p098
  p200
  p218
  p222
  bcrf_confidence
```

### 组合结论

当前已验证最佳组合分两类：

| 组合 | 内容 | Overall MAE | 状态 |
| --- | --- | ---: | --- |
| 最佳生理信号路径 | D2 + C5 over 098 | 28.7145 | 已完整验证 |
| 最佳输出头路径 | H3: 222_BCRF_onSCRF | 28.6868 | 已完整验证 |
| 目标最终路径 | H3 + small CCMI residual | 待验证 | 需要先做逐样本输出头缓存 |

因此下一步不是再随机试 20 个模型，而是：

```text
1. 先做输出头逐样本 OOF 缓存。
2. 再快速测试:
     y = p222 + lambda * CCMI
     y = p222 + lambda * CCMI * 1[CCMI 与 SCRF/BCRF 同号]
     y = p222 + lambda * CCMI * (1 - BCRF_confidence)
3. 如果这一步不提升，说明信号残差和 SCRF/BCRF 校准捕获的是同一类 valence 误差；
   如果提升，CCMI 就可以作为论文里真正和输出校准互补的多模态模块。
```
## Iteration 321-335: No-video/time-prior signal-only validation

Purpose: remove VideoTimeMean, PatternPrior, video ID, timestamp, and label-derived video-time cells, then test whether EEG/fNIRS signals alone can predict valence-arousal trajectories under the same subject-disjoint CV protocol.

Result file: `experiments/results/iteration_321_335_no_video_prior_signal.json`

| Method | Overall MAE | Valence MAE | Arousal MAE | Meaning |
| --- | ---: | ---: | ---: | --- |
| Official ASAC demo | 47.0087 | 49.2285 | 44.7890 | Official-style small model reference on fixed split |
| 321_Center128_noPrior | 47.5663 | 52.1980 | 42.9346 | No signal, no video/time prior |
| 333_PCAEarlyDirectRidge_c8_a10000p0_SignalSmooth5 | 47.0764 | 50.6582 | 43.4946 | Best direct EEG/fNIRS no-prior model in this batch |
| 222_BCRF_onSCRF | 28.6868 | 26.8958 | 30.4777 | Current global best with strong prior + credible residual calibration |

Key interpretation:

```text
1. Direct signal-only modeling improves Center128 by 0.4899 MAE, so EEG/fNIRS contain usable information.
2. Direct signal-only modeling is still 18.3896 MAE worse than the current best prior-calibrated framework.
3. This confirms that most of the current leaderboard gain comes from video/time label structure and output calibration.
4. For a publishable method, we should clearly separate:
   - no-prior physiological decoding ability;
   - video/time prior modeling;
   - reliable residual correction over that prior.
```

## Iteration 336-535: 200 physiological-only experiments

Dedicated log:

```text
PHYSIO_ONLY_ITERATION_LOG.md
```

Result files:

```text
experiments/results/iteration_336_355_no_prior_physio_dimwise.json
experiments/results/iteration_356_375_no_prior_physio_adaptive.json
experiments/results/iteration_376_395_no_prior_physio_huber.json
experiments/results/iteration_396_415_no_prior_physio_calibration.json
experiments/results/iteration_416_435_no_prior_physio_asym_refine.json
experiments/results/iteration_436_535_no_prior_physio_final100.json
```

Best physiological-only result after these 200 new candidates:

| Method | Overall MAE | Valence MAE | Arousal MAE |
| --- | ---: | ---: | ---: |
| 511_StateTrialStartCenterBlend40 | 46.4686 | 50.0025 | 42.9346 |

Core module:

```text
RAVC-S = Huber PCA16 valence decoder
       + asymmetric negative valence residual shrink
       + trial-onset center brake
       + center arousal
```
