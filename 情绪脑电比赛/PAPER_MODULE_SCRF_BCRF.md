# SCRF-BCRF 原创模块提案

## 模块名称

```text
SCRF-BCRF: Sign-Consistent Bayesian Credible Residual Field
中文名：符号一致贝叶斯可信残差场
```

## 要解决的问题

MER-PS 是 subject-disjoint 的连续情绪回归任务。前 200 组实验显示，大容量模型很容易学习到训练主体中的偶然残差，导致跨主体泛化不稳定。真正稳定的收益主要来自三类低自由度信息：

```text
1. video/time 情绪诱发先验；
2. valence 的系统性校准残差；
3. 对 arousal 的保守投影和限幅。
```

因此新模块不再追求更大的 backbone，而是回答一个更关键的问题：

```text
什么时候一个残差值得修正？
```

## 核心公式

给定基础预测：

```text
y_base = [v_base, a_base]
```

训练集 OOF 残差为：

```text
r_i = y_i - p_i
```

在多个低维条件视角中统计残差：

```text
g in {
  video-time,
  video-value,
  time-value,
  value-slope,
  video,
  time
}
```

每个残差桶估计：

```text
m_g   = median(r_i | i in g)
MAD_g = median(|r_i - m_g|)
n_g   = |g|

se_g = 1.4826 * MAD_g / sqrt(n_g)
z_g  = |m_g| / (se_g + 1)
s_g  = |mean(sign(r_i))|
```

可信权重：

```text
w_g =
  n_g / (n_g + k)
  * 1 / (1 + MAD_g / c)
  * sigmoid(z_g - tau)
  * s_g
```

多视角合成：

```text
c_g = w_g * m_g

mean_corr =
  sum_h(w_h * c_h) / sum_h(w_h)

sign_agreement =
  |sum_h(w_h * sign(c_h))| / sum_h(w_h)

dispersion_gate =
  1 / (1 + std(c_h) / d)

confidence =
  count_confidence * sign_agreement * dispersion_gate

delta =
  lambda * confidence * clip(mean_corr)
```

最终预测：

```text
y_hat = y_base + delta
```

当前实验证明，`delta` 只应主要作用于 valence：

```text
v_hat = v_base + delta_v
a_hat = a_base
```

## 与普通残差学习的区别

普通残差学习：

```text
delta = f(features)
```

风险是 `f` 的自由度较高，容易把主体噪声当成规律。

SCRF-BCRF：

```text
delta = credible_low_dim_residual_field(video, time, value, slope)
```

它限制了残差来源，并且要求：

```text
1. 样本数足够；
2. MAD 噪声足够低；
3. 残差方向符号一致；
4. 近似 z-score 支持残差非零；
5. 多个残差视角方向一致。
```

## 当前实验结果

使用 subject-disjoint folds，结果来自：

```text
experiments/results/iteration_221_228_bcrf_module_seed2026.json
```

| 方法 | Overall MAE | Valence MAE | Arousal MAE |
| --- | ---: | ---: | ---: |
| 200_CurrentManualFusion | 28.6912 | 26.9046 | 30.4777 |
| 218_SCRF_reference | 28.6869 | 26.8961 | 30.4777 |
| 222_BCRF_onSCRF | 28.6868 | 26.8958 | 30.4777 |

结论：

```text
1. SCRF 是当前实用层面的主要增益来源。
2. BCRF 在 SCRF 上只有极小提升，但提供了更清晰的可信度建模。
3. arousal 残差修正没有稳定收益，应保持冻结。
```

## 论文创新性评价

这个模块的创新点不在于提出一个更大的模型，而在于提出一种适合小样本跨主体生理情绪预测的可信残差校准机制：

```text
1. 从大量实验中归纳出 valence/arousal 非对称可校正性；
2. 把残差修正限制在 video-time-value-slope 的低维条件空间；
3. 用符号一致性和可信权重阻止过度修正；
4. 把传统 ensemble 后处理提升为可解释、可消融的模块。
```

适合的论文表述：

```text
For subject-disjoint physiological affect decoding, we propose a
Sign-Consistent Bayesian Credible Residual Field that estimates
when low-dimensional residual corrections are reliable, rather than
directly fitting high-capacity residual functions.
```

## 发表风险

当前模块可以作为论文方法雏形，但还不能直接宣称强 SOTA：

```text
1. 218 -> 222 的提升只有 0.0001 Overall MAE，数值上几乎打平；
2. 需要 bootstrap confidence interval 和 per-fold paired test；
3. 需要更多 ablation 证明 count shrink、MAD shrink、sign consistency、credible gate 分别有效；
4. 最好在官方 leaderboard 或额外数据集上验证。
```

## 下一步实验

建议优先做下面三件事：

```text
1. 统计显著性：
   paired bootstrap CI for 200 vs 218 vs 222。

2. 可解释性：
   可视化 video-time/value/slope 残差场。

3. 消融实验：
   去掉 sign consistency、MAD shrink、credible gate、multi-view aggregation。
```

如果这三类证据成立，这个模块可以支撑一篇方法型论文；如果只看当前分数，它更适合作为比赛方案中的稳健后处理模块。
