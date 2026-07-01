# MER-PS strategy report outline

- Current best: `222_BCRF_onSCRF` overall MAE `28.6868`, valence `26.8958`, arousal `30.4777`.
- Experiment scale: `181` result JSON files, `316` unique iteration IDs, numbering up to `335`, `2231` aggregate rows.
- Best EEG-fNIRS signal fusion: `CCMI PriorSlopeGate`, overall MAE `28.7145`.
- Best no-video/time direct signal model: `333_PCAEarlyDirectRidge_c8_a10000p0_SignalSmooth5` overall MAE `47.0764`.

Slides:
1. MER-PS 情绪脑电比赛策略复盘
2. 赛题与数据
3. 实验规模
4. 当前最优结果
5. 策略演进
6. 为什么不是盲目堆大模型
7. 数据预处理模块
8. CCMI 融合模块
9. 输出头模块
10. SCRF-BCRF 模块逻辑
11. 当前最佳框架
12. 下一组实验：去视频/时间前导
13. 阶段性结论

Added detailed innovation-module slides:
14. CRF-Fusion（可信残差融合）总览图
15. SCRF-BCRF（符号一致贝叶斯可信残差场）结构图
16. CCMI（保守跨模态交集）融合图
17. 数学拼接逻辑
18. 汇报话术：贡献、证据与风险

Image2 generated module figure and two-route submission slides:
19. Image2 模块图：CRF-Fusion（可信残差融合）
20. 提交路线 A：视频/时间先导高分模型
21. 提交路线 B：直接生理信号模型
22. 两种提交路线如何同时使用
