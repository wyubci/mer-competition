# No-Video/Time Prior Signal-Only Results

本轮用于回答：去掉 `VideoTimeMean` / `PatternPrior` / `video-time cell` 后，EEG+fNIRS 直接处理到底能到什么效果。

| 方法 | Overall MAE | Valence MAE | Arousal MAE | 说明 |
| --- | ---: | ---: | ---: | --- |
| Official ASAC demo | 47.0087 | 49.2285 | 44.7890 | 官方 demo，固定 `test_21-test_24` 评估 |
| 321_Center128_noPrior | 47.5663 | 52.1980 | 42.9346 | 无信号、无先验中心点 |
| 333_PCAEarlyDirectRidge_c8_a10000p0_SignalSmooth5 | 47.0764 | 50.6582 | 43.4946 | 本轮最佳 no-prior 直接信号模型 |
| 222_BCRF_onSCRF | 28.6868 | 26.8958 | 30.4777 | 当前全局最佳，含强先验与可信残差校准 |

- 相对 Center128，最佳直接信号模型改善 `0.4899` MAE。
- 相对全局最佳，no-prior 仍落后 `18.3896` MAE。
- 结论：生理信号有可用信息，但直接高维拟合不强；现阶段更合理的路线是低维信号表征 + 可信残差/保守融合。
