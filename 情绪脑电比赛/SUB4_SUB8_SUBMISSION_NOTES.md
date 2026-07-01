# MER-PS sub4-sub8 提交包说明

本文件只用于记录，不在任何提交 zip 内。五个 zip 均位于 `submissions/`，zip 根目录包含 `model.py`、`video_prior_222_bcrf_artifact.npz`、`submission_metadata.json`。

| Zip | 方法 | 本地 Overall MAE | Valence MAE | Arousal MAE | 目的 |
|---|---|---:|---:|---:|---|
| `sub4.zip` | `222_BCRF_onSCRF` | 28.6868 | 26.8958 | 30.4777 | 当前本地 full-CV 全局最好；SCRF 上叠加 BCRF 可信残差。 |
| `sub5.zip` | `218_SCRF_reference` | 28.6869 | 26.8961 | 30.4777 | 只用 SCRF 符号一致残差，不叠 BCRF；和 sub4 几乎等价但更保守。 |
| `sub6.zip` | `224_BCRF_BrakeSCRFDisagreement` | 28.6880 | 26.8983 | 30.4777 | BCRF/SCRF 分歧时刹车；public 如果 BCRF 过拟合，这个版本可能更稳。 |
| `sub7.zip` | `214_HierResidualField_signConsensus` | 28.6907 | 26.9040 | 30.4774 | 本地 arousal 最低的非 oracle 版本；两维都只在符号一致时做小残差。 |
| `sub8.zip` | `200_MilestoneSynthesisFusion_VBatch4_AConformalBand` | 28.6912 | 26.9046 | 30.4777 | 222 的底座版本；valence 风险专家 + arousal conformal median band，最少后续残差。 |

## 选择逻辑

- `sub4.zip` 是当前本地 full-CV 最优版本，对应之前 public 提交主路线。
- `sub5.zip` 去掉 BCRF 叠加，只保留 SCRF，作为更保守的 valence 校准。
- `sub6.zip` 在 BCRF/SCRF 分歧时不修正，测试 public 上的过拟合风险。
- `sub7.zip` 是本地 arousal 最低的非 oracle 版本，但 valence 略弱。
- `sub8.zip` 是 200 底座，不使用 218/222 的后续残差，作为稳定对照。

## 提交注意

不要把本 md 上传到 Codabench。每次只上传一个 zip，例如 `submissions/sub4.zip`。
