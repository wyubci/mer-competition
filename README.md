# MER Competition — 多模态情绪识别比赛

基于 EEG + fNIRS 融合的多模态情绪识别比赛方案。

## 比赛概述

通过视频诱发情绪，采集 EEG (脑电) 和 fNIRS (近红外) 信号，构建多模态融合模型进行情绪分类。

## 目录结构

```
emotion_eeg_fnirs/情绪脑电比赛/
├── train_baseline.py          # 基线模型训练
├── train_distill.py           # 知识蒸馏训练
├── submissions/               # 各被试提交版本
│   ├── sub4/ ~ sub9/          # 不同被试的模型
│   └── video_prior_222_bcrf/  # 视频先验 + BCRF 方案
├── tools/                     # 工具脚本
│   ├── cross_fold_*.py        # 交叉验证策略
│   ├── build_*_submission.py  # 提交构建脚本
│   └── validate_predictions.py
├── experiments/               # 实验配置和数据
└── assets/                    # 图表和文档
```

## 方法

- **多模态融合**: EEG + fNIRS 特征级/决策级融合
- **视频先验**: 视频情感标签作为弱监督信号
- **交叉验证**: 被试内/被试间多种策略
- **模型**: XGBoost, LightGBM, 神经网络, BCRF

## 环境

```bash
pip install numpy scipy scikit-learn xgboost lightgbm torch
```

---

**仓库**: https://github.com/wyubci/mer-competition
