from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.cross_fold_no_prior_physio_adaptive_batch import add_metric, finalize_metric, pca_ridge_predict  # noqa: E402
from tools.cross_fold_no_prior_physio_calibration_batch import exp_smooth, median_smooth, smooth_1d  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-video-prior physiological final 100 batch 436-535.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_436_535_no_prior_physio_final100.json")
    parser.add_argument("--top-k", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root=data_root,
        subjects=subjects,
        cache_path=Path(args.precompute_cache),
        fnirs_types=(0, 1, 2, 3, 4, 5),
        feature_normalization="none",
        baseline_correction=True,
    )
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}

    metric_acc: dict[str, dict[str, object]] = {}
    fold_outputs = []
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] train={len(train_subjects)} val={val_subjects}", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids)
        y_val = labels_to_array(labels, val_ids)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)
        center = np.full_like(y_val, 128.0)

        bundle = build_prediction_bundle(pre, train_idx, val_idx, y_train, train_ids, val_ids)
        candidates, notes = build_final100_candidates(center, y_train, val_ids, bundle)
        references = {
            "Reference_321_Center128_noPrior": center,
            "Reference_416_HuberAsymP10N06Valence_CenterArousal": from_va(center, bundle["ravc"]),
        }
        reference_notes = {
            "Reference_321_Center128_noPrior": "Reference center-only baseline.",
            "Reference_416_HuberAsymP10N06Valence_CenterArousal": "Current best physiological-only RAVC reference.",
        }
        fold_predictions = {**references, **candidates}
        all_notes = {**reference_notes, **notes}

        fold_results = []
        for name, pred in fold_predictions.items():
            pred = np.clip(pred.astype(np.float32), 1.0, 255.0)
            note = all_notes[name]
            fold_results.append(score(name, y_val, pred, note))
            add_metric(metric_acc, name, y_val, pred, note)
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"]))[: args.top_k],
            }
        )

    aggregate_results = sorted(
        [finalize_metric(name, payload) for name, payload in metric_acc.items()],
        key=lambda item: float(item["overall_mae"]),
    )
    output = {
        "method": "No-video-prior physiological final 100 module search",
        "iteration_range": "436-535",
        "note": (
            "No candidate uses video/time label priors. The final 100 candidates cover arousal probes, "
            "multi-expert fusion, subject reliability gates, state-space filters, and final RAVC combinations."
        ),
        "feature_shapes": feature_shapes,
        "aggregate_results": aggregate_results[: args.top_k],
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_prediction_bundle(
    pre: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[str],
    val_ids: list[str],
) -> dict[str, np.ndarray]:
    x_raw_train = pre["early_concat"][train_idx]
    x_raw_val = pre["early_concat"][val_idx]
    x_eeg_train = pre["eeg_lag"][train_idx]
    x_eeg_val = pre["eeg_lag"][val_idx]
    x_fnirs_train = pre["fnirs_slow"][train_idx]
    x_fnirs_val = pre["fnirs_slow"][val_idx]
    x_neuro_train = pre["neurovascular"][train_idx]
    x_neuro_val = pre["neurovascular"][val_idx]

    ridge_full = pca_ridge_predict(x_raw_train, y_train, x_raw_val, components=16, alpha=10000.0)
    ridge_v = smooth_1d(val_ids, ridge_full[:, 0], 5)
    ridge_a = smooth_1d(val_ids, ridge_full[:, 1], 5)
    huber_v_raw = robust_head(x_raw_train, y_train[:, 0], x_raw_val, "huber", 16)
    huber_v_s3 = smooth_1d(val_ids, huber_v_raw, 3)
    huber_v_s5 = smooth_1d(val_ids, huber_v_raw, 5)
    huber_v_s7 = smooth_1d(val_ids, huber_v_raw, 7)
    huber_v_s9 = smooth_1d(val_ids, huber_v_raw, 9)
    huber_v_s11 = smooth_1d(val_ids, huber_v_raw, 11)
    huber_v_s13 = smooth_1d(val_ids, huber_v_raw, 13)
    huber_v_exp02 = exp_smooth(val_ids, huber_v_raw, alpha=0.20)
    huber_v_exp03 = exp_smooth(val_ids, huber_v_raw, alpha=0.30)
    huber_v_exp04 = exp_smooth(val_ids, huber_v_raw, alpha=0.40)
    huber_v_med5 = median_smooth(val_ids, huber_v_raw, 5)
    huber_a = smooth_1d(val_ids, robust_head(x_raw_train, y_train[:, 1], x_raw_val, "huber", 16), 5)

    elastic_v = smooth_1d(val_ids, robust_head(x_raw_train, y_train[:, 0], x_raw_val, "elastic", 16), 5)
    bayes_v = smooth_1d(val_ids, robust_head(x_raw_train, y_train[:, 0], x_raw_val, "bayes", 16), 5)
    eeg_v = smooth_1d(val_ids, robust_head(x_eeg_train, y_train[:, 0], x_eeg_val, "huber", 8), 5)
    fnirs_v = smooth_1d(val_ids, robust_head(x_fnirs_train, y_train[:, 0], x_fnirs_val, "huber", 8), 5)
    neuro_v = smooth_1d(val_ids, robust_head(x_neuro_train, y_train[:, 0], x_neuro_val, "huber", min(8, x_neuro_train.shape[1])), 5)
    eeg_a = smooth_1d(val_ids, robust_head(x_eeg_train, y_train[:, 1], x_eeg_val, "huber", 8), 5)
    fnirs_a = smooth_1d(val_ids, robust_head(x_fnirs_train, y_train[:, 1], x_fnirs_val, "huber", 8), 5)
    neuro_a = smooth_1d(val_ids, robust_head(x_neuro_train, y_train[:, 1], x_neuro_val, "huber", min(8, x_neuro_train.shape[1])), 5)

    pca8_v = smooth_1d(val_ids, robust_head(x_raw_train, y_train[:, 0], x_raw_val, "huber", 8), 5)
    pca24_v = smooth_1d(val_ids, robust_head(x_raw_train, y_train[:, 0], x_raw_val, "huber", 24), 5)

    return {
        "huber_raw": huber_v_raw,
        "huber_s3": huber_v_s3,
        "huber_s5": huber_v_s5,
        "huber_s7": huber_v_s7,
        "huber_s9": huber_v_s9,
        "huber_s11": huber_v_s11,
        "huber_s13": huber_v_s13,
        "huber_exp02": huber_v_exp02,
        "huber_exp03": huber_v_exp03,
        "huber_exp04": huber_v_exp04,
        "huber_med5": huber_v_med5,
        "ravc": asym_scale(huber_v_s5, 1.0, 0.6),
        "ridge_v": ridge_v,
        "ridge_a": ridge_a,
        "huber_a": huber_a,
        "elastic_v": elastic_v,
        "bayes_v": bayes_v,
        "eeg_v": eeg_v,
        "fnirs_v": fnirs_v,
        "neuro_v": neuro_v,
        "eeg_a": eeg_a,
        "fnirs_a": fnirs_a,
        "neuro_a": neuro_a,
        "pca8_v": pca8_v,
        "pca24_v": pca24_v,
        "val_ids": np.asarray(val_ids, dtype=object),
        "train_mean_a": np.asarray([float(y_train[:, 1].mean())], dtype=np.float32),
    }


def build_final100_candidates(
    center: np.ndarray,
    y_train: np.ndarray,
    val_ids: list[str],
    b: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    candidates: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}

    def add(label: str, valence: np.ndarray, note: str, arousal: np.ndarray | None = None) -> None:
        index = 436 + len(candidates)
        name = f"{index}_{label}"
        candidates[name] = from_va(center, valence, arousal)
        notes[name] = note

    ravc = b["ravc"]
    ridge_a = b["ridge_a"]
    huber_a = b["huber_a"]
    train_mean_a = np.full(center.shape[0], float(b["train_mean_a"][0]), dtype=np.float32)

    # 436-455: arousal probes around RAVC.
    add("RAVC_CenterArousal", ravc, "Arousal probe reference: RAVC valence, center arousal.")
    add("RAVC_TrainMeanArousal", ravc, "Use train global arousal mean.", train_mean_a)
    for scale in (0.03, 0.05, 0.10, 0.15):
        add(f"RAVC_RidgeArousalScale{tag(scale)}", ravc, "Tiny Ridge arousal residual over center.", scaled_arousal(ridge_a, scale))
    for scale in (0.03, 0.05, 0.10, 0.15):
        add(f"RAVC_HuberArousalScale{tag(scale)}", ravc, "Tiny Huber arousal residual over center.", scaled_arousal(huber_a, scale))
    add("RAVC_RidgeHuberArousalAvgScale05", ravc, "Tiny average arousal residual.", scaled_arousal(0.5 * ridge_a + 0.5 * huber_a, 0.05))
    add("RAVC_EEGArousalScale05", ravc, "Tiny EEG-only arousal residual.", scaled_arousal(b["eeg_a"], 0.05))
    add("RAVC_FNIRSArousalScale05", ravc, "Tiny fNIRS-only arousal residual.", scaled_arousal(b["fnirs_a"], 0.05))
    add("RAVC_NeuroArousalScale05", ravc, "Tiny neurovascular arousal residual.", scaled_arousal(b["neuro_a"], 0.05))
    add("RAVC_ArousalAgreementScale10", ravc, "Use arousal residual only when Ridge and Huber agree.", agreement_arousal(ridge_a, huber_a, 0.10, 6.0))
    add("RAVC_ArousalSmallOnlyScale15", ravc, "Use arousal residual only when predicted residual is small.", small_only_arousal(ridge_a, 0.15, 8.0))
    add("RAVC_ArousalPositiveOnlyScale10", ravc, "Only positive arousal residual.", positive_only_arousal(ridge_a, 0.10))
    add("RAVC_ArousalNegativeOnlyScale10", ravc, "Only negative arousal residual.", negative_only_arousal(ridge_a, 0.10))
    add("RAVC_ArousalExpScale10", ravc, "Smoothed arousal residual.", scaled_arousal(exp_smooth(val_ids, ridge_a, 0.30), 0.10))
    add("RAVC_ArousalMedianScale10", ravc, "Median-smoothed arousal residual.", scaled_arousal(median_smooth(val_ids, ridge_a, 5), 0.10))

    # 456-475: multi-expert physiological valence fusion.
    experts = [
        asym_scale(b["huber_s5"], 1.0, 0.6),
        asym_scale(b["ridge_v"], 1.0, 0.6),
        asym_scale(b["elastic_v"], 1.0, 0.6),
        asym_scale(b["bayes_v"], 1.0, 0.6),
        asym_scale(b["fnirs_v"], 1.0, 0.6),
        asym_scale(b["eeg_v"], 1.0, 0.6),
        asym_scale(b["neuro_v"], 1.0, 0.6),
    ]
    add("ValenceMeanHuberRidgeElastic", mean_stack(experts[:3]), "Mean fusion of Huber/Ridge/Elastic valence experts.")
    add("ValenceMedianHuberRidgeElastic", median_stack(experts[:3]), "Median fusion of Huber/Ridge/Elastic valence experts.")
    add("ValenceWeightedHRE_721", 0.7 * experts[0] + 0.2 * experts[1] + 0.1 * experts[2], "Weighted Huber-dominant fusion.")
    add("ValenceWeightedHRE_811", 0.8 * experts[0] + 0.1 * experts[1] + 0.1 * experts[2], "More Huber-dominant fusion.")
    add("ValenceHuberFNIRS82", 0.8 * experts[0] + 0.2 * experts[4], "Huber plus fNIRS expert.")
    add("ValenceHuberEEG82", 0.8 * experts[0] + 0.2 * experts[5], "Huber plus EEG expert.")
    add("ValenceHuberNeuro82", 0.8 * experts[0] + 0.2 * experts[6], "Huber plus neurovascular expert.")
    add("ValenceAllExpertMedian", median_stack(experts), "Median over all physiological experts.")
    add("ValenceAllExpertTrimmed", trimmed_mean_stack(experts), "Trimmed mean over all physiological experts.")
    add("ValenceMinMagnitudeAll", min_magnitude_stack(experts), "Choose most conservative expert residual.")
    add("ValenceAgreementHuberRidgeT6", agreement_valence(experts[0], experts[1], 6.0), "Huber/Ridge agreement gate.")
    add("ValenceAgreementHuberElasticT6", agreement_valence(experts[0], experts[2], 6.0), "Huber/Elastic agreement gate.")
    add("ValenceAgreementHuberFNIRST8", agreement_valence(experts[0], experts[4], 8.0), "Huber/fNIRS agreement gate.")
    add("ValenceAgreementHuberNeuroT8", agreement_valence(experts[0], experts[6], 8.0), "Huber/neurovascular agreement gate.")
    add("ValencePCA8PCA16PCA24Mean", mean_stack([asym_scale(b["pca8_v"], 1.0, 0.6), experts[0], asym_scale(b["pca24_v"], 1.0, 0.6)]), "Low-rank dimension ensemble.")
    add("ValencePCA8PCA16PCA24Median", median_stack([asym_scale(b["pca8_v"], 1.0, 0.6), experts[0], asym_scale(b["pca24_v"], 1.0, 0.6)]), "Low-rank dimension median ensemble.")
    add("ValenceHuberBayes82", 0.8 * experts[0] + 0.2 * experts[3], "Huber plus Bayesian Ridge expert.")
    add("ValenceHuberBayesElasticMedian", median_stack([experts[0], experts[2], experts[3]]), "Median of robust linear heads.")
    add("ValenceHuberRidgeMinMagnitude", min_magnitude_stack([experts[0], experts[1]]), "Conservative Huber/Ridge intersection.")
    add("ValenceHuberRidgeMaxAgreement", max_agreement_value(experts[0], experts[1]), "Use stronger residual only when Huber/Ridge agree in sign.")

    # 476-495: reliability and subject/prediction confidence gates.
    add("ReliabilityDisagreeShrinkT4S70", disagreement_shrink(experts[0], experts[1], 4.0, 0.70), "Shrink if Huber/Ridge disagree.")
    add("ReliabilityDisagreeShrinkT6S70", disagreement_shrink(experts[0], experts[1], 6.0, 0.70), "Shrink if Huber/Ridge disagree.")
    add("ReliabilityDisagreeShrinkT8S70", disagreement_shrink(experts[0], experts[1], 8.0, 0.70), "Shrink if Huber/Ridge disagree.")
    add("ReliabilityDisagreeCenterT4", disagreement_center(experts[0], experts[1], 4.0), "Center fallback if Huber/Ridge disagree.")
    add("ReliabilityDisagreeCenterT8", disagreement_center(experts[0], experts[1], 8.0), "Center fallback if Huber/Ridge disagree.")
    add("ReliabilitySubjectStdShrink10", subject_std_shrink(val_ids, ravc, 10.0, 0.8), "Shrink high-variance subject predictions.")
    add("ReliabilitySubjectStdShrink14", subject_std_shrink(val_ids, ravc, 14.0, 0.8), "Shrink high-variance subject predictions.")
    add("ReliabilitySubjectStdShrink18", subject_std_shrink(val_ids, ravc, 18.0, 0.8), "Shrink high-variance subject predictions.")
    add("ReliabilityTrialStdShrink8", trial_std_shrink(val_ids, ravc, 8.0, 0.8), "Shrink high-variance trial predictions.")
    add("ReliabilityTrialStdShrink12", trial_std_shrink(val_ids, ravc, 12.0, 0.8), "Shrink high-variance trial predictions.")
    add("ReliabilitySubjectNegMeanBrake", subject_negative_mean_brake(val_ids, ravc, -8.0, 0.8), "Brake subjects with negative mean residual.")
    add("ReliabilitySubjectNegMeanBrakeStrong", subject_negative_mean_brake(val_ids, ravc, -8.0, 0.6), "Stronger brake for negative subject residual.")
    add("ReliabilitySubjectPosMeanKeepNegBrake", subject_signed_mean_gate(val_ids, ravc), "Subject signed mean gate.")
    add("ReliabilityAbsResidualShrink20", residual_abs_shrink(ravc, 20.0, 0.8), "Shrink large residuals.")
    add("ReliabilityAbsResidualShrink28", residual_abs_shrink(ravc, 28.0, 0.8), "Shrink very large residuals.")
    add("ReliabilityHuberElasticDisagreeShrink6", disagreement_shrink(experts[0], experts[2], 6.0, 0.75), "Shrink Huber/Elastic disagreement.")
    add("ReliabilityHuberFNIRSDisagreeShrink8", disagreement_shrink(experts[0], experts[4], 8.0, 0.75), "Shrink Huber/fNIRS disagreement.")
    add("ReliabilityHuberNeuroDisagreeShrink8", disagreement_shrink(experts[0], experts[6], 8.0, 0.75), "Shrink Huber/neuro disagreement.")
    add("ReliabilitySubjectMeanToTrainMeanLight", subject_mean_to_target(val_ids, ravc, target=128.0, amount=0.25), "Light subject mean recentering.")
    add("ReliabilitySubjectMeanToCenterMedium", subject_mean_to_target(val_ids, ravc, target=128.0, amount=0.50), "Medium subject mean recentering.")

    # 496-515: state-space and temporal-shape filters.
    add("StateSmooth11Asym", asym_scale(b["huber_s11"], 1.0, 0.6), "Longer moving-average state filter.")
    add("StateSmooth13Asym", asym_scale(b["huber_s13"], 1.0, 0.6), "Longer moving-average state filter.")
    add("StateExp02Asym", asym_scale(b["huber_exp02"], 1.0, 0.6), "EMA state filter.")
    add("StateExp03Asym", asym_scale(b["huber_exp03"], 1.0, 0.6), "EMA state filter.")
    add("StateExp04Asym", asym_scale(b["huber_exp04"], 1.0, 0.6), "EMA state filter.")
    add("StateMedianAsym", asym_scale(b["huber_med5"], 1.0, 0.6), "Median state filter.")
    add("StateSlopeLimit2", slope_limit(val_ids, ravc, 2.0), "Limit per-second prediction slope.")
    add("StateSlopeLimit4", slope_limit(val_ids, ravc, 4.0), "Limit per-second prediction slope.")
    add("StateSlopeLimit6", slope_limit(val_ids, ravc, 6.0), "Limit per-second prediction slope.")
    add("StateSlopeLimit8", slope_limit(val_ids, ravc, 8.0), "Limit per-second prediction slope.")
    add("StateSlopeLimit2ThenExp", exp_smooth(val_ids, slope_limit(val_ids, ravc, 2.0), 0.40), "Slope limit then EMA.")
    add("StateSlopeLimit4ThenExp", exp_smooth(val_ids, slope_limit(val_ids, ravc, 4.0), 0.40), "Slope limit then EMA.")
    add("StateTrendDamp30", trend_damp(val_ids, ravc, 0.30), "Dampen rapid trend component.")
    add("StateTrendDamp50", trend_damp(val_ids, ravc, 0.50), "Dampen rapid trend component.")
    add("StateTrialStartCenterBlend20", trial_start_center_blend(val_ids, ravc, 0.20), "Blend early trial prediction toward center.")
    add("StateTrialStartCenterBlend40", trial_start_center_blend(val_ids, ravc, 0.40), "Blend early trial prediction toward center.")
    add("StateLateResidualShrink20", late_residual_shrink(val_ids, ravc, after_t=40, scale=0.9), "Shrink late-trial residuals.")
    add("StateLateResidualShrink40", late_residual_shrink(val_ids, ravc, after_t=40, scale=0.8), "Shrink late-trial residuals.")
    add("StateCenteredEMAResidual", centered_ema_residual(val_ids, ravc, 0.30), "EMA on residual around center.")
    add("StateMedianExpBlend", 0.5 * asym_scale(b["huber_med5"], 1.0, 0.6) + 0.5 * asym_scale(b["huber_exp03"], 1.0, 0.6), "Blend median and EMA state filters.")

    # 516-535: final combinations built from previous winners.
    add("FinalNeg055", asym_scale(b["huber_s5"], 1.0, 0.55), "Final RAVC negative scale search.")
    add("FinalNeg050", asym_scale(b["huber_s5"], 1.0, 0.50), "Final RAVC negative scale search.")
    add("FinalNeg065", asym_scale(b["huber_s5"], 1.0, 0.65), "Final RAVC negative scale search.")
    add("FinalPos095Neg060", asym_scale(b["huber_s5"], 0.95, 0.60), "Final asymmetric scale search.")
    add("FinalPos105Neg060", asym_scale(b["huber_s5"], 1.05, 0.60), "Final asymmetric scale search.")
    add("FinalNeg060Exp03", asym_scale(b["huber_exp03"], 1.0, 0.60), "Final asymmetry with EMA.")
    add("FinalNeg060Smooth9", asym_scale(b["huber_s9"], 1.0, 0.60), "Final asymmetry with long smoothing.")
    add("FinalNeg060Blend75", 0.75 * asym_scale(b["huber_s5"], 1.0, 0.60) + 0.25 * asym_scale(b["ridge_v"], 1.0, 0.60), "Final Huber/Ridge asymmetric blend.")
    add("FinalNeg060Blend90", 0.90 * asym_scale(b["huber_s5"], 1.0, 0.60) + 0.10 * asym_scale(b["ridge_v"], 1.0, 0.60), "Final Huber/Ridge asymmetric blend.")
    add("FinalNeg060ElasticBlend90", 0.90 * asym_scale(b["huber_s5"], 1.0, 0.60) + 0.10 * asym_scale(b["elastic_v"], 1.0, 0.60), "Final Huber/Elastic asymmetric blend.")
    add("FinalNeg060DisagreeShrink", disagreement_shrink(asym_scale(b["huber_s5"], 1.0, 0.60), asym_scale(b["ridge_v"], 1.0, 0.60), 6.0, 0.85), "Final disagreement shrink.")
    add("FinalNeg060SubjectStd14", subject_std_shrink(val_ids, asym_scale(b["huber_s5"], 1.0, 0.60), 14.0, 0.9), "Final subject reliability shrink.")
    add("FinalNeg060Slope4", slope_limit(val_ids, asym_scale(b["huber_s5"], 1.0, 0.60), 4.0), "Final state-space slope limit.")
    add("FinalNeg060Slope6", slope_limit(val_ids, asym_scale(b["huber_s5"], 1.0, 0.60), 6.0), "Final state-space slope limit.")
    add("FinalNeg060MedianExp", 0.5 * asym_scale(b["huber_med5"], 1.0, 0.60) + 0.5 * asym_scale(b["huber_exp03"], 1.0, 0.60), "Final median/EMA blend.")
    add("FinalNeg060ArousalScale03", asym_scale(b["huber_s5"], 1.0, 0.60), "Final tiny arousal residual.", scaled_arousal(ridge_a, 0.03))
    add("FinalNeg060ArousalAgree05", asym_scale(b["huber_s5"], 1.0, 0.60), "Final agreement-gated arousal residual.", agreement_arousal(ridge_a, huber_a, 0.05, 6.0))
    add("FinalNeg060MeanShift", asym_scale(b["huber_s5"] + float(y_train[:, 0].mean() - 128.0) * 0.05, 1.0, 0.60), "Final light train mean shift.")
    add("FinalNeg060PCAEnsemble", mean_stack([asym_scale(b["pca8_v"], 1.0, 0.60), asym_scale(b["huber_s5"], 1.0, 0.60), asym_scale(b["pca24_v"], 1.0, 0.60)]), "Final low-rank ensemble.")
    add("FinalNeg060ExpertMedian", median_stack([asym_scale(b["huber_s5"], 1.0, 0.60), asym_scale(b["ridge_v"], 1.0, 0.60), asym_scale(b["elastic_v"], 1.0, 0.60)]), "Final expert median.")

    if len(candidates) != 100:
        raise RuntimeError(f"Expected exactly 100 candidates, got {len(candidates)}")
    return candidates, notes


def robust_head(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, kind: str, components: int) -> np.ndarray:
    if kind == "huber":
        reg = HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=300)
    elif kind == "elastic":
        reg = ElasticNet(alpha=0.02, l1_ratio=0.15, max_iter=3000, random_state=2026)
    elif kind == "bayes":
        reg = BayesianRidge()
    else:
        raise ValueError(kind)
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(components, x_train.shape[1]), svd_solver="randomized", random_state=2026),
        reg,
    )
    model.fit(sanitize(x_train), y_train)
    return np.clip(model.predict(sanitize(x_val)).astype(np.float32), 1.0, 255.0)


def from_va(center: np.ndarray, valence: np.ndarray, arousal: np.ndarray | None = None) -> np.ndarray:
    out = center.copy()
    out[:, 0] = np.asarray(valence, dtype=np.float32)
    if arousal is not None:
        out[:, 1] = np.asarray(arousal, dtype=np.float32)
    return out


def asym_scale(values: np.ndarray, pos: float, neg: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    scale = np.where(residual >= 0.0, pos, neg)
    return 128.0 + scale * residual


def scaled_arousal(values: np.ndarray, scale: float) -> np.ndarray:
    return 128.0 + scale * (np.asarray(values, dtype=np.float32) - 128.0)


def agreement_arousal(a: np.ndarray, b: np.ndarray, scale: float, threshold: float) -> np.ndarray:
    avg = 0.5 * (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32))
    return np.where(np.abs(a - b) <= threshold, scaled_arousal(avg, scale), 128.0)


def small_only_arousal(values: np.ndarray, scale: float, threshold: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    return np.where(np.abs(residual) <= threshold, 128.0 + scale * residual, 128.0)


def positive_only_arousal(values: np.ndarray, scale: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    return 128.0 + np.where(residual > 0.0, scale * residual, 0.0)


def negative_only_arousal(values: np.ndarray, scale: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    return 128.0 + np.where(residual < 0.0, scale * residual, 0.0)


def mean_stack(values: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values, axis=0), axis=0)


def median_stack(values: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(values, axis=0), axis=0)


def trimmed_mean_stack(values: list[np.ndarray]) -> np.ndarray:
    stack = np.sort(np.stack(values, axis=0), axis=0)
    if stack.shape[0] <= 2:
        return stack.mean(axis=0)
    return stack[1:-1].mean(axis=0)


def min_magnitude_stack(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values, axis=0)
    residual = stack - 128.0
    choice = np.argmin(np.abs(residual), axis=0)
    return stack[choice, np.arange(stack.shape[1])]


def agreement_valence(a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    avg = 0.5 * (a + b)
    return np.where(np.abs(a - b) <= threshold, avg, 128.0)


def max_agreement_value(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    residual_a = a - 128.0
    residual_b = b - 128.0
    same = np.sign(residual_a) == np.sign(residual_b)
    stronger = np.where(np.abs(residual_a) >= np.abs(residual_b), a, b)
    return np.where(same, stronger, 128.0)


def disagreement_shrink(a: np.ndarray, b: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    residual = np.asarray(a, dtype=np.float32) - 128.0
    shrink = np.where(np.abs(a - b) > threshold, scale, 1.0)
    return 128.0 + shrink * residual


def disagreement_center(a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.abs(a - b) > threshold, 128.0, a)


def residual_abs_shrink(values: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    shrink = np.where(np.abs(residual) > threshold, scale, 1.0)
    return 128.0 + shrink * residual


def group_key(sample_id: str, mode: str) -> str:
    subject, rest = sample_id.split("_V", 1)
    if mode == "subject":
        return subject
    video = rest.split("_T", 1)[0]
    return f"{subject}_V{video}"


def timestamp(sample_id: str) -> int:
    return int(sample_id.rsplit("_T", 1)[1])


def grouped_indices(sample_ids: list[str], mode: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[group_key(sample_id, mode)].append(index)
    return groups


def subject_std_shrink(sample_ids: list[str], values: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    return group_std_shrink(sample_ids, values, "subject", threshold, scale)


def trial_std_shrink(sample_ids: list[str], values: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    return group_std_shrink(sample_ids, values, "trial", threshold, scale)


def group_std_shrink(sample_ids: list[str], values: np.ndarray, mode: str, threshold: float, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, mode).values():
        idx = np.asarray(indices, dtype=np.int64)
        residual = values[idx] - 128.0
        if float(np.std(residual)) > threshold:
            out[idx] = 128.0 + scale * residual
    return out


def subject_negative_mean_brake(sample_ids: list[str], values: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, "subject").values():
        idx = np.asarray(indices, dtype=np.int64)
        residual = values[idx] - 128.0
        if float(np.mean(residual)) < threshold:
            out[idx] = 128.0 + scale * residual
    return out


def subject_signed_mean_gate(sample_ids: list[str], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, "subject").values():
        idx = np.asarray(indices, dtype=np.int64)
        residual = values[idx] - 128.0
        scale = 0.75 if float(np.mean(residual)) < -8.0 else 1.0
        out[idx] = 128.0 + scale * residual
    return out


def subject_mean_to_target(sample_ids: list[str], values: np.ndarray, target: float, amount: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, "subject").values():
        idx = np.asarray(indices, dtype=np.int64)
        shift = target - float(values[idx].mean())
        out[idx] = values[idx] + amount * shift
    return out


def slope_limit(sample_ids: list[str], values: np.ndarray, cap: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, "trial").values():
        idx = sorted(indices, key=lambda item: timestamp(sample_ids[item]))
        seq = values[idx]
        limited = seq.copy()
        for local in range(1, len(seq)):
            limited[local] = np.clip(limited[local], limited[local - 1] - cap, limited[local - 1] + cap)
        out[idx] = limited
    return out


def trend_damp(sample_ids: list[str], values: np.ndarray, amount: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    smooth = exp_smooth(sample_ids, values, alpha=0.30)
    return smooth + (1.0 - amount) * (values - smooth)


def trial_start_center_blend(sample_ids: list[str], values: np.ndarray, amount: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for indices in grouped_indices(sample_ids, "trial").values():
        idx = sorted(indices, key=lambda item: timestamp(sample_ids[item]))
        for local, global_index in enumerate(idx[:10]):
            weight = amount * (1.0 - local / 10.0)
            out[global_index] = (1.0 - weight) * values[global_index] + weight * 128.0
    return out


def late_residual_shrink(sample_ids: list[str], values: np.ndarray, after_t: int, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for index, sample_id in enumerate(sample_ids):
        if timestamp(sample_id) >= after_t:
            out[index] = 128.0 + scale * (values[index] - 128.0)
    return out


def centered_ema_residual(sample_ids: list[str], values: np.ndarray, alpha: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    return 128.0 + exp_smooth(sample_ids, residual, alpha)


def tag(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
