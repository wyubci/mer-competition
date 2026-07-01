# MER-PS Physiological-Only Iteration Log

This log records experiments that use EEG/fNIRS signals directly, without video/time label priors.

Definition of "physiological-only" in this log:

- No `VideoTimeMean`.
- No `PatternPrior`.
- No video ID label statistics.
- No timestamp label-cell prior.
- `sample_id` may only be used to align samples, group subjects/trials, and apply signal-domain temporal smoothing.

## Current Reference Points

| Method | Setting | Overall MAE | Valence MAE | Arousal MAE | Note |
| --- | --- | ---: | ---: | ---: | --- |
| Official ASAC demo | official starting kit style, fixed split | 47.0087 | 49.2285 | 44.7890 | Main interface reference, not a strong model |
| 321_Center128_noPrior | full 24-subject CV | 47.5663 | 52.1980 | 42.9346 | No signal, no prior |
| 333_PCAEarlyDirectRidge_c8_a10000p0_SignalSmooth5 | full 24-subject CV | 47.0764 | 50.6582 | 43.4946 | Previous best direct EEG/fNIRS model |

## Iteration 336-355: Dimwise Physiological Correction

Result file:

```text
experiments/results/iteration_336_355_no_prior_physio_dimwise.json
```

Core hypothesis:

```text
EEG/fNIRS signal is useful for valence, but arousal residual prediction is unstable.
Therefore use signal only to correct valence and keep arousal conservative.
```

Top results:

| Rank | Method | Overall MAE | Valence MAE | Arousal MAE | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | 354_AgreementGatedPCA16Valence_CenterArousal | 46.7142 | 50.4937 | 42.9346 | Best no-prior physiological result so far |
| 2 | 337_PCA16Valence_CenterArousal | 46.7335 | 50.5323 | 42.9346 | Low-rank signal helps valence |
| 3 | 352_LowRankEnsembleValence_CenterArousal | 46.7612 | 50.5878 | 42.9346 | Ensemble is stable but not better than gate |
| 4 | 336_PCA8Valence_CenterArousal | 46.7964 | 50.6582 | 42.9346 | Matches previous valence model while avoiding arousal damage |
| 5 | 338_PLS2Valence_CenterArousal | 46.8097 | 50.6847 | 42.9346 | Supervised low-rank projection is competitive |

Main conclusion:

```text
Compared with Center128, the best physiological-only model improves Overall MAE by 0.8521.
Compared with the previous best direct signal model, it improves Overall MAE by 0.3622.
The improvement comes from avoiding arousal over-correction while preserving valence correction.
```

Mathematical reading:

Let `y = [v, a]`, and let a signal model estimate residuals `r = f(x)`.

The direct model uses:

```text
v_hat = 128 + r_v
a_hat = 128 + r_a
```

The best 336-355 family uses:

```text
v_hat = 128 + g(x) * r_v
a_hat = 128
```

where `g(x)` is an agreement gate derived from multiple low-rank physiological views. This works because the empirical risk shows:

```text
E|a - (128 + r_a)| > E|a - 128|
E|v - (128 + gated_r_v)| < E|v - 128|
```

So the next useful direction is not "larger model", but better estimation of `g(x)` and safer cross-subject feature alignment.

## Next Batch 356-375

Planned module families:

| Family | Logic | Expected benefit |
| --- | --- | --- |
| Subject-adaptive normalization | Normalize each subject in signal-feature space using only unlabeled signal statistics | Reduce cross-subject amplitude/domain shift |
| Trial-adaptive normalization | Normalize each trial sequence separately | Preserve temporal shape while suppressing absolute offset |
| Residual amplitude control | Predict residual direction, then shrink magnitude toward center | Reduce overfitting under small subject count |
| Robust low-rank regression | Replace plain ridge with robust or supervised low-rank variants | Reduce fold-specific outlier damage |
| Confidence-gated arousal probe | Only allow arousal correction when signal confidence is high | Test whether arousal can be improved without damaging center baseline |

## Iteration 356-375: Adaptive Alignment and Robust Heads

Result file:

```text
experiments/results/iteration_356_375_no_prior_physio_adaptive.json
```

Core hypothesis:

```text
The 336-355 gain may be limited by cross-subject feature shift and MAE-sensitive outliers.
Try unlabeled subject/trial alignment and robust/nonlinear output heads, while keeping arousal conservative.
```

Top results:

| Rank | Method | Overall MAE | Valence MAE | Arousal MAE | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | 364_HuberPCA16Valence_CenterArousal | 46.6619 | 50.3892 | 42.9346 | New physiological-only best |
| 2 | Reference_354_AgreementGatedPCA16Valence_CenterArousal | 46.7142 | 50.4937 | 42.9346 | Previous best |
| 3 | 366_ElasticNetPCA16Valence_CenterArousal | 46.7310 | 50.5274 | 42.9346 | Similar to ridge, slight sparsity benefit |
| 4 | Reference_337_PCA16Valence_CenterArousal | 46.7335 | 50.5323 | 42.9346 | Plain low-rank ridge reference |
| 5 | 365_BayesianPCA16Valence_CenterArousal | 46.7344 | 50.5341 | 42.9346 | Uncertainty-style regularization is neutral |

Main conclusion:

```text
Huber loss improves Overall MAE by 0.0523 over the previous best no-prior physiological model.
This is small but meaningful because the target metric is MAE.
However, Huber increases overall MSE, so it is suppressing large-error influence rather than improving every sample.
```

Module interpretation:

```text
Ridge minimizes squared error:
  min sum_i (y_i - f(x_i))^2 + lambda ||w||^2

Huber minimizes a clipped quadratic/linear loss:
  L_delta(e) = 0.5 e^2                 if |e| <= delta
             = delta(|e| - 0.5 delta)  otherwise

Because MER-PS is scored by MAE, Huber is better aligned with the leaderboard objective than Ridge.
The result suggests that cross-subject physiological labels contain heavy-tailed residuals.
```

What did not work:

| Module | Observation | Reason |
| --- | --- | --- |
| Subject/trial z-score | Usually weaker than raw PCA16 | Absolute physiological level still carries useful valence signal |
| Trial delta only | Weaker than level features | Emotion state is not purely derivative; low-frequency state matters |
| Tiny arousal gates | Still worsened arousal MAE | Arousal signal residual remains less reliable than center |
| HistGradientBoosting | Worse overall | Nonlinear head overfits fold-specific subject structure |

Next batch 376-395:

```text
Focus on Huber as a useful module:
  1. Huber per modality: EEG-only, fNIRS-only, neurovascular-only.
  2. Huber under subject/trial transforms.
  3. Huber + agreement gate with Ridge/ElasticNet.
  4. Huber residual magnitude caps and shrinkage.
  5. Tree/boosting references to confirm nonlinear overfitting.
```

## Iteration 376-395: Huber Module Dissection

Result file:

```text
experiments/results/iteration_376_395_no_prior_physio_huber.json
```

Core hypothesis:

```text
If Huber is the useful module, changing its view, dimension, alignment, or residual amplitude should reveal why it helps.
```

Top results:

| Rank | Method | Overall MAE | Valence MAE | Arousal MAE | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Reference_364_HuberPCA16Valence_CenterArousal | 46.6619 | 50.3892 | 42.9346 | Current best still holds |
| 2 | 393_WinsorTargetHuberPCA16Valence_CenterArousal | 46.6701 | 50.4055 | 42.9346 | Target-tail clipping is close but not better |
| 3 | 392_HuberShrink80Valence_CenterArousal | 46.6792 | 50.4239 | 42.9346 | Shrink improves MSE but slightly hurts MAE |
| 4 | Reference_354_AgreementGatedPCA16Valence_CenterArousal | 46.7142 | 50.4937 | 42.9346 | Previous non-Huber best |
| 5 | 379_HuberSubjectZPCA16Valence_CenterArousal | 46.7446 | 50.5545 | 42.9346 | Subject z-score helps some folds but not aggregate |

Main conclusion:

```text
HuberPCA16 is a narrow optimum.
Changing PCA dimension, switching to EEG/fNIRS-only, adding tree/boosted nonlinear heads, or agreement-gating Huber all failed to improve aggregate MAE.
```

Why this matters:

```text
The useful part is not "Huber + any representation".
It is specifically:
  low-rank whole EEG/fNIRS representation
  + robust MAE-aligned regression
  + valence-only correction
  + center arousal

This suggests the next search should focus on output calibration of the Huber trajectory, not on larger or more nonlinear predictors.
```

Fold behavior:

```text
Some variants win individual folds, for example subject-z Huber on fold 2 and subject-agreement on fold 6.
But no unlabeled gate yet identifies these fold conditions reliably.
The next batch should test gates/calibrations based only on prediction distribution and temporal structure.
```

Next batch 396-415:

```text
Use HuberPCA16 as the base predictor and search:
  1. smoothing window and median/exponential smoothers;
  2. tanh/soft residual clipping;
  3. positive-vs-negative asymmetric residual scaling;
  4. train-prediction distribution calibration;
  5. subject/trial prediction-level centering using only unlabeled prediction statistics.
```

## Iteration 396-415: Output Calibration of Huber Trajectory

Result file:

```text
experiments/results/iteration_396_415_no_prior_physio_calibration.json
```

Core hypothesis:

```text
HuberPCA16 is already a strong physiological-only valence predictor.
Its remaining error may come from output trajectory shape: smoothing, sign bias, residual amplitude, or prediction-level distribution shift.
```

Top results:

| Rank | Method | Overall MAE | Valence MAE | Arousal MAE | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | 407_HuberAsymP10N08Valence_CenterArousal | 46.6216 | 50.3087 | 42.9346 | New physiological-only best |
| 2 | 415_HuberRidgeBlend75Valence_CenterArousal | 46.6390 | 50.3433 | 42.9346 | Blending Huber with Ridge helps slightly |
| 3 | 401_HuberExpA03Valence_CenterArousal | 46.6391 | 50.3435 | 42.9346 | Exponential smoothing improves over window-5 |
| 4 | 399_HuberSmooth9Valence_CenterArousal | 46.6435 | 50.3523 | 42.9346 | Longer smoothing helps |
| 5 | 398_HuberSmooth7Valence_CenterArousal | 46.6517 | 50.3688 | 42.9346 | Longer smoothing helps |

Main conclusion:

```text
The best output calibration is asymmetric:
  positive valence residual scale = 1.0
  negative valence residual scale = 0.8

This improves Overall MAE by 0.0403 over HuberPCA16.
```

Module interpretation:

Let the Huber model output `v_h`, and define residual `r = v_h - 128`.

The winning correction is:

```text
v_hat = 128 + r                         if r >= 0
v_hat = 128 + 0.8 * r                   if r < 0
a_hat = 128
```

This means:

```text
The physiological model's negative-valence corrections are too large on average.
Positive corrections should be mostly preserved.
```

What did not work:

| Module | Observation | Reason |
| --- | --- | --- |
| Mean/std or IQR calibration | Much worse | Train prediction spread does not transfer to unseen subjects |
| Hard or tanh clipping | Worse than sign-asymmetry | It suppresses useful positive residuals too much |
| Subject prediction centering | Worse | Subject-level mean prediction contains useful signal or at least should not be forced to 128 |
| No smoothing | Worse than smoother variants | Physiological prediction noise is temporally correlated |

Current best physiological-only path:

```text
EEG/fNIRS features
-> low-rank PCA16
-> Huber valence head
-> window-5 smoothing
-> asymmetric residual correction, pos=1.0, neg=0.8
-> arousal fixed at center

Best so far:
  Overall MAE  = 46.6216
  Valence MAE  = 50.3087
  Arousal MAE  = 42.9346
```

Next batch 416-435:

```text
Refine the asymmetric residual module:
  1. try nearby negative scales 0.6, 0.7, 0.9;
  2. combine asymmetry with longer/exponential smoothing;
  3. combine asymmetry with Huber-Ridge blending;
  4. compare asymmetry on Huber, Ridge, and Elastic heads;
  5. test magnitude-dependent negative shrinkage.
```

## Iteration 416-435: Asymmetric Residual Refinement

Result file:

```text
experiments/results/iteration_416_435_no_prior_physio_asym_refine.json
```

Core hypothesis:

```text
The winning 396-415 module found that negative valence residuals should be shrunk.
This batch refines the negative scale, smoothing, and Huber/Ridge blending around that module.
```

Top results:

| Rank | Method | Overall MAE | Valence MAE | Arousal MAE | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | 416_HuberAsymP10N06Valence_CenterArousal | 46.5819 | 50.2291 | 42.9346 | New physiological-only best |
| 2 | 425_HuberAsymBlend75Valence_CenterArousal | 46.5968 | 50.2589 | 42.9346 | Huber/Ridge blend helps but not enough |
| 3 | 423_HuberAsymP10N08Exp03Valence_CenterArousal | 46.5971 | 50.2596 | 42.9346 | Exponential smoothing plus asymmetry is close |
| 4 | 417_HuberAsymP10N07Valence_CenterArousal | 46.6017 | 50.2687 | 42.9346 | Negative scale 0.7 is close |
| 5 | 435_HuberAsymExpBlendValence_CenterArousal | 46.6042 | 50.2737 | 42.9346 | Combining smoothers helps slightly |

Main conclusion:

```text
The negative-valence residual should be shrunk more aggressively than the previous 0.8 estimate.
The best rule so far is:

  r = v_huber - 128
  v_hat = 128 + 1.0 * r  if r >= 0
  v_hat = 128 + 0.6 * r  if r < 0
  a_hat = 128
```

Why it works:

```text
HuberPCA16 captures useful valence direction.
But in subject-disjoint evaluation, negative predictions appear over-confident.
Asymmetric residual scaling keeps useful positive valence evidence and brakes negative over-correction.
```

## Stage Summary: 100 New Physiological-Only Experiments

Covered iterations:

```text
336-355  Dimwise signal correction
356-375  Adaptive alignment and robust heads
376-395  Huber module dissection
396-415  Output calibration of Huber trajectory
416-435  Asymmetric residual refinement
```

Best progression:

| Stage | Best method | Overall MAE | Gain vs previous best | Main useful module |
| --- | --- | ---: | ---: | --- |
| Reference | 333_PCAEarlyDirectRidge_c8_a10000p0_SignalSmooth5 | 47.0764 | - | Direct low-rank physiological signal |
| 336-355 | 354_AgreementGatedPCA16Valence_CenterArousal | 46.7142 | 0.3622 | Valence-only correction, center arousal |
| 356-375 | 364_HuberPCA16Valence_CenterArousal | 46.6619 | 0.0523 | Huber robust valence head |
| 396-415 | 407_HuberAsymP10N08Valence_CenterArousal | 46.6216 | 0.0403 | Asymmetric negative residual shrink |
| 416-435 | 416_HuberAsymP10N06Valence_CenterArousal | 46.5819 | 0.0397 | Stronger negative residual shrink |

Current physiological-only best:

```text
416_HuberAsymP10N06Valence_CenterArousal
Overall MAE = 46.5819
Valence MAE = 50.2291
Arousal MAE = 42.9346
```

Key evidence after 100 experiments:

```text
1. EEG/fNIRS signal is useful mainly for valence.
2. Arousal residual remains unreliable; center arousal is still the safest MAE choice.
3. Huber is better than Ridge for MAE because label residuals are heavy-tailed.
4. Negative valence residuals over-shoot; asymmetric scaling is a real module, not just a random tweak.
5. Larger nonlinear models, trial z-score, subject centering, and broad train-distribution calibration usually hurt.
```

Working module name:

```text
RAVC: Robust Asymmetric Valence Correction

RAVC = PCA16 EEG/fNIRS representation
     + Huber valence regression
     + temporal smoothing
     + asymmetric residual calibration
     + conservative center arousal
```

Next 100 experiments:

```text
436-455  Arousal conservative probes around RAVC
456-475  Multi-expert physiological prediction fusion
476-495  Subject reliability and prediction-confidence gates
496-515  State-space and temporal-shape filters
516-535  Final combination modules built from the strongest parts
```

## Iteration 436-535: Final 100 Physiological-Only Search

Result file:

```text
experiments/results/iteration_436_535_no_prior_physio_final100.json
```

This batch contains 100 new candidates plus 2 references:

```text
436-455  Arousal conservative probes around RAVC
456-475  Multi-expert physiological prediction fusion
476-495  Subject reliability and prediction-confidence gates
496-515  State-space and temporal-shape filters
516-535  Final combination modules
```

Top results:

| Rank | Method | Overall MAE | Valence MAE | Arousal MAE | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | 511_StateTrialStartCenterBlend40 | 46.4686 | 50.0025 | 42.9346 | New physiological-only best |
| 2 | 510_StateTrialStartCenterBlend20 | 46.5249 | 50.1152 | 42.9346 | Same module, weaker start brake |
| 3 | 494_ReliabilitySubjectMeanToTrainMeanLight | 46.5306 | 50.1265 | 42.9346 | Light subject mean recentering helps |
| 4 | 506_StateSlopeLimit2ThenExp | 46.5316 | 50.1287 | 42.9346 | State-space smoothing helps |
| 5 | 498_StateExp02Asym | 46.5329 | 50.1313 | 42.9346 | EMA after asymmetric correction helps |

Main new finding:

```text
Trial onset is a weak point.
Blending only the first 10 seconds of each trial toward center gives the largest new gain.
```

The winning state filter is:

```text
For each trial and each timestamp t in the first 10 seconds:

  w(t) = 0.40 * (1 - t / 10)
  v_hat(t) = (1 - w(t)) * RAVC(t) + w(t) * 128

For t >= 10:

  v_hat(t) = RAVC(t)

a_hat(t) = 128
```

Interpretation:

```text
The 5-second baseline and early stimulus transition create unstable physiology-to-valence mapping.
The model overreacts near onset, so an onset brake improves subject-disjoint MAE.
This is different from generic smoothing: it is a time-local reliability correction.
```

## Final Summary: 200 New Physiological-Only Experiments

Covered iterations:

```text
336-535
Total new candidates = 200
```

Best progression:

| Milestone | Best method | Overall MAE | Valence MAE | Arousal MAE | Main module |
| --- | --- | ---: | ---: | ---: | --- |
| Starting physiological reference | 333_PCAEarlyDirectRidge_c8_a10000p0_SignalSmooth5 | 47.0764 | 50.6582 | 43.4946 | Direct EEG/fNIRS Ridge |
| 336-355 | 354_AgreementGatedPCA16Valence_CenterArousal | 46.7142 | 50.4937 | 42.9346 | Valence-only, center arousal |
| 356-375 | 364_HuberPCA16Valence_CenterArousal | 46.6619 | 50.3892 | 42.9346 | Huber robust head |
| 396-415 | 407_HuberAsymP10N08Valence_CenterArousal | 46.6216 | 50.3087 | 42.9346 | Negative residual shrink 0.8 |
| 416-435 | 416_HuberAsymP10N06Valence_CenterArousal | 46.5819 | 50.2291 | 42.9346 | Negative residual shrink 0.6 |
| 436-535 | 511_StateTrialStartCenterBlend40 | 46.4686 | 50.0025 | 42.9346 | Trial-onset reliability brake |

Gains:

| Comparison | Overall MAE gain |
| --- | ---: |
| vs Center128 no-prior | 1.0977 |
| vs previous best direct signal model 333 | 0.6078 |
| vs official ASAC demo reference | 0.5401 |
| vs best after first 100 new experiments | 0.1133 |

Current best physiological-only framework:

```text
RAVC-S: Robust Asymmetric Valence Correction with State Onset Brake

1. Extract EEG/fNIRS physiological features.
2. Project full EEG/fNIRS features to PCA16.
3. Fit Huber valence regressor.
4. Smooth valence trajectory.
5. Apply asymmetric residual correction:
     positive residual scale = 1.0
     negative residual scale = 0.6
6. Apply trial-onset brake for the first 10 seconds:
     max center blend = 0.40 at t=0, linearly decays to 0 by t=10.
7. Keep arousal at 128.
```

What consistently helped:

| Module | Evidence | Likely reason |
| --- | --- | --- |
| Valence-only physiological correction | 336-355 | Signal improves valence but arousal residual hurts MAE |
| Huber head | 356-375 | Better aligned with MAE under heavy-tailed subject residuals |
| Negative residual shrink | 396-435 | Model over-corrects toward low valence |
| Trial-onset brake | 436-535 | Early physiology after video onset is less reliable |
| Light state filtering | 436-535 | Prediction noise is temporally correlated |

What consistently failed:

| Module | Evidence | Likely reason |
| --- | --- | --- |
| Direct arousal residual | 436-455 and earlier | Arousal signal is not robust under subject-disjoint split |
| Strong subject/trial z-score | 356-375 | Removes useful absolute physiological level |
| Strong train distribution calibration | 396-415 | Train prediction spread does not transfer to unseen subjects |
| Larger nonlinear heads | 356-395 | Overfit subject-specific structure |
| Hard residual clipping | 396-415 | Suppresses useful positive valence evidence |

Recommended submission variants:

```text
Physiology-only submission:
  511_StateTrialStartCenterBlend40

Prior-assisted submission:
  Keep the previous best 222_BCRF_onSCRF route separately.

Do not mix these blindly:
  physiology-only is now a cleaner no-prior decoder;
  prior-assisted remains stronger for leaderboard MAE.
```
