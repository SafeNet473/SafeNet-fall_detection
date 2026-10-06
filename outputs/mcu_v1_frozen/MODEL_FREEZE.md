# MCU v1 model freeze

**MCU v1 is frozen for implementation. Any future change to preprocessing, trigger, window, feature set, model weights, or feature ordering must create MCU v2 or a new experimental branch.**

**The logistic weights are frozen. The final hardware operating threshold is NOT frozen.** Neither documented operating point below is silently promoted to deployment. Five-feature and prospective seven-feature expansions are excluded and preserved only as comparison artifacts.

## Complete pipeline

Raw ADXL345 XYZ counts -> counts/256 in g -> causal two-biquad 5 Hz acceleration filter; in parallel raw g -> gravity EMA -> magnitude and gravity-aligned streams -> magnitude/jerk rising-edge candidate trigger -> circular prehistory and delayed 300-sample window -> three features -> training StandardScaler -> fixed logistic score -> explicitly chosen operating threshold -> FALL/non-fall. Filter/gravity/trigger/history run continuously; window aggregation and classifier inference run for completed candidates. Gyroscope is not used.

## Exact preprocessing

Sampling is 200 Hz. The causal filter is fourth-order Butterworth with 5 Hz nominal cutoff, two cascaded direct-form-II-transposed biquads per axis. SOS rows `[b0,b1,b2,a0,a1,a2]` are:

```json
[
  [
    3.123897691708262e-05,
    6.247795383416523e-05,
    3.123897691708262e-05,
    1.0,
    -1.7259333950369407,
    0.7474473719077911
  ],
  [
    1.0,
    2.0,
    1.0,
    1.0,
    -1.863800492075235,
    0.8870329996526946
  ]
]
```

Per section: `y=b0*x+z1; z1_new=b1*x-a1*y+z2; z2_new=b2*x-a2*y`. Initialize `zi=sosfilt_zi(SOS)*first_raw_sample_g` for each axis; keep states across triggers/windows and reset only per independent trial. The manifest includes unit-input zi values as a reproducibility aid. There is no backward pass or future-dependent phase compensation.

Gravity uses raw, unfiltered acceleration in g: `G[t]=alpha*r[t]+(1-alpha)*G[t-1]`, `alpha=1-exp(-2*pi*0.5/200)=0.01558523664828626`, `G[-1]=r[0]`. Preserve history across windows. Normalize with `u=G/max(norm(G),1e-8)`; the denominator floor is part of the frozen behavior.

Numeric reference: float64 filter/gravity; filtered samples cast float32 then back to float64 before derived calculations; float64 features, scaler and logistic inference. An all-float32 port is not yet certified equivalent.

## Trigger and event window

`m[t]=norm(a[t])`; `j[t]=200*abs(m[t]-m[t-1])`, with `j[0]=0`. Combined condition `H[t]=(m[t]>=1.1 g) OR (j[t]>=5.0 g/s)`. Accept only `H[t] AND NOT H[t-1]` and at least 300 samples (1.5 s) since the previous accepted trigger. Start with previous-high=false and last-trigger=-300. Crossings in refractory are discarded; remaining high does not retrigger. Even an accepted trigger with an incomplete window updates refractory state.

`pre100_post200` means `[k-100,k+200)` for trigger k: 100 pre-trigger samples, then 200 samples including k through k+199, total 300 (nominal 1.5 s). Decision occurs at k+200, exactly 1 s after trigger, and that current sample is excluded. Require k>=100 and k+200<trial_length; discard incomplete windows, without padding/shifting. The reference ring has 301 rows of m, h, abs(p). Post-trigger samples are causal because inference waits until they arrive.

## Exact feature definitions and order

For each window sample, use filtered `a`, raw-driven gravity `G`, `u=G/max(norm(G),1e-8)`, signed `p=a dot u`, residual `b=a-p*u`, and `h=norm(b)`. The ordered vector is:

1. **ga_C2:** `max_i h[i]`, i=0..299, in g.
2. **jerk_abs_mean:** `sum_(i=1..299) abs(200*(m[i]-m[i-1])) / 299`, in g/s. No cross-window difference is included.
3. **ga_parallel_peak:** `max_i abs(p[i])`, i=0..299, in g.

The jerk is the derivative of scalar magnitude, not vector jerk magnitude. Parallel acceleration retains gravity; no 1 g subtraction is performed. Epsilon normalization need not produce a unit vector at near-zero gravity.

## Frozen logistic model

Train-only StandardScaler, population standard deviation. Feature order is the table order. L2 logistic regression, C=10.0, class_weight=balanced, solver=liblinear, max_iter=5000, tol=1e-8, random_state=473. The manifest copies every exported estimator hyperparameter, including defaults, from the authoritative joblib object. No refit occurs.

| Feature | Mean | Scale | Coefficient |
|---|---:|---:|---:|
| ga_C2 | 0.4885001226818366 | 0.3393389000402486 | 3.8564795859958036 |
| jerk_abs_mean | 4.842116154469659 | 4.666720996263801 | -4.541012509245931 |
| ga_parallel_peak | 1.830923027297042 | 0.743175050800754 | 1.8653532808946023 |

Intercept: `-3.3544620174694724`. `z=intercept+sum(w_i*(x_i-mean_i)/scale_i)`; `score=1/(1+exp(-z))`; predict FALL when `score>=tau`.

Algebraic unstandardized-feature form: `w_raw=[11.36468464281104, -0.9730627806722294, 2.5099783407485585]`, `b_raw=-8.790005992210238`. These are derived as `w/scale` and `intercept-sum(w_raw*mean)`, not refitted. Compare `dot(w_raw,x)+b_raw >= log(tau/(1-tau))` to omit runtime sigmoid. Fusion/float32 parity has not been verified, and the standardized equation remains the verified reference.

## Documented operating points and validation

| Operating point | Exact threshold | Sensitivity | ADL FA/hour | Event precision | ADL trial specificity | Trial BA |
|---|---:|---:|---:|---:|---:|---:|
| balanced_reference | 0.8850929456953477 | 0.9333333333333333 | 4.889271580022436 | 0.9308510638297872 | 0.9772727272727273 | 0.9553030303030303 |
| sensitivity_oriented_candidate | 0.7742031648776018 | 0.952 | 12.712106108058334 | 0.8707317073170732 | 0.9423076923076923 | 0.9471538461538461 |

Balanced reference: 350 detected / 25 missed falls, 376 positive decisions and 15 ADL alarms. Sensitivity candidate: 357 detected / 18 missed, 410 positive decisions and 39 ADL alarms. Neither is a final hardware threshold.

Validation comprises 375 fall trials and 572 ADL trials, 3.0679416666666666 ADL hours. Matching uses trigger within +/-1 s of a fall trial’s raw-magnitude argmax proxy and a complete window. Credit at most one detected fall per trial. Precision divides credited detections by all positive candidate decisions, counting unmatched/duplicate fall-recording alarms as false positives. Trial BA averages fall-event sensitivity and the fraction of ADL trials with no alarm; it is not event specificity. Candidate recall is 370/375=98.6667%, an end-to-end ceiling; five no-matched-trigger misses cannot be recovered by classifier threshold changes.

## Verification and provenance

The serialized manifest preserves exact feature ordering, scaler/model parameters and estimator hyperparameters. Reconstructed inference produces bit-for-bit identical float64 scores for all 4,387 validation candidates against both authoritative logistic.joblib and saved CSV scores, with zero reference prediction mismatches. Both documented thresholds have zero prediction mismatches against the authoritative model. Reconstruction only assigns saved fitted attributes and calls transform/predict_proba; it never calls fit. All authoritative source hashes remained unchanged. See verification.json and provenance.json.

Key sources: ../causal_trained_models_v1/model_parameters.json and logistic.joblib; ../causal_event_study/experiment_plan.json and selection.json; ../causal_sensitivity_analysis/operating_points.csv; ../sisfall/causal_events.py and orientation_features.py. provenance.json provides workspace-relative paths and SHA-256 hashes of every authoritative input and code file used, including the freeze script. Git HEAD is supplementary because individual files may be uncommitted.

## Known limitations and unresolved work

SisFall is waist-mounted, simulated-fall data, not wrist-domain validation. Raw-magnitude peak matching is proxy supervision, not annotated onset/impact. Validation has been reused for trigger/window/model/threshold development and is not an independent performance guarantee. Balanced logistic scores are not automatically calibrated probabilities. Short scripted ADL recordings do not establish real-world daily alarm rates. Desktop verification does not establish float32 firmware parity, actual nRF5340 execution time, RAM high-water mark, power, sensor timing, or live boot/shutdown behavior. No test optimization or new test evaluation was performed.

| Frozen for MCU v1 | Unresolved |
|---|---|
| 200 Hz accelerometer units/conversion; saved causal SOS, initialization and numeric reference | Hardware acquisition/calibration and float32 parity |
| Raw gravity EMA, alpha and normalization floor | Wrist gravity-estimation behavior |
| Selected trigger, rising-edge/refractory semantics and pre100_post200 bounds | Real wrist performance and annotated timing validation |
| Exactly three features and their order; saved scaler, logistic weights and hyperparameters | Final hardware operating threshold |
| Two named documented thresholds retained as alternatives | Runtime, RAM, power measurements and firmware qualification |
| Source paths/hashes and desktop verification | Any algorithm/model/feature change requires MCU v2 or an experimental branch |
