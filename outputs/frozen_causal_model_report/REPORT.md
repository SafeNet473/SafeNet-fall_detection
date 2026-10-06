# Frozen causal fall-detection system: technical report

**Documentation date: October 4, 2026.** This report describes the saved three-feature causal logistic-regression model and its validation operating points. It is documentation, not a new experiment: no estimator was fitted, no threshold was selected again, and no preprocessing, trigger, feature, window, or prior artifact was changed.

The model weights are fixed. The saved balanced-accuracy threshold remains the reference; the sensitivity-oriented threshold is an alternative operating point. **A final hardware deployment threshold has not been designated.** “Frozen for MCU v1” below means the algorithmic baseline to be ported, not a completed or hardware-qualified product.

Stored parameter values are reproduced at their saved precision. Percentage/rate summaries are rounded for readability; linked CSVs retain their full stored values. Fused coefficients and arithmetic/storage budgets are explicitly labeled derivations rather than new model exports or hardware measurements.

## 1. System overview

The detector first identifies candidate motion events, then classifies each completed event window. The trigger and the classifier have separate thresholds and separate jobs.

```text
ADXL345 raw XYZ counts at 200 Hz
                 |
          convert counts to g
                 |
        +--------+-------------------------+
        |                                  |
causal 4th-order 5 Hz               causal 0.5 Hz gravity EMA
Butterworth filter                         |
        | a[t]                             | G[t]
        +----------------+-----------------+
                         |
      magnitude + gravity-aligned scalar streams
                         |
            circular history buffer
                         |
     magnitude/jerk rising-edge candidate trigger
             with 1.5 s refractory
                         |
       100 samples before + 200 from trigger onward
               wait 1.0 s after trigger
                         |
      [ga_C2, jerk_abs_mean, ga_parallel_peak]
                         |
          training-fitted scaler + fixed logistic model
                         |
         classifier score >= operating threshold?
                         |
                  FALL / ADL
```

The acceleration filter, gravity EMA, magnitude/jerk trigger, and history buffer run continuously. Gravity is driven by **raw acceleration in g**, not the output of the 5 Hz filter. The three scalar streams used for window features are magnitude, perpendicular magnitude, and absolute parallel acceleration. They can be calculated continuously and retained in the ring buffer.

A trigger schedules a candidate decision; it does not itself declare a fall. Once the post-trigger interval is available, the detector computes the three window aggregates, applies the logistic model, and compares the score with the chosen classifier threshold. No candidate means no event decision, not a stream of counted “true-negative events.” A negative candidate decision is described as ADL/non-fall.

**Source:** [causal_events.py](../sisfall/causal_events.py), `Frontend`, `derived`, `triggers`, `ring_windows`, and `feature_row`; [train_causal_models.py](../sisfall/train_causal_models.py), `NAMES` and model inference.

## 2. Input data

| Item | Current setting |
|---|---|
| Sensor used | ADXL345 accelerometer, XYZ; columns 1–3 of each nine-column SisFall trial |
| Stated sensor range/resolution | ±16 g, 13-bit |
| Gyroscope | Not used by this three-feature causal model |
| Second accelerometer | Not used |
| Sampling frequency | 200 Hz; 5 ms between samples |
| Physical unit | g, not m/s² |
| Conversion | `acceleration_g = raw_count × (2 × 16 / 2^13) = raw_count / 256` |
| Stored conversion factor | `0.00390625` g/count |
| Trial identifier | `<activity>_<subject>_<trial>.txt`, e.g. `F13_SA05_R01.txt` |
| Source placement | Waist-mounted SisFall sensor |

**SisFall is waist-mounted data. This is not wrist-domain validation.** Constant-coordinate rotation robustness cannot reproduce arm motion or moving the sensor from the waist to the wrist. Fall recordings are simulated/scripted falls rather than a prospective stream of naturally occurring wrist-recorded falls.

The subject partitions remain those in [split_subjects.json](../../processed/baseline_v2/split_subjects.json). The current model was fitted on training candidates only; validation subjects are SA05, SA08, SA09, SA13, SA19, SE08, SE09, and SE11. Validation contains 375 fall trials and 572 ADL trials, with **3.0679416666666666 recorded ADL hours**. These cohorts and recordings should not be treated as a representative daily-life event prevalence.

**Sources:** [baseline config](../../processed/baseline_v2/config.json), `sampling_rate_hz`, `channels`, and `counts_to_g`; [pipeline.py](../sisfall/pipeline.py), `SCALE` and `load_trial`; [SisFall Readme](../../data/Readme.txt), file/channel/activity definitions; [orientation methods](../sisfall/ORIENTATION_METHODS.md), waist/wrist limitations; [sensitivity provenance](../causal_sensitivity_analysis/provenance.json), validation counts and exposure. The old baseline config is cited for input metadata only—its zero-phase filter and 200-sample windows are **not** the current causal settings.

## 3. Causal signal preprocessing

### 3.1 Acceleration filter

The current acceleration filter is a **fourth-order Butterworth low-pass filter**, nominal cutoff **5 Hz**, sampling frequency **200 Hz**. It is designed as `butter(4, 5, fs=200, output='sos')` and applied once forward in time using two cascaded direct-form-II-transposed biquads per axis.

The saved SOS coefficients are rows of `[b0, b1, b2, a0, a1, a2]`:

```text
[3.123897691708262e-05, 6.247795383416523e-05,
 3.123897691708262e-05, 1.0,
 -1.7259333950369407, 0.7474473719077911]

[1.0, 2.0, 1.0, 1.0,
 -1.863800492075235, 0.8870329996526946]
```

For each section and axis, with input `v` and previous states `z1,z2`, the implemented update is:

```text
y      = b0*v + z1
z1_new = b1*v - a1*y + z2
z2_new = b2*v - a2*y
```

The output of one section becomes the next section's input. At the first observed sample of a trial, each axis's section states are initialized as `sosfilt_zi(SOS) × first_raw_sample`. This is steady-state initialization for a constant signal equal to that first observation; it does not inspect later samples or guarantee removal of motion/startup transients.

Filter states persist across all samples, candidate triggers, and windows in a trial. They reset only at trial boundaries in replay. A continuously operating device must retain its state between events; its boot/restart behavior must implement the documented initialization rather than resetting at every trigger.

This filter is causal because each output uses only the current input and stored past state. The previous `sosfiltfilt` pipeline ran forward and backward over a whole trial. Its backward pass used future samples, eliminated phase delay, and changed the effective magnitude response/order relative to a single forward pass. The same nominal order and cutoff therefore do not imply identical filtered values. No future-dependent phase compensation is applied in the current pipeline.

**Sources:** [causal experiment plan](../causal_event_study/experiment_plan.json), `sos`, `fs`, and `filter_initialization`; [causal_events.py](../sisfall/causal_events.py), `Frontend`; earlier [pipeline.py](../sisfall/pipeline.py), `preprocess`.

### 3.2 Gravity estimator

Let `r[t]` be the raw acceleration vector after conversion to g. The separate gravity state is:

\[
G[t]=(1-\alpha)G[t-1]+\alpha r[t],
\qquad \alpha=1-\exp\left(-\frac{2\pi\,0.5}{200}\right)
=0.01558523664828626.
\]

This is a first-order low-frequency EMA with a **nominal/effective cutoff approximately 0.5 Hz**, using the exponential pole mapping shown above. Its corresponding continuous-time time constant is approximately 0.3183 s; it is not another fourth-order filter. The exact deployed specification is the discrete recurrence and alpha, not an assumed ideal brick-wall gravity separator.

Initialize `G[-1] = r[0]` at each trial start and carry the state continuously. The code evaluates `alpha*raw + (1-alpha)*gravity`. The algebraic form `G += alpha*(r-G)` is equivalent in real arithmetic but may round differently.

Gravity provides a local estimated vertical direction, permitting projection without assuming that a particular sensor axis is vertical. This supports invariance to a constant rotation of both acceleration and its gravity history. It is not a full attitude estimator: dynamic acceleration can contaminate it, and it can lag time-varying wrist rotation.

**Sources:** [orientation_features.py](../sisfall/orientation_features.py), `FS`, `ALPHA`, `GRAVITY_EPS`; [orientation provenance](../orientation_study/provenance.json), `gravity_cutoff_hz=0.5`, `gravity_alpha=0.01558523664828626`; [causal_events.py](../sisfall/causal_events.py), `Frontend.push`.

### 3.3 Numeric interface

The desktop causal filter and gravity states use float64. Each filtered acceleration sample is then cast to float32 and back to float64 before magnitude/projection calculations, preserving the earlier saved-window numeric interface. Features and logistic inference use float64. This intermediate quantization is part of the saved reference behavior; using float32 for all states is a different numerical implementation and has not been verified equivalent on an MCU.

**Source:** [causal_events.py](../sisfall/causal_events.py), `derived`; [causal training verification](../causal_trained_models_v1/verification.json), recorded Python/library versions.

## 4. Event trigger

For filtered acceleration `a[t]`, define:

\[
m[t]=\sqrt{a[t]\cdot a[t]},\qquad
j[t]=200\,|m[t]-m[t-1]|,
\]

with `j[0]=0`. The trigger condition is:

\[
H[t]=(m[t]\ge1.1\text{ g})\ \lor\ (j[t]\ge5.0\text{ g/s}).
\]

A candidate trigger occurs on a **rising edge of this combined OR condition**, `H[t] AND NOT H[t-1]`, provided at least **300 samples = 1.5 s** have passed since the last accepted trigger. Before the first sample the high state is false; an initially high magnitude can therefore cause an edge. There is no independent retrigger when one branch rises while the other already keeps the OR condition high.

Crossings during refractory are discarded, not queued for later. A continuously high condition does not generate periodic events when the refractory expires; a later low-to-high transition is needed. Triggering uses neither activity labels nor the offline impact proxy. The count of accepted triggers is distinct from the count of complete candidate windows: boundary triggers may not have sufficient prehistory or post-trigger samples.

This is a **high-recall candidate generator**, not the final fall classifier. The selected trigger/window achieves validation localized complete-candidate recall of **0.9866666666666667 = 370/375 = 98.67%**. The original 99% target was not met. This recall is the maximum end-to-end sensitivity possible by classifier-threshold adjustment under the current matching convention. **100% end-to-end sensitivity is impossible with the fixed trigger/window.**

**Sources:** [causal selection](../causal_event_study/selection.json), `trigger=[1.1,5.0]`, `design=pre100_post200`, and `target_99_percent_met=false`; [causal_events.py](../sisfall/causal_events.py), `REFRACTORY=300` and `triggers`; [sensitivity provenance](../causal_sensitivity_analysis/provenance.json), `candidate_recall_ceiling`.

## 5. Event window

The selected design is **`pre100_post200`**. For a trigger at sample index `k`:

| Quantity | Exact convention |
|---|---|
| Pre-trigger samples | `k-100` through `k-1`: 100 samples, nominally 0.5 s |
| Trigger-and-after samples | `k` through `k+199`: 200 samples, nominally 1.0 s |
| Window bounds | Half-open `[k-100, k+200)` |
| Total | 300 samples, nominally 1.5 s |
| First-to-last sample-center span | 299/200 = 1.495 s |
| Decision sample | `k+200`, exactly 1.0 s after the trigger |

The phrase “200 post-trigger samples” includes the trigger sample itself in this implementation. The sample at `k+200` is received at decision time but is **excluded** from the feature window. This deliberate convention avoids an off-by-one change to the saved behavior.

A circular buffer preserves history before the trigger. After a trigger, the state machine waits until `k+200` arrives, then aggregates the designated window. The desktop reference ring has **301 rows × 3 scalar channels**; its extra row retains the excluded current sample without overwriting the oldest window sample.

Using post-trigger samples is causal because no decision is emitted until those samples are available. “Causal” does not mean “zero delay.” The scheduled delay is exactly 1 s relative to the trigger, whereas delay relative to a real fall onset or impact additionally depends on trigger timing and filter delay.

Require `k>=100` and `k+200<recording_length`. Incomplete boundary windows are discarded, never shifted, padded, or completed using invented future samples. Trigger refractory state still follows accepted trigger times even when an associated window cannot be completed. The offline replay can precompute the schedule, but prefix-invariance tests verify that shortening a recording does not alter earlier emitted events. A live device would expire or cancel an incomplete pending event at shutdown rather than knowing the recording length in advance.

**Sources:** [causal_events.py](../sisfall/causal_events.py), `DESIGNS`, `ring_windows`; [causal experiment plan](../causal_event_study/experiment_plan.json), `boundary` and `longer_window`; [saved regression results](../causal_event_study/regression_tests.json) and [test definitions](../sisfall/test_causal_events.py).

## 6. Exact feature vector

The trained logistic model's order is exactly:

\[
x=[x_1,x_2,x_3]
=[\mathrm{ga\_C2},\ \mathrm{jerk\_abs\_mean},\ \mathrm{ga\_parallel\_peak}].
\]

For each sample in the 300-sample window, use filtered `a` and the corresponding raw-driven gravity state `G`:

\[
D=\max(\|G\|,10^{-8}\text{ g}),\qquad u=G/D,
\qquad p=a\cdot u,\qquad b=a-pu.
\]

When `||G||` is above the floor, `u` is a unit vector along estimated gravity. The signed scalar `p` is acceleration parallel to that direction. The vector `b` is the perpendicular residual. Neither is automatically “linear acceleration with gravity removed”: `p` retains the gravitational contribution, and both depend on the quality of the direction estimate.

The denominator floor is part of the equation. Below it, `u` is not a unit vector and `b` is not a strict orthogonal projection. Firmware must retain this behavior rather than silently using a different fallback or unconditional Pythagorean shortcut.

Let `m_i=||a_i||`, `i=0,...,299` within the event window:

| Feature | Exact equation | Physical meaning | Units | Gravity needed? | Incremental aggregation |
|---|---|---|---|---|---|
| `ga_C2` | `max_i sqrt(b_i · b_i)` | Peak acceleration magnitude perpendicular to estimated gravity | g | Yes | Running maximum; equivalently retain max squared norm and take one final sqrt, subject to floating-point verification |
| `jerk_abs_mean` | `(1/299) Σ_(i=1..299) abs(200 × (m_i-m_(i-1)))` | Average absolute rate of change of acceleration magnitude | g/s | No | Previous magnitude plus sum; exclude the difference crossing the window's start |
| `ga_parallel_peak` | `max_i abs(p_i)` | Largest absolute acceleration component parallel to estimated gravity | g | Yes | Running maximum of absolute projection |

The jerk feature has **299 differences**, not 300. It is the derivative of scalar magnitude, not `||a_i-a_(i-1)|| × 200`. The maxima are over the full selected event window, not over additional candidate windows. No gyro statistic, variance, raw axis-specific C2/C8 feature, or additional feature enters the model.

For a window with known boundaries, the three aggregates require only two maxima, the previous magnitude, a sum, and a counter, in addition to shared gravity/filter state. Retrospective pre-trigger inclusion still requires the circular history or equivalent retained summaries. The current reference scans the buffered scalar window; an incremental MCU implementation must reproduce these same sample bounds and equations.

**Sources:** [model_parameters.json](../causal_trained_models_v1/model_parameters.json), `logistic.feature_names`; [causal_events.py](../sisfall/causal_events.py), `derived` and `feature_row`. [orientation_features.py](../sisfall/orientation_features.py) supplies the original projection definitions and epsilon; its older 200-sample shape restriction is not the 300-sample causal extraction function.

## 7. Logistic-regression model

### 7.1 Training and scaling

The model was fitted from scratch on **12,253 training candidates**, including **1,081 positive candidate labels**. A positive training label means a candidate from a fall trial whose trigger lies within ±200 samples of the raw-magnitude peak proxy. Other candidates—including unmatched candidates in a fall trial—were negative examples. This is proxy supervision, not independently annotated event timing.

`StandardScaler` was fitted on training features only, using its population-standard-deviation convention (`ddof=0`). Logistic regression used **L2 regularization, `C=10.0`, `class_weight='balanced'`, `solver='liblinear'`, `max_iter=5000`, `tol=1e-8`, `random_state=473`**. Balanced class weighting is computed from training candidate class frequencies, not from real-world fall prevalence. Neither scaler nor model was refitted on validation. The selected C came from `[0.01,0.1,1,10]` using validation metrics.

| Feature, in model order | Scaler mean μ | Scaler scale s | Learned standardized coefficient w |
|---|---:|---:|---:|
| `ga_C2` | 0.4885001226818366 | 0.3393389000402486 | 3.8564795859958036 |
| `jerk_abs_mean` | 4.842116154469659 | 4.666720996263801 | -4.541012509245931 |
| `ga_parallel_peak` | 1.830923027297042 | 0.743175050800754 | 1.8653532808946023 |

**Intercept:** `-3.3544620174694724`.

The exact standardized inference equation is:

\[
z=-3.3544620174694724
+3.8564795859958036\frac{x_1-0.4885001226818366}{0.3393389000402486}
-4.541012509245931\frac{x_2-4.842116154469659}{4.666720996263801}
+1.8653532808946023\frac{x_3-1.830923027297042}{0.743175050800754},
\]

\[
q=\sigma(z)=\frac{1}{1+\exp(-z)}.
\]

The negative jerk coefficient is a learned conditional association given the other features, not a general rule that high jerk makes motion safe. The logistic output is used as a **classifier score**. Balanced weighting, proxy labels, candidate sampling, and population shift mean it should not automatically be interpreted as a calibrated real-world probability of falling.

**Sources:** [model_parameters.json](../causal_trained_models_v1/model_parameters.json), `logistic`; [training plan](../causal_trained_models_v1/experiment_plan.json); [train_causal_models.py](../sisfall/train_causal_models.py), scaler/model construction; [verification.json](../causal_trained_models_v1/verification.json), training counts. The completed `_v1` directory is authoritative; the earlier unversioned training directory contains an explicitly marked incomplete run.

### 7.2 Algebraically fused raw-feature equation

For MCU inference, derive `w_raw_i=w_i/s_i` and `b_raw=b-Σ(w_raw_i μ_i)`. “Raw-feature” here means **unstandardized extracted features**, not raw accelerometer counts.

Using the saved parameters, ordinary double-precision arithmetic gives:

```text
w_raw = [11.36468464281104,
         -0.9730627806722294,
          2.5099783407485585]
b_raw = -8.790005992210238
```

Thus the algebraically equivalent equation is:

\[
z=11.36468464281104x_1
-0.9730627806722294x_2
+2.5099783407485585x_3
-8.790005992210238.
\]

These numbers are **documentation derivations from the saved scaler/weights**, not newly fitted coefficients or a replacement model artifact. Reassociation and conversion to float32 may change rounding, especially for scores on a threshold boundary; the derivation is not a bit-for-bit firmware validation.

For binary decisions, sigmoid is unnecessary in real arithmetic:

\[
q\ge\tau\iff z\ge\log\left(\frac{\tau}{1-\tau}\right),\quad 0<\tau<1.
\]

The derived logit thresholds are `2.041569084741143` for the BA reference and `1.2321986907376832` for the sensitivity candidate. Alternatively, fold the logit threshold into `b_raw` and compare the resulting dot product with zero; the corresponding derived intercepts are `-10.831575076951381` and `-10.02220468294792`. This removes runtime sigmoid/exp and division for binary output, but still requires numerical equivalence checks before adoption. No saved threshold or inference artifact was changed for this report.

## 8. Decision threshold and operating point

Training determines the scaler and logistic coefficients. Operating-point selection determines the score cutoff **after those weights are fixed**. Changing only the cutoff changes the sensitivity/false-alarm tradeoff without retraining the model.

The artifacts contain two useful choices, not a declared final hardware threshold:

| Validation quantity | Balanced reference | Sensitivity-oriented candidate |
|---|---:|---:|
| Exact classifier threshold τ | **0.8850929456953477** | **0.7742031648776018** |
| Fall-event sensitivity | 0.9333333333333333 | 0.952 |
| ADL false alarms/hour | 4.889271580022436 | 12.712106108058334 |
| Event precision | 0.9308510638297872 | 0.8707317073170732 |
| ADL trial specificity | 0.9772727272727273 | 0.9423076923076923 |
| Trial balanced accuracy | 0.9553030303030303 | 0.9471538461538461 |
| Detected / missed fall trials | 350 / 25 | 357 / 18 |
| Emitted positive decisions | 376 | 410 |
| ADL false alarms | 15 | 39 |

The sensitivity candidate is the **highest validation score boundary attaining at least 95% sensitivity**, as saved in the threshold analysis. Relative to the reference it detects seven additional falls (+1.8667 percentage points of sensitivity), adds 24 ADL alarms (+7.8228 alarms/hour), and reduces precision by about 6.0119 percentage points. These are existing validation results, not a new threshold recommendation.

Comparison semantics are `score >= threshold`. The exact reference is not rounded `0.8851`, and the sensitivity cutoff is not rounded `0.774203` in computation.

**Source:** [operating_points.csv](../causal_sensitivity_analysis/operating_points.csv), rows `current_BA_reference` and `highest_threshold_sensitivity_ge_0.950000`; [model_parameters.json](../causal_trained_models_v1/model_parameters.json), stored reference `logistic.threshold`.

## 9. Validation results and metric definitions

All three models below use the **same causal candidate windows**, trigger, matching convention, and validation subjects. The offline-trained tree is a frozen transfer reference, not a newly trained causal model.

| Model and own selected operating point | Sensitivity | ADL false alarms/hour | Event precision | ADL trial specificity | Trial balanced accuracy | Tree depth / nodes |
|---|---:|---:|---:|---:|---:|---|
| Causal logistic, BA reference | 93.33% | 4.8893 | 93.09% | 97.73% | 95.53% | Not a tree |
| Causal shallow tree | 94.13% | 47.9149 | 67.75% | 90.73% | 92.43% | 2 / 7 |
| Frozen offline-trained tree | 84.80% | 12.3862 | 87.85% | 96.33% | 90.56% | 3 / 7 simplified binary export |

The new causal tree's selected positive-score cutoff is `0.971060451555281`; its four leaves and seven nodes were not pruned for this report. Although offered the same three features, its splits use only `ga_C2` and `jerk_abs_mean`. The frozen offline tree's binary export retains its original deployed decisions from positive-score threshold `0.7705231308225133`.

Logistic regression is the preferred current development reference because it gives almost the causal tree's sensitivity with substantially fewer ADL alarms (15 versus 147 over the same exposure), higher precision, and higher trial BA. The causal tree detects only three more validation falls (353 versus 350) at its selected operating point. This does not establish superiority on new subjects or wrist data, and does not itself select a final deployment threshold.

The evaluation assumes one fall event per fall trial. A complete candidate is matched when its **trigger time**, not its decision time, lies within ±1 s of that trial's raw-acceleration-magnitude argmax. Let `TP` be the number of fall trials with at least one matched positive decision, `N_F` the number of fall trials, `A` all positive candidate decisions, `A_ADL` positive decisions in ADL trials, `H_ADL` ADL recording hours, and `N_ADL` the number of ADL trials:

\[
\text{sensitivity}=TP/N_F,\qquad
\text{event precision}=TP/A,\qquad
\text{ADL false alarms/hour}=A_{ADL}/H_{ADL}.
\]

\[
\text{ADL trial specificity}
=\frac{\text{ADL trials with no positive decision}}{N_{ADL}},
\qquad
\text{trial BA}=\tfrac12(\text{sensitivity}+\text{ADL trial specificity}).
\]

Only one true positive is credited per fall trial. Duplicate or unmatched positive decisions in fall recordings count against event precision, even though they are not part of the **ADL-only** false-alarm rate. At the reference, 350 credited detections among 376 positive decisions leave 26 unmatched/duplicate false-positive decisions, including 15 in ADLs. Full recorded ADL time, including startup and tails, is the rate denominator. No extra alarm-merging policy is applied beyond the fixed trigger refractory.

There is no natural number of true-negative **events** in a continuous stream. Trial BA is therefore explicitly a mixed trial-level measure; multiple alarms within one ADL trial reduce its specificity only once. Candidate-level BA in other saved tables is a different quantity based on candidate proxy labels and excludes untriggered falls. Precision is undefined at a no-alarm boundary and is blank in the sweep.

Measured latency, when evaluated, is decision time minus the impact proxy for the earliest matched detection and is conditional on detection. It is not annotated onset latency. This report does not copy the previous offline-tree causal diagnostic's latency numbers onto the new logistic model; its fixed post-trigger delay is known, but no new timing evaluation was run here.

**Sources:** [causal model metrics](../causal_trained_models_v1/metrics.csv), validation rows; [model parameters](../causal_trained_models_v1/model_parameters.json); [simplified offline tree](../simplified_decision_tree/simplified_tree.json); [training evaluation code](../sisfall/train_causal_models.py), `metrics`; [sensitivity sweep code](../sisfall/causal_sensitivity_sweep.py), accounting and no-alarm handling.

## 10. Sensitivity / false-alarm tradeoff

The completed analysis swept **4,388 boundaries**: every unique positive score on the 4,387 saved validation candidates, plus `nextafter(max_score,+infinity)` for no alarms. Model weights and pipeline settings were held fixed. For false-alarm budgets, sensitivity was maximized first; exact sensitivity ties favored fewer ADL alarms/hour, then fewer emitted positives, then the higher threshold. No unstated “nearly equal” tolerance was introduced; one fall is 1/375, or about 0.2667 percentage points of sensitivity.

| Saved operating point | Exact threshold | Sensitivity | ADL alarms/hour | Event precision | Detected / missed |
|---|---:|---:|---:|---:|---:|
| BA reference; also best sensitivity at ≤5/hour | 0.8850929456953477 | 93.33% | 4.8893 | 93.09% | 350 / 25 |
| Maximum sensitivity at ≤10/hour | 0.8621999918050488 | 93.87% | 6.8450 | 91.67% | 352 / 23 |
| Maximum sensitivity at ≤20/hour | 0.7352880730771258 | 96.80% | 17.2754 | 84.03% | 363 / 12 |
| Highest threshold attaining ≥95% | 0.7742031648776018 | 95.20% | 12.7121 | 87.07% | 357 / 18 |
| Highest threshold attaining ≥97% | 0.6101532700558683 | 97.07% | 24.7723 | 79.13% | 364 / 11 |
| Highest threshold attaining the candidate ceiling | 0.025037875550837807 | 98.67% | 340.6193 | 20.80% | 370 / 5 |
| Literal lowest boundary; accept all candidates | 1.4766279401201908e-09 | 98.67% | 1072.0543 | 8.43% | 370 / 5 |

The sensitivity–false-alarm frontier consists of operating points for which no other threshold achieves at least as much sensitivity with fewer ADL alarms, or higher sensitivity with no more ADL alarms. It describes the best observed tradeoffs, not a smooth physical law or a guaranteed field rate. Lowering a threshold adds positive decisions; it may increase sensitivity, or merely add false alarms while sensitivity stays unchanged.

The earlier request for the **lowest** threshold meeting 95%, 97%, or maximum sensitivity was interpreted literally within the finite sweep: all three select the minimum score and accept every candidate. The more useful selective alternatives above are separately labeled **highest threshold attaining** a target. At maximum attainable sensitivity, the higher cutoff reduces ADL alarms from about 1072/hour to 341/hour without losing credited detections, but neither setting recovers a fall lacking a matched candidate.

**Sources:** [threshold_sweep.csv](../causal_sensitivity_analysis/threshold_sweep.csv), all aggregate and per-activity values at every boundary; [operating_points.csv](../causal_sensitivity_analysis/operating_points.csv); [sensitivity report](../causal_sensitivity_analysis/REPORT.md); [independent verification](../causal_sensitivity_analysis/independent_verification.json).

## 11. Remaining failure modes

### 11.1 Trigger failure versus classifier rejection

At the balanced logistic reference, the 25 missed validation falls consist of **5 without a matched trigger** and **20 with complete matched candidates rejected by the classifier**. At the approximately 95% sensitivity candidate, 18 misses remain: the same five trigger failures plus 13 classifier rejections. There are no matched-trigger/incomplete-window fall misses in these saved validation results.

At maximum classifier sensitivity, all five remaining failures are:

| Fall trial | Subject | Fall type | Failure category |
|---|---|---|---|
| `F01_SA05_R01.txt` | SA05 | F01 | No trigger matched to the impact proxy |
| `F04_SA05_R05.txt` | SA05 | F04 | No trigger matched to the impact proxy |
| `F06_SA05_R01.txt` | SA05 | F06 | No trigger matched to the impact proxy |
| `F11_SA05_R04.txt` | SA05 | F11 | No trigger matched to the impact proxy |
| `F08_SA19_R02.txt` | SA19 | F08 | No trigger matched to the impact proxy |

“No matched trigger” does not mean no trigger anywhere in the trial. The saved rows have other complete candidates, but none within the ±1 s matching tolerance. No logistic threshold can turn those unmatched candidates into a matched detection. At the all-candidate threshold there are zero matched complete candidates rejected by the classifier, so there are no rejected matched-candidate feature vectors to invent or report.

**Sources:** [maximum_sensitivity_misses.csv](../causal_sensitivity_analysis/maximum_sensitivity_misses.csv); [classifier_rejected_candidates.csv](../causal_sensitivity_analysis/classifier_rejected_candidates.csv), intentionally header-only; [causal validation trial results](../causal_event_study/validation_trial_results.csv), selected `g00/pre100_post200` rows; [causal model trial results](../causal_trained_models_v1/trial_results.csv).

### 11.2 Difficult fall types

F13 (forward fall while sitting) is the weakest reference fall type at **80%**. It rises to **96%** at the sensitivity-oriented cutoff. F01 remains at **88%** at both cutoffs; F03/F05/F06/F08/F11 remain at 92%. Not all residual errors are seated falls. Selected comparisons are:

| Fall type | Balanced reference sensitivity | Sensitivity-oriented candidate |
|---|---:|---:|
| F01 | 88% | 88% |
| F04 | 92% | 96% |
| F07 | 92% | 96% |
| F13: forward seated fall | 80% | 96% |
| F14: backward seated fall | 100% | 100% |
| F15: lateral seated fall | 92% | 96% |

Each validation fall type has 25 trials, so a four-percentage-point type-specific change corresponds to one trial. These small counts limit precision of activity-level conclusions.

### 11.3 False-positive ADL decisions

The logistic reference has no D03/D04 jogging alarms in this validation set, unlike the new shallow tree. Nonzero logistic ADL rates, plus the specifically monitored jogging activities, are shown below:

| ADL code | Reference alarms/hour | Sensitivity-candidate alarms/hour |
|---|---:|---:|
| D03: slow jogging | 0 | 0 |
| D04: fast jogging | 0 | 0 |
| D06: quick stairs | 5.7603 | 23.0411 |
| D08 | 0 | 7.5002 |
| D10 | 0 | 7.5001 |
| D11 | 7.5005 | 15.0011 |
| D13 | 12.0002 | 96.0016 |
| D14 | 22.5000 | 60.0000 |
| D18: stumble | 24.0012 | 60.0030 |
| D19: gentle jump | 84.0000 | 120.0000 |

All other ADL codes have zero alarms at these two saved points. D19, D18, D13, and D14 illustrate the price of lowering the classifier threshold. Activity rates divide by each activity's own recorded exposure; they must not be averaged unweighted to obtain the overall ADL rate, or extrapolated directly into daily alarm counts.

**Sources:** [selected_activity_metrics.csv](../causal_sensitivity_analysis/selected_activity_metrics.csv), `current_BA_reference` and `highest_threshold_sensitivity_ge_0.950000`; [activity_metrics.csv](../causal_trained_models_v1/activity_metrics.csv); [SisFall activity descriptions](../../data/Readme.txt). These tables report observed failures; they do not establish a physical causal explanation for each error.

## 12. MCU deployment implications

### 12.1 What must run on an nRF5340

An implementation needs the 200 Hz accelerometer acquisition and scaling; two causal biquads per axis; the three-component raw-driven gravity EMA; magnitude and magnitude-jerk trigger state; a circular buffer and pending-event state machine; the three feature aggregates; the fixed logistic calculation; and one final comparison. Gyroscope acquisition is not required by this model.

With a 1.5 s trigger refractory and a 1.0 s decision delay, a single selected window design has at most one pending post-trigger event. Sampling/filter/gravity/trigger updates continue while it is pending. No full-trial storage or backward filtering is algorithmically necessary, although the Python diagnostic archives full derived streams for analysis.

The algorithm is causally implementable; the current deliverable is a desktop reference, not nRF5340 firmware. The saved tests establish chronological filter equivalence to the desktop SOS reference, prefix invariance, window ordering/delay, feature consistency, and matching/accounting checks. Saved parameter verification reproduces training/validation predictions, but does not exercise a float32 MCU implementation.

### 12.2 Storage estimates

The following are data-size estimates, excluding stack, RTOS, sensor/DMA buffers, Python overhead, alignment beyond the explicitly stated scalar sizes, and library scratch memory:

| Item | State/parameters | Float32 size | Float64 size where applicable |
|---|---|---:|---:|
| Reference-compatible scalar ring | 301 × 3 values | 3612 bytes | 7224 bytes |
| Acceleration SOS state | 2 sections × 2 states × 3 axes | 48 bytes | 96 bytes |
| Gravity EMA | 3 values | 12 bytes | 24 bytes |
| Previous magnitude for trigger | 1 value | 4 bytes | 8 bytes |
| Subtotal of these persistent numeric arrays | 919 values | **3676 bytes** | **7352 bytes** |
| Three feature outputs | 3 values, plus temporary aggregation state | 12 bytes | 24 bytes |
| Standardized logistic parameters including one threshold | 3 means + 3 scales + 3 weights + intercept + threshold | 44 bytes | 88 bytes |
| Fused model with separate logit threshold | 3 weights + intercept + threshold | 20 bytes | 40 bytes |
| Fused model with threshold folded into intercept | 3 weights + decision intercept | 16 bytes | 32 bytes |

Add a previous-high flag, ring index/sample counters, last accepted trigger index, pending-event deadline, temporary vectors, and any window accumulators. Model/filter constants may reside in flash. Two normalized biquads require ten coefficients if `a0=1` is omitted (40 bytes float32), or twelve if the original SOS rows are retained (48 bytes); coefficients are shared across axes. Gravity alpha and trigger constants also require storage. No whole-program RAM or flash benchmark has been measured.

The ring and filter-state sizes come directly from the implemented array shapes. The older [MCU audit](../robust_classifiers/MCU.md) describes different, larger feature/model sets: its 21-feature model size and one-second window budget must **not** be copied onto this three-feature, 300-sample system.

### 12.3 Arithmetic estimates, not measured cycles

Count additions/subtractions as A, multiplications as M, square roots as S, and division/reciprocal as D. Exclude comparisons, absolute values, addressing, loads/stores, and integer state updates. The following is a straightforward scalar arithmetic reading of the current equations; compiler fusion and library reduction order may differ.

| Continuous work per sample | A | M | S | D |
|---|---:|---:|---:|---:|
| Raw XYZ count conversion | 0 | 3 | 0 | 0 |
| Two biquads on three axes | 24 | 30 | 0 | 0 |
| Gravity recurrence as currently written | 3 | 6 | 0 | 0 |
| Shared magnitude, normalized gravity, projection and perpendicular magnitude | 11 | 15 | 3 | 3 |
| Magnitude-jerk trigger after first sample | 1 | 1 | 0 | 0 |
| **Total continuous arithmetic/sample** | **39** | **55** | **3** | **3** |

The three divisions are the literal component-wise gravity normalization. One reciprocal plus three multiplications is an algebraic alternative, not the exact operation ordering currently documented. The existing orientation audit's normalization/EMA cost conventions differ because it budgets that reciprocal and `G += alpha*(r-G)`; do not add that shared cost again to this table.

Over an arbitrary 300-sample interval after startup, the continuous work corresponds to approximately **11,700 A, 16,500 M, 900 S, and 900 D**, whether or not a candidate is emitted. It is a rate budget, not an extra per-event charge: shared preprocessing runs once per physical sample.

At candidate completion, scanning the already-derived 300-row ring for the two maxima needs 598 comparisons. A direct scalar jerk aggregation uses 299 differences, 299 multiplications by 200, 298 sum additions (initializing from the first term), and one division by 299: **597 A + 299 M + 1 D**, plus absolute values. Accumulation order and algebraic simplification can change floating-point rounding.

Fused binary logistic inference then needs **3 multiplications, approximately 3 additions, and one comparison**. No exp, sqrt, or division is required at this final stage if comparing the logit directly. Producing the sigmoid score instead adds an exponential and division. Gravity normalization and acceleration magnitudes, rather than the three-weight classifier, are the expensive floating-point parts.

The arithmetic above is derived from [causal_events.py](../sisfall/causal_events.py) and the equations in Section 7, consistent with the biquad state/cost guidance in [ORIENTATION_METHODS.md](../sisfall/ORIENTATION_METHODS.md). It is not a cycle, energy, execution-time, throughput, or memory-usage benchmark on the nRF5340. No such measured numbers are available in the cited artifacts.

### 12.4 Verification status

The saved causal replay regression record reports **five tests, zero failures, zero errors**. The saved causal model export checks report **zero prediction mismatches** on 12,253 training and 4,387 validation candidates for the desktop parameter reconstructions, together with training-only scaler and event-accounting checks. The threshold analysis also records independent selected-point counting and budget-optimality checks. These are existing results, not tests rerun for this report.

**Sources:** [causal regression results](../causal_event_study/regression_tests.json); [export verification](../causal_trained_models_v1/export_verification.json); [sweep verification](../causal_sensitivity_analysis/verification.json); [independent sweep verification](../causal_sensitivity_analysis/independent_verification.json).

## 13. Known limitations

1. **Waist-to-wrist domain gap.** All reported model results concern waist-mounted SisFall data, not a wrist-worn device. Constant rotation invariance does not validate arm movement, time-varying rotation, mounting differences, or wrist impacts.
2. **Simulated falls and constrained activities.** Scripted trials do not reproduce the prevalence, diversity, environment, or behavior of naturally occurring falls in long-term use.
3. **Timing proxy.** Raw-magnitude argmax is an evaluation/training proxy rather than annotated fall onset or impact. A trigger outside its ±1 s neighborhood is considered unmatched even if it captures meaningful motion. Conversely, a matched trigger need not identify the true physiological onset.
4. **Weak candidate labels.** Unmatched candidates from fall trials are treated as negatives. This convention may mislabel parts of a longer fall process; it was retained, not resolved, in causal training.
5. **Reused development validation.** The validation subjects have been used for trigger/window selection, model/hyperparameter selection, and operating-point analysis. Their metrics are development results and can be optimistic. No new generalization estimate is created by this report.
6. **Different experiment generations.** The earlier fixed-test causal-transfer diagnostic evaluated the frozen offline tree. Those outcomes must not be presented as test results for the later causal-trained logistic model. Its saved training and sensitivity analyses did not use test outcomes; this report does not evaluate it on test.
7. **Numeric portability.** Float64 desktop state, intermediate float32 quantization, final float64 features, coefficient fusion, logit comparison, compiler reassociation, and MCU float32 arithmetic can produce different boundary decisions. Desktop parameter parity is not firmware equivalence.
8. **Gravity estimate limitations.** A 0.5 Hz accelerometer EMA can lag rapid rotation and absorb dynamic acceleration. The normalization floor protects division but does not guarantee a valid gravity direction.
9. **Candidate ceiling and boundaries.** Five validation falls have no localized trigger. Lower classifier thresholds cannot recover them. Startup and recording-end candidates can be discarded for insufficient history/future arrivals; live boot/shutdown behavior must be consistent.
10. **Exposure and calibration.** SisFall false alarms/hour reflect short, selected activity recordings, not expected daily-life alarms/hour. Trial BA does not fully penalize repeated alarms within a trial. Event precision depends on candidate and fall prevalence, and the balanced-weight logistic score is not a calibrated field probability.
11. **Hardware unverified.** No actual nRF5340 timing, RAM high-water mark, energy, sensor synchronization, numeric-equivalence, or wrist-domain validation is established by the saved artifacts. The report's resource counts are estimates, not measurements.

**Sources:** [orientation limitations](../sisfall/ORIENTATION_METHODS.md); [causal training report](../causal_trained_models_v1/REPORT.md); [sensitivity report](../causal_sensitivity_analysis/REPORT.md); the exact implementation and verification sources cited above. No new features or model families are proposed here.

## 14. Frozen versus unresolved design choices

| Frozen for MCU v1 | Still unresolved / requires hardware or wrist validation |
|---|---|
| Algorithmic reference: ADXL345 XYZ-equivalent acceleration in g at 200 Hz; counts/256 for SisFall | Actual device sensor configuration, calibration, axis mapping and acquisition timing must match the intended physical units |
| Causal fourth-order 5 Hz Butterworth; saved two-section coefficients and first-sample initialization | Quantized coefficient/state behavior, boot transients and implementation-specific numeric equivalence |
| Raw-driven gravity EMA, alpha `0.01558523664828626`, denominator floor `1e-8` g | Gravity quality under real wrist rotation and dynamic acceleration |
| Trigger magnitude ≥1.1 g OR absolute magnitude jerk ≥5 g/s; combined rising edge; 300-sample refractory | Five no-matched-trigger misses remain; no trigger change is authorized or proposed by this report |
| `pre100_post200`: 300-sample half-open window; decision at trigger+200; no boundary padding | Live sensor buffering, scheduling, lost samples, and restart/shutdown handling on hardware |
| Exactly `[ga_C2, jerk_abs_mean, ga_parallel_peak]` with the equations in Section 6 | Float32 incremental feature equivalence; no feature expansion is proposed |
| Saved scaler, three logistic coefficients, intercept, and `C=10` model | Calibrated field probabilities, wrist-domain validity, and independent generalization remain unestablished |
| BA reference threshold `0.8850929456953477`; sensitivity candidate `0.7742031648776018` documented separately | **Final deployment threshold remains unresolved.** Neither this report nor the sweep replaces the reference |
| Fixed evaluation definitions, subject partitions and saved validation outcomes | Annotated timing, prospective real-world exposure, and actual wrist data needed to establish field behavior |
| Causal desktop algorithm, saved regression/export checks and source artifacts preserved | Actual nRF5340 firmware correctness, float32 decision parity, runtime, RAM, power, and end-to-end latency not yet measured |
