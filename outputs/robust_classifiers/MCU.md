# MCU-oriented complexity estimates

These are arithmetic/storage estimates for an incremental implementation, not measured MCU cycles or energy. Existing Python evaluation uses float64 features. Feature equations and individual costs remain in [ORIENTATION_METHODS.md](../sisfall/ORIENTATION_METHODS.md). No firmware implementation or float32 accuracy claim is made here.

## Models

**Logistic regression:** 21 features. Fuse the training scaler into exported raw-feature weights: `w_raw = w / scale`, `b_raw = b - dot(w_raw, mean)`. Classify `dot(w_raw,x) + b_raw >= log(tau/(1-tau))`. The final comparison offset can be folded into the intercept. This costs 21 multiplications, roughly 21 additions and one comparison; no sigmoid/exp, sqrt or division is required for a binary decision. Outputting the sigmoid score adds exp and division. Store 21 weights plus fused decision intercept: 22 float32 values = **88 bytes**, preferably flash. `model_parameters.json` retains full-precision coefficients and threshold for conversion.

**Decision tree:** actual depth **3**, **15 nodes**, **8 leaves**, and **5 distinct used features**: jerk_abs_mean, jerk_rms, jerk_abs_peak, ga_C2, ga_parallel_peak. Inference requires at most 3 split comparisons plus a final leaf-score threshold comparison, no multiplications, divisions or roots. Precompute leaf decisions to remove the final comparison. A simple all-node table with int16 children/feature and float32 split/leaf score takes about 16 bytes/node after alignment: **240 bytes**, preferably flash. Compiler packing and pruning can reduce this. A node index requires only a few runtime bytes.

Only tree-used features are required at inference; exporting this subset follows the fitted training tree and does not use test feature selection. The Python evaluation retained the full 21-column matrix. Logistic regression retains every feature regardless of small or correlated coefficients.

This fitted tree uses only acceleration-derived features: it does not require gyroscope input at inference, even though gyroscope candidates were available during training. Its incremental implementation can omit gyro filtering and magnitude calculation; a causal accelerometer-only SOS filter would require 48 bytes of state rather than the 96-byte six-axis budget below.

## Window and feature extraction

200 samples at 200 Hz, one-second windows, unchanged 100-sample ADL stride and event-centered fall windows. Jerk has 199 differences; pre/post variance uses two 100-sample halves. Gravity context precedes the window, so its state must be continuously carried from past raw samples.

`mcu_complexity.csv` gives conservative sums of independently computing each required feature. These deliberately double-count shared magnitude/EMA work and are upper estimates, not a recommended implementation. Expensive per-sample operations are XYZ magnitude roots and gravity normalization reciprocal/sqrt.

**Shared full-21 implementation:** computing accelerometer magnitude, gyro magnitude and gravity/perpendicular projections once per sample, then sharing moment accumulators, costs approximately **8,000 additions, 7,000 multiplications, 810 square roots and 200 reciprocals per 200-sample feature window**, excluding filtering, sensor scaling and model inference. This is an illustrative budget, not an optimized lower bound. Roots can be avoided for some squared-norm statistics, but gyro/acceleration magnitude means and jerk still require magnitudes; gravity normalization remains costly. Constant reciprocals (1/N, 1/(N−1), fs and alpha) compile to multiplications.

A future causal fourth-order filter with two biquads per axis would add about 60 multiplies + 48 adds per incoming sample for six sensor axes, with 96 bytes of filter state. Six count-to-unit conversions add six multiplies/sample. **This causal filter does not reproduce current forward/backward filtering.** Current offline filtering needs reverse access/full-trial storage and cannot be made causal without changing the evaluation.

## Approximate RAM

For all 21 features, a practical active-window state comprises about 29 float accumulators/extrema plus counters: roughly **132 bytes**. Two staggered accumulator sets for stride 100 use about **264 bytes**; gravity state adds 12 bytes; a prospective causal six-axis SOS filter adds 96 bytes; feature output is 84 bytes; allow ~48 bytes for transient vectors plus a few bytes for inference. Thus approximately **0.5–0.7 KiB of data RAM** can suffice for an optimized incremental full-feature implementation, excluding stack, DMA/sensor buffers, library workspace and model parameters kept in flash. If weights/tree tables reside in RAM, add the model storage above.

The fitted tree can omit all unused feature streams and use only 20 bytes for its feature output; the full-feature **0.5–0.7 KiB** estimate is a conservative shared-buffer budget for either model, not a measured tree-specific minimum.

A straightforward implementation storing a whole acceleration and gyroscope window uses **4,800 bytes** for the two 200×3 float32 arrays, plus states and feature vector. Storing gravity per sample adds another 2,400 bytes, but is unnecessary in a streaming implementation. Overlapping windows can instead use staggered accumulators for sums, moments, extrema, jerk and trapezoid area. Reset jerk/trapezoid previous-sample state at each window start. Pre/post half summaries also update incrementally.

Variance can use Welford updates for numerical stability at a somewhat higher arithmetic cost than sum/sum-of-squares estimates. Model parameters should be stored in flash. Scaler fusion was checked against sklearn in float64; exported tree evaluation was checked exactly, including sklearn float32 input casting. Quantized coefficients, fixed-point features and embedded numerical drift are unverified.

## Causality and wrist limitations

The gravity EMA itself is causal. The **preserved end-to-end protocol is not**: full-trial zero-phase filtering and finding the full-trial largest peak for fall localization use future information. A realistic MCU trigger needs a rolling pre-event buffer and delayed post-event statistics; the current whole-trial peak rule has no causal equivalent without redefining the detector. Constant coordinate rotations also do not model moving a sensor from the waist to the wrist or time-varying wrist orientation. These costs guide the next engineering step, not deployment readiness.
