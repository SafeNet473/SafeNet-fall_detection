# Orientation experiment definitions and deployment limits

## Preserved waist reference

Use the saved `processed/baseline_v2` windows, labels, subject partitions and five-file exclusion. Copy the previous paper metrics and thresholds without modification. C2/C3/C8/C9/C13 are imported from `paper_features.py` unchanged; [their definitions and ambiguities](FEATURE_EQUATIONS.md) remain applicable. Existing acceleration is the same ADXL345 data in g, with full-trial fourth-order 5 Hz Butterworth SOS forward/backward filtering. No window relocalization follows rotations.

New feature thresholds use **unrotated validation only**, maximizing balanced accuracy with the same lowest-threshold tie policy and `score >= threshold` decision. All new features have a predefined higher-means-fall direction. No direction selection, rotation augmentation tuning, test threshold tuning, feature fusion, or learned classifier is used. Original validation and test performance are reported for every feature.

## Acceleration-magnitude features

N=200, fs=200 Hz; `m[j]=sqrt(ax[j]^2+ay[j]^2+az[j]^2)`, j=0,...,199, from existing filtered acceleration. All standard deviations/variances use sample denominator L−1 over their L samples.

| Name | Equation | Units |
|---|---|---|
| mag_mean | `sum(m)/N` | g |
| mag_std | `sqrt(sum((m-mean(m))^2)/(N-1))` | g |
| mag_rms | `sqrt(sum(m^2)/N)` | g |
| mag_peak | `max(m)` | g |
| mag_ptp | `max(m)-min(m)` | g |
| jerk_abs_mean | `mean(abs(d))`, `d[j]=fs*(m[j+1]-m[j])`, j=0,...,198 | g/s |
| jerk_std | `std(d, ddof=1)` | g/s |
| jerk_rms | `sqrt(mean(d^2))` | g/s |
| jerk_abs_peak | `max(abs(d))` | g/s |
| pre_variance | `var(m[0:100], ddof=1)` | g² |
| post_variance | `var(m[100:200], ddof=1)` | g² |

Magnitude jerk is the **derivative of magnitude**, not the magnitude of the 3D derivative. This distinction is intentional. Pre/post halves are **window-relative proxies**, applied to ADL and fall windows identically. They are not event annotations: boundary-shifted fall windows may not place the peak at index 100, and ADL windows have no event. Post variance is available only after the second half has arrived; it implies latency, not access to unavailable future samples at that decision time. No label is used in these equations.

## Gyroscope magnitude

The raw source has ITG3200 columns 4–6. Convert counts using `4000/65536` deg/s per count from the stated ±2000 deg/s and 16-bit range. No undocumented bias calibration or alternate datasheet sensitivity is substituted. Apply the existing acceleration filter design/phase convention to each full gyro trial (fourth-order 5 Hz Butterworth, offline `sosfiltfilt`) and use the exact same saved sample bounds. Raw data remains untouched; source hashes must match the ingestion manifest.

`w[j]=sqrt(gx[j]^2+gy[j]^2+gz[j]^2)`; gyro_mean/std/rms/peak/ptp use the same five equations above with w replacing m (units deg/s). These are new single-feature experiments, not changes to any paper baseline. The gyro source and initial gravity state require only validation/test trials; no training fit is needed.

## Causal gravity estimation and aligned features

Use **raw**, unfiltered acceleration in g for gravity state. Set `alpha=1-exp(-2*pi*0.5/200)` (0.5 Hz exponential smoothing, time constant about 0.318 s). At trial start initialize `G[-1]=a_raw[0]`; thereafter `G[t]=G[t-1]+alpha*(a_raw[t]-G[t-1])`. This is a first-order low-frequency component, not a validated attitude estimator. The initial sample can contain motion, and there is no warm-up exclusion because labels/windows must stay fixed.

For a window beginning at s, carry the full-trial state G[s−1], then update through its 200 raw samples. At s=0 use the first raw sample initialization. No current or future validation/test labels enter gravity estimation. EMA state is computed chronologically from each independent trial, never across trials or subjects. The raw count scale is exactly representable in the saved float32 windows, and raw-window parity is checked.

At time t, `u[t]=G[t]/max(norm(G[t]),1e-8 g)`; `v[t]=dot(a_filtered[t],u[t])`; `b[t]=a_filtered[t]-v[t]*u[t]`; `h[t]=norm(b[t])`. For a valid unit u, v is signed parallel acceleration and h is perpendicular magnitude. If gravity norm is tiny, the denominator floor prevents division by zero; the vector is no longer a reliable unit gravity direction. The source count of such cases is saved in provenance. No fixed-axis fallback is used (it would break rotation equivariance).

| Name | Equation over the same N=200 samples | Units |
|---|---|---|
| ga_C2 | `max(h)` | g |
| ga_C8 | `sqrt(var(bx)+var(by)+var(bz))`, sample variance | g |
| ga_C13 | `sum((h[j]+h[j+1])/2, j=0..198)` | g·sample |
| ga_parallel_peak | `max(abs(v))` | g |
| ga_parallel_std | `std(v, ddof=1)` | g |
| ga_parallel_ptp | `max(v)-min(v)` | g |

When gravity is constant and aligned with y, ga_C2/C8/C13 reduce to their x/z equivalents. ga_C8 is the covariance-trace magnitude of the **perpendicular vector**, not std(h). With changing estimated gravity, it is an explicit new definition that remains equivariant under a constant coordinate rotation. It is not a choice of arbitrary perpendicular basis at each sample.

Gravity estimation itself is causal and tested by changing later raw samples without affecting earlier state. However, the preserved filtered a and gyro remain noncausal, and fall windows remain selected using the full raw recording's largest peak. Consequently **the overall detector is not a causal MCU implementation**. A future deployment study must replace offline filtering and event selection explicitly and re-evaluate; this experiment does not silently change either.

## Synthetic rotations and interpretation

Default ten replicates, seeds 473–482. Draw an independent unit quaternion from four normalized standard Gaussians for each window, giving uniform SO(3) rotations. Apply one orthogonal matrix R to all 200 acceleration and gyro vectors and the associated gravity-state trajectory. Rotating EMA history is exactly equivalent to rotating raw history and its initial state before running the same EMA; the equivalence is tested. Rotating only a and leaving gravity in the old frame would be an invalid aligned-feature test.

Additional controls: 90° around sensor y preserves the x/z plane; 90° around x mixes y/z. These paired controls distinguish coordinate dependence from random sampling variability. Every scenario uses the same original examples, partitions and thresholds. Each rotation realization is shared by all features. Overlapping windows receive independent rotations; therefore this is a window-level coordinate perturbation, not one physically continuous orientation trajectory.

C9 is already invariant under a constant 3D rotation because the trace of a covariance matrix is invariant. The norm of axis-wise peak-to-peak ranges (C3) is generally not. All magnitude and aligned features are invariant under these rotations when gravity history is transformed consistently. Raw numerical deviations are saved and checked with rtol=atol=1e-10. After this check, invariant scores reuse their original values to prevent floating-roundoff threshold ties from creating spurious classification changes. The old baseline retains its exact float32-derived scores, with its float64 magnitude used for the numerical rotation check.

Metrics include sensitivity, specificity, balanced accuracy, precision, F1, trapezoidal PR-AUC and separate average precision. The ten-replicate min/max/std describe orientation draws; they are not confidence intervals over subjects. All 34 activity error rates are saved, with a separate requested-code summary. Outputs preserve denominators; ADL overlapping windows are not independent alarm events.

**Wrist limitation:** arbitrary fixed rotations approximate unknown mounting orientation only. They do not reproduce sensor translation from waist to wrist, arm motion, wrist impacts, time-varying frame rotation, gyro frame-rate effects, or accelerometer/gyro misalignment. A 0.5 Hz accelerometer-only gravity estimate can lag wrist rotation and mistake dynamic acceleration for gravity. Rotation invariance is necessary for some mounting scenarios but is not evidence of wrist fall-detection accuracy.

## MCU operation estimates (float32 streaming implementations)

Estimates below count scalar adds/subtracts A, multiplications M, square roots S per 200-sample decision window. Comparisons, abs, branches, division/reciprocal and startup are called out separately. Reciprocal constant scaling counts as one multiply. These are arithmetic estimates, not measured cycles, energy, latency or compiler benchmarks. Use running sums/Welford on an MCU, not the vectorized Python implementation. Cost formulas use N=200, J=199; float32 occupies four bytes.

Shared magnitude front-end B = `2N A + 3N M + N S` = **400 A, 600 M, 200 S**. Gyro magnitude has the same cost. Reuse it across scalar statistics if measuring several, although each experiment's decision uses only one feature. For a scalar stream of L samples: mean costs `(L-1) A + 1 M`; RMS adds `L M + (L-1) A + 1 M + 1 S`; sample variance from sum/sum-of-squares costs `(2L-1) A + (L+3) M`; std adds one S. The sum/squares formula can lose precision; Welford needs extra operations but avoids cancellation. Negative roundoff variances need clamping.

| Feature | A per window | M per window | S per window | Additional considerations |
|---|---:|---:|---:|---|
| baseline / mag_peak | 400 | 600 | 200 | N−1 comparisons |
| mag_mean | 599 | 601 | 200 | running sum |
| mag_std | 799 | 803 | 201 | running sum and sum squares |
| mag_rms | 599 | 801 | 201 | can optimize: sum XYZ squares directly, then one final sqrt |
| mag_ptp | 401 | 600 | 200 | 2(N−1) comparisons |
| jerk_abs_mean | 797 | 800 | 200 | B + J differences/scales + mean; J abs |
| jerk_std | 996 | 1001 | 201 | signed jerk; B + J differences/scales + std over J |
| jerk_rms | 797 | 999 | 201 | B + J differences/scales + RMS over J |
| jerk_abs_peak | 599 | 799 | 200 | J abs and J−1 comparisons |
| pre_variance | 399 | 403 | 100 | magnitude of first 100 samples only |
| post_variance | 399 | 403 | 100 | magnitude of last 100 samples only |
| gyro_mean | 599 | 601 | 200 | gyro magnitude front-end |
| gyro_std | 799 | 803 | 201 | gyro magnitude front-end |
| gyro_rms | 599 | 801 | 201 | direct XYZ-square optimization possible |
| gyro_peak | 400 | 600 | 200 | N−1 comparisons |
| gyro_ptp | 401 | 600 | 200 | 2(N−1) comparisons |
| C2 | 200 | 400 | 200 | x/z norm, then max |
| C3 | 5 | 3 | 1 | 6(N−1) min/max comparisons; three axis ranges |
| C8 | 799 | 406 | 1 | two axis sum/squares streams |
| C9 | 1199 | 609 | 1 | three axis sum/squares streams |
| C13 | 597 | 401 | 200 | x/z norms + trapezoids |
| ga_C2 | 3000 | 3600 | 400 | aligned front-end + max; N reciprocals |
| ga_C8 | 3799 | 3609 | 201 | residual-vector moments; no per-sample perpendicular sqrt; N reciprocals |
| ga_C13 | 3397 | 3601 | 400 | aligned front-end + trapezoids; N reciprocals |
| ga_parallel_peak | 2000 | 2400 | 200 | parallel front-end; N reciprocals, abs/max |
| ga_parallel_std | 2399 | 2603 | 201 | parallel front-end + std; N reciprocals |
| ga_parallel_ptp | 2001 | 2400 | 200 | parallel front-end + range; N reciprocals |

Aligned front-end including gravity EMA and perpendicular norm costs 15 A + 18 M + 2 S + one reciprocal per sample. Parallel-only costs 10 A + 12 M + 1 S + one reciprocal. Residual-vector-only costs 13 A + 15 M + 1 S + one reciprocal. ga_C8 adds three scalar variance streams and their sum/sqrt. Continuously update the gravity EMA even between decisions; its state is three floats (12 bytes). An rsqrt instruction may fuse normalization operations but is hardware-dependent.

### Storage and incremental suitability

For every scalar mean/std/RMS/variance feature: approximately 2–4 accumulator floats plus counters, **8–16 bytes**, excluding filter state. Peak needs one float, peak-to-peak two (4–8 bytes). Jerk adds the previous magnitude (4 bytes). Trapezoidal area requires previous magnitude and sum (8 bytes). C3 uses six extrema (24 bytes); C8 four moment accumulators (16 bytes); C9 and ga_C8 six (24 bytes). Aligned features add G (12 bytes) and transient projection vectors; no gravity-window storage is needed. All these are feasible in a single forward pass over a nonoverlapping window.

To reproduce stride-100 overlapping windows with N=200, maintain two staggered sets of accumulators and reset each every 200 samples: roughly double the listed window-state memory and reduction work on shared incoming samples. Each receives N updates per output window, matching the table. Jerk/trapezoid state must reset at each window boundary to exclude cross-window differences. Pre/post variance can use consecutive 100-sample summaries with constant storage. No 200-sample ring buffer is needed for these fixed known start times. Arbitrary trailing windows could instead use an N-float ring (800 bytes per scalar stream), and extrema may need a monotonic queue or recomputation.

Storing raw XYZ windows costs 200×3×4 = **2,400 bytes** per sensor; both accelerometer and gyro cost 4,800 bytes. This is unnecessary for incremental statistics but may be needed for a different event trigger with retrospective extraction. The existing event-center protocol cannot be reproduced as a causal trigger merely by using small accumulators.

Filtering costs are separate. A causal fourth-order filter as two direct-form-II-transposed biquads costs roughly **10 M + 8 A per axis per sample**, with four states per axis (**48 bytes per tri-axis sensor**) plus coefficients. Raw-count scaling adds three multiplies per sensor sample. The preserved full-trial forward/backward offline filter is approximately two passes (double arithmetic plus edge setup) and requires whole-trial/reverse access; it cannot be implemented as this single-pass causal filter without changing results. Magnitude RMS/peak squared thresholds can remove many sqrt operations, but must be implemented and validated separately; current experiments evaluate the documented features directly.
