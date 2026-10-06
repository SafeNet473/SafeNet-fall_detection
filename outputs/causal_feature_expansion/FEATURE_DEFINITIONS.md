# Predeclared causal accelerometer feature progression

The fixed baseline is `[ga_C2, jerk_abs_mean, ga_parallel_peak]`. The 5-feature set adds `late_mag_variance` and `late_minus_pre_jerk`. Only if that step passes the predeclared material-improvement gate is the 7-feature set fitted, adding `perp_active_fraction` and `late_parallel_std`. There is no combinatorial subset search. All four prospective features are extracted in one replay to avoid repeating source processing; extraction does not imply fitting the gated 7-feature model.

## Shared signal and timing definitions

Raw acceleration in g drives the unchanged 0.5 Hz gravity EMA. The unchanged fourth-order 5 Hz causal Butterworth produces acceleration `a`; retain the existing float32 acceleration interface before float64 feature arithmetic. Let `u=G/max(norm(G),1e-8)`, `p=a·u`, `b=a-pu`, `m=norm(a)`, and `h=norm(b)`. Here **p is signed**; its absolute value is used only for the baseline peak.

For a trigger at k, the unchanged 300-sample window is `[k-100,k+200)`, available for decision at k+200. Local indices 0:100 are the pre-trigger half-second. Indices 200:300 are the **late** half-second `[k+100,k+200)`, from +0.5 to +0.995 seconds relative to the trigger. Late means relative to the trigger, not an annotated fall impact. Every feature is available at the existing decision time; no new waiting interval is introduced.

## Exact added features

| Feature | Definition | Interpretation | Units |
|---|---|---|---|
| `late_mag_variance` | `(1/100) sum_(i=200..299) (m_i - mean(m_200..299))^2` | Residual late-window motion; smaller values can indicate settling | g² |
| `late_minus_pre_jerk` | `(200/99) [sum_(i=201..299) abs(m_i-m_(i-1)) - sum_(i=1..99) abs(m_i-m_(i-1))]` | Change in motion rate from pre-trigger to late window; negative means less late motion | g/s |
| `perp_active_fraction` | `(1/300) sum_(i=0..299) 1[h_i > 0.5 g]` | Duration of sustained gravity-perpendicular activity rather than its peak | Dimensionless, fraction 0..1 |
| `late_parallel_std` | `sqrt((1/100) sum_(i=200..299) (p_i - mean(p_200..299))^2)` | Late gravity-parallel variability, retaining directional acceleration information lost by magnitude alone | g |

Both variance and standard deviation use population normalization (`ddof=0`). Each jerk segment has **99 differences**; differences crossing segment boundaries are excluded. The 0.5 g activity cutoff and late-segment length were fixed before fitting and are not tuned. Strict `>` is used for the activity indicator.

The first two features are magnitude-based and require no gravity estimate. The latter two use the existing gravity projection. All are robust to a constant coordinate rotation when gravity history is rotated consistently, up to floating-point rounding. They do not establish robustness to real wrist motion or time-varying mounting orientation.

## MCU state and approximate arithmetic

These are scalar operation estimates, not measured MCU cycles, energy or RAM high-water marks. They assume m/h/p are already available from the baseline front end, fixed reciprocal constants, and add/subtract counted together. Baseline filtering/normalization costs are not charged again.

| Feature | Incremental state | Rough arithmetic per affected sample | Window finalization / total incremental arithmetic |
|---|---|---|---|
| Late magnitude variance | Sum and sum of squares, 2 floats (8 bytes float32), plus shared phase count | On late 100 samples: 2 adds + 1 multiply | Moment-form finalization: 1 subtract + 3 multiplies; total 201 adds/subtracts + 103 multiplies |
| Late-minus-pre jerk | Two segment sums and previous magnitude: 3 floats (12 bytes), previous magnitude can be shared | For each of 198 within-segment differences: 1 subtract, abs, 1 accumulate | Subtract sums and multiply by 200/99: total 397 adds/subtracts + 1 multiply. Reuse of the already-computed magnitude jerk can reduce this work |
| Perpendicular active fraction | One integer count, typically 4 bytes | One comparison per window sample; increment on true | 300 comparisons, up to 300 integer increments, 1 multiplication by 1/300 |
| Late signed-parallel std | Sum and sum of squares, 2 floats (8 bytes), plus shared phase count | On late 100 samples: 2 adds + 1 multiply | Same moment arithmetic as variance plus one sqrt: 201 adds/subtracts + 103 multiplies + 1 sqrt |

The two added 5-feature statistics need approximately 20 bytes of additional accumulator state if previous magnitude is not shared, excluding counters. The 7-feature set adds approximately 12 bytes more. They require no new raw sensor or gyro stream. These are forward aggregation state sizes, not a claim that pre-trigger history can be discarded: the existing circular history still supplies the first 100 samples after a trigger is known.

The replay reference retains a 301×4 float64 ring (m,h,abs(p),signed p), 9632 bytes, to extract every predeclared feature at its causal availability time. For a 5-feature implementation the existing 301×3 scalar ring suffices (3612 bytes in float32). A direct 7-feature port keeping signed p as an additional ring channel uses 4816 bytes float32; alternatively the late-only signed-p moments can be accumulated after the trigger with two scalars without storing a full signed-p history. This alternative changes implementation, not the feature equation, and is not firmware-verified here.

Moment-form variance is algebraically equivalent to the centered reference calculation but can suffer cancellation and may round differently. The actual reference uses NumPy centered population variance/std; float32 MCU equivalence is not claimed. The added-feature tests check constant signals, segment boundaries, independent moment equations, constant rotations, and causal prefix behavior.

With training-scaler fusion, an F-feature logistic binary decision uses F multiplications, about F additions, and a logit-threshold comparison, without runtime sigmoid. Fused weights plus intercept and a separate threshold require `(F+2)*4` bytes float32: 20/28/36 bytes for F=3/5/7. Folding a fixed threshold into the intercept saves 4 bytes but does not select a deployment threshold here. Stored standardized parameters (means/scales/weights/intercept/threshold) need `(3F+2)*4` bytes: 44/68/92 bytes. These are arithmetic/storage derivations, not deployed parameter quantizations.
