# Four predefined physical additions

These are exploratory MCU v2 features, not part of frozen MCU v1. Only the six requested sets are evaluated. No feature timing, threshold, epsilon, or variance convention is optimized on validation.

## Shared timing and signals

For trigger k, the window is `[k-100,k+200)`, with a decision at k+200. Its 300 local indices are 0..299. Pre-trigger is indices0:100, or `[k-100,k)`. Late post-trigger is indices200:300, or `[k+100,k+200)`. The latter covers +0.5 through +0.995 seconds relative to the trigger; it is not necessarily post-impact.

Use the unchanged filtered acceleration `a` and raw-driven gravity EMA `G`. The saved interface casts filtered acceleration to float32 then back to float64. Set `u=G/max(norm(G),1e-8 g)`, signed `p=a·u`, `b=a-pu`, `h=norm(b)`, `m=norm(a)`. Gravity history remains continuous, and all windows use only samples that have arrived by k+200.

## Definitions

| Feature | Exact equation and range | Units | Physical interpretation |
|---|---|---|---|
| `ga_C8` | `sqrt(sum_(axis=x,y,z) sum_(i=0..299) (b_axis[i]-mean(b_axis))² / 299)` | g | Overall perpendicular-vector variability, distinct from peak magnitude |
| `post_mag_variance` | `(1/100) sum_(i=200..299) (m[i]-mean(m[200:300]))²` | g² | Late motion / settling |
| `pre_post_energy_ratio` | `mean(h[200:300]²) / (mean(h[0:100]²) + 1e-6)` | dimensionless | Relative late perpendicular energy; values below roughly1 can indicate settling |
| `gravity_direction_change` | Let `Gpre=mean(G[0:100])`, `Gpost=mean(G[200:300])`. If both norms >=1e-8 g, return `acos(clamp(dot(Gpre/norm(Gpre),Gpost/norm(Gpost)),-1,1))`; otherwise return exactly0 | radians | Persistent change in estimated gravity direction, a proxy for posture/orientation change |

**Variance convention:** `ga_C8` retains the existing named feature's sample variance (`ddof=1`), as implemented in `outputs/sisfall/orientation_features.py`; its full-window denominator here is299. The user allowed retaining an existing convention. It is the square root of the trace of the residual-vector covariance, **not** standard deviation of scalar h. The new `post_mag_variance` uses population variance (`ddof=0`), matching the earlier late-settling statistic. No baseline feature definition is changed.

**Energy epsilon:** exactly **1e-6 g²**, added to the mean pre-energy. It is fixed before fitting and is not the gravity epsilon. Zero pre-energy produces a finite ratio; no clipping, logarithm, or learned transform is applied beyond training StandardScaler. Large ratios can occur for near-zero pre-motion.

**Gravity fallback:** if either mean-vector norm is strictly less than **1e-8 g**, return0 radians and increment a diagnostic count. The count is not a model feature. This deterministic fallback is neutral numerically but does not mean physical posture is unchanged. Norm equal to the floor is normalized normally. Mean raw gravity vectors are formed first, then normalized: averaging unit vectors instead would change the definition.

## Orientation and causality

All four features are invariant in real arithmetic to a **constant** common rotation of acceleration and the corresponding gravity history. Covariance trace, norms, energies and dot products are rotation invariant. This does not make them invariant to actual changing wrist posture or dynamics—those may be the physical information captured. Gravity estimation can lag or be contaminated by motion. Quantization causes small floating-point discrepancies.

All are computable incrementally once segment membership is known, with very small accumulator state. Pre-trigger statistics require the existing history buffer or equivalent rolling statistics; incremental does not mean that retrospective history can be ignored. No feature adds delay beyond the existing k+200 decision.

## MCU arithmetic and state estimates

Assume the existing front end supplies b, h, m and G; do not count its filter, normalization or projection work again. Counts below are real-arithmetic moment implementations, not measured MCU instructions or exact NumPy evaluation order. Add includes subtract; constants such as1/100 and1/299 can be precomputed. Counts exclude absolute values, comparisons unless stated, loads, indexing and loop counters.

| Feature | Extra state (float32) | Per-sample work | Aggregate per event, including finalization |
|---|---|---|---|
| `ga_C8` | Three vector sums plus one sum of squared norm:4 floats,16 bytes | All300: accumulate bx/by/bz (3 adds), h² and its sum (1 multiply+1 add) | About1203 adds,305 multiplies,1 sqrt,0 runtime divisions. Compute `sqrt((sum(h²)-norm(sum(b))²/300)/299)` mathematically |
| `post_mag_variance` | Sum and squared sum:2 floats,8 bytes | Last100:2 adds+1 multiply | About201 adds,103 multiplies,0 sqrt/div/acos |
| `pre_post_energy_ratio` | Two energy sums:2 floats,8 bytes | Pre100 and late100:1 square+1 accumulate | About201 adds,202 multiplies,1 division. The two mean scalings can instead cancel if epsilon is scaled by100 consistently |
| `gravity_direction_change` | Two vector sums:6 floats,24 bytes | Pre100 and late100:3 adds | About606 adds,15 multiplies,2 sqrt,6 component divisions,1 acos, plus2 clamp comparisons and2 norm-floor comparisons. Using two reciprocals+six multiplies is an algebraic alternative |

Combined extra accumulator state is approximately **56 bytes float32**, excluding shared counters and scratch; energy/norm work can be shared. Moment subtraction can suffer cancellation. The desktop reference uses centered NumPy variances, so a moment-based float32 implementation needs separate numerical verification; no such firmware is claimed here.

The diagnostic replay uses a 301×9 float64 ring containing `(m,h,abs(p),bx,by,bz,Gx,Gy,Gz)`, or21672 bytes, to make causal availability explicit. A direct float32 copy would use10836 bytes, **not** the minimal deployment state. Incremental accumulation can avoid storing full b/G histories after the trigger, and pre-trigger sums can be drawn from retained/rolling history. No actual firmware RAM layout was built in this study.

For F-feature logistic inference, scaler fusion gives F multiplies, aboutF adds and one logit comparison, with `(F+2)*4` bytes for float32 weights, intercept and a separate threshold:20/24/36 bytes for3/4/7 features. No sigmoid is needed for a binary logit decision in real arithmetic. These are model/storage estimates, not float32 parity or hardware runtime/power measurements.
