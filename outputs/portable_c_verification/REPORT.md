# Portable C MCU v1 engineering report

The default mixed-precision C implementation **passed** the raw-recording Python-vs-C golden comparison: 19 cases, 99,113 samples, 217 completed events per operating point, both documented operating points (38 comparisons). All trigger/high/edge flags, event bounds, and final labels agree exactly. Authoritative files remained unchanged: True. No algorithm redesign, fitting, threshold tuning, or held-out test evaluation occurred.

## Files and reproduction

- `embedded/fall_detector.h/.c`: reusable one-sample API, continuous state, event scheduling, current trace and completed event diagnostics.
- `embedded/filters.h/.c`: numeric type and continuous SOS/gravity state.
- `embedded/features.h/.c`: maxima and within-window jerk accumulation.
- `embedded/classifier.h/.c`, `embedded/model_params.h`: folded logistic inference and both named operating points.
- `embedded/README.md`: API units, build/run instructions and stream lifetime.
- `tests/replay_test.c`: host-only raw SisFall replay and intermediate/event CSV output.
- `tools/export_model.py`: deterministic export from the frozen manifest, no fit.
- `tools/generate_golden_reference.py`: builds both precisions, validates constants against saved estimator, replays raw counts, compares intermediate values.
- `tools/write_portable_c_report.py`: this report from measured results.
- `outputs/portable_c_verification/comparison.json`: all per-case maximum/mean errors, fixed tolerances, counts, build commands, compiler version and hashes. Paired Python/C CSVs are preserved in case directories.

Run `python tools/export_model.py`, then `python tools/generate_golden_reference.py --compiler PATH`, then `python tools/write_portable_c_report.py`. The local host compiler is under `outputs/host_toolchain`; scientific dependencies are the existing `outputs/deps`. Compiler used: `0.16.0`. Strict C99 compilation passed with `-O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off`. No fast-math. Core uses no Python/scikit-learn, runtime, I/O, OS, dynamic allocation, or platform code.

## Reference audit and discrepancies

Authoritative causal preprocessing, derived scalar streams, rising-edge triggers, ring bounds and features come from `outputs/sisfall/causal_events.py` (`Frontend`, `derived`, `triggers`, `ring_windows`, `feature_row`), `orientation_features.py`, and `pipeline.py` (`load_trial`, counts/256). The manifest and saved `causal_trained_models_v1/logistic.joblib` provide the frozen logistic model. Exact constants/order are checked against that saved object.

The request describes a frozen operating threshold, but the freeze explicitly leaves **final hardware threshold unresolved**. Both documented thresholds are exposed; balanced is clearly designated as the host-test default, not a deployment selection. The reference uses double filter/gravity/derived/features arithmetic with a float32 filtered-sample interface; default C preserves this. An all-float port is a separately measured experiment.

The earlier causal replay module still calls the old decision tree in `ring_windows`; golden tooling uses its exact window/feature results and discards that tree prediction, then calls the saved frozen scaler/logistic. The baseline config describes an older offline zero-phase pipeline; it is used only for raw data location, never preprocessing. No original files were modified to resolve these historical mismatches.

The Python ring has 301 scalar rows, whereas C retains only 100 prior scalar rows and accumulates the active event. This storage change reproduces the exact tested bounds/features. `ga_C2` is a maximum, so it does not need an event-sized buffer. Refractory spacing (300) exceeds active duration (200), so accepted events cannot overlap.

## Signal formulas and filters

Raw ADXL345 first three integer columns: `r=(x,y,z)/256` g at 200 Hz; gyro and other sensor columns are ignored. Counts conversion is exact for int16 input in float. `push_sample` instead accepts g; arbitrary double-valued sensor samples are outside its float-input contract.

Two cascaded fourth-order 5 Hz Butterworth SOS, per axis, direct form II transposed. Rows `[b0,b1,b2,a0,a1,a2]`:

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

For each section: `y=b0*x+z1`, `z1_new=b1*x-a1*y+z2`, `z2_new=b2*x-a2*y`. First sample initializes each axis to `sosfilt_zi(SOS)*r[0]`; unit-input zi rows:

```json
[
  [
    0.005776887917447799,
    -0.00431003020598291
  ],
  [
    0.99419187310563,
    -0.8812248727583252
  ]
]
```

Filter/gravity state persists across all events; only detector initialization resets it. Filter output is cast to float32, then promoted to double, giving feature input `a`.

Gravity input is **raw g**, `alpha=0.01558523664828626=1-exp(-2*pi*0.5/200)`. Initialize `G[-1]=r[0]`, and apply the EMA even on the first push: `G[t]=alpha*r[t]+(1-alpha)*G[t-1]`. `u=G/max(sqrt(G dot G),1e-8)`, `m=sqrt(a dot a)`, signed `p=a dot u`, residual `b=a-p*u`, `h=sqrt(b dot b)`. Parallel includes gravity, with no 1 g subtraction. At near-zero gravity the denominator floor is preserved. Trigger jerk `j[0]=0`; for t>=1, `j[t]=abs(200*(m[t]-m[t-1]))` in g/s; this is scalar magnitude jerk, not vector jerk.

## Event state machine and exact indexing

Evaluate current-sample `H[t]=(m[t]>=1.1)||(j[t]>=5.0)`. A crossing is `H[t] && !H[t-1]`. Start previous-high=false, last-accepted-trigger=-300. Accept when crossing and `t-last>=300`. Every sample updates previous-high, including refractory crossings; suppressed edges are discarded, and a sustained high never retriggers. Refractory starts at the accepted trigger, including triggers with insufficient prehistory or an eventual incomplete tail.

For trigger k>=100, seed the accumulator with prior samples k-100..k-1 in chronological ring order, then include the current sample k. Continue through k+199. At k+200 finalize before adding that current sample; the feature count is 300, jerk count 299. Event `[k-100,k+200)` is half-open; `end=decision=k+200`. Classification waits exactly 1 s, but the decision sample is excluded. Require that decision sample actually arrive; EOF at k+199 gives no decision. No padding, shifting or flushing. Maintain the 100-row history continuously even while active; seed before overwriting the oldest slot. Triggers k<100 still update refractory but never activate an event.

## Frozen features

Feature order is exactly `[ga_C2, jerk_abs_mean, ga_parallel_peak]`. All use the 300 event samples, local i=0..299:

| Feature | Exact definition and source | Accumulator/finalization |
|---|---|---|
| ga_C2 | `max(h[i])`, i=0..299; h is gravity-perpendicular residual magnitude, g | running maximum; final maximum |
| jerk_abs_mean | `sum(abs(200*(m[i]-m[i-1])))/299`, i=1..299, g/s | previous in-window magnitude, sum, count; divide by 299 |
| ga_parallel_peak | `max(abs(p[i]))`, i=0..299, g | running absolute maximum; final maximum |

No cross-boundary jerk difference enters the event feature. No full event buffer is required. Seed accumulation from the 100 scalar prehistory rows, then update with arriving event samples.

## Classifier and folding

Saved train-fitted StandardScaler mean: `[0.4885001226818366, 4.842116154469659, 1.830923027297042]`; scale: `[0.3393389000402486, 4.666720996263801, 0.743175050800754]`. Logistic weights: `[3.8564795859958036, -4.541012509245931, 1.8653532808946023]`, intercept `-3.3544620174694724`.

`w_folded[i]=w[i]/scale[i]`, `b_folded=intercept-sum(w_folded[i]*mean[i])`. Exported values: `[11.36468464281104, -0.9730627806722294, 2.5099783407485585]`, bias `-8.790005992210238`. C computes `z=b_folded+sum(w_folded[i]*feature[i])`, then compares z to the chosen logit with `>=`; no runtime sigmoid.

| Documented point | Probability threshold | Equivalent logit threshold |
|---|---:|---:|
| balanced_reference | 0.88509294569534769 | 2.041569084741143 |
| sensitivity_oriented_candidate | 0.77420316487760177 | 1.2321986907376832 |

Algebraic folding changes floating-point summation, so logit scores are tested numerically, not claimed bitwise identical. Golden labels use the saved estimator's `predict_proba >= tau` as authority; C uses folded logits. Exact-threshold boundary behavior beyond these tested inputs is not guaranteed by a numerical tolerance.

## Golden results

Validation recordings: D01 walking, D03/D04 jogging, D05/D06 stairs, D11 collapsing into chair, F01 forward walking fall, F05 jogging fall, and F13/F14/F15 seated falls, supplemented with an already documented detected F01 fall and missed seated fall. They are representative recordings, not a new performance evaluation. Synthetic inputs: zero gravity, initial high magnitude/no prehistory, repeated pulses/refractory edges, seeded random counts, EOF before decision, and EOF including decision. C receives raw counts sample-by-sample, never precomputed features. Explicit checks verify an accepted edge exactly 300 samples after the prior accepted edge, discarded refractory edges, first-sample trigger without an event, finite zero-gravity output, prefix invariance, no EOF decision before the delay, and an identical first event once the excluded decision sample arrives.

Per-case absolute tolerances were specified before replay; logic/index/labels require zero error. The table gives the maximum across cases and the largest per-case mean absolute error (not a pooled mean). Diagnostics `f` are native filter output, `a` the float32 interface, `g` gravity, `u` normalized gravity, `b` residual, `m` magnitude, `h` perpendicular, `p` signed parallel. Units are g except unit vectors, jerk (g/s), and logit.

| Quantity | Reference C max abs error | Largest case mean abs error | Absolute tolerance | Experimental float max abs error |
|---|---:|---:|---:|---:|
| fx | 0 | 0 | 2e-10 | 1.9592961e-05 |
| fy | 0 | 0 | 2e-10 | 2.4221952e-05 |
| fz | 0 | 0 | 2e-10 | 2.3543184e-05 |
| ax | 0 | 0 | 2e-10 | 1.9550323e-05 |
| ay | 0 | 0 | 2e-10 | 2.4199486e-05 |
| az | 0 | 0 | 2e-10 | 2.3603439e-05 |
| gx | 0 | 0 | 2e-10 | 1.0877279e-06 |
| gy | 0 | 0 | 2e-10 | 1.3923999e-06 |
| gz | 0 | 0 | 2e-10 | 1.2049976e-06 |
| ux | 0 | 0 | 2e-10 | 3.1509124e-06 |
| uy | 0 | 0 | 2e-10 | 1.9820867e-06 |
| uz | 0 | 0 | 2e-10 | 1.6438288e-06 |
| bx | 0 | 0 | 2e-10 | 1.0593501e-05 |
| by | 0 | 0 | 2e-10 | 6.4966577e-06 |
| bz | 0 | 0 | 2e-10 | 9.7583095e-06 |
| m | 0 | 0 | 2e-10 | 2.5484289e-05 |
| h | 0 | 0 | 2e-10 | 1.4141336e-05 |
| p | 0 | 0 | 2e-10 | 2.523981e-05 |
| jerk | 0 | 0 | 5e-08 | 0.0040804642 |
| ga_C2 | 0 | 0 | 2e-10 | 1.0654013e-05 |
| jerk_abs_mean | 2.1316282e-14 | 9.3258734e-15 | 2e-10 | 7.2541586e-05 |
| ga_parallel_peak | 0 | 0 | 2e-10 | 1.6303272e-05 |
| z | 2.1316282e-14 | 9.1038288e-15 | 5e-09 | 0.00011392899 |

Default reference precision: **all 38 comparisons passed**. Float mode: passed the reference's strict numeric criteria in 4/38 comparisons. Exact trigger/edge/high/index/label agreement in float mode across the golden cases: **True**. Its numerical discrepancies are visible from the first stream stage and propagate into features/scores. No tolerance, coefficient, or threshold was changed to make it pass. This mode remains uncertified for general numeric parity; exact labels on a finite corpus do not establish all-input equivalence.

## Memory and deployment scope

Measured host `sizeof(FallDetector)`: **2856 bytes** for reference precision; **1472 bytes** for experimental float. This is the complete persistent state including diagnostic trace and last event, excluding caller input/output storage, compiler-dependent stack frames, standard math implementation, and read-only constants. The 100x3 history alone is 2400/1200 bytes respectively. `FallFilters` stores 12 SOS states + 3 gravity values; features store four numeric accumulators plus a count. No dynamic allocation or event-sized temporary arrays. ABI padding can change target sizeof; host sizeof is not a firmware RAM high-water measurement.

Signal processing, trigger/event state machine, feature accumulation and classifier are portable C and suitable to integrate directly as the reference implementation. Actual MCU correctness/performance must still be measured; double math may cost more than float on the target.

Remaining platform integration: ADXL345 configuration/acquisition at 200 Hz with the frozen scale, timestamps and dropped-sample handling, stream startup/reset/shutdown policy, build/compiler FP settings, linking math support, interrupt/task ownership, target replay, runtime/stack/RAM/power measurement, and eventual output/BLE integration. No nRF5340, Zephyr, BLE, interrupts, or sensor-driver code was added. Final hardware operating threshold and all-float qualification remain explicit non-platform decisions; wrist-domain validation remains outside this PC-port task.
