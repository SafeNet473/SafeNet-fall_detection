# Checking portability to nRF5340

The portable C detector and timestamp-aware acquisition bridge have been validated on the host. Portability to nRF5340 still needs evidence from the target compiler, sensor driver and board. Examine two boundaries separately: **the frozen 200 Hz detector**, then **the timestamped sensor-to-detector adapter**. This makes numerical errors distinguishable from acquisition errors.

This is a verification plan, not a claim of completed board validation. Start with the [architecture README](README.md), [detector API](embedded/README.md) and [acquisition contract](acquisition/README.md).

## 1. Establish a reproducible target build

Build the existing portable modules into the target application:

```text
sensor_adapters/lsm6dsox_input.c
acquisition/timing_adapter.c
acquisition/acquisition.c
embedded/filters.c
embedded/features.c
embedded/classifier.c
embedded/fall_detector.c
```

Record the board/application core, SDK and compiler versions, optimization flags, floating-point ABI/options, numeric mode and selected operating point. Keep the frozen constants and algorithm unchanged during comparison. Start without fast-math and disable floating-point contraction where the compiler supports it, matching the host validation build. Confirm that the required C99, integer types and math functions compile and link in the actual firmware configuration.

Run both builds: the default mixed/double implementation as the numerical reference, and `FALL_USE_FLOAT` as the deployment candidate. Apply the definition consistently across translation units that share detector types. The timing adapter deliberately retains double interpolation in either build; a float detector does not imply an entirely float-only acquisition path. Measure the cost of both configurations rather than assuming their target performance.

## 2. Isolate detector portability first

Replay identical **200 Hz XYZ samples in g, including gravity**, directly into `fall_detector_push_sample()`. Bypass the sensor driver and timing adapter for this stage. Initialize a fresh state for each independent recording and use the same operating point as the host reference.

Use the inputs and expected results described in [host validation](outputs/host_pc_validation/REPORT.md). For example, `c_events.csv` under a recording/operating-point directory is a host event reference. Capture target results in the same field order and compare:

- Sample count, accepted trigger indices, window start/end indices and decision indices.
- Feature order: `ga_C2`, `jerk_abs_mean`, `ga_parallel_peak`.
- Feature values, classifier logits and final FALL/ADL labels.
- Where needed to locate a discrepancy: filtered XYZ, gravity, magnitude and gravity-aligned streams before comparing the final score.

Indices and labels should match the corresponding same-input reference. Quantify numerical differences separately; do not silently widen tolerances or retune a threshold to obtain agreement. Host float32 behavioral regression passed, but strict numerical parity with the mixed reference failed. Target float behavior must therefore be checked independently. A small logit difference near the chosen threshold deserves explicit inspection even when aggregate errors are small.

Include startup, insufficient prehistory, refractory edges, overlapping candidate activity, and a recording that ends before an active event completes. Confirm that a trigger at `k` uses `[k-100, k+200)`, exactly 300 samples, and decides on arrival of `k+200` while excluding that arriving sample. The existing host boundary checks, including the `before_*` artifacts, provide useful cases; consult their generating tests for the exact inputs and expected semantics.

## 3. Isolate timing-adapter portability

Before connecting a live IMU, replay known timestamped samples through `timing_adapter_push()`, then through `acquisition_push_lsm6dsox_sample()`. Capture generated XYZ, logical timestamps, later-bracket measurement timestamps, stream IDs and detector indices. Compare against a host run of the **same values and timestamps**, not just the same nominal ODR.

Check these properties:

- The first measurement anchors the epoch; no output appears until a second valid measurement supplies a bracket.
- Logical outputs advance by exactly 5 ms in the configured integer tick units, without accumulated drift.
- All axes use the same interpolation factor; coincident endpoints reproduce their input samples.
- No extrapolation, EOF flush or interpolation across a rejected gap occurs.
- Timestamp jitter changes interpolation/availability while the logical grid remains uniform.
- Duplicate/reversed timestamps, excessive gaps and timestamp overflow produce the documented restart behavior and discard incomplete events.
- Sensor reset, ODR change and range change notifications block input until explicit reconfiguration.

Reuse the constant/ramp/sine, jitter, large-epoch, long-duration and discontinuity cases documented in the [acquisition validation report](outputs/acquisition_validation/REPORT.md). Include epochs beyond double's exact-integer range: the adapter must preserve absolute integer timestamps and convert only relative differences for interpolation.

After adapter outputs agree, replay them into the detector and compare events with the standalone same-input detector run from step 2. This separates timing/conversion differences from core arithmetic differences. Raw-count proxies and unquantized g inputs are different inputs: quantization and clipping can change events, as the acquisition report documents.

## 4. Connect real acquisition

For Nordic/Zephyr acceleration channels already in m/s², call
`acquisition_push_lsm6dsox_ms2_sample()` using `sensor_value_to_double()` for
each axis. The bridge divides by 9.80665 to obtain g before timing adaptation.
Do not apply register-count sensitivity to SI readings. The existing raw-count
entry point remains appropriate only for actual signed register samples; see
the [acquisition usage example](acquisition/README.md).

The hardware driver is still a target integration task. Configure and read back the **effective +/-16 g range and nominal 208 Hz ODR**, then pass those confirmed settings explicitly to acquisition initialization. Verify signed XYZ decoding, axis convention/mounting and g conversion with controlled stationary orientations and motion. Preserve gravity; do not feed gravity-removed acceleration into the frozen model.

Use measurement/FIFO acquisition timestamps, not callback or ISR arrival times. Verify the sensor clock units, wrap extension, monotonic ordering, FIFO sample-to-timestamp association and clock calibration. Convert to a monotonic integer timebase whose tick rate is divisible by 200. The host's 52 MHz test timebase is only a convenient exact representation, not a required target clock.

Measure actual timestamp intervals and test FIFO batching, delayed handling, missing samples and sensor resets. Set and document an explicit maximum-gap policy. Report hardware changes to the bridge before feeding more data. Logical grid time, later-bracket measurement time and actual processing time must remain distinct in logs.

The adapter creates a uniform logical stream; it does not prove correct sensor timestamps, adequate bandwidth or alias rejection. Real noise, bias, saturation and the physical mounting must be assessed separately from implementation equivalence.

## 5. Measure target resources and scheduling

Keep each acquisition/detector state persistent and under one serialized owner. Callbacks are synchronous; they must not reenter or reconfigure the state. Copy completed event data needed after the callback. Integrate execution into a context whose timing and stack budget can be measured.

Measure target `sizeof` values, linker RAM/flash usage and stack high-water use. Host state sizes are reference information only, not target measurements. Include driver/FIFO buffers, logging and application queues in the total budget.

Measure typical and worst observed processing time for an input push, a push that emits multiple logical samples, and event completion/classification. Exercise realistic FIFO bursts and concurrent application activity. Compare processing and queueing time with acquisition deadlines; inspect queue growth, dropped samples and end-to-end decision latency. Include interpolation availability, post-trigger collection and scheduling/transport delay when reporting latency. Measure power on the board with the intended sensor and application configuration.

## Evidence to retain before deployment

| Evidence | What it establishes |
|---|---|
| Target build configuration and frozen-source identity | Reproducible implementation and numeric mode |
| Same-input 200 Hz host/target replay comparison | Detector arithmetic and event-state portability |
| Timestamped host/target adapter replay comparison | Conversion, interpolation and restart portability |
| Real sensor configuration/read-back and timestamp logs | Actual acquisition contract is satisfied |
| Runtime, RAM/flash, stack and power measurements | Target resource and scheduling feasibility |
| Live labeled recordings with the intended mounting | Hardware behavior and deployment performance |

Passing replay checks establishes implementation portability on the tested cases. It does not establish real-world fall-detection accuracy. Keep the final hardware operating threshold explicitly unresolved until the required hardware/application validation supports its selection.
