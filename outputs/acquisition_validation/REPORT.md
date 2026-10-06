# Timestamp-aware LSM6DSOX acquisition implementation and host validation

## Result

**Host acquisition validation passed.** Seven 208 Hz reconstructed recordings were replayed through both g-boundary and raw-int16 paths, in mixed and float builds, at both existing operating points: **56 comparisons, 528 mixed decisions and 528 float decisions**. Each path emits 56,992 logical samples per operating point. Mathematical interpolation samples/timestamps are exact; downstream C streams/features/logits/events/labels match the same-precision standalone 200 Hz C oracle **bitwise**. Accepted triggers, event indices and final labels match the corresponding same-input Python Strategy-A reference. All frozen sources/parameters/reference hashes remain unchanged: **True**.

```json
{
  "recordings": 7,
  "replay_comparisons": 56,
  "mixed_decisions": 528,
  "float_decisions": 528,
  "generated_sample_mismatches": 0,
  "detector_oracle_mismatches": 0,
  "trigger_mismatches": 0,
  "event_index_mismatches": 0,
  "classification_mismatches": 0,
  "unmatched_events": 0
}
```

This is **same-input implementation equivalence**. The raw LSM6DSOX-format proxy differs from the earlier unquantized 208 Hz surrogate because of ADC representation and one clipped offline-reconstruction value. Those input effects, and the prototype's endpoint cancellation, are disclosed below. No detector thresholds or tolerances were altered to hide them. Float remains a candidate, not strict numerical parity with Python mixed precision.

## Architecture and API

```text
Driver / FIFO / sensor-clock reconstruction (not implemented here)
    -> signed int16 XYZ + acquisition timestamp + confirmed effective config
sensor_adapters/lsm6dsox_input.c: explicit-range counts -> g
    -> timestamped AccelSample, gravity included
acquisition/timing_adapter.c: bracketed linear interpolation on exact 5 ms grid
    -> frozen fall_detector_push_sample()
acquisition/acquisition.c: orchestration, restart policy, optional observer
```

All new runtime modules are portable C99 with caller-owned deterministic state, standard math/string support, and no heap/OS/SDK/hardware-register/BLE/alarm dependencies. `acquisition_push_lsm6dsox_sample(state,x,y,z,timestamp)` converts, resamples, and synchronously pushes every generated sample into the existing detector. Its structured result reports status, number emitted and whether an active event was discarded. An optional observer receives each logical timestamp/index, later bracket measurement time, sample/result and a read-only detector view. It must copy anything retained and cannot reenter/mutate acquisition state.

`AcquisitionConfig` requires explicit effective ±16 g, nominal 208 Hz, timestamp tick rate, maximum gap and one existing operating point. The caller/driver must establish those settings through configuration/read-back. Incompatible ranges/ODRs fail closed, not silently fall back. The standalone sensor converter still supports all four explicit ranges; the selected acquisition contract enforces ±16 g. Conversion is unchanged at 0.000488 g/LSB for that range. Register decoding and the effective range-mode bit remain driver responsibilities.

## Logical time, startup and bounded work

The first finite measurement anchors the output epoch. **No output before a second measurement**. On the next valid measurement, emit all grid positions in the closed bracket, including the original first sample at k=0. For each axis use `previous + alpha*(current-previous)` with a shared double alpha from relative integer timestamps, then cast the output to float once. Exact endpoints copy the original sample. Absolute uint64 epochs never pass through double.

Output timestamps are `epoch+k*period_ticks`. Tick rate must be divisible by 200, so `period_ticks=ticks_per_second/200` is exactly 5 ms. Real acquisition/FIFO timestamps are supported now; nominal uniform timing is simply one caller-supplied source. Host ideal replay uses 52 MHz units: input period 250,000 ticks, output period 260,000 ticks. Nanosecond tests use 5,000,000 tick outputs and nominal inputs computed from integer index/208, not an accumulated rounded period.

`bracket_timestamp` is the later bracketing **measurement time**, distinct from logical time and from ISR/software processing time. Nominal first-output availability is delayed by 4.807692 ms; nominal steady-state bracket delay is at most 4.615385 ms. FIFO/transport/scheduling can add delay. Event indices and the frozen `[trigger-100,trigger+200)` / decision-time exclusion refer solely to the uniform logical grid.

The explicit test/example max gap is 1.5 nominal sensor periods: 375,000 ticks at 52 MHz, or 7,211,538 ns. A missing nominal 208 Hz frame is rejected, not bridged. The reusable module permits a max gap <=2 output periods, limiting a push to at most three outputs at startup and two thereafter. A preflight overflow check rejects an entire batch before any emission. No extrapolation or EOF flush occurs.

## Discontinuities

| Condition | Resampler action | Acquisition/core action |
|---|---|---|
| Duplicate timestamp | Clear timing; discard duplicate | Reset detector and advance stream ID |
| Reversal | Restart with current sample as anchor; no output | Reset detector and advance stream ID |
| Gap above configured threshold | Restart with current sample as anchor; no output across gap | Reset detector and advance stream ID |
| Nonfinite independent g sample | Clear timing and discard input | Raw-int16 converter cannot produce this; receiver must reset on status |
| uint64 output-timestamp overflow | Clear timing; no partial output batch | Reset detector and advance stream ID |
| Sensor reset / ODR / full-scale change | Driver explicitly notifies wrapper; clear timing | Block input, reset detector, advance stream ID, require fresh read-back/reconfiguration |
| Reconfigure | Reset even if new config is valid | Discard incomplete event/history; invalid settings remain blocked |

Stream restart means a **new independent stream**, so filter, gravity, trigger/refractory and event state reset together. That is not a filter reset at an event boundary. Logical index restarts at zero; `stream_id` distinguishes restarts, including a timestamp clock that goes backward after reset. All active events are explicitly discarded. Gap/reversal current sample becomes the held first sample of the new stream; duplicate/invalid/overflow inputs are dropped. Hardware changes cannot be inferred reliably from XYZ alone and must be notified by the acquisition driver.

## Synthetic tests

Both precision builds passed constant, XYZ ramp, three-axis low-frequency sine, differing XYZ values, exact coincidences, jitter, startup, duplicate/reversal/gap/nonfinite/overflow, acquisition event-discard, configuration changes and invalid effective settings. All interpolation values match an independent double expression rounded to output float, with exact endpoint preservation. Ramp and sine values are also checked against their analytic signals; sine error is bounded by `max|f''|*interval²/8` plus float rounding, rather than mistaken for exact sine reconstruction.

Across 100 nominal 26-input/25-output interval cycles: 2601 input measurements -> 2501 outputs, with 100 exact later-endpoint coincidences. Jitter can legitimately add coincidences (166 in the test), but every grid position/bracket is checked. The first sample is included, so interval ratios must not be confused with finite-stream inclusive sample counts.

**One-hour test:** 748,801 input measurements -> 720,001 outputs, every timestamp exactly 5 ms apart, last logical timestamp exactly one hour after the epoch, zero accumulated phase drift. Tests also use epochs above double's exact integer range, verifying integer-difference handling. `unit_mixed.txt` and `unit_float.txt` preserve every measured analytic error and host sizeof result. Assertions are enabled with `-UNDEBUG`; strict C99 warnings, no fast-math and `-ffp-contract=off` are used.

## Replay comparison results

Two separate paths prevent confusing changed inputs with implementation errors:

1. **g boundary:** reuse exactly the archived float32 208 Hz g surrogate from the earlier Strategy-A study. Compare generated logical samples to its interpolation and all mixed C streams/events to the archived Strategy-A C output. Detector output is bitwise identical. Only near-zero endpoint rounding differs in generated samples, as explained next.
2. **full raw path:** derive a declared LSM6DSOX-format int16 proxy by rounding that surrogate to 0.000488 g/LSB and explicitly clipping int16 overflow. Python decodes those same signed counts with the float conversion constant and applies the existing Strategy-A interpolation; the C converter/resampler/detector must match this **same-input** oracle. It does so exactly at the boundary and against standalone C. This is a host proxy, not real LSM6DSOX recorded data.

Each precision also feeds the generated Strategy-A samples directly through an independent standalone 200 Hz C replay. Integrated and standalone detector output is bitwise identical in all 56 cases. Mixed C-vs-Python errors remain within the original tolerances; float differences remain visible and are not labeled strict parity.

| Case | Precision | Path | Point | Logical samples | Decisions | Mathematical samples exact | Standalone C exact | Label mismatches |
|---|---|---|---|---:|---:|---|---|---:|
| F01_SA05_R02 | mixed | g | balanced_reference | 2999 | 5 | True | True | 0 |
| F01_SA05_R02 | mixed | g | sensitivity_oriented_candidate | 2999 | 5 | True | True | 0 |
| F01_SA05_R02 | mixed | raw | balanced_reference | 2999 | 5 | True | True | 0 |
| F01_SA05_R02 | mixed | raw | sensitivity_oriented_candidate | 2999 | 5 | True | True | 0 |
| D01_SA05_R01 | mixed | g | balanced_reference | 19998 | 49 | True | True | 0 |
| D01_SA05_R01 | mixed | g | sensitivity_oriented_candidate | 19998 | 49 | True | True | 0 |
| D01_SA05_R01 | mixed | raw | balanced_reference | 19998 | 49 | True | True | 0 |
| D01_SA05_R01 | mixed | raw | sensitivity_oriented_candidate | 19998 | 49 | True | True | 0 |
| D03_SA05_R01 | mixed | g | balanced_reference | 19999 | 62 | True | True | 0 |
| D03_SA05_R01 | mixed | g | sensitivity_oriented_candidate | 19999 | 62 | True | True | 0 |
| D03_SA05_R01 | mixed | raw | balanced_reference | 19999 | 62 | True | True | 0 |
| D03_SA05_R01 | mixed | raw | sensitivity_oriented_candidate | 19999 | 62 | True | True | 0 |
| D05_SA05_R01 | mixed | g | balanced_reference | 4999 | 10 | True | True | 0 |
| D05_SA05_R01 | mixed | g | sensitivity_oriented_candidate | 4999 | 10 | True | True | 0 |
| D05_SA05_R01 | mixed | raw | balanced_reference | 4999 | 10 | True | True | 0 |
| D05_SA05_R01 | mixed | raw | sensitivity_oriented_candidate | 4999 | 10 | True | True | 0 |
| F13_SA05_R01 | mixed | g | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R01 | mixed | g | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F13_SA05_R01 | mixed | raw | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R01 | mixed | raw | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | mixed | g | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | mixed | g | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | mixed | raw | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | mixed | raw | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F05_SA05_R01 | mixed | g | balanced_reference | 2999 | 4 | True | True | 0 |
| F05_SA05_R01 | mixed | g | sensitivity_oriented_candidate | 2999 | 4 | True | True | 0 |
| F05_SA05_R01 | mixed | raw | balanced_reference | 2999 | 4 | True | True | 0 |
| F05_SA05_R01 | mixed | raw | sensitivity_oriented_candidate | 2999 | 4 | True | True | 0 |
| F01_SA05_R02 | float | g | balanced_reference | 2999 | 5 | True | True | 0 |
| F01_SA05_R02 | float | g | sensitivity_oriented_candidate | 2999 | 5 | True | True | 0 |
| F01_SA05_R02 | float | raw | balanced_reference | 2999 | 5 | True | True | 0 |
| F01_SA05_R02 | float | raw | sensitivity_oriented_candidate | 2999 | 5 | True | True | 0 |
| D01_SA05_R01 | float | g | balanced_reference | 19998 | 49 | True | True | 0 |
| D01_SA05_R01 | float | g | sensitivity_oriented_candidate | 19998 | 49 | True | True | 0 |
| D01_SA05_R01 | float | raw | balanced_reference | 19998 | 49 | True | True | 0 |
| D01_SA05_R01 | float | raw | sensitivity_oriented_candidate | 19998 | 49 | True | True | 0 |
| D03_SA05_R01 | float | g | balanced_reference | 19999 | 62 | True | True | 0 |
| D03_SA05_R01 | float | g | sensitivity_oriented_candidate | 19999 | 62 | True | True | 0 |
| D03_SA05_R01 | float | raw | balanced_reference | 19999 | 62 | True | True | 0 |
| D03_SA05_R01 | float | raw | sensitivity_oriented_candidate | 19999 | 62 | True | True | 0 |
| D05_SA05_R01 | float | g | balanced_reference | 4999 | 10 | True | True | 0 |
| D05_SA05_R01 | float | g | sensitivity_oriented_candidate | 4999 | 10 | True | True | 0 |
| D05_SA05_R01 | float | raw | balanced_reference | 4999 | 10 | True | True | 0 |
| D05_SA05_R01 | float | raw | sensitivity_oriented_candidate | 4999 | 10 | True | True | 0 |
| F13_SA05_R01 | float | g | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R01 | float | g | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F13_SA05_R01 | float | raw | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R01 | float | raw | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | float | g | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | float | g | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | float | raw | balanced_reference | 2999 | 1 | True | True | 0 |
| F13_SA05_R02 | float | raw | sensitivity_oriented_candidate | 2999 | 1 | True | True | 0 |
| F05_SA05_R01 | float | g | balanced_reference | 2999 | 4 | True | True | 0 |
| F05_SA05_R01 | float | g | sensitivity_oriented_candidate | 2999 | 4 | True | True | 0 |
| F05_SA05_R01 | float | raw | balanced_reference | 2999 | 4 | True | True | 0 |
| F05_SA05_R01 | float | raw | sensitivity_oriented_candidate | 2999 | 4 | True | True | 0 |

| Precision | Continuous quantity / feature / logit | Maximum absolute Python difference |
|---|---|---:|
| mixed | fx | 0 |
| mixed | fy | 0 |
| mixed | fz | 0 |
| mixed | gx | 0 |
| mixed | gy | 0 |
| mixed | gz | 0 |
| mixed | m | 0 |
| mixed | h | 0 |
| mixed | p | 0 |
| mixed | jerk | 0 |
| mixed | ga_C2 | 0 |
| mixed | jerk_abs_mean | 1.42108547e-14 |
| mixed | ga_parallel_peak | 0 |
| mixed | z | 1.59872116e-14 |
| float | fx | 7.93920801e-06 |
| float | fy | 1.93882866e-05 |
| float | fz | 1.62873743e-05 |
| float | gx | 3.76460761e-07 |
| float | gy | 1.30775691e-06 |
| float | gz | 8.63517745e-07 |
| float | m | 1.91720169e-05 |
| float | h | 4.79092032e-06 |
| float | p | 1.94173786e-05 |
| float | jerk | 0.00129751888 |
| float | ga_C2 | 4.01179151e-06 |
| float | jerk_abs_mean | 5.48645256e-05 |
| float | ga_parallel_peak | 1.46585507e-05 |
| float | z | 5.55759662e-05 |

All accepted trigger/event indices and final labels agree with their same-input reference. A float high-condition flag differs at one stair sample (below), but does not change a rising edge or accepted trigger. Existing float numerical differences are not caused by resampling, since the same-precision standalone outputs are exact.

```json
[
  {
    "case": "D05_SA05_R01",
    "path": "g",
    "flag": "high",
    "max_abs": 1.0,
    "mean_abs": 0.00020004000800160032,
    "absolute_tolerance": 0,
    "passed": false,
    "first_failed_row": 2113
  }
]
```

### Exact-endpoint difference in the old prototype

The previous Python interpolation always evaluates `previous + 1*(current-previous)` at coincidences. On tiny floating-point near-zero components from offline polyphase reconstruction, this can lose information by cancellation. The new module copies the exact measured endpoint as the timestamp contract requires. Maximum generated-sample difference versus the old prototype is **2.9026840961330529e-18 g**, exclusively at exact coincidences. Mathematical endpoint samples, all non-endpoint samples, timestamps and the entire raw-int16 path agree **exactly**. Frozen mixed detector streams/events remain bitwise identical to the archived Strategy-A experiment; no tolerance/threshold was loosened to claim that exact comparison. JSON contains the prototype's strict-equality failures alongside the separate endpoint-contract exact comparison.

### Input quantization and saturation are not hidden

| Proxy recording | Clipped axis values | Max decoded-input change, g | Trigger-index symmetric difference vs unquantized | Unmatched events vs unquantized | Shared-event label changes |
|---|---:|---:|---:|---:|---:|
| F01_SA05_R02 | 1 | 2.15049553 | 0 | 0 | 0 |
| D01_SA05_R01 | 0 | 0.000244021416 | 2 | 2 | 0 |
| D03_SA05_R01 | 0 | 0.000244021416 | 0 | 0 | 0 |
| D05_SA05_R01 | 0 | 0.000243961811 | 0 | 0 | 0 |
| F13_SA05_R01 | 0 | 0.000243976712 | 0 | 0 | 0 |
| F13_SA05_R02 | 0 | 0.000243961811 | 0 | 0 | 0 |
| F05_SA05_R01 | 0 | 0.000243976712 | 0 | 0 | 0 |

The F01 offline polyphase surrogate overshoots the ±16 g conversion span at one axis sample; it is explicitly saturated to int16 in the host proxy, producing an approximately 2.15 g input change there. This is neither a real sensor measurement nor proof of its physical peak. The prior raw SisFall peak was 15.671875 g; offline interpolation can overshoot it. No range/model/threshold was changed to hide this.

In D01, quantization changes the accepted trigger **7921 -> 7920**, a one-sample/5 ms shift. The index-set symmetric difference is two because one old and one new trigger are counted. Relative to their own decoded-input Strategy-A reference, both paths' triggers/windows/features/scores/labels still agree exactly in logic and within the original mixed numerical tolerances. Identical input representation is essential: demanding bitwise equality to a differently quantized/clipped surrogate would not test implementation equivalence. All shared-event classifications and decision counts remain unchanged in this corpus; input effects remain documented.

## State memory and arithmetic

Host sizes (64-bit ABI; target alignment/pointers can change totals):

| State | Mixed bytes | Float bytes |
|---|---:|---:|
| Sensor conversion persistent state | 0 | 0 |
| TimingAdapter | 56 | 56 |
| AcquisitionConfig | 32 | 32 |
| Frozen FallDetector | 2856 | 1472 |
| Complete AcquisitionState (includes all above + observer/stream metadata) | 2976 | 1592 |

The sensor range/ODR/timing settings live in AcquisitionConfig, not in an additional allocated object. Conversion has three float multiplications per input plus dispatch. Timing push has finite/config/time/gap checks and bounded uint64 operations. Each non-endpoint output uses one shared double division, relative integer differences/conversions, and three double subtract/multiply/add triples; final XYZ rounds to float. Endpoints avoid the XYZ arithmetic. The frozen detector runs once per logical output: approximately 36 filter/gravity multiplications and 27 additions, then magnitude/projection calculations, three vector norms (three square roots), three direction divisions and scalar jerk/trigger logic. Active events add running maxima and a magnitude-difference accumulation; trigger activation seeds 100 prior scalar rows, and completion performs one feature division and a three-term classifier dot product. These are operation counts, not target benchmarks.

No dynamic allocation, full-output queue or event buffer is added. Callback views and local temporaries use bounded stack, excluded from persistent sizeof. Double interpolation remains intentional even when detector state is float; Cortex-M target FP64 costs must be measured. Existing float norm calls also use the unchanged double math interface. Actual target stack/RAM high-water mark, timing/power and acquisition-driver buffers remain unmeasured. No premature fixed-point or math rewrite was introduced.

## Files and remaining target work

Added: `acquisition/timing_adapter.h/.c`, `acquisition/acquisition.h/.c`, `acquisition/README.md`, `tests/timing_adapter_test.c`, `tests/acquisition_replay.c`, `tools/validate_acquisition.py`, `tools/write_acquisition_report.py`, report/JSON/paired CSV/log artifacts. The explicit-range LSM6DSOX converter was reused unchanged. `sensor_adapters/INPUT_CONTRACT.md` and `input_contract.json` now record the selected architecture and required read-back. The older sensor-study report generator now writes its historical proposal snapshot into that study's output directory instead of overwriting the current deployment contract.

Run `python tools/validate_acquisition.py --compiler PATH`, then `python tools/write_acquisition_report.py`. All frozen source/model/reference hashes are checked before and after the run. `validation.json` includes build commands, memory sizes, original tolerances and every mismatch/error; intermediate CSVs and source proxy inputs remain on disk. No original authoritative files were modified.

Target-facing software is prepared for nRF5340 integration. Remaining work is the real LSM6DSOX driver/configuration/read-back, sensor/FIFO timestamp calibration and wrap extension, gap/change detection/ownership, real-sensor bandwidth/noise/clipping validation, resampler latency/spectral assessment, target compiler/math replay, runtime/stack/RAM/power qualification and eventual alarm/BLE integration. No hardware driver, Zephyr, ISR, BLE or alarm behavior is implemented here. Linear interpolation does not supply a guaranteed anti-alias stopband; real sensor spectra/filtering remain to be validated. Mixed remains the reference and float remains a candidate. The final hardware classifier operating point still requires an explicit choice between existing documented points or a separately authorized threshold decision.
