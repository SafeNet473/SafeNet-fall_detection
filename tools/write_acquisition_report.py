"""Render measured acquisition validation and the selected input contract."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/acquisition_validation'


def main():
    result = json.loads((OUT / 'validation.json').read_text())
    cases = result['cases']
    size_mixed = result['unit_tests']['mixed']['state_bytes']
    size_float = result['unit_tests']['float']['state_bytes']
    rows = []
    for row in cases:
        rows.append(f'| {row["name"]} | {row["precision"]} | {row["path"]} | {row["point"]} | '
                    f'{row["outputs"]} | {row["decisions"]} | {row["generated_samples"]["passed"]} | '
                    f'{row["direct_200_c_bitwise_equal"]} | {row["logic"]["labels"]} |')
    error_rows = []
    for mode in ('mixed', 'float'):
        selected = [row for row in cases if row['precision'] == mode]
        for group, names in [('python_streams', ['fx', 'fy', 'fz', 'gx', 'gy', 'gz', 'm', 'h', 'p', 'jerk']),
                             ('python_events', ['ga_C2', 'jerk_abs_mean', 'ga_parallel_peak', 'z'])]:
            for name in names:
                error = max(row[group]['columns'][name]['max_abs'] for row in selected)
                error_rows.append(f'| {mode} | {name} | {error:.9g} |')
    prototype_error = max(row['archived_prototype_samples']['columns'][name]['max_abs']
                          for row in cases for name in ('ax', 'ay', 'az'))
    quantization_rows = []
    for row in result['quantization']:
        logic = row['comparison_to_unquantized']['balanced_reference']
        quantization_rows.append(f'| {row["name"]} | {row["clipped_axis_values"]} | '
                                 f'{row["maximum_decoded_g_difference"]:.9g} | '
                                 f'{logic["trigger_indices"]} | {logic["unmatched_events"]} | {logic["labels"]} |')
    float_flags = []
    for row in cases:
        if row['precision'] != 'float' or row['point'] != 'balanced_reference':
            continue
        for name in ('high', 'edge', 'trigger'):
            info = row['python_streams']['columns'][name]
            if info['max_abs']:
                float_flags.append(dict(case=row['name'], path=row['path'], flag=name, **info))
    unique_outputs = sum(row['outputs'] for row in cases if row['precision'] == 'mixed'
                         and row['path'] == 'raw' and row['point'] == 'balanced_reference')
    report = f'''# Timestamp-aware LSM6DSOX acquisition implementation and host validation

## Result

**Host acquisition validation passed.** Seven 208 Hz reconstructed recordings were replayed through both g-boundary and raw-int16 paths, in mixed and float builds, at both existing operating points: **56 comparisons, 528 mixed decisions and 528 float decisions**. Each path emits {unique_outputs:,} logical samples per operating point. Mathematical interpolation samples/timestamps are exact; downstream C streams/features/logits/events/labels match the same-precision standalone 200 Hz C oracle **bitwise**. Accepted triggers, event indices and final labels match the corresponding same-input Python Strategy-A reference. All frozen sources/parameters/reference hashes remain unchanged: **{result['frozen_inputs_unchanged']}**.

```json
{json.dumps(result['summary'], indent=2)}
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
{chr(10).join(rows)}

| Precision | Continuous quantity / feature / logit | Maximum absolute Python difference |
|---|---|---:|
{chr(10).join(error_rows)}

All accepted trigger/event indices and final labels agree with their same-input reference. A float high-condition flag differs at one stair sample (below), but does not change a rising edge or accepted trigger. Existing float numerical differences are not caused by resampling, since the same-precision standalone outputs are exact.

```json
{json.dumps(float_flags, indent=2)}
```

### Exact-endpoint difference in the old prototype

The previous Python interpolation always evaluates `previous + 1*(current-previous)` at coincidences. On tiny floating-point near-zero components from offline polyphase reconstruction, this can lose information by cancellation. The new module copies the exact measured endpoint as the timestamp contract requires. Maximum generated-sample difference versus the old prototype is **{prototype_error:.17g} g**, exclusively at exact coincidences. Mathematical endpoint samples, all non-endpoint samples, timestamps and the entire raw-int16 path agree **exactly**. Frozen mixed detector streams/events remain bitwise identical to the archived Strategy-A experiment; no tolerance/threshold was loosened to claim that exact comparison. JSON contains the prototype's strict-equality failures alongside the separate endpoint-contract exact comparison.

### Input quantization and saturation are not hidden

| Proxy recording | Clipped axis values | Max decoded-input change, g | Trigger-index symmetric difference vs unquantized | Unmatched events vs unquantized | Shared-event label changes |
|---|---:|---:|---:|---:|---:|
{chr(10).join(quantization_rows)}

The F01 offline polyphase surrogate overshoots the ±16 g conversion span at one axis sample; it is explicitly saturated to int16 in the host proxy, producing an approximately 2.15 g input change there. This is neither a real sensor measurement nor proof of its physical peak. The prior raw SisFall peak was 15.671875 g; offline interpolation can overshoot it. No range/model/threshold was changed to hide this.

In D01, quantization changes the accepted trigger **7921 -> 7920**, a one-sample/5 ms shift. The index-set symmetric difference is two because one old and one new trigger are counted. Relative to their own decoded-input Strategy-A reference, both paths' triggers/windows/features/scores/labels still agree exactly in logic and within the original mixed numerical tolerances. Identical input representation is essential: demanding bitwise equality to a differently quantized/clipped surrogate would not test implementation equivalence. All shared-event classifications and decision counts remain unchanged in this corpus; input effects remain documented.

## State memory and arithmetic

Host sizes (64-bit ABI; target alignment/pointers can change totals):

| State | Mixed bytes | Float bytes |
|---|---:|---:|
| Sensor conversion persistent state | 0 | 0 |
| TimingAdapter | {size_mixed['timing']} | {size_float['timing']} |
| AcquisitionConfig | {size_mixed['config']} | {size_float['config']} |
| Frozen FallDetector | {size_mixed['detector']} | {size_float['detector']} |
| Complete AcquisitionState (includes all above + observer/stream metadata) | {size_mixed['acquisition']} | {size_float['acquisition']} |

The sensor range/ODR/timing settings live in AcquisitionConfig, not in an additional allocated object. Conversion has three float multiplications per input plus dispatch. Timing push has finite/config/time/gap checks and bounded uint64 operations. Each non-endpoint output uses one shared double division, relative integer differences/conversions, and three double subtract/multiply/add triples; final XYZ rounds to float. Endpoints avoid the XYZ arithmetic. The frozen detector runs once per logical output: approximately 36 filter/gravity multiplications and 27 additions, then magnitude/projection calculations, three vector norms (three square roots), three direction divisions and scalar jerk/trigger logic. Active events add running maxima and a magnitude-difference accumulation; trigger activation seeds 100 prior scalar rows, and completion performs one feature division and a three-term classifier dot product. These are operation counts, not target benchmarks.

No dynamic allocation, full-output queue or event buffer is added. Callback views and local temporaries use bounded stack, excluded from persistent sizeof. Double interpolation remains intentional even when detector state is float; Cortex-M target FP64 costs must be measured. Existing float norm calls also use the unchanged double math interface. Actual target stack/RAM high-water mark, timing/power and acquisition-driver buffers remain unmeasured. No premature fixed-point or math rewrite was introduced.

## Files and remaining target work

Added: `acquisition/timing_adapter.h/.c`, `acquisition/acquisition.h/.c`, `acquisition/README.md`, `tests/timing_adapter_test.c`, `tests/acquisition_replay.c`, `tools/validate_acquisition.py`, `tools/write_acquisition_report.py`, report/JSON/paired CSV/log artifacts. The explicit-range LSM6DSOX converter was reused unchanged. `sensor_adapters/INPUT_CONTRACT.md` and `input_contract.json` now record the selected architecture and required read-back. The older sensor-study report generator now writes its historical proposal snapshot into that study's output directory instead of overwriting the current deployment contract.

Run `python tools/validate_acquisition.py --compiler PATH`, then `python tools/write_acquisition_report.py`. All frozen source/model/reference hashes are checked before and after the run. `validation.json` includes build commands, memory sizes, original tolerances and every mismatch/error; intermediate CSVs and source proxy inputs remain on disk. No original authoritative files were modified.

Target-facing software is prepared for nRF5340 integration. Remaining work is the real LSM6DSOX driver/configuration/read-back, sensor/FIFO timestamp calibration and wrap extension, gap/change detection/ownership, real-sensor bandwidth/noise/clipping validation, resampler latency/spectral assessment, target compiler/math replay, runtime/stack/RAM/power qualification and eventual alarm/BLE integration. No hardware driver, Zephyr, ISR, BLE or alarm behavior is implemented here. Linear interpolation does not supply a guaranteed anti-alias stopband; real sensor spectra/filtering remain to be validated. Mixed remains the reference and float remains a candidate. The final hardware classifier operating point still requires an explicit choice between existing documented points or a separately authorized threshold decision.
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    contract = dict(
        frozen_algorithm='MCU_v1', algorithm_sample_rate_hz=200,
        input_units='g', gravity_included=True, input='chronological uniform 5 ms XYZ grid',
        selected_sensor='ST LSM6DSOX', selected_effective_range_g=16, selected_nominal_odr_hz=208,
        actual_hardware_readback_verified=False,
        range_readback_policy='explicit confirmed effective range; no register-bit assumptions',
        sensitivity_g_per_lsb={'2': .000061, '4': .000122, '8': .000244, '16': .000488},
        conversion_api='lsm6dsox_counts_to_g; acquisition accepts confirmed effective ±16 g only',
        timing_api='timing_adapter_push; configurable integer acquisition timestamp units',
        acquisition_api='acquisition_push_lsm6dsox_sample',
        logical_period_seconds=.005, startup='anchor first valid sample; emit only after second arrives',
        interpolation='shared double weight; float output; exact endpoint copies',
        timestamp_policy='acquisition/FIFO/sensor-clock time, not ISR arrival time; integer output grid',
        max_gap_policy='explicit config; replay uses 1.5 nominal input periods; no interpolation beyond limit',
        discontinuity_policy='discard active event; restart stream; hardware changes block until reconfigure/read-back',
        authoritative_precision='unchanged mixed-precision detector', float_status='deployment candidate; numerical differences remain',
        native208_status='historical isolated experiment, not selected deployment',
        final_hardware_threshold=None, resampler_status='implemented and host-validated; real sensor/target qualification pending')
    (ROOT / 'sensor_adapters/input_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    (ROOT / 'sensor_adapters/INPUT_CONTRACT.md').write_text('''# Selected detector / LSM6DSOX input contract

The deployment architecture is **LSM6DSOX effective ±16 g, nominal 208 Hz ->
explicit counts-to-g conversion -> timestamp-aware linear timing adapter ->
uniform 200 Hz g-valued XYZ including gravity -> unchanged frozen MCU v1 core**.

The driver must configure/read back the effective range and ODR before calling
`acquisition_init`. Register access is outside the unit converter and detector.
The converter supports all four explicit ranges; the selected acquisition path
accepts only ±16 g and nominal 208 Hz, failing closed for other configurations.
Sensitivity at ±16 g is 0.000488 g/LSB. The range enum is physical g, not FS bits.
Never pass LSM6DSOX raw counts to the ADXL345 `/256` detector helper.

Timestamp units are explicit in AcquisitionConfig, with a tick rate divisible
by 200. The first measurement anchors the logical grid. Two measurements are
required before any output; exact endpoints are copied, and other XYZ values
share one interpolation weight. Logical timestamps are exactly 5 ms apart.
The later bracket measurement timestamp is not the ISR/processing timestamp.
Actual driver/FIFO timebase reconstruction and counter extension remain driver
responsibilities. No missing data is filled across a rejected gap.

Duplicate, reversed or excessive-gap timing resets detector/event history as
an independent stream. Sensor reset, ODR change and full-scale change require
explicit notification and block further samples until fresh configuration and
read-back. No output extrapolation or EOF flush is allowed.

Mixed precision remains the numerical reference. Float remains the deployment
candidate, with exact integrated-vs-standalone behavior on the host suite but
known numerical differences versus mixed/Python. The final hardware classifier
operating threshold remains an explicit unresolved choice.

Host validation passed both precision builds, all 56 replay comparisons and
1056 decisions, with the prototype endpoint and sensor-quantization differences
reported separately. No frozen detector/model/reference file was changed.

See `acquisition/README.md` for the target-facing API and restart policy,
`outputs/acquisition_validation/REPORT.md` for measured evidence, and
`input_contract.json` for selected settings versus unverified hardware read-back.
No MCU driver, Zephyr, interrupts, BLE or alarm implementation is included.
''', encoding='utf-8')
    print('Acquisition report and selected input contract written.')


if __name__ == '__main__':
    main()
