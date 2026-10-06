"""Generate a measured report and proposed acquisition contract; no refitting."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sensor_contract_validation'
DATASHEET = 'https://www.st.com/resource/en/datasheet/lsm6dsox.pdf'
ST_DRIVER = 'https://raw.githubusercontent.com/STMicroelectronics/lsm6dsox-pid/master/lsm6dsox_reg.c'
SCIPY = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html'


def main():
    result = json.loads((OUT / 'validation.json').read_text())
    summary = result['float_summary']
    native = result['native208_configuration']
    shifts = result['fixed_window_feature_shift']
    cases = result['float_cases']
    def maximum(group, name):
        return max(row[group]['columns'][name]['max_abs'] for row in cases)
    stream_rows = []
    for name in ('fx', 'fy', 'fz', 'gx', 'gy', 'gz', 'm', 'h', 'p', 'jerk'):
        mean = max(row['streams']['columns'][name]['mean_abs'] for row in cases)
        limit = cases[0]['streams']['columns'][name]['absolute_tolerance']
        stream_rows.append(f'| {name} | {maximum("streams", name):.9g} | {mean:.9g} | {limit:.9g} |')
    feature_rows = []
    for name in ('ga_C2', 'jerk_abs_mean', 'ga_parallel_peak', 'z'):
        mean = max(row['events']['columns'][name]['mean_abs'] for row in cases)
        limit = cases[0]['events']['columns'][name]['absolute_tolerance']
        feature_rows.append(f'| {name} | {maximum("events", name):.9g} | {mean:.9g} | {limit:.9g} |')
    case_rows = []
    for row in cases:
        logic = row['logic']
        case_rows.append(f'| {row["name"]} | {row["point"]} | {logic["reference_decisions"]} | '
                         f'{logic["trigger_index_mismatches"]} | {logic["event_index_mismatches"]} | '
                         f'{logic["classification_mismatches"]} | {row["events"]["columns"]["z"]["max_abs"]:.9g} |')
    sampling_rows = []
    for row in result['strategy_cases']:
        for strategy in ('A_linear', 'B_native208'):
            comparison = row[strategy]
            sampling_rows.append(f'| {row["name"]} | {row["point"]} | {strategy} | '
                                 f'{comparison["matched_events"]} | {comparison["unmatched_original"]} | '
                                 f'{comparison["unmatched_variant"]} | {comparison["matched_class_changes"]} | '
                                 f'{comparison["original_positive_decisions"]} | {comparison["variant_positive_decisions"]} |')
    distribution_rows = []
    for strategy, shift in shifts.items():
        for index, feature in enumerate(('ga_C2', 'jerk_abs_mean', 'ga_parallel_peak')):
            distribution_rows.append(f'| {strategy} | {feature} | {shift["max_abs"][index]:.9g} | '
                                     f'{shift["mean_abs"][index]:.9g} | {shift["rms_standardized_delta"][index]:.9g} | '
                                     f'{shift["max_abs_standardized_delta"][index]:.9g} |')
    minimum_margin = min(row['minimum_reference_decision_margin'] for row in cases)
    threshold_rows = []
    for point in ('balanced_reference', 'sensitivity_oriented_candidate'):
        example = next(row for row in cases if row['point'] == point)
        cli = 'balanced' if point == 'balanced_reference' else 'sensitivity'
        with (ROOT / 'outputs/host_pc_validation' / example['name'] / cli / 'python_events.csv').open() as stream:
            expected = next(csv.DictReader(stream))
        with (OUT / example['name'] / ('float_' + cli) / 'c_events.csv').open() as stream:
            actual = next(csv.DictReader(stream))
        old, new = float(expected['threshold']), float(actual['threshold'])
        threshold_rows.append(f'| {point} | {old:.17g} | {new:.17g} | {abs(old-new):.9g} |')
    range_rows = []
    for row in result['raw_sensor_range_diagnostics']:
        counts = row['axis_samples_above_range']
        range_rows.append(f'| {row["name"]} | {row["peak_abs_axis_g"]:.9g} | {counts["2"]} | '
                          f'{counts["4"]} | {counts["8"]} | {counts["16"]} |')
    changed_events = [row for row in cases if row['behavior_diagnostic'] is not None]
    if changed_events:
        diagnosis = json.dumps(changed_events, indent=2)
    else:
        diagnosis = ('No float trigger or classification changed, so there is no behavior-changing '
                     'numerical stage in this suite. Numerical divergence begins in filter/gravity '
                     'coefficient/state arithmetic, then propagates through scalar streams, jerk and '
                     'feature accumulation. All per-case errors and same-input C-reference comparisons '
                     'are preserved; the numerical parity failures are not hidden.')
    report = f'''# Float32 and LSM6DSOX input/sampling contract study

## Outcome and scope

The existing `FALL_USE_FLOAT` build produced **264/264 identical decisions**, zero accepted-trigger index mismatches, zero event-bound mismatches and zero unmatched events on the same seven recordings and two operating points as the successful mixed-precision PC suite. It **did not meet the original numerical tolerances**. It is behaviorally equivalent on this finite suite, not certified for arbitrary inputs or substituted for the reference.

The mixed-precision reference was rebuilt and reproduced the previous C golden streams/events bitwise. All protected embedded sources, frozen parameters and Python reference inputs remained unchanged: **{result['protected_inputs_unchanged']}**. Both resampling strategies were tested in isolation. The native-208 C variant matched its own independent Python implementation within the unchanged tolerances: **{all(row['passed'] for row in result['native208_c_vs_python'])}**.

No training, threshold adjustment, production resampler, Zephyr/nRF5340 integration or sensor driver was added. The actual sensor full-scale setting and ODR have not been supplied; the adapter has no default selection. Recommendations below are proposals, not hardware read-back facts.

## A. Existing single-precision detector

Exact input contract: chronological XYZ g from the existing SisFall loader; no conversion back to integer counts. Every accepted trigger, completed event, feature, logit, threshold and label was compared to the same archived Python golden data and a freshly rebuilt validated C reference. `FALL_USE_FLOAT` changes stored coefficients/state/scalars/accumulators/model arithmetic to float; the existing `sqrt`/`fabs` calls still promote arguments to double internally. This study measures that existing implementation, not a new all-`sqrtf` firmware rewrite.

All builds use strict C99, `-O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off`; test assertions are forced active with `-UNDEBUG`. No fast-math or altered tolerance is used. Exact build commands and hashes are in `validation.json`.

The same suite's isolated classifier vectors, chronological history indices, mutated decision-sample exclusion and EOF checks were also repeated in float. Their measured results are:

```json
{json.dumps(dict(classifier_only=result['float_classifier_only'], boundary_checks=result['float_boundary_checks']), indent=2)}
```

### Actual errors

Maximum requested stream error is **{summary['maximum_requested_stream_error']:.9g}** (the largest is jerk in g/s). Reporting one combined maximum mixes units, so the per-stream table is the useful engineering measure. Means below are the largest per-case means, not pooled means.

| Stream | Maximum absolute error | Largest case mean absolute error | Original absolute tolerance |
|---|---:|---:|---:|
{chr(10).join(stream_rows)}

`f` = native filter output, `g` = gravity, `m` = magnitude, `h` = perpendicular, `p` = signed parallel. All are in g except jerk (g/s). Additional normalized/residual streams and per-case errors are in JSON/CSV.

| Feature / classifier | Maximum absolute error | Largest case mean absolute error | Original absolute tolerance |
|---|---:|---:|---:|
{chr(10).join(feature_rows)}

Maximum feature error: **{summary['maximum_feature_error']:.9g}**. Maximum replay logit error: **{summary['maximum_classifier_score_error']:.9g}**. These include accumulated feature differences and rounded folded coefficients. The smallest reference decision margin in this suite is **{minimum_margin:.9g} logits**, substantially above the observed error, explaining the label agreement; this is not an all-input error bound.

| Operating point | Reference logit threshold | Float logit threshold | Absolute rounding error |
|---|---:|---:|---:|
{chr(10).join(threshold_rows)}

This is ordinary float representation of the unchanged constants, not retuning. No epsilon band or tolerance is applied to actual FALL/ADL comparisons.

### Per-recording logic results

| Recording | Point | Decisions | Trigger-index mismatches | Event-index mismatches | Label mismatches | Max logit error |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(case_rows)}

{diagnosis}

Conclusion: single precision is a promising target candidate and passed this behavioral regression suite. It is not strict numerical parity and should not yet replace the mixed reference. Broader recordings, synthetic threshold-neighborhood checks and target/compiler replay remain necessary before treating float as a generally qualified numerical contract.

## B. Sensor-specific boundary

The detector consumes **raw acceleration including gravity, in g**, through `fall_detector_push_sample`. The `/256.0f` helper is ADXL345/SisFall-only and must never receive LSM6DSOX counts. The new adapter is outside `embedded/`:

```text
signed int16 LSM6DSOX XYZ + explicitly known configured range
    -> lsm6dsox_counts_to_g(...)
    -> g-valued samples with acquisition timestamps
    -> qualified 208-to-200 timing adapter
    -> fall_detector_push_sample() at exactly the frozen 200 Hz grid
```

Conversion is `a_g=counts*sensitivity_mg_per_lsb/1000`:

| Configured full-scale range | Sensitivity mg/LSB | g/LSB |
|---|---:|---:|
| ±2 g | 0.061 | 0.000061 |
| ±4 g | 0.122 | 0.000122 |
| ±8 g | 0.244 | 0.000244 |
| ±16 g | 0.488 | 0.000488 |

Values were verified in ST DS12814 Rev 4 Table 2 and ST's conversion functions. The range enum represents the physical configured range, not register bits. Table 52 maps FS bits `00/01/10/11` to ±2/±16/±4/±8 g when `XL_FS_MODE=0`; `01` instead means ±2 g when `XL_FS_MODE=1`. Therefore read back the **effective range**, including that mode bit. [ST datasheet]({DATASHEET}), [ST reference conversion functions]({ST_DRIVER}).

The portable adapter requires a range on every conversion, rejects unknown ranges and null output, and leaves output unchanged on failure. It performs no register operations, no implicit default, no ODR conversion, no filtering, no calibration and no gravity subtraction. Tests passed for all four modes, sign, zero, int16 endpoints and invalid arguments. Different multiplication order versus ST's mg-returning helper can differ by float rounding; this adapter directly uses the datasheet g/LSB constants. Nominal conversion is clean, while real scale/offset/noise still require sensor checks.

### Recommended initial full-scale setting: ±16 g

This is an engineering recommendation to preserve impact headroom and the training data's ±16 g range. The selected recordings themselves demonstrate why smaller ranges can clip before the 5 Hz software filter:

| Recording | Peak absolute raw axis, g | Axis samples >2 g | >4 g | >8 g | >16 g |
|---|---:|---:|---:|---:|---:|
{chr(10).join(range_rows)}

The observed peak is 15.671875 g, so ±8 g would clip existing fall examples. At ±16 g the nominal 0.000488 g/LSB resolution is finer than SisFall's 1/256 g resolution; this makes initial headroom the stronger concern in these recordings. This comparison is not a noise-floor guarantee or proof that real impacts cannot exceed ±16 g. The actual setting must be selected/read back explicitly before using the conversion; ±16 g requires the correct effective mode. Hardware clipping, sensor noise and low-g resolution still need measurement.

## C. Every 200 Hz dependency and the 208 Hz issue

ST's accelerometer ODR table includes nominal 208 Hz but no 200 Hz setting. [ST datasheet, Table 51]({DATASHEET}). Feeding those samples one-for-one into the frozen detector would change physical behavior:

| Dependency | Current location/value | Consequence if 208 Hz is fed unchanged | Native-208 experiment |
|---|---|---|---|
| Causal acceleration filter | `embedded/model_params.h`: saved SOS/zi, 5 Hz at 200 Hz | Physical cutoff becomes 5.2 Hz | New 5 Hz SOS/zi at 208 Hz |
| Gravity EMA | `model_params.h` FALL_ALPHA; `filters.c` uses it | Nominal cutoff becomes 0.52 Hz | alpha={native['alpha']:.17g} |
| Trigger jerk | `fall_detector.c` SAMPLE_RATE_HZ=200 | g/s derivative underestimated by 3.846% | multiply differences by 208 |
| Feature jerk | `features.c` explicit factor 200 | Same derivative bias | multiply differences by 208 |
| Prehistory | detector PRE_TRIGGER_SAMPLES and header history[100] | 0.480769 s rather than 0.5 s | 104 samples |
| Post/decision delay | POST_TRIGGER_SAMPLES=200 | 0.961538 s rather than 1 s | 208 samples |
| Refractory + initial last trigger | REFRACTORY_SAMPLES=300 and init | 1.442308 s rather than 1.5 s | 312 samples / initial last=-312 |
| Event feature count | 300 samples, 299 within-window differences | Nominal window duration becomes 1.442308 s | 312 samples, 311 differences |
| Event bounds + ring modulo | detector uses pre/post/history constants | Physical boundaries change | `[k-104,k+208)`, decide k+208 |
| Golden/reference generation | causal_events.py FS, SOS, REFRACTORY, DESIGNS, feature_row | Index/time/feature authority remains 200 Hz | isolated parameterized 208 reference |
| Orientation Python reference | orientation_features.py FS/ALPHA, derivative scaling; older 200-row feature API | Assumptions cannot be reused for 312-row features | explicit 208 derivation |
| Host assertions + reporting | host_validation.c modulo/count/bounds; validate_host_pipeline.py / generate_golden_reference.py fs, time offsets, counts | Old tests assert 300 and old timing | separate generated experimental runner |
| Model export/manifest/data provenance | export_model.py reads the 200 Hz freeze; manifest; baseline config | Reference constants and training feature distribution remain rate-specific | copied model, changed experimental preprocessing only |
| Evaluation-only matching/latency | causal_events.py MATCH=200 (±1 s), latency divides by FS | Evaluation time conversions must also change if used | unused for performance claims; study matches by seconds |

Magnitude/jerk trigger levels (1.1 g, 5 g/s), feature order, scaler, model weights, operating thresholds and counts-to-g conversion do not themselves encode a sample rate. Their interpretation still depends on correctly derived streams/windows. The ADXL345 conversion is sensor-specific, not rate-specific.

### Strategy A: 208 Hz acquisition -> qualified 200 Hz stream -> frozen detector

The isolated lightweight prototype keeps two adjacent XYZ samples. Output k is at `k/200`, corresponding to nominal 208 Hz position `k*26/25`. On arrival of the sample bracketing that position, emit `previous + fraction*(current-previous)` with the same interpolation weight for all axes. Integer phase avoids accumulated float clock drift; it emits 25 intervals for every 26 input intervals. It neither drops every 26th sample nor extrapolates future values. State is one previous XYZ plus input/output integer indices and current XYZ during push; arithmetic is roughly three subtractions, multiplications and additions per output, about 1,800 scalar arithmetic operations/s before bookkeeping.

Measured bracketing latency: max **{result['interpolation_latency_seconds']['maximum']*1000:.9g} ms**, mean **{result['interpolation_latency_seconds']['mean']*1000:.9g} ms**; nominal theoretical maximum 4.615385 ms, under one 208 Hz interval. Decision wall-clock availability adds this latency to the frozen 1 s sample-grid delay. No benchmark or MCU cycle measurement is claimed.

Unit tests validate constant/ramp reproduction, exactly 1001 outputs from 1041 inputs spanning 5 s, no future extrapolation, output-prefix invariance and bounded availability delay. Analytic interpolation errors (unit-amplitude sine, same timestamps):

```json
{json.dumps(result['linear_resampler_checks'], indent=2)}
```

Linear interpolation attenuates high frequencies and is **not a guaranteed anti-alias filter**. The 5 Hz software filter reduces out-of-band acceleration, but gravity uses raw input and projection/nonlinear magnitude behavior prevent assuming this solves all aliasing. If spectrum measurements show relevant energy near/above the 100 Hz output Nyquist, use an explicitly designed causal polyphase FIR ahead of the frozen detector. A nominal 25/26 Kaiser prototype of 521 taps at the upsampled 5200 Hz grid would require about 21 input frames per phase, 25 coefficient phases and about 12,600 three-axis MAC/s at 200 outputs/s; group delay is 260/5200=50 ms. These are design estimates, not measured code or verified stopband specifications. A real FIR must be designed to explicit passband/stopband requirements and its startup/delay policy tested. No FIR was inserted into production.

Timing contract: use sensor acquisition/FIFO timestamps or a validated reconstruction from sensor clock, not ISR/host arrival times. The simple prototype assumes ideal uniform nominal 208 Hz. In real deployment, anchor a 5 ms output grid to a documented clock/epoch; use actual bracketing acquisition times for interpolation, account for clock drift/timestamp wrap and label outputs by their grid time rather than emission time. Scheduling bursts must retain chronological output order. Do not synthesize a long gap or silently hold/drop/repeat a sample; explicitly detect discontinuity, with a defined stream-reset/event-discard policy validated separately. Changes in ODR/range/filter mode must be explicit stream boundaries. Transport latency is separate from sample time.

### Strategy B: isolated native 208 Hz detector

Generated files are under `experiments/sampling_rate/native208/`, and compiled only into a separate host executable. It keeps the frozen logistic parameters and trigger levels, with these **explicitly changed experimental** settings:

- 4th-order causal 5 Hz Butterworth redesigned at fs=208; initialize `sosfilt_zi * first sample`.
- Raw-driven 0.5 Hz EMA, alpha={native['alpha']:.17g}; same initialization/floor.
- Jerk differences multiply by 208 in the trigger and feature.
- 104 pre + 208 from trigger onward, 312 total, 311 differences; 312-sample refractory.
- Window `[k-104,k+208)`; wait k+208 and exclude its sample. Physical durations remain 0.5/1.0/1.5 s.

Exact 208 SOS rows:

```json
{json.dumps(native['sos'], indent=2)}
```

Unit-input zi:

```json
{json.dumps(native['zi'], indent=2)}
```

Changing 200 to 208 across the production code is not permitted. This variant is an experiment/new baseline, not MCU v1.

### Principled offline input reconstruction and limits

Existing 200 Hz SisFall cannot supply independent 208 Hz sensor ground truth or recover unrecorded frequencies. The primary surrogate uses floating-point g samples and `resample_poly(up=26,down=25,window=('kaiser',5),padtype='edge')`. It preserves the time origin, uses a band-limited reconstruction with explicit edge extension, and trims output to timestamps <= the original final sample. The offline FIR is phase-compensated and uses future samples; it is not the proposed live causal adapter. [SciPy resample_poly documentation]({SCIPY}). Samples are cast to float32 for the C g-input API, with the same quantized input passed to the independent 208 Python reference.

To avoid interpreting a resampling artifact as a rate effect, a second native-208 diagnostic reconstructs the same input with linear interpolation. Fixed-window comparisons use original 200 Hz event times; native-208 anchors are nearest samples, within 2.404 ms. Windows within 100 ms of recording boundaries are excluded, leaving **131 physically paired windows**. This isolates feature shifts from trigger/candidate changes; it does not demand exact cross-rate index equality. Reference-vs-float comparisons separately require exact indices. Dynamic event matching across strategies uses a predeclared 25 ms one-to-one tolerance, reports unmatched events explicitly, and is not a performance metric.

### Dynamic candidate comparison

| Case | Point | Strategy | Matched | Unmatched original | Unmatched variant | Matched class changes | Original positives | Variant positives |
|---|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sampling_rows)}

Strategy A retains all 132 original events per point within the 25 ms limit, with no matched class changes. Native-208 matches 121/132 per point; 11 original and 11 variant events are unmatched. All unmatched indices/times/labels and matched timing offsets are listed in `validation.json`. A small shift in a threshold crossing can select a different refractory edge, so continuous-value closeness is not candidate equivalence. Zero class flips among matched events does not erase unmatched candidates.

### Fixed physical windows: feature/model shifts

| Strategy / reconstruction | Feature | Max absolute shift | Mean absolute shift | RMS shift / training scale | Max shift / training scale |
|---|---|---:|---:|---:|---:|
{chr(10).join(distribution_rows)}

For Strategy A, largest fixed-window logit shift is **{shifts['A_linear']['decisions']['balanced_reference']['max_abs_logit_shift']:.9g}**; native-208 primary reconstruction is **{shifts['B_native208']['decisions']['balanced_reference']['max_abs_logit_shift']:.9g}**, and the alternative reconstruction is **{shifts['B_linear_reconstruction']['decisions']['balanced_reference']['max_abs_logit_shift']:.9g}**. All 131 fixed-window labels agree at both operating points in these three comparisons. The largest native-208 feature shift is about 0.069 training-scale units; low average changes do not prove distribution equivalence near thresholds.

Same trained logistic model: reasonable to **reuse provisionally as an experimental comparison**, because features keep physical units/durations and measured shifts are modest on this small corpus. It is not justified to certify native-208 as the frozen model/pipeline: candidate sets differ and decision scores shift materially compared with float-rounding errors. A larger locked-data equivalence study and real-sensor data are needed to determine whether reuse is acceptable. This study neither proves retraining is necessary nor authorizes it.

## D. Recommendation and remaining sensor validation

1. **Float32:** 264 decisions agree with exact triggers/bounds, but original numerical parity fails. Keep the mixed build as authority; qualify float against broader/near-boundary data and on-target compiler/math behavior before making it the deployment default.
2. **Input conversion:** use the independent adapter and an explicitly known/read-back range. g-valued XYZ remains the core interface. Never use the ADXL345 helper for this sensor.
3. **Full scale:** propose ±16 g initially, given recorded impact peaks above ±8 g and training headroom. Actual configuration remains unspecified, not silently defaulted.
4. **208 Hz consequence:** one-for-one feeding changes filters, derivative units and physical window/refractory times. A timing adapter or a declared new-rate pipeline is necessary.
5. **Preferred path:** Strategy A preserves the frozen 200 Hz processing/model contract and showed smaller shifts with full candidate retention here. Qualify the timestamp-aware resampler and anti-aliasing requirement on real data; the linear prototype is not automatically production-approved. Keep native-208 isolated unless intentionally accepting a new pipeline/version.
6. **Real sensor:** verify effective full scale/mode and raw decoding/axis signs; stationary ±1 g orientation checks; bias/scale/noise/temperature; high-impact saturation; sensor-internal bandwidth/filter/gravity retention; actual ODR/clock drift/jitter; FIFO order/gaps/timestamp wrap; interpolation/FIR spectral response and delay; reset/first-window behavior; paired raw sensor replay under both numeric builds; target math/FP contraction, runtime/stack/RAM and classifier margins. Actual placement/domain performance remains outside desktop equivalence.

`sensor_adapters/INPUT_CONTRACT.md` and `input_contract.json` distinguish the frozen algorithm input from proposed/unknown acquisition settings. Final hardware threshold is still unfrozen. The numerical/reference contract is retained, while the actual sensor configuration, production resampler and general float qualification remain explicit unresolved items.

## Build-test discrepancy found and fixed

This local Zig/Windows optimized build defines `NDEBUG=1`, which suppresses C `assert`. A new adapter test caught that through `-Werror` on assertion-only values. The new builds now use `-UNDEBUG`; host test sources reject assertion-disabled compilation. `tools/validate_host_pipeline.py` also now requests active assertions. This is a host-test build fix, not a detector bug; the earlier Python stream/index/feature comparisons were unaffected. The current same-recording mixed rebuild and native-208 tests execute the C assertions, including event sample counts/bounds. No genuine algorithm bug was found.

## Artifacts and reproduction

- `sensor_adapters/lsm6dsox_input.h/.c`: explicit-range portable unit adapter; `tests/lsm6dsox_input_test.c`: conversion checks.
- `experiments/sampling_rate/linear_resampler.py`: isolated causal interpolation prototype with timing/prefix tests.
- `experiments/sampling_rate/native208/*`: generated independent 208 C sources/parameters/host runner and provenance manifest.
- `tools/validate_sensor_contract.py`: same-input float replay, archived golden checks, native C/Python parity and strategy experiments.
- `tools/write_sensor_contract_report.py`: measured report and contract proposal.
- `outputs/sensor_contract_validation/validation.json`: all errors, mismatch counts, timing shifts, unmatched candidates and hashes; paired C CSVs, surrogate inputs and console output stay locally available. Original Python golden CSVs remain in `outputs/host_pc_validation`.

Run `python tools/validate_sensor_contract.py --compiler PATH`, then `python tools/write_sensor_contract_report.py`. Reproduction does not write frozen parameters or call fit. Experiment pass means its implementation checks passed; it does **not** mean float met strict parity or that either rate strategy is deployment-qualified.
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    contract = dict(
        frozen_algorithm='MCU_v1', algorithm_sample_rate_hz=200, input_units='g',
        input='AccelSample XYZ; finite raw acceleration including gravity; chronological uniform 5 ms grid',
        authoritative_precision='default mixed precision; reference remains unchanged',
        float_status='behavioral parity on 7 recordings/264 decisions; strict numeric parity failed; not default',
        sensor='ST LSM6DSOX', configured_effective_range_g=None, configured_odr_hz=None,
        recommended_initial_range_g=16, recommended_range_status='proposal; require actual selection/read-back',
        sensitivity_g_per_lsb={'2': .000061, '4': .000122, '8': .000244, '16': .000488},
        range_selection='explicit Lsm6dsoxRange; no default; not register bit encoding',
        nominal_nearest_odr_hz=208,
        timing_strategy='prefer qualified 208->200 resampling outside core',
        production_resampler='not selected/implemented; isolated linear prototype only',
        native208_status='host experiment only; modified preprocessing/windows, not MCU_v1',
        final_hardware_threshold=None,
        missing_data_policy='must explicitly detect discontinuity and validate stream reset; no silent padding/drops',
        source_datasheet=DATASHEET,
    )
    # Historical study snapshots must not overwrite the subsequently selected
    # deployment contract under sensor_adapters/.
    (OUT / 'input_contract_at_study.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    document = f'''# Detector / LSM6DSOX input contract

The validated core remains **MCU v1 at 200 Hz**, mixed precision by default.
`fall_detector_push_sample` consumes finite XYZ in **g, including gravity**, one
chronological sample per uniform 5 ms algorithm-grid interval. Detector state is
continuous across events and initialized only for a new independent stream.
The ADXL345-specific `fall_detector_push_counts` helper is for SisFall/ADXL345 only.

`lsm6dsox_counts_to_g` converts signed int16 XYZ using the **explicitly configured,
effective full-scale range**. No range is assumed by default. The enum represents
±2/±4/±8/±16 g, not the hardware FS register bit encoding. Sensitivities are
0.061/0.122/0.244/0.488 mg/LSB respectively; g is mg divided by 1000.
Confirm effective range/read-back, including XL_FS_MODE. Unknown ranges and null
outputs fail without modifying output. [ST datasheet, Tables 2 and 52]({DATASHEET}).

The actual project's configured range and ODR remain **unknown**. ±16 g is the
initial recommendation based on the current recorded impact peaks and training
range; it is not an implicit configuration or approved hardware default.

The sensor's nominal 208 Hz cannot be fed directly to the frozen detector as if
it were 200 Hz. Prefer an independently qualified, timestamp-aware 208->200
adapter. Acquisition timestamps/clock/FIFO order, anti-aliasing, output-grid
epoch, startup/delay, clock drift and discontinuity policy must be specified and
tested before production. The isolated linear resampler is a prototype, and
native-208 is a separately named experiment. No production rate adapter exists.

Float replay agrees in all 264 tested decisions but fails the unchanged strict
numerical tolerances. Keep mixed precision as the reference until broader data,
threshold-neighborhood behavior and target/compiler numerical behavior qualify
float. Do not retune thresholds to compensate for sensor/rate/precision changes.
The final hardware classifier operating point is still explicitly unfrozen.

See `input_contract.json` for machine-readable frozen, proposed and unknown
fields, and `outputs/sensor_contract_validation/REPORT.md` for measured evidence.
No Zephyr, nRF5340, register access, interrupts or BLE integration is included.
'''
    (OUT / 'INPUT_CONTRACT_AT_STUDY.md').write_text(document, encoding='utf-8')
    print('Wrote sensor study report and archived proposal snapshot.')


if __name__ == '__main__':
    main()
