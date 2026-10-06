# Fall-detection pipeline

This repository contains a portable C fall detector and a host-validated acquisition bridge for the selected LSM6DSOX prototype. The bridge adapts nominal 208 Hz sensor measurements to the frozen detector's exact 200 Hz input contract. The hardware driver and live target integration remain to be implemented and validated.

## End-to-end architecture

```text
LSM6DSOX: selected +/-16 g, nominal 208 Hz
        |
raw signed XYZ counts + acquisition/FIFO timestamp
        |
LSM6DSOX counts-to-g conversion
        |
timestamp-aware linear timing adapter: 208 -> 200 Hz
        |
uniform 200 Hz XYZ acceleration in g, gravity included
-------- frozen detector boundary --------------------
        |
causal 4th-order 5 Hz Butterworth + raw-input 0.5 Hz gravity EMA
        |
magnitude / gravity-aligned scalar streams
        |
magnitude/jerk rising-edge trigger + 1.5 s refractory
        |
100 pre-trigger + 200 trigger/post-trigger samples
        |
[ga_C2, jerk_abs_mean, ga_parallel_peak]
        |
frozen Logistic Regression score + selected threshold
        |
FALL / ADL (activities of daily living)
```

## Acquisition and logical time

The planned driver supplies signed LSM6DSOX XYZ counts, an acquisition/FIFO timestamp, and explicitly configured/read-back effective range and ODR. Register access, signed decoding, FIFO ordering, timestamp wrap extension and clock calibration belong to that driver. The current bridge accepts the selected effective +/-16 g and nominal 208 Hz configuration; it cannot verify hardware registers itself.

Counts are converted to g before rate adaptation. The timing adapter anchors a uniform 5 ms grid at the first valid measurement. Once a second measurement arrives, it generates grid samples bracketed by the previous and current measurements, using the same linear interpolation factor for all three axes. It neither extrapolates nor flushes an incomplete bracket at EOF.

Three times have different meanings:

- **Logical sample time:** the exact 200 Hz grid position used by the detector.
- **Physical measurement time:** the sensor's acquisition timestamp. The later bracketing measurement determines the earliest availability of an interpolated sample.
- **Processing time:** when software handles the data, potentially later because of FIFO, transport or scheduling delays.

The acquisition boundary is sensor-specific; its timing adapter is portable and independent of sensor registers. Linear interpolation satisfies the timing contract but is not an anti-aliasing guarantee. See the [acquisition README](acquisition/README.md) and [sensor input contract](sensor_adapters/INPUT_CONTRACT.md) for configuration and timing details.

## Frozen detector contract and processing

The detector accepts **XYZ acceleration in g, with gravity included, at an exact logical rate of 200 Hz**. It does not know the IMU model, original ODR, register configuration, raw count representation, or SPI/I2C/FIFO implementation.

Within each independent stream, causal Butterworth filters run continuously. A separate gravity EMA follows the **raw** input, providing the gravity direction. Filtered acceleration produces magnitude, signed gravity-parallel acceleration and perpendicular magnitude.

A rising edge of the magnitude/jerk candidate condition can accept a trigger, subject to a 1.5 s refractory interval. For trigger index `k`, the event window is `[k - 100, k + 200)`: exactly 300 samples, including the trigger and ending at `k + 199`. Classification is emitted when sample `k + 200` arrives; that arriving sample is excluded from the completed event. Events require full prehistory and posthistory, with no padding or EOF completion. Features accumulate as samples arrive, without retaining a full event buffer.

See the [embedded README](embedded/README.md) and [model freeze](outputs/mcu_v1_frozen/MODEL_FREEZE.md) for the API and frozen parameters.

## Features and classifier

The frozen feature order is:

1. `ga_C2`: peak acceleration magnitude perpendicular to the estimated gravity direction.
2. `jerk_abs_mean`: mean absolute rate of change of acceleration magnitude over the 299 differences within the event.
3. `ga_parallel_peak`: peak absolute gravity-parallel acceleration, including gravity.

The training StandardScaler is algebraically folded into the Logistic Regression weights and intercept. Runtime classification is a linear logit score followed by a threshold comparison; no scikit-learn or ML runtime is required on target. Existing operating points are available, but the final hardware operating threshold is not yet frozen.

## Main module boundaries

| Module | Responsibility |
|---|---|
| `sensor_adapters/lsm6dsox_input.*` | Explicit-range LSM6DSOX raw counts to g |
| `acquisition/timing_adapter.*` | Timestamp-aware interpolation onto the 200 Hz grid |
| `acquisition/acquisition.*` | Conversion/resampling orchestration, restart policy and bridge into the detector |
| `embedded/filters.*` | Causal Butterworth filtering and gravity EMA |
| `embedded/features.*` | Streaming feature extraction |
| `embedded/classifier.*` | Frozen Logistic Regression score and threshold comparison |
| `embedded/fall_detector.*` | Trigger/event state machine and detector orchestration |
| `embedded/model_params.h` | Frozen filter, trigger and classifier constants |

The main call flow is:

```text
driver (planned)
  -> acquisition_push_lsm6dsox_sample()
       -> counts-to-g
       -> timing_adapter_push()
            -> generated_sample() callback, for each 200 Hz output
                 -> fall_detector_push_sample()
                      -> preprocessing -> trigger -> features -> classifier
```

## Discontinuities

Duplicate/reversed timestamps and excessive gaps reset the stream and discard incomplete events. The adapter does not interpolate across rejected regions. Sensor reset, ODR change or full-scale change must be explicitly reported by the driver; input then remains blocked until fresh configuration/read-back and reconfiguration. Each restart initializes a new independent detector stream. Detailed restart semantics remain in the [acquisition documentation](acquisition/README.md) and tests.

## Validation status

Host comparisons of the Python causal reference and portable mixed/double C core matched trigger/event indices and classifier decisions, with numerical differences near floating-point rounding. Float32 passed behavioral regression on the current suite, including accepted triggers, event windows and labels, but **did not pass strict numerical parity** with the mixed-precision reference. See the [core verification](outputs/portable_c_verification/REPORT.md), [host pipeline validation](outputs/host_pc_validation/REPORT.md) and [float/sensor-contract validation](outputs/sensor_contract_validation/REPORT.md).

Strategy A, timestamp-aware 208 -> 200 Hz adaptation, is the selected prototype path. The integrated acquisition implementation passed host comparisons against the earlier Strategy-A behavior and a same-input standalone C detector oracle. The [acquisition validation report](outputs/acquisition_validation/REPORT.md) documents a tiny endpoint-rounding difference in the older interpolation formula and separate quantization/clipping effects when constructing raw-count proxy inputs. Equivalence claims apply to corresponding inputs, not arbitrary sensor representations.

These tests use reconstructed recordings and synthetic inputs. They do **not** validate real LSM6DSOX timestamp accuracy, sensor noise/bias/saturation, nRF5340 runtime/RAM/stack, power behavior or live hardware performance.

## Current prototype decisions

- Sensor: LSM6DSOX; selected effective accelerometer range: +/-16 g.
- Sensor ODR: nominal 208 Hz; detector rate: fixed 200 Hz.
- Rate adaptation: timestamp-aware linear interpolation.
- Detector numeric deployment candidate: float32; numerical reference: mixed/double precision.
- Final hardware classifier operating threshold: unresolved.

Actual hardware configuration/read-back and real sensor/target validation remain pending. The separate [native-208 Hz experiment](experiments/sampling_rate/README.md) is exploratory and is not the selected deployment architecture.
