# Timestamp-aware acquisition boundary

The selected architecture is LSM6DSOX **effective ±16 g, nominal 208 Hz** ->
sensor conversion -> timestamp-aware linear interpolation -> frozen **200 Hz**
detector. All files under `embedded/` remain unchanged. Mixed precision remains
the numerical reference; `FALL_USE_FLOAT` remains the deployment candidate.

The implementation is C99, caller-owned state, no allocation, OS, SDK, register
access, BLE, alarm delivery or driver dependency. Build these files alongside
the existing core:

```text
sensor_adapters/lsm6dsox_input.c
acquisition/timing_adapter.c
acquisition/acquisition.c
embedded/{filters,features,classifier,fall_detector}.c
```

## Target-facing use

### Nordic/Zephyr acceleration already in m/s²

Zephyr accelerometer channels return SI acceleration in **m/s²**, rather than
signed raw register counts. Use `acquisition_push_lsm6dsox_ms2_sample()` for this
representation. It divides each axis by **9.80665 m/s² per g**, then uses the
same timestamp-aware adapter and frozen detector. For example, +9.80665 m/s²
becomes +1 g. Gravity stays included; no magnitude normalization is performed.

After a successful driver fetch/read, with `struct sensor_value accel[3]`
obtained from `SENSOR_CHAN_ACCEL_XYZ`, call:

```c
/* Target code includes <zephyr/drivers/sensor.h> and "acquisition.h".
 * state was initialized with confirmed range/ODR and timestamp units.
 * acquisition_timestamp is the corresponding measurement/FIFO time. */
AcquisitionResult result = acquisition_push_lsm6dsox_ms2_sample(
    &state,
    sensor_value_to_double(&accel[0]),
    sensor_value_to_double(&accel[1]),
    sensor_value_to_double(&accel[2]),
    acquisition_timestamp);
```

Use the conversion helper on the complete `sensor_value`; reading only `val1`
loses the fractional part. Do not cast these SI readings to `int16_t`, apply
the raw-count sensitivity, or divide by 9.80665 again before this call.
See the [Zephyr sensor API](https://docs.zephyrproject.org/latest/doxygen/html/group__sensor__interface.html).

The existing `acquisition_push_lsm6dsox_sample()` remains for actual signed
register counts. Choose one representation for a stream. Both entry points
retain the explicit effective range/ODR, timestamp and discontinuity contract.
Nonfinite or unrepresentable SI values return `TIMING_INVALID_SAMPLE`, clear
timing/detector history and discard an incomplete event; the next valid sample
anchors a new stream. Configuration remains valid unless hardware changes.

Check stationary orientations on the board: acceleration magnitude should be
about 9.81 m/s² before conversion and about 1 g afterward, with signs/axes
following the mounting. This sanity check does not replace sensor validation.

### Raw-register-count input

After the driver has configured and read back the **effective** range/ODR, pass
that configuration explicitly. The range enum is physical g, not register bits.
The adapter retains all four conversion modes; this selected acquisition path
accepts only ±16 g and nominal 208 Hz. Incompatible configurations fail closed.
Choose one existing classifier operating point explicitly; the final hardware
threshold is still a separate unresolved choice.

Example using nanosecond acquisition timestamps:

```c
AcquisitionState state; /* Allocate persistently, not inside an ISR stack frame. */
AcquisitionConfig config = {
    .effective_range = LSM6DSOX_RANGE_16G, /* Confirmed by driver read-back. */
    .configured_odr_hz = 208,
    .timestamp_ticks_per_second = UINT64_C(1000000000),
    .max_input_gap_ticks = UINT64_C(7211538), /* Explicit 1.5 nominal-period policy. */
    .operating_point = FALL_BALANCED_REFERENCE /* Host example, not hardware selection. */
};

if (!acquisition_init(&state, &config, observer, observer_context)) {
    /* Configuration rejected: do not feed samples. */
}

/* Called in acquisition order, with a decoded signed XYZ sample and its
 * acquisition/FIFO timestamp. ISR arrival time is not a substitute. */
AcquisitionResult result = acquisition_push_lsm6dsox_sample(
    &state, raw_x, raw_y, raw_z, acquisition_timestamp_ns);
/* Inspect result.status and result.active_event_discarded. */
```

The driver is responsible for read-back, signed decoding, timestamp extension
across hardware counter wrap, clock/tick calibration, FIFO ordering, and timely
change notifications. No code here independently verifies a sensor register.
Convert sensor timestamps into any monotonic integer tick unit whose tick rate
is divisible by 200. Nanoseconds work; actual sensor tick units may work directly
if they meet that requirement. Do not use raw unextended wrapping timestamps.

The host replay uses a **52,000,000 tick/s** timebase solely to represent ideal
208 Hz (250,000 ticks) and 200 Hz (260,000 ticks) exactly. It is not a required
hardware clock. Tests also exercise nanoseconds, timestamp jitter, and epochs
larger than the exact-integer range of double. Only relative timestamp
differences are converted to floating point.

## Startup and interpolation

The first valid measurement establishes `epoch_timestamp`; output timestamps
are `epoch + k*period_ticks`. Hold that sample until a second valid measurement
arrives. Then emit any grid positions in the closed bracket, including k=0.
There is no earlier output and no extrapolation. The first sample is delayed,
not discarded. An isolated one-sample stream emits nothing. EOF does not flush.

For each bracket, use one double interpolation factor for all XYZ components,
then cast the final XYZ to `AccelSample` float once. Exact endpoint coincidences
copy the original sample directly to avoid cancellation near zero. Double
interpolation is deliberate in both core precision builds to retain a common
input contract. Output sample k is passed immediately, synchronously to the
unchanged `fall_detector_push_sample`.

`logical_timestamp` is the uniform output-grid time; `bracket_timestamp` is the
later bracketing sample's **measurement time**, not processing time. It gives
the earliest data-availability time before FIFO/transport/scheduling delay.
Actual ISR/wall-clock processing time can be later and is managed by the caller.
Observer `logical_index` matches detector sample indices within `stream_id`.
On a decision, `detector->event` is valid for the callback. Copy retained data;
the callback cannot reenter or reconfigure the state. Serialize all ownership
of each state; the API is not internally thread-safe.

Nominal startup delay for k=0 is one input interval (4.807692 ms); steady-state
bracketing delay is at most 4.615385 ms under ideal 208 Hz timing. Jitter changes
availability, while grid timestamps remain exact. No target runtime is claimed.
Linear interpolation is not an anti-aliasing guarantee; sensor bandwidth and
real input spectra still need validation. No extra filtering was inserted.

## Explicit discontinuity policy

The max-gap threshold is part of the caller's explicit configuration. The
example uses 1.5 nominal input periods (about 7.211538 ms), so an ideal missing
208 Hz frame creates a rejected gap. A different documented policy can be
selected, bounded by two output periods to cap per-call work. Never silently
increase the threshold to cover missing data.

| Input/change | Action |
|---|---|
| Duplicate timestamp | Reject current sample; clear timing and reset detector; next valid sample anchors a new stream |
| Timestamp reversal | Reset detector/timing; current finite sample becomes a new anchor; emit nothing on this call |
| Gap exceeding configured limit | Same restart/anchor policy; do not bridge the gap |
| Nonfinite g sample in independent adapter | Reject it and clear timing; acquisition raw-int16 conversion cannot create one |
| Timestamp arithmetic overflow | Reject entire batch before any emission; clear timing/reset detector |
| Sensor reset, ODR change, full-scale change | Driver calls `acquisition_notify_discontinuity`; block further input until explicit read-back/reconfiguration |
| `acquisition_reconfigure` | Always discard pending event/history and restart; invalid configuration remains blocked |

Every restart advances `stream_id`; logical index restarts at zero and filter,
gravity, refractory and event history are freshly initialized. This is a new
independent stream, not an event-boundary filter reset. Incomplete events are
discarded. The gap/reversal sample is held as the new first sample and emitted
only when the next bracket arrives. Duplicate/invalid/overflow samples are
discarded entirely. Long gaps are never filled. Absolute timestamps may go
backward across stream IDs after a sensor reset; ordering is required within
each stream. Driver-reported hardware changes require fresh configuration.

## Verification and resources

Run `python tools/validate_acquisition.py --compiler PATH`, then
`python tools/write_acquisition_report.py`. Strict builds disable FP contraction,
keep fast-math off, and enable assertions with `-UNDEBUG` in host tests.

The report in `outputs/acquisition_validation/REPORT.md` includes mixed/float
state sizes, per-case errors, exact timestamp/value/core comparisons, and the
separate sensor-quantization/clipping effects. Archived unquantized Strategy-A
detector streams/events match bitwise; mathematical endpoint samples match
exactly, with the old prototype's near-zero cancellation disclosed separately.

Do not take the host tests as validation of a real IMU, acquisition timestamp
accuracy, sensor spectrum, target math/runtime, or power. These modules prepare
the interface for target integration; no nRF5340/Zephyr driver is included.
