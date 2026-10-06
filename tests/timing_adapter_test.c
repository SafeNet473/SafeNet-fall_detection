#include "acquisition.h"

#ifdef NDEBUG
#error "Host tests require -UNDEBUG"
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { INPUT_PERIOD = 250000, OUTPUT_PERIOD = 260000, MAX_GAP = 375000 };
/* A 52 MHz integer timebase represents ideal 208 and 200 Hz exactly. */
static const uint64_t TICKS_PER_SECOND = 52000000;

typedef struct {
    unsigned count;
    uint64_t epoch;
    uint64_t previous_output;
    uint64_t previous_input;
    uint64_t current_input;
    AccelSample previous_sample;
    AccelSample current_sample;
    unsigned coincidences;
    unsigned mode;
    double max_error;
    double analytic_max_error;
} Capture;

static void verify_output(void *context, AccelSample value, uint64_t logical, uint64_t bracket) {
    Capture *capture = context;
    double weight = (double)(logical - capture->previous_input)
                    / (double)(capture->current_input - capture->previous_input);
    float expected[3];
    float actual[3] = {value.ax, value.ay, value.az};
    int axis;

    assert(logical == capture->epoch + (uint64_t)capture->count * OUTPUT_PERIOD);
    if (capture->count) {
        assert(logical - capture->previous_output == OUTPUT_PERIOD);
    }
    assert(logical >= capture->previous_input && logical <= capture->current_input);
    assert(bracket == capture->current_input && bracket >= logical);
    assert(weight >= 0 && weight <= 1);
    expected[0] = (float)((double)capture->previous_sample.ax
                         + weight * ((double)capture->current_sample.ax - capture->previous_sample.ax));
    expected[1] = (float)((double)capture->previous_sample.ay
                         + weight * ((double)capture->current_sample.ay - capture->previous_sample.ay));
    expected[2] = (float)((double)capture->previous_sample.az
                         + weight * ((double)capture->current_sample.az - capture->previous_sample.az));
    if (logical == capture->previous_input) {
        expected[0] = capture->previous_sample.ax;
        expected[1] = capture->previous_sample.ay;
        expected[2] = capture->previous_sample.az;
    } else if (logical == capture->current_input) {
        expected[0] = capture->current_sample.ax;
        expected[1] = capture->current_sample.ay;
        expected[2] = capture->current_sample.az;
    }
    for (axis = 0; axis < 3; ++axis) {
        double error = fabs((double)actual[axis] - expected[axis]);
        assert(actual[axis] == expected[axis]);
        if (error > capture->max_error) {
            capture->max_error = error;
        }
    }
    {
        double time = (double)(logical - capture->epoch) / (double)TICKS_PER_SECOND;
        double interval_seconds = (double)(capture->current_input - capture->previous_input)
                                  / (double)TICKS_PER_SECOND;
        double analytic[3];
        double frequency[3] = {.5, 5., 2.};
        double amplitude[3] = {1., 1., 2.};
        if (capture->mode == 0) {
            analytic[0] = 1.25; analytic[1] = -2.5; analytic[2] = 3.75;
        } else if (capture->mode == 1) {
            analytic[0] = .3 * time; analytic[1] = -2 * time + 1; analytic[2] = time + .5;
        } else {
            analytic[0] = sin(2 * 3.141592653589793 * .5 * time);
            analytic[1] = cos(2 * 3.141592653589793 * 5 * time);
            analytic[2] = 2 * sin(2 * 3.141592653589793 * 2 * time) + .3;
        }
        for (axis = 0; axis < 3; ++axis) {
            double error = fabs((double)actual[axis] - analytic[axis]);
            double bound = 3e-6; /* Input/output float rounding on this ramp range. */
            if (capture->mode == 2) {
                double omega = 2 * 3.141592653589793 * frequency[axis];
                /* Independent linear-interpolation remainder: max |f''|*dt²/8. */
                bound += amplitude[axis] * omega * omega * interval_seconds * interval_seconds / 8;
            }
            assert(error <= bound);
            if (error > capture->analytic_max_error) capture->analytic_max_error = error;
        }
    }
    if (logical == capture->current_input) {
        ++capture->coincidences;
        assert(value.ax == capture->current_sample.ax);
        assert(value.ay == capture->current_sample.ay);
        assert(value.az == capture->current_sample.az);
    }
    capture->previous_output = logical;
    ++capture->count;
}

static void signal_test(unsigned mode, bool jitter) {
    TimingAdapter adapter;
    Capture capture;
    unsigned index;
    uint64_t previous_time = 0;
    AccelSample previous_value = {0, 0, 0};

    memset(&capture, 0, sizeof(capture));
    capture.mode = mode;
    capture.epoch = UINT64_C(9007199254740993); /* Not exactly representable in double. */
    assert(timing_adapter_init(&adapter, OUTPUT_PERIOD, MAX_GAP));
    for (index = 0; index <= 2600; ++index) {
        int64_t offset = jitter && index && index % 26 ? ((int64_t)(index % 3) - 1) * 10000 : 0;
        uint64_t timestamp = capture.epoch + (uint64_t)((int64_t)index * INPUT_PERIOD + offset);
        double time = (double)(timestamp - capture.epoch) / (double)TICKS_PER_SECOND;
        AccelSample value;
        TimingResult result;

        if (mode == 0) {
            value.ax = 1.25f; value.ay = -2.5f; value.az = 3.75f;
        } else if (mode == 1) {
            value.ax = (float)(.3 * time);
            value.ay = (float)(-2 * time + 1);
            value.az = (float)(time + .5);
        } else {
            value.ax = (float)sin(2 * 3.141592653589793 * .5 * time);
            value.ay = (float)cos(2 * 3.141592653589793 * 5 * time);
            value.az = (float)(2 * sin(2 * 3.141592653589793 * 2 * time) + .3);
        }
        capture.previous_input = previous_time;
        capture.current_input = timestamp;
        capture.previous_sample = previous_value;
        capture.current_sample = value;
        result = timing_adapter_push(&adapter, value, timestamp, verify_output, &capture);
        if (index == 0) {
            assert(result.status == TIMING_BUFFERED && result.emitted == 0 && capture.count == 0);
        } else {
            assert(result.status == TIMING_BUFFERED || result.status == TIMING_EMITTED);
        }
        previous_time = timestamp;
        previous_value = value;
    }
    assert(capture.count == 2501); /* 100 cycles of 26/25 intervals plus grid origin */
    if (jitter) {
        /* Jitter may also move otherwise off-grid samples onto the grid. */
        assert(capture.coincidences >= 100);
    } else {
        assert(capture.coincidences == 100);
    }
    assert(capture.previous_output == capture.epoch + 100 * UINT64_C(6500000));
    printf("signal %u jitter=%d: %u outputs, %u exact coincidences, interpolation error %.17g, analytic error %.17g\n",
           mode, jitter, capture.count, capture.coincidences, capture.max_error, capture.analytic_max_error);
}

typedef struct { uint64_t count, last; } LongCapture;
static void count_nanosecond_output(void *context, AccelSample value, uint64_t logical, uint64_t bracket) {
    LongCapture *capture = context;
    assert(logical == UINT64_C(1000000000) + capture->count * UINT64_C(5000000));
    assert(bracket >= logical);
    assert(value.ax == 1.25f && value.ay == -2.5f && value.az == 3.75f);
    capture->last = logical;
    ++capture->count;
}

static void long_duration_test(void) {
    TimingAdapter adapter;
    LongCapture capture = {0, 0};
    AccelSample value = {1.25f, -2.5f, 3.75f};
    uint64_t index;
    assert(timing_adapter_init(&adapter, 5000000, 7211538));
    for (index = 0; index <= UINT64_C(208) * 3600; ++index) {
        uint64_t timestamp = UINT64_C(1000000000) + index * UINT64_C(1000000000) / 208;
        TimingResult result = timing_adapter_push(&adapter, value, timestamp, count_nanosecond_output, &capture);
        assert(result.status == TIMING_BUFFERED || result.status == TIMING_EMITTED);
    }
    assert(capture.count == UINT64_C(200) * 3600 + 1);
    assert(capture.last == UINT64_C(3601000000000));
    puts("one-hour nanosecond timing: 748801 inputs, 720001 outputs, zero grid drift");
}

static void ignore_output(void *context, AccelSample value, uint64_t logical, uint64_t bracket) {
    unsigned *count = context;
    (void)value; (void)logical; (void)bracket;
    ++*count;
}

static void discontinuity_test(void) {
    TimingAdapter adapter;
    AccelSample value = {1, -2, 3};
    unsigned outputs = 0;
    TimingResult result;

    assert(!timing_adapter_init(&adapter, 0, MAX_GAP));
    assert(!timing_adapter_init(&adapter, OUTPUT_PERIOD, 2 * OUTPUT_PERIOD + 1));
    assert(timing_adapter_init(&adapter, OUTPUT_PERIOD, MAX_GAP));
    timing_adapter_push(&adapter, value, 100, ignore_output, &outputs);
    result = timing_adapter_push(&adapter, value, 100, ignore_output, &outputs);
    assert(result.status == TIMING_DUPLICATE_TIMESTAMP && !adapter.has_previous && outputs == 0);
    timing_adapter_push(&adapter, value, 1000, ignore_output, &outputs);
    result = timing_adapter_push(&adapter, value, 900, ignore_output, &outputs);
    assert(result.status == TIMING_TIMESTAMP_REVERSAL && adapter.epoch_timestamp == 900 && outputs == 0);
    result = timing_adapter_push(&adapter, value, 900 + MAX_GAP + 1, ignore_output, &outputs);
    assert(result.status == TIMING_EXCESSIVE_GAP && outputs == 0 && adapter.has_previous);
    assert(adapter.epoch_timestamp == 900 + MAX_GAP + 1);
    value.ax = NAN;
    result = timing_adapter_push(&adapter, value, 1000000, ignore_output, &outputs);
    assert(result.status == TIMING_INVALID_SAMPLE && !adapter.has_previous && outputs == 0);
    value.ax = 1;
    timing_adapter_push(&adapter, value, UINT64_MAX - OUTPUT_PERIOD / 2, ignore_output, &outputs);
    result = timing_adapter_push(&adapter, value, UINT64_MAX - 1, ignore_output, &outputs);
    assert(result.status == TIMING_TIMESTAMP_OVERFLOW && outputs == 0 && !adapter.has_previous);
    timing_adapter_reset(&adapter);
    assert(adapter.configured && !adapter.has_previous);
    puts("duplicate/reversal/gap/nonfinite/overflow/reset checks passed");
}

static void acquisition_test(void) {
    AcquisitionConfig config = {LSM6DSOX_RANGE_16G, 208, 52000000, MAX_GAP, FALL_BALANCED_REFERENCE};
    AcquisitionState state;
    AcquisitionResult result;
    unsigned reason;

    assert(acquisition_init(&state, &config, NULL, NULL));
    result = acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, 0);
    assert(result.emitted == 0 && state.detector.next_index == 0);
    result = acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, INPUT_PERIOD);
    assert(result.emitted == 1 && state.detector.next_index == 1);
    /* Controlled state setup exercises event-discard policy, independent of
     * whether a synthetic waveform happens to pass the frozen trigger. */
    state.detector.active = 1;
    result = acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, INPUT_PERIOD);
    assert(result.status == TIMING_DUPLICATE_TIMESTAMP && result.active_event_discarded);
    assert(state.detector.next_index == 0 && !state.detector.active && !state.timing.has_previous);
    acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, 1000);
    state.detector.active = 1;
    result = acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, 900);
    assert(result.status == TIMING_TIMESTAMP_REVERSAL && result.active_event_discarded);
    assert(state.timing.epoch_timestamp == 900 && state.detector.next_index == 0);
    state.detector.active = 1;
    result = acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, 900 + MAX_GAP + 1);
    assert(result.status == TIMING_EXCESSIVE_GAP && result.active_event_discarded);
    assert(state.detector.next_index == 0);
    for (reason = ACQUISITION_SENSOR_RESET; reason <= ACQUISITION_FULL_SCALE_CHANGE; ++reason) {
        state.detector.active = 1;
        assert(acquisition_notify_discontinuity(&state, (AcquisitionDiscontinuity)reason));
        assert(!state.configured && !state.detector.active);
        result = acquisition_push_lsm6dsox_sample(&state, 0, 0, 2049, 0);
        assert(result.status == TIMING_NOT_CONFIGURED && result.emitted == 0);
        assert(acquisition_reconfigure(&state, &config));
    }
    config.configured_odr_hz = 104;
    assert(!acquisition_reconfigure(&state, &config));
    config.configured_odr_hz = 208;
    config.effective_range = LSM6DSOX_RANGE_8G;
    assert(!acquisition_reconfigure(&state, &config));
    config.effective_range = LSM6DSOX_RANGE_16G;
    config.timestamp_ticks_per_second = 1000000001;
    assert(!acquisition_reconfigure(&state, &config));
    config.timestamp_ticks_per_second = 1000000000;
    config.max_input_gap_ticks = 7211538;
    assert(acquisition_reconfigure(&state, &config));
    puts("acquisition startup/event discard/configuration notifications passed");
}

int main(void) {
    unsigned mode;
    for (mode = 0; mode < 3; ++mode) {
        signal_test(mode, false);
        signal_test(mode, true);
    }
    discontinuity_test();
    long_duration_test();
    acquisition_test();
    printf("sizeof: timing=%zu config=%zu detector=%zu acquisition=%zu\n",
           sizeof(TimingAdapter), sizeof(AcquisitionConfig), sizeof(FallDetector), sizeof(AcquisitionState));
    return 0;
}
