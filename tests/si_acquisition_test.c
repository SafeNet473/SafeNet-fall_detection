#include "acquisition.h"

#ifdef NDEBUG
#error "Host tests require -UNDEBUG"
#endif
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    AcquisitionOutput output;
    FallEvent event;
    unsigned count;
    unsigned decisions;
} Capture;

static void capture(void *context, const AcquisitionOutput *output,
                    const FallDetector *detector) {
    Capture *saved = context;
    saved->output = *output;
    saved->event = detector->event;
    ++saved->count;
    if (output->result != FALL_NO_DECISION) ++saved->decisions;
}

static void conversion_test(void) {
    AccelSample g = {7, 8, 9};
    const double invalid[] = {NAN, INFINITY, -INFINITY, DBL_MAX, -DBL_MAX};
    unsigned i, axis;
    assert(lsm6dsox_ms2_to_g(0, 9.80665, -19.6133, &g));
    assert(g.ax == 0 && g.ay == 1 && g.az == -2);
    assert(lsm6dsox_ms2_to_g(4.903325, -2.4516625, 0, &g));
    assert(g.ax == .5f && g.ay == -.25f && g.az == 0);
    assert(!lsm6dsox_ms2_to_g(0, 0, 0, NULL));
    for (i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        for (axis = 0; axis < 3; ++axis) {
            double xyz[3] = {0, 0, 9.80665};
            xyz[axis] = invalid[i];
            assert(!lsm6dsox_ms2_to_g(xyz[0], xyz[1], xyz[2], &g));
            assert(g.ax == .5f && g.ay == -.25f && g.az == 0);
        }
    }
}

static void replay_test(FallOperatingPoint point) {
    AcquisitionConfig config = {LSM6DSOX_RANGE_16G, 208, 52000000, 375000, point};
    AcquisitionState raw, si;
    Capture a = {0}, b = {0};
    unsigned i, j;
    assert(acquisition_init(&raw, &config, capture, &a));
    assert(acquisition_init(&si, &config, capture, &b));
    for (i = 0; i < 3000; ++i) {
        /* Repeated isolated pulses after full prehistory, producing completed
         * events. SI inputs represent exactly the same physical g samples. */
        int16_t x = (i % 600 >= 200 && i % 600 < 240) ? 6000 : 0;
        AccelSample g;
        uint64_t t = UINT64_C(9007199254740993) + (uint64_t)i * 250000;
        AcquisitionResult ra, rb;
        assert(lsm6dsox_counts_to_g(x, 0, 2049, LSM6DSOX_RANGE_16G, &g));
        ra = acquisition_push_lsm6dsox_sample(&raw, x, 0, 2049, t);
        rb = acquisition_push_lsm6dsox_ms2_sample(&si, (double)g.ax * 9.80665,
                                                (double)g.ay * 9.80665,
                                                (double)g.az * 9.80665, t);
        assert(ra.status == rb.status && ra.emitted == rb.emitted);
        assert(ra.active_event_discarded == rb.active_event_discarded);
        assert(a.count == b.count && a.decisions == b.decisions);
        if (!a.count) continue;
        assert(a.output.sample.ax == b.output.sample.ax);
        assert(a.output.sample.ay == b.output.sample.ay);
        assert(a.output.sample.az == b.output.sample.az);
        assert(a.output.logical_timestamp == b.output.logical_timestamp);
        assert(a.output.bracket_timestamp == b.output.bracket_timestamp);
        assert(a.output.logical_index == b.output.logical_index);
        assert(a.output.result == b.output.result);
        assert(raw.detector.trace.accepted_trigger == si.detector.trace.accepted_trigger);
        assert(raw.detector.trace.magnitude == si.detector.trace.magnitude);
        if (a.output.result != FALL_NO_DECISION) {
            assert(a.event.trigger == b.event.trigger && a.event.start == b.event.start);
            assert(a.event.end == b.event.end && a.event.decision == b.event.decision);
            for (j = 0; j < 3; ++j) assert(a.event.features[j] == b.event.features[j]);
            assert(a.event.score == b.event.score && a.event.result == b.event.result);
        }
    }
    assert(a.decisions >= 4);
    printf("SI/raw same-input replay: %u outputs, %u matching decisions\n", a.count, a.decisions);
    si.detector.active = 1;
    {
        uint64_t stream = si.stream_id;
        AcquisitionResult r = acquisition_push_lsm6dsox_ms2_sample(&si, NAN, 0, 9.80665, 0);
        assert(r.status == TIMING_INVALID_SAMPLE && r.emitted == 0 && r.active_event_discarded);
        assert(si.stream_id == stream + 1 && si.detector.next_index == 0);
        assert(!si.timing.has_previous && si.configured);
        r = acquisition_push_lsm6dsox_ms2_sample(&si, 0, 0, 9.80665, 0);
        assert(r.status == TIMING_BUFFERED && !r.emitted);
        r = acquisition_push_lsm6dsox_ms2_sample(&si, 0, 0, 9.80665, 250000);
        assert(r.status == TIMING_EMITTED && b.output.sample.az == 1);
        si.detector.active = 1;
        r = acquisition_push_lsm6dsox_ms2_sample(&si, 0, 0, 9.80665, 250000);
        assert(r.status == TIMING_DUPLICATE_TIMESTAMP && r.active_event_discarded);
        acquisition_notify_discontinuity(&si, ACQUISITION_SENSOR_RESET);
        r = acquisition_push_lsm6dsox_ms2_sample(&si, 0, 0, 9.80665, 0);
        assert(r.status == TIMING_NOT_CONFIGURED && !r.emitted);
    }
}

int main(void) {
    conversion_test();
    replay_test(FALL_BALANCED_REFERENCE);
    replay_test(FALL_SENSITIVITY_CANDIDATE);
    puts("SI conversion, replay, invalid input and restart checks passed");
    return 0;
}
