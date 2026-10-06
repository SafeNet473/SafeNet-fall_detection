#include "timing_adapter.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

bool timing_adapter_init(TimingAdapter *state, uint64_t period_ticks, uint64_t max_gap_ticks) {
    memset(state, 0, sizeof(*state));
    if (!period_ticks || period_ticks > UINT64_MAX / 2 || !max_gap_ticks
        || max_gap_ticks > 2 * period_ticks) {
        return false;
    }
    state->period_ticks = period_ticks;
    state->max_gap_ticks = max_gap_ticks;
    state->configured = true;
    return true;
}

void timing_adapter_reset(TimingAdapter *state) {
    state->has_previous = false;
    state->epoch_timestamp = 0;
    state->previous_timestamp = 0;
    state->next_output_timestamp = 0;
}

static void anchor(TimingAdapter *state, AccelSample sample, uint64_t timestamp) {
    state->previous = sample;
    state->previous_timestamp = timestamp;
    state->epoch_timestamp = timestamp;
    state->next_output_timestamp = timestamp;
    state->has_previous = true;
}

static float interpolate(float previous, float current, double fraction) {
    /* Use double for interpolation even with a float detector. This matches
     * the validated Strategy-A input boundary and rounds only the output. */
    double start = previous;
    return (float)(start + fraction * ((double)current - start));
}

TimingResult timing_adapter_push(TimingAdapter *state, AccelSample sample,
                                 uint64_t timestamp, TimingOutputFn output, void *context) {
    TimingResult result = {TIMING_BUFFERED, 0};
    uint64_t interval;
    uint64_t check_timestamp;

    if (!state->configured || output == NULL) {
        result.status = TIMING_NOT_CONFIGURED;
        return result;
    }
    if (!isfinite(sample.ax) || !isfinite(sample.ay) || !isfinite(sample.az)) {
        timing_adapter_reset(state);
        result.status = TIMING_INVALID_SAMPLE;
        return result;
    }
    if (!state->has_previous) {
        anchor(state, sample, timestamp);
        return result;
    }
    if (timestamp == state->previous_timestamp) {
        timing_adapter_reset(state);
        result.status = TIMING_DUPLICATE_TIMESTAMP;
        return result;
    }
    if (timestamp < state->previous_timestamp) {
        timing_adapter_reset(state);
        anchor(state, sample, timestamp);
        result.status = TIMING_TIMESTAMP_REVERSAL;
        return result;
    }
    interval = timestamp - state->previous_timestamp;
    if (interval > state->max_gap_ticks) {
        timing_adapter_reset(state);
        anchor(state, sample, timestamp);
        result.status = TIMING_EXCESSIVE_GAP;
        return result;
    }

    /* Check the entire bounded batch before emitting: overflow never causes
     * a partially emitted batch followed by a discontinuity. */
    check_timestamp = state->next_output_timestamp;
    while (check_timestamp <= timestamp) {
        if (UINT64_MAX - check_timestamp < state->period_ticks) {
            timing_adapter_reset(state);
            result.status = TIMING_TIMESTAMP_OVERFLOW;
            return result;
        }
        check_timestamp += state->period_ticks;
    }

    while (state->next_output_timestamp <= timestamp) {
        AccelSample generated;
        double fraction = (double)(state->next_output_timestamp - state->previous_timestamp)
                          / (double)interval;

        /* next_output lies in [previous_timestamp, timestamp]. */
        /* At a coincident endpoint, retain the original value exactly. The
         * general expression can cancel a tiny component near a sine zero. */
        if (state->next_output_timestamp == state->previous_timestamp) {
            generated = state->previous;
        } else if (state->next_output_timestamp == timestamp) {
            generated = sample;
        } else {
            generated.ax = interpolate(state->previous.ax, sample.ax, fraction);
            generated.ay = interpolate(state->previous.ay, sample.ay, fraction);
            generated.az = interpolate(state->previous.az, sample.az, fraction);
        }
        output(context, generated, state->next_output_timestamp, timestamp);
        state->next_output_timestamp += state->period_ticks;
        ++result.emitted;
    }
    state->previous = sample;
    state->previous_timestamp = timestamp;
    if (result.emitted) {
        result.status = TIMING_EMITTED;
    }
    return result;
}
