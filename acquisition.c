#include "acquisition.h"

#include <string.h>

static bool valid_config(const AcquisitionConfig *config) {
    uint64_t period;
    if (config == NULL || config->effective_range != LSM6DSOX_RANGE_16G
        || config->configured_odr_hz != 208 || !config->timestamp_ticks_per_second
        || config->timestamp_ticks_per_second % 200 != 0
        || (config->operating_point != FALL_BALANCED_REFERENCE
            && config->operating_point != FALL_SENSITIVITY_CANDIDATE)) {
        return false;
    }
    period = config->timestamp_ticks_per_second / 200;
    return period <= UINT64_MAX / 2 && config->max_input_gap_ticks <= 2 * period
           && config->max_input_gap_ticks > config->timestamp_ticks_per_second / 208;
}

static bool restart_detector(AcquisitionState *state) {
    bool discarded = state->detector.active != 0;
    fall_detector_init_operating_point(&state->detector, state->config.operating_point);
    ++state->stream_id;
    return discarded;
}

bool acquisition_init(AcquisitionState *state, const AcquisitionConfig *config,
                       AcquisitionObserver observer, void *context) {
    memset(state, 0, sizeof(*state));
    state->observer = observer;
    state->observer_context = context;
    if (!valid_config(config)) {
        return false;
    }
    state->config = *config;
    state->configured = timing_adapter_init(&state->timing,
                                           config->timestamp_ticks_per_second / 200,
                                           config->max_input_gap_ticks);
    fall_detector_init_operating_point(&state->detector, config->operating_point);
    return state->configured;
}

bool acquisition_reconfigure(AcquisitionState *state, const AcquisitionConfig *config) {
    timing_adapter_reset(&state->timing);
    restart_detector(state);
    state->configured = false;
    if (!valid_config(config)) {
        return false;
    }
    state->config = *config;
    state->configured = timing_adapter_init(&state->timing,
                                           config->timestamp_ticks_per_second / 200,
                                           config->max_input_gap_ticks);
    fall_detector_init_operating_point(&state->detector, config->operating_point);
    return state->configured;
}

bool acquisition_notify_discontinuity(AcquisitionState *state, AcquisitionDiscontinuity reason) {
    /* All hardware changes use the same fail-closed policy. The acquisition
     * driver owns detection and effective configuration read-back. */
    (void)reason;
    timing_adapter_reset(&state->timing);
    state->configured = false;
    return restart_detector(state);
}

static void generated_sample(void *context, AccelSample sample,
                              uint64_t logical_timestamp, uint64_t bracket_timestamp) {
    AcquisitionState *state = context;
    AcquisitionOutput output;

    output.stream_id = state->stream_id;
    output.logical_index = (uint64_t)state->detector.next_index;
    output.logical_timestamp = logical_timestamp;
    output.bracket_timestamp = bracket_timestamp;
    output.sample = sample;
    output.result = fall_detector_push_sample(&state->detector, sample);
    if (state->observer != NULL) {
        state->observer(state->observer_context, &output, &state->detector);
    }
}

AcquisitionResult acquisition_push_lsm6dsox_sample(AcquisitionState *state,
                                                  int16_t raw_x, int16_t raw_y, int16_t raw_z,
                                                  uint64_t sensor_timestamp) {
    AcquisitionResult result = {TIMING_NOT_CONFIGURED, 0, false};
    TimingResult timing;
    AccelSample sample;

    if (!state->configured) {
        return result;
    }
    if (!lsm6dsox_counts_to_g(raw_x, raw_y, raw_z, state->config.effective_range, &sample)) {
        result.active_event_discarded = acquisition_notify_discontinuity(state, ACQUISITION_FULL_SCALE_CHANGE);
        return result;
    }
    timing = timing_adapter_push(&state->timing, sample, sensor_timestamp, generated_sample, state);
    result.status = timing.status;
    result.emitted = timing.emitted;
    if (timing.status != TIMING_BUFFERED && timing.status != TIMING_EMITTED) {
        /* Timing adapter emitted nothing on a rejected batch. It already
         * either cleared timing or retained the new gap/reversal anchor. */
        result.active_event_discarded = restart_detector(state);
    }
    return result;
}
