#include "fall_detector.h"

#include <math.h>
#include <string.h>

/* These sample counts define the frozen [trigger - 100, trigger + 200) window. */
enum {
    PRE_TRIGGER_SAMPLES = 100,
    POST_TRIGGER_SAMPLES = 200,
    REFRACTORY_SAMPLES = 300,
    SAMPLE_RATE_HZ = 200
};

static FallReal vector_magnitude(const FallReal vector[3]) {
    return (FallReal)sqrt((double)(vector[0] * vector[0]
                                  + vector[1] * vector[1]
                                  + vector[2] * vector[2]));
}

void fall_detector_init_operating_point(FallDetector *detector, FallOperatingPoint point) {
    memset(detector, 0, sizeof(*detector));
    detector->last_trigger = -REFRACTORY_SAMPLES;
    detector->logit_threshold = fall_classifier_threshold(point);
}

void fall_detector_init(FallDetector *detector) {
    fall_detector_init_operating_point(detector, FALL_BALANCED_REFERENCE);
}

FallResult fall_detector_push_sample(FallDetector *detector, AccelSample sample) {
    FallReal raw[3] = {sample.ax, sample.ay, sample.az};
    FallTrace *trace = &detector->trace;
    FallScalars scalars;
    FallResult result = FALL_NO_DECISION;
    int64_t now = detector->next_index++;
    int axis;
    FallReal gravity_magnitude;

    /* 1. Update continuous filters, then resolve acceleration along gravity. */
    fall_filters_push(&detector->filters, raw, trace->filtered, trace->gravity);
    gravity_magnitude = vector_magnitude(trace->gravity);
    if (gravity_magnitude < (FallReal)1e-8) {
        gravity_magnitude = (FallReal)1e-8;
    }

    trace->parallel = 0;
    for (axis = 0; axis < 3; ++axis) {
        /* The float cast is part of the Python reference's sample interface. */
        trace->acceleration[axis] = (FallReal)(float)trace->filtered[axis];
        trace->unit[axis] = trace->gravity[axis] / gravity_magnitude;
        trace->parallel += trace->acceleration[axis] * trace->unit[axis];
    }
    for (axis = 0; axis < 3; ++axis) {
        trace->residual[axis] = trace->acceleration[axis]
                               - trace->parallel * trace->unit[axis];
    }
    trace->magnitude = vector_magnitude(trace->acceleration);
    trace->perpendicular = vector_magnitude(trace->residual);

    /* 2. Detect a current-sample rising edge outside the refractory interval. */
    trace->jerk = now
        ? (FallReal)fabs((double)((trace->magnitude - detector->previous_magnitude)
                                 * (FallReal)SAMPLE_RATE_HZ))
        : 0;
    detector->previous_magnitude = trace->magnitude;
    trace->high = trace->magnitude >= (FallReal)1.1 || trace->jerk >= (FallReal)5;
    trace->rising_edge = trace->high && !detector->previous_high;
    trace->accepted_trigger = trace->rising_edge
                              && now - detector->last_trigger >= REFRACTORY_SAMPLES;
    /* Track suppressed edges too: they must not be postponed until release. */
    detector->previous_high = trace->high;

    scalars.magnitude = trace->magnitude;
    scalars.perpendicular = trace->perpendicular;
    scalars.parallel_abs = (FallReal)fabs((double)trace->parallel);

    /* 3. Finish a due event BEFORE adding the excluded decision-time sample. */
    if (detector->active && now == detector->active_trigger + POST_TRIGGER_SAMPLES) {
        FallEvent *event = &detector->event;

        event->trigger = detector->active_trigger;
        event->start = event->trigger - PRE_TRIGGER_SAMPLES;
        event->end = now;
        event->decision = now;
        fall_features_finish(&detector->features, event->features);
        event->score = fall_classifier_score(event->features);
        event->result = event->score >= detector->logit_threshold ? FALL_DETECTED : FALL_ADL;
        result = event->result;
        detector->active = 0;
    }

    /* 4. Seed a new event with the prior 100 samples in chronological order. */
    if (trace->accepted_trigger) {
        /* Even a trigger lacking prehistory starts the refractory interval. */
        detector->last_trigger = now;
        if (now >= PRE_TRIGGER_SAMPLES) {
            int64_t history_index;

            detector->active = 1;
            detector->active_trigger = now;
            fall_features_reset(&detector->features);
            for (history_index = now - PRE_TRIGGER_SAMPLES; history_index < now; ++history_index) {
                fall_features_add(&detector->features,
                                  detector->history[history_index % PRE_TRIGGER_SAMPLES]);
            }
        }
    }

    /* 5. Include the trigger/current sample, then overwrite the oldest history. */
    if (detector->active) {
        fall_features_add(&detector->features, scalars);
    }
    detector->history[now % PRE_TRIGGER_SAMPLES] = scalars;
    return result;
}

FallResult fall_detector_push_counts(FallDetector *detector, int16_t x, int16_t y, int16_t z) {
    AccelSample sample = {
        (float)x / 256.0f,
        (float)y / 256.0f,
        (float)z / 256.0f
    };
    return fall_detector_push_sample(detector, sample);
}
