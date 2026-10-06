/* HOST EXPERIMENT ONLY: native 208 Hz, not MCU v1. */
#ifndef FALL_DETECTOR_H
#define FALL_DETECTOR_H

#include <stdint.h>
#include "filters.h"
#include "features.h"
#include "classifier.h"

/* Input acceleration in g. Use push_counts for raw ADXL345 values. */
typedef struct {
    float ax;
    float ay;
    float az;
} AccelSample;

typedef enum {
    FALL_NO_DECISION,
    FALL_ADL,
    FALL_DETECTED
} FallResult;

/* Current-sample diagnostics; vectors use XYZ axis order. */
typedef struct {
    FallReal filtered[3];      /* Butterworth output before float32 cast, g */
    FallReal acceleration[3];  /* Feature input after float32 cast, g */
    FallReal gravity[3];       /* Raw-driven gravity EMA, g */
    FallReal unit[3];          /* Gravity direction with denominator floor */
    FallReal residual[3];      /* Acceleration perpendicular to gravity, g */
    FallReal magnitude;
    FallReal perpendicular;
    FallReal parallel;         /* Signed projection, including gravity, g */
    FallReal jerk;             /* Absolute scalar-magnitude derivative, g/s */
    int high;
    int rising_edge;
    int accepted_trigger;
} FallTrace;

/* Most recently completed event; valid when push returns a decision. */
typedef struct {
    int64_t trigger;
    int64_t start;             /* First included sample */
    int64_t end;               /* Exclusive end */
    int64_t decision;          /* Arrival of excluded sample at trigger + 208 */
    FallReal features[3];      /* ga_C2, jerk_abs_mean, ga_parallel_peak */
    FallReal score;            /* Logistic logit */
    FallResult result;
} FallEvent;

typedef struct {
    /* Continuous signal processing and circular prehistory. */
    FallFilters filters;
    FallScalars history[104]; /* prior 104 samples, continuously maintained */
    FallFeatures features;
    FallTrace trace; /* current sample diagnostics */
    FallEvent event; /* most recently completed event */
    FallReal previous_magnitude;
    FallReal logit_threshold;

    /* Sample indices and trigger/event state. */
    int64_t next_index;
    int64_t last_trigger;
    int64_t active_trigger;
    int previous_high;
    int active;
} FallDetector;

/* One state per chronological 208 Hz stream. Reset only at trial/stream start.
 * Default is a documented host reference point, NOT a deployment choice.
 * Finite inputs required. Do not compile with fast-math or FP contraction.
 * EOF: do not pad/flush; incomplete active events produce no decision. */
void fall_detector_init(FallDetector *detector);
void fall_detector_init_operating_point(FallDetector *detector, FallOperatingPoint point);
FallResult fall_detector_push_sample(FallDetector *detector, AccelSample sample);
FallResult fall_detector_push_counts(FallDetector *detector, int16_t x, int16_t y, int16_t z);

#endif
