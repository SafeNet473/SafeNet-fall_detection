/* HOST EXPERIMENT ONLY: native 208 Hz, not MCU v1. */
#ifndef FALL_FEATURES_H
#define FALL_FEATURES_H

#include "filters.h"

/* The three scalar streams retained in prehistory, all in g. */
typedef struct {
    FallReal magnitude;
    FallReal perpendicular;
    FallReal parallel_abs;
} FallScalars;

typedef struct {
    FallReal perpendicular_max;
    FallReal parallel_max;
    FallReal jerk_sum;
    FallReal previous_magnitude;
    unsigned count;
} FallFeatures;

void fall_features_reset(FallFeatures *state);
/* Add samples in event-window order, including buffered prehistory. */
void fall_features_add(FallFeatures *state, FallScalars sample);
/* Requires at least two samples; detector calls this after exactly 312. */
void fall_features_finish(const FallFeatures *state, FallReal out[3]);

#endif
