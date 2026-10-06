#include "features.h"

#include <math.h>
#include <string.h>

void fall_features_reset(FallFeatures *state) {
    memset(state, 0, sizeof(*state));
}

void fall_features_add(FallFeatures *state, FallScalars sample) {
    if (sample.perpendicular > state->perpendicular_max) {
        state->perpendicular_max = sample.perpendicular;
    }
    if (sample.parallel_abs > state->parallel_max) {
        state->parallel_max = sample.parallel_abs;
    }

    /* Skip the first difference: jerk must not cross the window boundary. */
    if (state->count) {
        state->jerk_sum += (FallReal)fabs((double)((sample.magnitude - state->previous_magnitude)
                                                 * (FallReal)200));
    }
    state->previous_magnitude = sample.magnitude;
    ++state->count;
}

void fall_features_finish(const FallFeatures *state, FallReal out[3]) {
    /* Frozen classifier order; 300 samples give 299 magnitude differences. */
    out[0] = state->perpendicular_max;                  /* ga_C2 */
    out[1] = state->jerk_sum / (FallReal)(state->count - 1); /* jerk_abs_mean */
    out[2] = state->parallel_max;                       /* ga_parallel_peak */
}
