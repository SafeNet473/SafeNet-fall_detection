#include "filters.h"
#include "model_params.h"

void fall_filters_push(FallFilters *state, const FallReal raw[3],
                       FallReal filtered[3], FallReal gravity[3]) {
    int axis, section;

    /* Initialize for constant input equal to the stream's first raw sample. */
    if (!state->initialized) {
        for (axis = 0; axis < 3; ++axis) {
            state->gravity[axis] = raw[axis];
            for (section = 0; section < 2; ++section) {
                state->zi[section][0][axis] = FALL_ZI[section][0] * raw[axis];
                state->zi[section][1][axis] = FALL_ZI[section][1] * raw[axis];
            }
        }
        state->initialized = 1;
    }

    for (axis = 0; axis < 3; ++axis) {
        FallReal section_input = raw[axis];

        /* SOS columns: b0, b1, b2, a0 (= 1), a1, a2.
         * Direct form II transposed: preserve update order and old z2. */
        for (section = 0; section < 2; ++section) {
            const FallReal *coefficients = FALL_SOS[section];
            FallReal section_output = coefficients[0] * section_input
                                      + state->zi[section][0][axis];

            state->zi[section][0][axis] = coefficients[1] * section_input
                                        - coefficients[4] * section_output
                                        + state->zi[section][1][axis];
            state->zi[section][1][axis] = coefficients[2] * section_input
                                        - coefficients[5] * section_output;
            section_input = section_output;
        }
        filtered[axis] = section_input;

        /* Gravity follows raw acceleration, independently of Butterworth. */
        state->gravity[axis] = FALL_ALPHA * raw[axis]
                              + ((FallReal)1 - FALL_ALPHA) * state->gravity[axis];
        gravity[axis] = state->gravity[axis];
    }
}
