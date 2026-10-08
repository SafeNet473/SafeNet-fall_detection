#include "lsm6dsox_input.h"
#include <stddef.h>

bool lsm6dsox_counts_to_g(float x, float y, float z,
                         Lsm6dsoxRange configured_range, AccelSample *sample_g) {
    float sensitivity_g_per_lsb = 1/9.80665f;

    if (sample_g == NULL) {
        return false;
    }
    sample_g->ax = (float)x * sensitivity_g_per_lsb;
    sample_g->ay = (float)y * sensitivity_g_per_lsb;
    sample_g->az = (float)z * sensitivity_g_per_lsb;
    return true;
}
