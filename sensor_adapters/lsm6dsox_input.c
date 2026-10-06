#include "lsm6dsox_input.h"
#include <stddef.h>

bool lsm6dsox_counts_to_g(int16_t x, int16_t y, int16_t z,
                         Lsm6dsoxRange configured_range, AccelSample *sample_g) {
    float sensitivity_g_per_lsb;

    if (sample_g == NULL) {
        return false;
    }
    switch (configured_range) {
        case LSM6DSOX_RANGE_2G:
            sensitivity_g_per_lsb = 0.000061f;
            break;
        case LSM6DSOX_RANGE_4G:
            sensitivity_g_per_lsb = 0.000122f;
            break;
        case LSM6DSOX_RANGE_8G:
            sensitivity_g_per_lsb = 0.000244f;
            break;
        case LSM6DSOX_RANGE_16G:
            sensitivity_g_per_lsb = 0.000488f;
            break;
        default:
            return false;
    }
    sample_g->ax = (float)x * sensitivity_g_per_lsb;
    sample_g->ay = (float)y * sensitivity_g_per_lsb;
    sample_g->az = (float)z * sensitivity_g_per_lsb;
    return true;
}
