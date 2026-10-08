#ifndef LSM6DSOX_INPUT_H
#define LSM6DSOX_INPUT_H

#include <stdbool.h>
#include <stdint.h>
#include "fall_detector.h"

/* Actual configured/read-back range, NOT the hardware register bit encoding.
 * No default range: the caller must know the sensor configuration. */
typedef enum {
    LSM6DSOX_RANGE_2G = 2,
    LSM6DSOX_RANGE_4G = 4,
    LSM6DSOX_RANGE_8G = 8,
    LSM6DSOX_RANGE_16G = 16
} Lsm6dsoxRange;

/* ST DS12814 Rev 4, Table 2: sensitivity in mg/LSB divided by 1000.
 * False for an unknown range or null output; output is unchanged on failure.
 * This converts units only. It does NOT turn 208 Hz into the required 200 Hz.
 * Signed XYZ counts must already be decoded correctly by the acquisition layer. */
bool lsm6dsox_counts_to_g(int16_t x, int16_t y, int16_t z,
                         Lsm6dsoxRange configured_range, AccelSample *sample_g);

/* For driver outputs already expressed in m/s^2 (e.g. Zephyr sensor values),
 * NOT raw register counts. Divide each axis by standard gravity, 9.80665 m/s^2.
 * Preserve gravity and signs. No range sensitivity is applied a second time.
 * Reject nonfinite/unrepresentable output; leave sample_g unchanged on failure. */
bool lsm6dsox_ms2_to_g(double x, double y, double z, AccelSample *sample_g);

#endif
