#include "lsm6dsox_input.h"

/* Validation must fail at build time if compiler defaults suppress assertions. */
#ifdef NDEBUG
#error "Build this host test with -UNDEBUG"
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
    const Lsm6dsoxRange ranges[4] = {
        LSM6DSOX_RANGE_2G, LSM6DSOX_RANGE_4G, LSM6DSOX_RANGE_8G, LSM6DSOX_RANGE_16G
    };
    /* Datasheet values in g/LSB; comparison includes float representation error. */
    const double sensitivities[4] = {0.000061, 0.000122, 0.000244, 0.000488};
    unsigned index;
    AccelSample sample = {1.0f, 2.0f, 3.0f};

    for (index = 0; index < 4; ++index) {
        assert(lsm6dsox_counts_to_g(1024, -32768, 32767, ranges[index], &sample));
        assert(fabs(sample.ax - 1024 * sensitivities[index]) < 2e-6);
        assert(fabs(sample.ay + 32768 * sensitivities[index]) < 2e-6);
        assert(fabs(sample.az - 32767 * sensitivities[index]) < 2e-6);
        printf("%d,%.17g,%.17g,%.17g\n", ranges[index],
               (double)sample.ax, (double)sample.ay, (double)sample.az);
        assert(lsm6dsox_counts_to_g(0, 0, 0, ranges[index], &sample));
        assert(sample.ax == 0 && sample.ay == 0 && sample.az == 0);
    }
    sample.ax = 1; sample.ay = 2; sample.az = 3;
    assert(!lsm6dsox_counts_to_g(1, 2, 3, (Lsm6dsoxRange)0, &sample));
    assert(sample.ax == 1 && sample.ay == 2 && sample.az == 3);
    assert(!lsm6dsox_counts_to_g(1, 2, 3, (Lsm6dsoxRange)3, &sample));
    assert(!lsm6dsox_counts_to_g(1, 2, 3, LSM6DSOX_RANGE_16G, NULL));
    return 0;
}
