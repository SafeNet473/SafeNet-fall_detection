/* HOST EXPERIMENT ONLY: native 208 Hz, not MCU v1. */
#ifndef FALL_FILTERS_H
#define FALL_FILTERS_H

/* Default matches the frozen mixed-precision reference. The experimental
 * FALL_USE_FLOAT build is measured separately, not certified by default. */
#ifdef FALL_USE_FLOAT
typedef float FallReal;
#else
typedef double FallReal;
#endif

typedef struct {
    FallReal zi[2][2][3]; /* [SOS section][z1 or z2][XYZ axis] */
    FallReal gravity[3];  /* Previous gravity EMA value, g */
    int initialized;
} FallFilters;

/* State must initially be zeroed; first push performs signal initialization. */
void fall_filters_push(FallFilters *state, const FallReal raw[3],
                       FallReal filtered[3], FallReal gravity[3]);

#endif
