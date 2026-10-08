#ifndef FALL_ACQUISITION_H
#define FALL_ACQUISITION_H

#include "lsm6dsox_input.h"
#include "timing_adapter.h"

typedef struct {
    Lsm6dsoxRange effective_range; /* Explicit driver configuration/read-back */
    unsigned configured_odr_hz;   /* Selected architecture requires nominal 208 */
    uint64_t timestamp_ticks_per_second; /* Shared monotonic acquisition timebase */
    uint64_t max_input_gap_ticks; /* Explicit gap policy, not an inferred clock */
    FallOperatingPoint operating_point;
} AcquisitionConfig;

typedef enum {
    ACQUISITION_SENSOR_RESET,
    ACQUISITION_ODR_CHANGE,
    ACQUISITION_FULL_SCALE_CHANGE
} AcquisitionDiscontinuity;

typedef struct {
    uint64_t stream_id;
    uint64_t logical_index;
    uint64_t logical_timestamp;
    uint64_t bracket_timestamp;
    AccelSample sample;
    FallResult result;
} AcquisitionOutput;

/* Observer called after the generated sample has entered the frozen detector.
 * No reentrant pushes/reconfiguration; no concurrent mutation of this state. */
typedef void (*AcquisitionObserver)(void *context, const AcquisitionOutput *output,
                                    const FallDetector *detector);

typedef struct {
    AcquisitionConfig config;
    TimingAdapter timing;
    FallDetector detector;
    AcquisitionObserver observer;
    void *observer_context;
    uint64_t stream_id;
    bool configured;
} AcquisitionState;

typedef struct {
    TimingStatus status;
    unsigned emitted;
    bool active_event_discarded;
} AcquisitionResult;

/* No register I/O. Accepts only explicitly confirmed ±16 g / nominal 208 Hz.
 * Timestamp rate must be an integer multiple of 200 so the 5 ms grid is exact.
 * max_input_gap_ticks must be > nominal period and <= 2 output periods.
 * The caller chooses one of the existing operating points explicitly. */
bool acquisition_init(AcquisitionState *state, const AcquisitionConfig *config,
                       AcquisitionObserver observer, void *context);

/* Always discards detector/timing history and advances stream_id. Invalid
 * configurations fail closed. Valid configurations wait for two new samples. */
bool acquisition_reconfigure(AcquisitionState *state, const AcquisitionConfig *config);

/* Reset/ODR/range notifications block further input until configuration has
 * been confirmed again via acquisition_reconfigure. Returns event discarded. */
bool acquisition_notify_discontinuity(AcquisitionState *state, AcquisitionDiscontinuity reason);

AcquisitionResult acquisition_push_lsm6dsox_sample(AcquisitionState *state,
                                                  int16_t raw_x, int16_t raw_y, int16_t raw_z,
                                                  uint64_t sensor_timestamp);

#endif
