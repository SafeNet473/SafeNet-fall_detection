#ifndef TIMING_ADAPTER_H
#define TIMING_ADAPTER_H

#include <stdbool.h>
#include <stdint.h>
#include "fall_detector.h"

typedef enum {
    TIMING_BUFFERED,
    TIMING_EMITTED,
    TIMING_DUPLICATE_TIMESTAMP,
    TIMING_TIMESTAMP_REVERSAL,
    TIMING_EXCESSIVE_GAP,
    TIMING_INVALID_SAMPLE,
    TIMING_TIMESTAMP_OVERFLOW,
    TIMING_NOT_CONFIGURED
} TimingStatus;

typedef struct {
    TimingStatus status;
    unsigned emitted;
} TimingResult;

/* Invoked synchronously. bracket_timestamp is the later measurement time;
 * it is NOT the ISR/processing time. The receiver must copy anything retained. */
typedef void (*TimingOutputFn)(void *context, AccelSample sample,
                               uint64_t logical_timestamp, uint64_t bracket_timestamp);

typedef struct {
    uint64_t period_ticks;
    uint64_t max_gap_ticks;
    uint64_t epoch_timestamp;
    uint64_t previous_timestamp;
    uint64_t next_output_timestamp;
    AccelSample previous;
    bool has_previous;
    bool configured;
} TimingAdapter;

/* Caller supplies timestamp units. For 200 Hz, period_ticks must represent 5 ms.
 * Require 0 < max_gap_ticks <= 2*period_ticks, bounding work per push (<=3
 * outputs at startup, <=2 thereafter). No allocation or extrapolation. */
bool timing_adapter_init(TimingAdapter *state, uint64_t period_ticks, uint64_t max_gap_ticks);
void timing_adapter_reset(TimingAdapter *state);

/* First valid sample anchors the grid; no output until a second sample arrives.
 * On reversal/gap, current sample becomes the new anchor, without emission.
 * Duplicate/nonfinite/overflow inputs clear timing and are discarded.
 * These statuses require the receiver to reset its detector too. */
TimingResult timing_adapter_push(TimingAdapter *state, AccelSample sample,
                                 uint64_t timestamp, TimingOutputFn output, void *context);

#endif
