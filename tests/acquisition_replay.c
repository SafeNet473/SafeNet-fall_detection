/* Host-only I/O for the separable timing boundary and complete raw path. */
#include "acquisition.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    FILE *samples;
    FILE *streams;
    FILE *events;
    FallDetector *direct_detector;
    unsigned long long emitted;
} ReplayOutput;

static void vector(FILE *output, const FallReal *value, unsigned count) {
    unsigned index;
    for (index = 0; index < count; ++index) {
        fprintf(output, ",%.17g", (double)value[index]);
    }
}

static void record_output(void *context, const AcquisitionOutput *output, const FallDetector *detector) {
    ReplayOutput *files = context;
    const FallTrace *trace = &detector->trace;
    fprintf(files->samples, "%llu,%llu,%llu,%.17g,%.17g,%.17g\n",
            (unsigned long long)output->logical_index, (unsigned long long)output->logical_timestamp,
            (unsigned long long)output->bracket_timestamp,
            (double)output->sample.ax, (double)output->sample.ay, (double)output->sample.az);
    fprintf(files->streams, "%llu", (unsigned long long)output->logical_index);
    vector(files->streams, trace->filtered, 3); vector(files->streams, trace->acceleration, 3);
    vector(files->streams, trace->gravity, 3); vector(files->streams, trace->unit, 3);
    vector(files->streams, trace->residual, 3);
    fprintf(files->streams, ",%.17g,%.17g,%.17g,%.17g,%d,%d,%d\n",
            (double)trace->magnitude, (double)trace->perpendicular, (double)trace->parallel,
            (double)trace->jerk, trace->high, trace->rising_edge, trace->accepted_trigger);
    if (output->result != FALL_NO_DECISION) {
        const FallEvent *event = &detector->event;
        fprintf(files->events, "%lld,%lld,%lld,%lld", (long long)event->trigger,
                (long long)event->start, (long long)event->end, (long long)event->decision);
        vector(files->events, event->features, 3);
        fprintf(files->events, ",%.17g,%d\n", (double)event->score, output->result == FALL_DETECTED);
    }
    ++files->emitted;
}

static void direct_sample(void *context, AccelSample sample, uint64_t logical, uint64_t bracket) {
    ReplayOutput *files = context;
    AcquisitionOutput output;
    output.stream_id = 0;
    output.logical_index = (uint64_t)files->direct_detector->next_index;
    output.logical_timestamp = logical;
    output.bracket_timestamp = bracket;
    output.sample = sample;
    output.result = fall_detector_push_sample(files->direct_detector, sample);
    record_output(context, &output, files->direct_detector);
}

int main(int argc, char **argv) {
    AcquisitionConfig config = {LSM6DSOX_RANGE_16G, 208, 52000000, 375000, FALL_BALANCED_REFERENCE};
    AcquisitionState acquisition;
    TimingAdapter timing;
    FallDetector direct_detector;
    ReplayOutput files;
    FILE *input;
    char line[256];
    bool raw_mode;

    if (argc != 7) {
        fprintf(stderr, "Usage: acquisition_replay raw|g input.csv samples.csv streams.csv events.csv balanced|sensitivity\n");
        return 2;
    }
    raw_mode = strcmp(argv[1], "raw") == 0;
    if (!raw_mode && strcmp(argv[1], "g") != 0) return 2;
    if (strcmp(argv[6], "sensitivity") == 0) config.operating_point = FALL_SENSITIVITY_CANDIDATE;
    else if (strcmp(argv[6], "balanced") != 0) return 2;
    memset(&files, 0, sizeof(files));
    input = fopen(argv[2], "r");
    files.samples = fopen(argv[3], "w");
    files.streams = fopen(argv[4], "w");
    files.events = fopen(argv[5], "w");
    if (!input || !files.samples || !files.streams || !files.events) return 2;
    fprintf(files.samples, "index,logical_ticks,bracket_ticks,ax,ay,az\n");
    fprintf(files.streams, "index,fx,fy,fz,ax,ay,az,gx,gy,gz,ux,uy,uz,bx,by,bz,m,h,p,jerk,high,edge,trigger\n");
    fprintf(files.events, "trigger,start,end,decision,ga_C2,jerk_abs_mean,ga_parallel_peak,z,label\n");
    if (!acquisition_init(&acquisition, &config, record_output, &files)) return 2;
    if (!timing_adapter_init(&timing, 260000, 375000)) return 2;
    fall_detector_init_operating_point(&direct_detector, config.operating_point);
    files.direct_detector = &direct_detector;
    while (fgets(line, sizeof(line), input)) {
        unsigned long long timestamp;
        TimingStatus status;
        if (raw_mode) {
            int x, y, z;
            AcquisitionResult result;
            if (sscanf(line, " %llu , %d , %d , %d", &timestamp, &x, &y, &z) != 4
                || x < INT16_MIN || x > INT16_MAX || y < INT16_MIN || y > INT16_MAX
                || z < INT16_MIN || z > INT16_MAX) return 3;
            result = acquisition_push_lsm6dsox_sample(&acquisition, (int16_t)x, (int16_t)y,
                                                     (int16_t)z, (uint64_t)timestamp);
            status = result.status;
        } else {
            double x, y, z;
            AccelSample sample;
            TimingResult result;
            if (sscanf(line, " %llu , %lf , %lf , %lf", &timestamp, &x, &y, &z) != 4) return 3;
            sample.ax = (float)x; sample.ay = (float)y; sample.az = (float)z;
            result = timing_adapter_push(&timing, sample, (uint64_t)timestamp, direct_sample, &files);
            status = result.status;
        }
        if (status != TIMING_BUFFERED && status != TIMING_EMITTED) {
            fprintf(stderr, "Discontinuity in replay: status=%d timestamp=%llu\n", status, timestamp);
            return 4;
        }
    }
    if (ferror(input)) return 5;
    fclose(input);
    if (fclose(files.samples) || fclose(files.streams) || fclose(files.events)) return 5;
    return 0; /* No EOF extrapolation or partial-event flushing. */
}
