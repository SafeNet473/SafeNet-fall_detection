/* PC validation only. All detector/model behavior remains in embedded/. */
#include "fall_detector.h"

#ifdef NDEBUG
#error "Host validation requires active assertions; compile with -UNDEBUG"
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void write_vector(FILE *output, const FallReal *values, unsigned count) {
    unsigned index;
    for (index = 0; index < count; ++index) {
        fprintf(output, ",%.17g", (double)values[index]);
    }
}

static int classify(FILE *input, FILE *output, FallOperatingPoint point) {
    double values[3];
    int index = 0;
    int fields;
    FallReal threshold = fall_classifier_threshold(point);

    fprintf(output, "index,ga_C2,jerk_abs_mean,ga_parallel_peak,z,threshold,label\n");
    while ((fields = fscanf(input, " %lf , %lf , %lf", &values[0], &values[1], &values[2])) == 3) {
        FallReal features[3] = {(FallReal)values[0], (FallReal)values[1], (FallReal)values[2]};
        FallReal score = fall_classifier_score(features);

        fprintf(output, "%d", index++);
        write_vector(output, features, 3);
        fprintf(output, ",%.17g,%.17g,%d\n", (double)score, (double)threshold, score >= threshold);
    }
    return fields == EOF && !ferror(input) ? 0 : 3;
}

static int replay(FILE *input, FILE *streams, FILE *events, FILE *history,
                  FallOperatingPoint point) {
    FallDetector detector;
    double raw[3];
    int fields;

    fall_detector_init_operating_point(&detector, point);
    fprintf(streams, "index,fx,fy,fz,ax,ay,az,gx,gy,gz,ux,uy,uz,bx,by,bz,m,h,p,jerk,high,edge,trigger\n");
    fprintf(events, "trigger,start,end,decision,ga_C2,jerk_abs_mean,ga_parallel_peak,z,label,threshold,count\n");
    fprintf(history, "trigger,index,m,h,abs_p\n");

    while ((fields = fscanf(input, " %lf , %lf , %lf", &raw[0], &raw[1], &raw[2])) == 3) {
        int64_t now = detector.next_index;
        FallTrace *trace = &detector.trace;
        /* This slot is overwritten by push. Save it to inspect all 100 prior
         * samples afterward without changing or instrumenting the C core. */
        FallScalars oldest = detector.history[now % 100];
        AccelSample sample = {(float)raw[0], (float)raw[1], (float)raw[2]};
        FallResult result;

        if (!isfinite(raw[0]) || !isfinite(raw[1]) || !isfinite(raw[2])) {
            return 3;
        }
        result = fall_detector_push_sample(&detector, sample);
        fprintf(streams, "%lld", (long long)now);
        write_vector(streams, trace->filtered, 3);
        write_vector(streams, trace->acceleration, 3);
        write_vector(streams, trace->gravity, 3);
        write_vector(streams, trace->unit, 3);
        write_vector(streams, trace->residual, 3);
        fprintf(streams, ",%.17g,%.17g,%.17g,%.17g,%d,%d,%d\n",
                (double)trace->magnitude, (double)trace->perpendicular,
                (double)trace->parallel, (double)trace->jerk,
                trace->high, trace->rising_edge, trace->accepted_trigger);

        if (trace->accepted_trigger && now >= 100) {
            int64_t index;
            for (index = now - 100; index < now; ++index) {
                FallScalars prior = index == now - 100 ? oldest : detector.history[index % 100];
                fprintf(history, "%lld,%lld,%.17g,%.17g,%.17g\n",
                        (long long)now, (long long)index, (double)prior.magnitude,
                        (double)prior.perpendicular, (double)prior.parallel_abs);
            }
        }

        if (result != FALL_NO_DECISION) {
            FallEvent *event = &detector.event;
            assert(event->start == event->trigger - 100);
            assert(event->end == event->trigger + 200);
            assert(event->decision == event->end);
            assert(event->end - event->start == 300);
            assert(detector.features.count == 300);
            fprintf(events, "%lld,%lld,%lld,%lld", (long long)event->trigger,
                    (long long)event->start, (long long)event->end, (long long)event->decision);
            write_vector(events, event->features, 3);
            fprintf(events, ",%.17g,%d,%.17g,%u\n", (double)event->score,
                    result == FALL_DETECTED, (double)detector.logit_threshold, detector.features.count);
        }
    }
    /* No tail padding or EOF flushing. */
    return fields == EOF && !ferror(input) ? 0 : 3;
}

int main(int argc, char **argv) {
    FallOperatingPoint point;
    FILE *input, *output;
    int status;

    if (argc != 5 && argc != 7) {
        fprintf(stderr, "Usage: host_validation classify input.csv output.csv balanced|sensitivity\n"
                        "   or: host_validation replay input_g.csv streams.csv events.csv history.csv balanced|sensitivity\n");
        return 2;
    }
    if (strcmp(argv[argc - 1], "balanced") == 0) {
        point = FALL_BALANCED_REFERENCE;
    } else if (strcmp(argv[argc - 1], "sensitivity") == 0) {
        point = FALL_SENSITIVITY_CANDIDATE;
    } else {
        return 2;
    }
    input = fopen(argv[2], "r");
    output = fopen(argv[3], "w");
    if (!input || !output) {
        return 2;
    }
    if (argc == 5 && strcmp(argv[1], "classify") == 0) {
        status = classify(input, output, point);
    } else if (argc == 7 && strcmp(argv[1], "replay") == 0) {
        FILE *events = fopen(argv[4], "w");
        FILE *history = fopen(argv[5], "w");
        if (!events || !history) {
            return 2;
        }
        status = replay(input, output, events, history, point);
        if (fclose(events) != 0 || fclose(history) != 0) {
            status = 4;
        }
    } else {
        status = 2;
    }
    if (fclose(input) != 0 || fclose(output) != 0) {
        status = 4;
    }
    return status;
}
