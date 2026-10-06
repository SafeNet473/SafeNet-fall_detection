#ifndef FALL_CLASSIFIER_H
#define FALL_CLASSIFIER_H

#include "filters.h"

/* Documented reference points; final hardware operating point is unresolved. */
typedef enum {
    FALL_BALANCED_REFERENCE,
    FALL_SENSITIVITY_CANDIDATE
} FallOperatingPoint;

FallReal fall_classifier_score(const FallReal features[3]); /* logit, no sigmoid */
FallReal fall_classifier_threshold(FallOperatingPoint point);

#endif
