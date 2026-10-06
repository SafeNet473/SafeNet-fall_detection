#include "classifier.h"
#include "model_params.h"

FallReal fall_classifier_score(const FallReal features[3]) {
    FallReal logit = FALL_FOLDED_INTERCEPT;
    int feature_index;

    /* StandardScaler is already folded into these weights and the intercept. */
    for (feature_index = 0; feature_index < 3; ++feature_index) {
        logit += FALL_FOLDED_WEIGHTS[feature_index] * features[feature_index];
    }
    return logit;
}

FallReal fall_classifier_threshold(FallOperatingPoint point) {
    return point == FALL_SENSITIVITY_CANDIDATE
        ? FALL_SENSITIVITY_LOGIT
        : FALL_BALANCED_LOGIT;
}
