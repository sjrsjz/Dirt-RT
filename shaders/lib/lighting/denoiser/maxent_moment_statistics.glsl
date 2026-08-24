#ifndef MAXENT_MOMENT_STATISTICS_GLSL
#define MAXENT_MOMENT_STATISTICS_GLSL

#include "/lib/math/statistics.glsl"

// The denoiser operates on m=(E[R u], E[R]). No decoder distribution is
// involved. For one raw sample z=(R u,R), |z|^2=2R^2.

float maxentMomentDifferenceCorrelationForSpatialStep(int stepRadius) {
    if (stepRadius <= 1) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_16;
    return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_32;
}

float maxentMomentPropagationCorrelationForSpatialStep(int stepRadius) {
    if (stepRadius <= 1) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_16;
    return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_32;
}

float maxentMomentDistanceSq(vec4 a, vec4 b) {
    vec4 delta = a - b;
    return dot(delta, delta);
}

float maxentTemporalCurrentEstimatorVariance(float standardDeviation) {
    standardDeviation = max(standardDeviation, 0.0);
    return standardDeviation * standardDeviation
        + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
}

float maxentTemporalProposalCorrectedStandardDeviation(
        float proposalStddev, float currentStddev,
        float correctionCurrentWeight) {
    float proposalVariance = max(proposalStddev, 0.0)
        * max(proposalStddev, 0.0);
    float currentVariance = maxentTemporalCurrentEstimatorVariance(
        currentStddev);
    // The configured statistical closure assumes zero correlation.
    return sqrt(statisticsIndependentBlendVariance(
        proposalVariance, currentVariance,
        correctionCurrentWeight));
}

// Minimum-MSE update between two independent denoised estimators:
//   H = previous final filtered output, reprojected to this frame;
//   C = current Raw RT observation filtered with the proposal's exact spatial
//       weights, but never temporally premixed with H.
// Their variances are directly comparable and require neither beta
// deconvolution nor subtraction of nearly equal variances. In the stationary
// limit V_H=V0/N and V_C=V0 this naturally becomes 1/(N+1).
float maxentTemporalMinimumMseAlpha(float alphaFloor,
        vec4 currentMoment, float currentEstimatorVariance,
        vec4 historyMoment,
        float historyStddev, float historyEffectiveSamples,
        out float varianceOptimalAlpha) {
    varianceOptimalAlpha = -1.0;
    alphaFloor = isnan(alphaFloor) || isinf(alphaFloor)
        ? 1.0 : clamp(alphaFloor, 0.0, 1.0);
    if (any(isnan(currentMoment))
            || any(isinf(currentMoment))
            || any(isnan(historyMoment)) || any(isinf(historyMoment))
            || !(currentEstimatorVariance >= 0.0)
            || isnan(currentEstimatorVariance)
            || isinf(currentEstimatorVariance)
            || !(historyStddev >= 0.0) || isnan(historyStddev)
            || isinf(historyStddev)
            || !statisticsValidEffectiveSampleCount(
                historyEffectiveSamples))
        return alphaFloor;

    float historyVariance = historyStddev * historyStddev;
    float currentVariance = currentEstimatorVariance
        + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
    float distanceSq = maxentMomentDistanceSq(
        currentMoment, historyMoment);
    float innovationVariance = historyVariance + currentVariance;
    varianceOptimalAlpha = statisticsMinimumMseIndependentCurrentWeight(
        0.0, historyVariance, currentVariance);
    float estimatedSquaredBias = statisticsPositivePartSquaredBias(
        distanceSq, 0*innovationVariance);
    float optimalAlpha = statisticsMinimumMseIndependentCurrentWeight(
        estimatedSquaredBias, historyVariance, currentVariance);
    // // varianceOptimalAlpha is explicitly independent of distanceSq and is not
    // // the alpha used to update temporal history.
    // // The MSE optimum remains bounded by equal-weight Kish warm-up and
    // // reprojection confidence.  Bypassing alphaFloor would overweight an old
    // // first sample or retain history whose footprint was only partly valid.
    // return max(alphaFloor, optimalAlpha);

    return mix(1.0, 1.0 / (1.0 + historyEffectiveSamples), exp(-0.01 * sqrt(distanceSq / (historyVariance + currentVariance))));
}

#endif // MAXENT_MOMENT_STATISTICS_GLSL
