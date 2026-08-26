#ifndef MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
#define MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/math/statistics.glsl"

// The denoiser operates on m=(E[R u], E[R]). No decoder distribution is
// involved. For one raw sample z=(R u,R), |z|^2=2R^2.

float maxentTemporalCurrentEstimatorVariance(float estimatorStdDev) {
    estimatorStdDev = max(estimatorStdDev, 0.0);
    return estimatorStdDev * estimatorStdDev + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
}

float maxentTemporalProposalCorrectedStandardDeviation(float proposalEstimatorStdDev, float currentEstimatorStdDev,
        float correctionCurrentWeight) {
    float proposalVariance = max(proposalEstimatorStdDev, 0.0) * max(proposalEstimatorStdDev, 0.0);
    float currentVariance = maxentTemporalCurrentEstimatorVariance(currentEstimatorStdDev);
    // The configured statistical closure assumes zero correlation.
    return sqrt(statisticsIndependentBlendVariance(proposalVariance, currentVariance, correctionCurrentWeight));
}

// H is the reprojected previous filtered output; C is the independently spatially filtered current Raw RT output.
// currentEstimatorVariance and historyEstimatorStdDev^2 are estimator variances in the same four-moment metric.
// noiseOnlyCurrentWeight exposes V_H/(V_H+V_C) for diagnostics. The returned current weight is the user-tuned
// response: large denoised moment differences approach 1, while a stationary signal approaches 1/(N_eff+1).
float maxentTemporalResponseAlpha(float alphaFloor, vec4 currentMoment, float currentEstimatorVariance,
        vec4 historyMoment, float historyEstimatorStdDev, float historyEffectiveSamples, out float noiseOnlyCurrentWeight) {
    noiseOnlyCurrentWeight = -1.0;
    alphaFloor = isnan(alphaFloor) || isinf(alphaFloor)
        ? 1.0 : clamp(alphaFloor, 0.0, 1.0);
    if (any(isnan(currentMoment))
            || any(isinf(currentMoment))
            || any(isnan(historyMoment)) || any(isinf(historyMoment))
            || !(currentEstimatorVariance >= 0.0)
            || isnan(currentEstimatorVariance)
            || isinf(currentEstimatorVariance)
            || !(historyEstimatorStdDev >= 0.0) || isnan(historyEstimatorStdDev)
            || isinf(historyEstimatorStdDev)
            || !statisticsValidEffectiveSampleCount(
                historyEffectiveSamples))
        return alphaFloor;

    float historyVariance = historyEstimatorStdDev * historyEstimatorStdDev;
    float currentVariance = currentEstimatorVariance + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
    float distanceSq = maxentLightSampleDistanceSq(currentMoment, historyMoment);
    noiseOnlyCurrentWeight = statisticsMinimumVarianceIndependentCurrentWeight(historyVariance, currentVariance);
    float combinedVariance = historyVariance + currentVariance;
    float responseAlpha = mix(1.0, 1.0 / (1.0 + historyEffectiveSamples), combinedVariance / (combinedVariance + MAXENT_TEMPORAL_RESPONSE_DISTANCE_SCALE * distanceSq));
    return max(alphaFloor, responseAlpha);
}

#endif // MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
