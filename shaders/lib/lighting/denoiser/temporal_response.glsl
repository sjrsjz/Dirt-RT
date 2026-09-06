#ifndef MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
#define MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/math/statistics.glsl"

// The denoiser mean remains the linear state m=(E[R u],E[R]). Differences use
// its 2x2 PSD Bures geometry. Variance preparation uses the alpha=1 g^-3 family
// only as an explicit closure for missing R^2-weighted angular moments;
// lighting reconstruction remains independent of this filtering policy.
// These functions are the active Bures mode (MAXENT_TEMPORAL_CONFIDENCE_CLAMP=0).
// The experimental linear-moment estimator ABI lives in temporal_confidence.glsl.

float maxentTemporalEstimatorVariance(float monteCarloStandardDeviation) {
    if (!denoiserSigmaKnown(monteCarloStandardDeviation)) return DENOISER_UNKNOWN_UNCERTAINTY;
    return monteCarloStandardDeviation * monteCarloStandardDeviation;
}

float maxentTemporalMixEstimatorStandardDeviation(float historyStandardDeviation,
        float currentStandardDeviation, float currentWeight) {
    return denoiserMixEstimatorSigma(historyStandardDeviation, currentStandardDeviation, currentWeight, 0.0);
}

// H is the reprojected previous filtered output; C is the final A-Trous center estimate of the independently
// filtered current Raw RT output. Resolve accumulates this same C into filtered history with the returned alpha.
// Both sigmas describe estimator uncertainty. No further sample-count division.
// H/C covariance is approximated as zero; adaptive weights remain an approximation.
// Raw temporal N_eff controls only the baseline update rate.
float maxentTemporalResponseAlpha(float alphaFloor, vec4 currentMoment, float currentMonteCarloStandardDeviation,
        vec4 historyMoment, float historyMonteCarloStandardDeviation,
        float temporalHistoryEffectiveSamples, out float noiseOnlyCurrentWeight) {
    noiseOnlyCurrentWeight = -1.0;
    alphaFloor = isnan(alphaFloor) || isinf(alphaFloor)
        ? 1.0 : clamp(alphaFloor, 0.0, 1.0);
    // Missing uncertainty disables only the statistical response. Preserve the
    // ordinary temporal update and geometry-driven disocclusion floor.
    if (!denoiserSigmaKnown(currentMonteCarloStandardDeviation)
            || !denoiserSigmaKnown(historyMonteCarloStandardDeviation))
        return max(alphaFloor, statisticsValidEffectiveSampleCount(temporalHistoryEffectiveSamples)
            ? 1.0 / (1.0 + temporalHistoryEffectiveSamples) : 1.0);
    if (any(isnan(currentMoment))
            || any(isinf(currentMoment))
            || any(isnan(historyMoment)) || any(isinf(historyMoment))
            || !(currentMonteCarloStandardDeviation >= 0.0) || isnan(currentMonteCarloStandardDeviation)
            || isinf(currentMonteCarloStandardDeviation)
            || !(historyMonteCarloStandardDeviation >= 0.0) || isnan(historyMonteCarloStandardDeviation)
            || isinf(historyMonteCarloStandardDeviation)
            || !statisticsValidEffectiveSampleCount(temporalHistoryEffectiveSamples))
        return alphaFloor;

    float historyVariance = maxentTemporalEstimatorVariance(historyMonteCarloStandardDeviation);
    float currentVariance = maxentTemporalEstimatorVariance(currentMonteCarloStandardDeviation);
    float distanceSq = maxentLightSampleDistanceSq(currentMoment, historyMoment);
    float combinedVariance = historyVariance + currentVariance + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
    noiseOnlyCurrentWeight = statisticsMinimumVarianceIndependentCurrentWeight(historyVariance,
        currentVariance + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE);
    float responseAlpha = mix(1.0, 1.0 / (1.0 + temporalHistoryEffectiveSamples), combinedVariance / (combinedVariance + MAXENT_TEMPORAL_RESPONSE_DISTANCE_SCALE * distanceSq));
    return max(alphaFloor, responseAlpha);
}

#endif // MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
