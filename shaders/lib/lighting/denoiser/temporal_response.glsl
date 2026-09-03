#ifndef MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
#define MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/math/statistics.glsl"

// The denoiser operates on m=(E[R u], E[R]). No decoder distribution is
// involved. For one raw sample z=(R u,R), |z|^2=2R^2.

float maxentTemporalEstimatorVariance(float monteCarloStandardDeviation, float effectiveSamples) {
    monteCarloStandardDeviation = max(monteCarloStandardDeviation, 0.0);
    return monteCarloStandardDeviation * monteCarloStandardDeviation / max(effectiveSamples, 1.0);
}

float maxentTemporalMixMonteCarloStandardDeviation(float historyStandardDeviation,
        float currentStandardDeviation, float currentWeight) {
    float historyVariance = max(historyStandardDeviation, 0.0) * max(historyStandardDeviation, 0.0);
    float currentVariance = max(currentStandardDeviation, 0.0) * max(currentStandardDeviation, 0.0);
    return sqrt(mix(historyVariance, currentVariance, clamp(currentWeight, 0.0, 1.0)));
}

// H is the reprojected previous filtered output; C is the final A-Trous center estimate of the independently
// filtered current Raw RT output. Resolve accumulates this same C into filtered history with the returned alpha.
// Their standard deviations both describe linearly filtered MC observation variance. Estimator variances are
// constructed only here by dividing each MC variance by its own Kish N_eff.
// noiseOnlyCurrentWeight exposes V_H/(V_H+V_C) for diagnostics. The returned current weight is the user-tuned
// response: large denoised moment differences approach 1, while a stationary signal approaches the Raw temporal
// update 1/(N_eff+1). Filtered-estimator N_eff controls only innovation variance; it must not replace Raw history N_eff.
float maxentTemporalResponseAlpha(float alphaFloor, vec4 currentMoment, float currentMonteCarloStandardDeviation,
        float currentEffectiveSamples, vec4 historyMoment, float historyMonteCarloStandardDeviation,
        float historyEffectiveSamples, float temporalHistoryEffectiveSamples, out float noiseOnlyCurrentWeight) {
    noiseOnlyCurrentWeight = -1.0;
    alphaFloor = isnan(alphaFloor) || isinf(alphaFloor)
        ? 1.0 : clamp(alphaFloor, 0.0, 1.0);
    if (any(isnan(currentMoment))
            || any(isinf(currentMoment))
            || any(isnan(historyMoment)) || any(isinf(historyMoment))
            || !(currentMonteCarloStandardDeviation >= 0.0) || isnan(currentMonteCarloStandardDeviation)
            || isinf(currentMonteCarloStandardDeviation)
            || !(historyMonteCarloStandardDeviation >= 0.0) || isnan(historyMonteCarloStandardDeviation)
            || isinf(historyMonteCarloStandardDeviation)
            || !statisticsValidEffectiveSampleCount(currentEffectiveSamples)
            || !statisticsValidEffectiveSampleCount(historyEffectiveSamples)
            || !statisticsValidEffectiveSampleCount(temporalHistoryEffectiveSamples))
        return alphaFloor;

    float historyVariance = maxentTemporalEstimatorVariance(historyMonteCarloStandardDeviation, historyEffectiveSamples);
    float currentVariance = maxentTemporalEstimatorVariance(currentMonteCarloStandardDeviation, currentEffectiveSamples);
    float distanceSq = maxentLightSampleDistanceSq(currentMoment, historyMoment);
    float combinedVariance = historyVariance + currentVariance + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
    noiseOnlyCurrentWeight = statisticsMinimumVarianceIndependentCurrentWeight(historyVariance,
        currentVariance + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE);
    float responseAlpha = mix(1.0, 1.0 / (1.0 + temporalHistoryEffectiveSamples), combinedVariance / (combinedVariance + MAXENT_TEMPORAL_RESPONSE_DISTANCE_SCALE * distanceSq));
    return max(alphaFloor, responseAlpha);
}

#endif // MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
