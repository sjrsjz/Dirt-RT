#ifndef MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
#define MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/math/statistics.glsl"

// The denoiser operates on m=(E[R u], E[R]). No decoder distribution is
// involved. For one raw sample z=(R u,R), |z|^2=2R^2.

float maxentTemporalCurrentVariance(float standardDeviation) {
    standardDeviation = max(standardDeviation, 0.0);
    return standardDeviation * standardDeviation + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
}

float maxentTemporalProposalCorrectedStandardDeviation(float proposalStandardDeviation, float currentStandardDeviation,
        float correctionCurrentWeight) {
    float proposalVariance = max(proposalStandardDeviation, 0.0) * max(proposalStandardDeviation, 0.0);
    float currentVariance = maxentTemporalCurrentVariance(currentStandardDeviation);
    // The configured statistical closure assumes zero correlation.
    return sqrt(statisticsIndependentBlendVariance(proposalVariance, currentVariance, correctionCurrentWeight));
}

// H is the reprojected previous filtered output; C is the independently spatially filtered current Raw RT output.
// Spatial passes have already propagated both variance inputs. Only H represents a temporal weighted mean, so
// its propagated variance is divided by the reprojected Kish N_eff here. C is an independent current frame.
// noiseOnlyCurrentWeight exposes V_H/(V_H+V_C) for diagnostics. The returned current weight is the user-tuned
// response: large denoised moment differences approach 1, while a stationary signal approaches 1/(N_eff+1).
float maxentTemporalResponseAlpha(float alphaFloor, vec4 currentMoment, float currentPropagatedVariance,
        vec4 historyMoment, float historyPropagatedStandardDeviation, float historyEffectiveSamples,
        out float noiseOnlyCurrentWeight) {
    noiseOnlyCurrentWeight = -1.0;
    alphaFloor = isnan(alphaFloor) || isinf(alphaFloor)
        ? 1.0 : clamp(alphaFloor, 0.0, 1.0);
    if (any(isnan(currentMoment))
            || any(isinf(currentMoment))
            || any(isnan(historyMoment)) || any(isinf(historyMoment))
            || !(currentPropagatedVariance >= 0.0)
            || isnan(currentPropagatedVariance)
            || isinf(currentPropagatedVariance)
            || !(historyPropagatedStandardDeviation >= 0.0) || isnan(historyPropagatedStandardDeviation)
            || isinf(historyPropagatedStandardDeviation)
            || !statisticsValidEffectiveSampleCount(
                historyEffectiveSamples))
        return alphaFloor;

    float historyVariance = historyPropagatedStandardDeviation * historyPropagatedStandardDeviation
        / historyEffectiveSamples;
    float currentVariance = currentPropagatedVariance + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
    float distanceSq = maxentLightSampleDistanceSq(currentMoment, historyMoment);
    noiseOnlyCurrentWeight = statisticsMinimumVarianceIndependentCurrentWeight(historyVariance, currentVariance);
    float combinedVariance = historyVariance + currentVariance;
    float responseAlpha = mix(1.0, 1.0 / (1.0 + historyEffectiveSamples), combinedVariance / (combinedVariance + MAXENT_TEMPORAL_RESPONSE_DISTANCE_SCALE * distanceSq));
    return max(alphaFloor, responseAlpha);
}

#endif // MAXENT_DENOISER_TEMPORAL_RESPONSE_GLSL
