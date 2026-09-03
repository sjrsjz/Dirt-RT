#ifndef MAXENT_DENOISER_ATROUS_FILTER_GLSL
#define MAXENT_DENOISER_ATROUS_FILTER_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/lighting/denoiser/geometry.glsl"
#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/math/statistics.glsl"

struct DenoiserSpatialAccumulator {
    vec4 maxEntY;
    vec2 CoCg;
    float monteCarloVariance;
    float weight;
    float virtualDistance;
    float virtualWeight;
};

struct DenoiserSpatialEffectiveSampleAccumulator {
    float squaredWeightOverEffectiveSamples;
    float weightOverRootEffectiveSamples;
};

float denoiserSpatialRejectionConfidenceForStep(int stepRadius) {
    if (stepRadius <= 1) return MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_16;
    return MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_32;
}

float denoiserSpatialEffectiveSampleCorrelationForStep(int stepRadius) {
    if (stepRadius <= 1) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_16;
    return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_32;
}

DenoiserSpatialEffectiveSampleAccumulator denoiserSpatialBeginEffectiveSampleAccumulation(float effectiveSamples) {
    float inverseRootSamples = inversesqrt(max(effectiveSamples, 1.0));
    DenoiserSpatialEffectiveSampleAccumulator accum;
    accum.squaredWeightOverEffectiveSamples = inverseRootSamples * inverseRootSamples;
    accum.weightOverRootEffectiveSamples = inverseRootSamples;
    return accum;
}

void denoiserSpatialAccumulateEffectiveSamples(inout DenoiserSpatialEffectiveSampleAccumulator accum,
        float effectiveSamples, float weight) {
    float inverseRootSamples = inversesqrt(max(effectiveSamples, 1.0));
    accum.squaredWeightOverEffectiveSamples += weight * weight * inverseRootSamples * inverseRootSamples;
    accum.weightOverRootEffectiveSamples += weight * inverseRootSamples;
}

float denoiserSpatialResolveEffectiveSamples(DenoiserSpatialEffectiveSampleAccumulator accum, float weightSum,
        int stepRadius) {
    float effectiveSamples = statisticsCorrelatedEffectiveSampleCount(weightSum,
        accum.squaredWeightOverEffectiveSamples, accum.weightOverRootEffectiveSamples,
        denoiserSpatialEffectiveSampleCorrelationForStep(stepRadius));
    return clamp(effectiveSamples, 1.0, DENOISER_SPATIAL_FP16_MAX);
}

vec3 denoiserSpatialVirtualWorldPosition(
    vec3 primaryRay, float virtualDistance) {
    return primaryRay * virtualDistance;
}

float denoiserSpatialVirtualRejectionScale(
    float centerGgxAlpha, float centerVirtualDistance) {
    if (centerVirtualDistance <= 0.0) return 0.0;
    float planeTolerance = float(MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE);
    return (1.0 - centerGgxAlpha)
        / max(planeTolerance * centerVirtualDistance, 1e-5);
}

vec3 denoiserSpatialVirtualNormal(vec3 tangentX, vec3 tangentY, vec3 fallback) {
    return denoiserSpatialSafeDirection(cross(tangentX, tangentY), fallback);
}

float denoiserSpatialVirtualPlaneDepthExponent(
    vec3 centerVirtualPosition, vec3 centerVirtualNormal,
    vec3 samplePrimaryRay, float sampleVirtualDistance,
    float virtualRejectionScale) {
    vec3 sampleVirtualPosition = denoiserSpatialVirtualWorldPosition(
            samplePrimaryRay, sampleVirtualDistance);
    vec3 virtualDelta = sampleVirtualPosition - centerVirtualPosition;
    float planeDepth = abs(dot(centerVirtualNormal, virtualDelta));
    return virtualRejectionScale * planeDepth;
}

float denoiserSpatialWeight(DenoiserMaxEntSignal centerSignal,
    DenoiserMaxEntSignal sampleSignal,
    vec3 samplePrimaryRay, float surfaceGeometryExponent,
    float kernelWeight,
    float phiLuminance,
    float virtualDistanceAlpha, vec3 centerVirtualPosition,
    vec3 centerVirtualNormal, float virtualRejectionScale,
    out float virtualDistanceWeight) {

    float signalExponent = surfaceGeometryExponent
        + denoiserSpatialVirtualPlaneDepthExponent(
            centerVirtualPosition, centerVirtualNormal, samplePrimaryRay,
            sampleSignal.virtualDistance, virtualRejectionScale);
    float distanceSq = maxentLightSampleDistanceSq(
            centerSignal.maxEntY, sampleSignal.maxEntY);
    // Difference noise uses both MC variance fields. phiLuminance already contains sqrt(N_eff_center), so the
    // complete ratio is N_eff_center*distanceSq/(V_MC_center+V_MC_sample); the sample N_eff is intentionally absent.
    float combinedMonteCarloVariance = centerSignal.standardDeviation * centerSignal.standardDeviation
        + sampleSignal.standardDeviation * sampleSignal.standardDeviation;
    signalExponent += phiLuminance * sqrt(distanceSq / max(combinedMonteCarloVariance, 1e-20));
    float surfaceWeight = kernelWeight * exp(-signalExponent);
    virtualDistanceWeight = kernelWeight * virtualDistanceAlpha
        * exp(-surfaceGeometryExponent / max(virtualDistanceAlpha, 1e-5));
    return surfaceWeight;
}

DenoiserSpatialAccumulator denoiserSpatialBeginAccumulation(
    DenoiserMaxEntSignal center) {
    DenoiserSpatialAccumulator accum;
    accum.maxEntY = center.maxEntY;
    accum.CoCg = center.CoCg;
    // MC variance is an independently supplied field. It is filtered with the
    // signal weights and is never inverted into an assumed second moment.
    accum.monteCarloVariance = center.standardDeviation * center.standardDeviation;
    accum.weight = 1.0;
    accum.virtualDistance = center.virtualDistance;
    accum.virtualWeight = 1.0;
    return accum;
}

void denoiserSpatialAccumulate(inout DenoiserSpatialAccumulator accum,
    DenoiserMaxEntSignal neighbor, float weight,
    float virtualDistanceWeight) {
    accum.maxEntY += neighbor.maxEntY * weight;
    accum.CoCg += neighbor.CoCg * weight;
    accum.monteCarloVariance += weight * neighbor.standardDeviation * neighbor.standardDeviation;
    accum.weight += weight;
    accum.virtualDistance += neighbor.virtualDistance
        * virtualDistanceWeight;
    accum.virtualWeight += virtualDistanceWeight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(DenoiserSpatialAccumulator accum) {
    // Both sums start at one and only receive nonnegative exponential weights.
    float invWeight = 1.0 / accum.weight;
    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = accum.maxEntY * invWeight;
    outputSignal.CoCg = accum.CoCg * invWeight;
    outputSignal.standardDeviation = sqrt(accum.monteCarloVariance * invWeight);
    outputSignal.virtualDistance = accum.virtualDistance
        / accum.virtualWeight;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_DENOISER_ATROUS_FILTER_GLSL
