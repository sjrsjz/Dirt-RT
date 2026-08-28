#ifndef MAXENT_DENOISER_ATROUS_FILTER_GLSL
#define MAXENT_DENOISER_ATROUS_FILTER_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/lighting/denoiser/geometry.glsl"
#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/math/statistics.glsl"

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

struct DenoiserSpatialAccumulator {
    vec4 maxEntY;
    vec2 CoCg;
    vec2 varianceEnergy;
    float weight;
    float virtualDistance;
    float virtualWeight;
};

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
    float momentCorrelation,
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
    float variance = statisticsDifferenceVarianceFromStandardDeviations(
            centerSignal.standardDeviation,
            sampleSignal.standardDeviation,
            momentCorrelation);
    // Both variances may be exactly zero; without the floor, identical
    // signals produce 0/0 and poison the exponential with NaN.
    signalExponent += phiLuminance * sqrt(distanceSq / max(variance, 1e-20));
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
    // x = sum w_i^2 V_i, y = sum w_i sqrt(V_i).  These are the two
    // sufficient accumulators for the constant-correlation expansion.
    accum.varianceEnergy = vec2(
        center.standardDeviation * center.standardDeviation,
        center.standardDeviation);
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
    accum.varianceEnergy += vec2(
            weight * weight * neighbor.standardDeviation * neighbor.standardDeviation,
            weight * neighbor.standardDeviation);
    accum.weight += weight;
    accum.virtualDistance += neighbor.virtualDistance
        * virtualDistanceWeight;
    accum.virtualWeight += virtualDistanceWeight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(
    DenoiserSpatialAccumulator accum, float propagationCorrelation) {
    // Both sums start at one and only receive nonnegative exponential weights.
    float invWeight = 1.0 / accum.weight;
    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = accum.maxEntY * invWeight;
    outputSignal.CoCg = accum.CoCg * invWeight;
    outputSignal.standardDeviation = statisticsWeightedMeanStandardDeviation(
        accum.varianceEnergy.x, accum.varianceEnergy.y, invWeight, propagationCorrelation);
    outputSignal.virtualDistance = accum.virtualDistance
        / accum.virtualWeight;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_DENOISER_ATROUS_FILTER_GLSL
