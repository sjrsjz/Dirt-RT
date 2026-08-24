#ifndef MAXENT_SPATIAL_COMMON_GLSL
#define MAXENT_SPATIAL_COMMON_GLSL

#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"
#include "/lib/lighting/denoiser/maxent_moment_statistics.glsl"

struct DenoiserSpatialAccumulator {
    f16vec4 maxEntY;
    f16vec2 CoCg;
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
    float distanceSq = maxentMomentDistanceSq(
            centerSignal.maxEntY, sampleSignal.maxEntY);
    float variance = statisticsDifferenceVarianceFromStandardDeviations(
            centerSignal.standardDeviation,
            sampleSignal.standardDeviation,
            momentCorrelation);
    // Both variances may be exactly zero; without the floor, identical
    // signals produce 0/0 and poison the exponential with NaN.
    signalExponent += phiLuminance * sqrt(distanceSq / max(variance, 1e-12));
    float surfaceWeight = kernelWeight * exp(-signalExponent);
    virtualDistanceWeight = kernelWeight * virtualDistanceAlpha
        * exp(-surfaceGeometryExponent / max(virtualDistanceAlpha, 1e-5));
    return surfaceWeight;
}

DenoiserSpatialAccumulator denoiserSpatialBeginAccumulation(
    DenoiserMaxEntSignal center) {
    DenoiserSpatialAccumulator accum;
    // The largest supported kernel mass is 7.2517605. Scaling every lighting
    // term by the exact power of two 1/8 keeps all sanitized FP16 inputs below
    // overflow while the six long-lived lighting accumulators stay packed.
    accum.maxEntY = f16vec4(center.maxEntY * 0.125);
    accum.CoCg = f16vec2(center.CoCg * 0.125);
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
    float scaledWeight = weight * 0.125;
    accum.maxEntY += f16vec4(neighbor.maxEntY * scaledWeight);
    accum.CoCg += f16vec2(neighbor.CoCg * scaledWeight);
    accum.varianceEnergy += vec2(
            weight * weight * neighbor.standardDeviation
                * neighbor.standardDeviation,
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
    float lightNormalization = 8.0 * invWeight;
    outputSignal.maxEntY = vec4(accum.maxEntY) * lightNormalization;
    outputSignal.CoCg = vec2(accum.CoCg) * lightNormalization;
    outputSignal.standardDeviation =
        statisticsWeightedMeanStandardDeviation(
            accum.varianceEnergy.x, accum.varianceEnergy.y, invWeight,
            propagationCorrelation);
    outputSignal.virtualDistance = accum.virtualDistance
        / accum.virtualWeight;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_SPATIAL_COMMON_GLSL
