#ifndef MAXENT_SPATIAL_COMMON_GLSL
#define MAXENT_SPATIAL_COMMON_GLSL

#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"
#include "/lib/lighting/denoiser/maxent_bures.glsl"

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
    DenoiserSpatialBuresData centerBures,
    DenoiserMaxEntSignal sampleSignal,
    DenoiserSpatialBuresData sampleBures,
    vec3 samplePrimaryRay, float surfaceGeometryExponent,
    float kernelWeight,
    float phiLuminance,
    float virtualDistanceAlpha, vec3 centerVirtualPosition,
    vec3 centerVirtualNormal, float virtualRejectionScale,
    out float virtualDistanceWeight) {
    virtualDistanceWeight = kernelWeight * virtualDistanceAlpha
        * exp(-surfaceGeometryExponent);
    float signalExponent = surfaceGeometryExponent
        + denoiserSpatialVirtualPlaneDepthExponent(
            centerVirtualPosition, centerVirtualNormal, samplePrimaryRay,
            sampleSignal.virtualDistance, virtualRejectionScale);
    float distanceSq = denoiserSpatialBuresDistanceSq(
            centerSignal.maxEntY, centerBures,
            sampleSignal.maxEntY, sampleBures);
    float variance = centerSignal.variance + sampleSignal.variance;
    // Both variances may be exactly zero; without the floor, identical
    // signals produce 0/0 and poison the exponential with NaN.
    signalExponent += phiLuminance * (distanceSq / max(variance, 1e-12));
    return kernelWeight * exp(-signalExponent);
}

float denoiserSpatialVariancePower(int stepRadius) {
    float coefficient;
    if (stepRadius <= 1) coefficient = 0.3339015144;
    else if (stepRadius <= 2) coefficient = 0.4375201036;
    else if (stepRadius <= 4) coefficient = 0.4592464660;
    else if (stepRadius <= 8) coefficient = 0.4644479501;
    else if (stepRadius <= 16) coefficient = 0.4657344365;
    else coefficient = 0.4660551979;
    // The exposed adaptation range is [0, 2], while coefficient <=
    // 0.4660551979, so this expression is bounded below by 1.0678896042.
    return 2.0 - MAXENT_SPATIAL_VARIANCE_ADAPTATION * coefficient;
}

DenoiserSpatialAccumulator denoiserSpatialBeginAccumulation(
    DenoiserMaxEntSignal center) {
    DenoiserSpatialAccumulator accum;
    // The largest supported kernel mass is 7.2517605. Scaling every lighting
    // term by the exact power of two 1/8 keeps all sanitized FP16 inputs below
    // overflow while the six long-lived lighting accumulators stay packed.
    accum.maxEntY = f16vec4(center.maxEntY * 0.125);
    accum.CoCg = f16vec2(center.CoCg * 0.125);
    accum.varianceEnergy = vec2(center.variance);
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
    float weightedVariance = weight * neighbor.variance;
    accum.varianceEnergy += vec2(weightedVariance,
            weight * weightedVariance);
    accum.weight += weight;
    accum.virtualDistance += neighbor.virtualDistance
        * virtualDistanceWeight;
    accum.virtualWeight += virtualDistanceWeight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(
    DenoiserSpatialAccumulator accum, int stepRadius) {
    // Both sums start at one and only receive nonnegative exponential weights.
    float invWeight = 1.0 / accum.weight;
    DenoiserMaxEntSignal outputSignal;
    float lightNormalization = 8.0 * invWeight;
    outputSignal.maxEntY = vec4(accum.maxEntY) * lightNormalization;
    outputSignal.CoCg = vec2(accum.CoCg) * lightNormalization;
    float power = denoiserSpatialVariancePower(stepRadius);
    float varianceMix = 2.0 - exp2(2.0 - power);
    outputSignal.variance = mix(accum.varianceEnergy.x, accum.varianceEnergy.y, varianceMix) * pow(invWeight, power);
    outputSignal.virtualDistance = accum.virtualDistance
        / accum.virtualWeight;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_SPATIAL_COMMON_GLSL
