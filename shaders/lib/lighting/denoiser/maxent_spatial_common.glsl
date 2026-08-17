#ifndef MAXENT_SPATIAL_COMMON_GLSL
#define MAXENT_SPATIAL_COMMON_GLSL

#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"

struct DenoiserSpatialBuresData {
    vec2 stddev;
    float trace;
};

struct DenoiserSpatialAccumulator {
    f16vec4 maxEntY;
    f16vec2 CoCg;
    vec2 varianceEnergy;
    float weight;
    float virtualDistance;
    float virtualWeight;
};

DenoiserSpatialBuresData denoiserSpatialMakeBuresData(vec4 maxEntY) {
    DenoiserSpatialBuresData data;
    float parallelExcess = 0.5 * dot(maxEntY.xyz, maxEntY.xyz);
    // Spatial inputs have crossed the sanitizing pack boundary, which owns
    // the nonnegative total-energy invariant.
    float energy = maxEntY.w;
    float energy2 = energy * energy;
    // Eliminate the intermediate rho/kappa reconstruction. Substituting
    // rho=4*kappa/(3+kappa^2) into maxent_eigen_std gives the covariance
    // trace directly from the stored first moment and total energy.
    // Independent FP16 rounding can move an otherwise valid moment just
    // outside its cone (and subnormals can violate it substantially), so this
    // is a real sqrt-domain guard rather than a redundant nonnegative clamp.
    float traceRoot = sqrt(max(4.0 * energy2 - 1.5 * parallelExcess, 0.0));
    float analyticTrace = (2.0 * energy2 + energy * traceRoot) * (1.0 / 3.0) - parallelExcess;
    float perpendicularVariance = max((analyticTrace - parallelExcess) * (1.0 / 3.0), 0.0);
    // Reconstruct the trace from the clamped PSD axes. This only differs from
    // analyticTrace if independent FP16 rounding made |mean| slightly exceed
    // total energy at an earlier pack boundary.
    data.trace = 3.0 * perpendicularVariance + parallelExcess;
    vec2 axisVariance = vec2(perpendicularVariance, perpendicularVariance + parallelExcess);
    data.stddev = sqrt(axisVariance);
    return data;
}

DenoiserSpatialBuresData denoiserSpatialMakeBuresDataFromStddev(vec2 stddev) {
    DenoiserSpatialBuresData data;
    data.stddev = stddev;
    vec2 axisVariance = stddev * stddev;
    data.trace = 2.0 * axisVariance.x + axisVariance.y;
    return data;
}

float denoiserSpatialBuresDistanceSq(vec4 centerMaxEntY,
    DenoiserSpatialBuresData centerData, vec4 sampleMaxEntY,
    DenoiserSpatialBuresData sampleData) {
    // For this MaxEnt model rho = 4*kappa/(3+kappa^2), while
    // sigma_parallel^2 - sigma_perpendicular^2
    //     = 8*omega^2*kappa^2/(3+kappa^2)^2
    //     = 0.5*|mean|^2.
    // Hence cos(theta)^2*A_center*A_sample is exactly
    // 0.25*dot(mean_center, mean_sample)^2. No normalized mean length or
    // separately retained anisotropy is required.
    float meanDot = dot(centerMaxEntY.xyz, sampleMaxEntY.xyz);
    float crossAxes = centerData.stddev.x * sampleData.stddev.y + centerData.stddev.y * sampleData.stddev.x;
    float cross2d = sqrt(crossAxes * crossAxes + 0.25 * meanDot * meanDot);
    float crossTrace = centerData.stddev.x * sampleData.stddev.x + cross2d;
    vec3 meanDelta = centerMaxEntY.xyz - sampleMaxEntY.xyz;
    // Cancellation can make an exactly-zero metric slightly negative. It must
    // not enter the variance-normalized exponent as a negative distance.
    return max(dot(meanDelta, meanDelta) + centerData.trace + sampleData.trace - 2.0 * crossTrace, 0.0);
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
    signalExponent += phiLuminance * distanceSq / max(variance, 1e-12);
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
