#ifndef MAXENT_SPATIAL_COMMON_GLSL
#define MAXENT_SPATIAL_COMMON_GLSL

#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_pdf_direction.glsl"

// Geometry-side contract. PDF direction is the sole spatial-domain geometry
// descriptor; material IDs do not participate in A-trous filtering.
struct DenoiserSpatialGeometry {
    vec3 pdfDirection;
    vec3 primaryRay;
    float roughness;
    float surfaceDistance;
    float virtualScale;
    float ggxAlpha;
    bool valid;
};

struct DenoiserSpatialBuresData {
    vec2 stddev;
    float trace;
};

struct DenoiserSpatialAccumulator {
    vec4 maxEntY;
    vec2 CoCg;
    vec2 varianceEnergy;
    float weight;
    float hitDistance;
    float hitWeight;
};

DenoiserSpatialBuresData denoiserSpatialMakeBuresData(vec4 maxEntY) {
    DenoiserSpatialBuresData data;
    float meanLength2 = dot(maxEntY.xyz, maxEntY.xyz);
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
    float traceRoot = sqrt(max(4.0 * energy2
            - 3.0 * meanLength2, 0.0));
    float analyticTrace = (2.0 * energy2 + energy * traceRoot)
            * (1.0 / 3.0) - 0.5 * meanLength2;
    float parallelExcess = 0.5 * meanLength2;
    float perpendicularVariance = max(
        (analyticTrace - parallelExcess) * (1.0 / 3.0), 0.0);
    vec2 axisVariance = vec2(perpendicularVariance,
        perpendicularVariance + parallelExcess);
    data.stddev = sqrt(axisVariance);
    // Reconstruct the trace from the clamped PSD axes. This only differs from
    // analyticTrace if independent FP16 rounding made |mean| slightly exceed
    // total energy at an earlier pack boundary.
    data.trace = 3.0 * perpendicularVariance + parallelExcess;
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
    float crossAxes = centerData.stddev.x * sampleData.stddev.y
            + centerData.stddev.y * sampleData.stddev.x;
    float cross2d = sqrt(crossAxes * crossAxes + 0.25 * meanDot * meanDot);
    float crossTrace = centerData.stddev.x * sampleData.stddev.x
            + cross2d;
    vec3 meanDelta = centerMaxEntY.xyz - sampleMaxEntY.xyz;
    // Cancellation can make an exactly-zero metric slightly negative. It must
    // not enter the variance-normalized exponent as a negative distance.
    return max(dot(meanDelta, meanDelta) + centerData.trace
            + sampleData.trace - 2.0 * crossTrace, 0.0);
}

float denoiserSpatialLightSourceToleranceScale(float roughness) {
    return 1.0;
}

vec3 denoiserSpatialVirtualWorldPosition(
        DenoiserSpatialGeometry geometry, float hitDistance) {
    float virtualWorldDistance = geometry.surfaceDistance
        + geometry.virtualScale * hitDistance;
    return geometry.primaryRay * virtualWorldDistance;
}

float denoiserSpatialVirtualRejectionScale(
        DenoiserSpatialGeometry centerGeometry, float centerHitDistance) {
    if (centerHitDistance <= 0.0) return 0.0;
    float planeTolerance = float(MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE);
    return (1.0 - centerGeometry.ggxAlpha)
        / max(planeTolerance * centerHitDistance, 1e-5);
}

vec3 denoiserSpatialVirtualNormal(vec3 tangentX, vec3 tangentY,
        vec3 fallback) {
    return denoiserSpatialSafeDirection(cross(tangentX, tangentY), fallback);
}

float denoiserSpatialVirtualPlaneDepthExponent(
        vec3 centerVirtualPosition, vec3 centerVirtualNormal,
        DenoiserSpatialGeometry sampleGeometry, float sampleHitDistance,
        float virtualRejectionScale) {
    vec3 sampleVirtualPosition = denoiserSpatialVirtualWorldPosition(
        sampleGeometry, sampleHitDistance);
    vec3 virtualDelta = sampleVirtualPosition - centerVirtualPosition;
    float planeDepth = abs(dot(centerVirtualNormal, virtualDelta));
    return virtualRejectionScale * planeDepth;
}

float denoiserSpatialWeight(DenoiserMaxEntSignal centerSignal,
    DenoiserSpatialBuresData centerBures,
    DenoiserMaxEntSignal sampleSignal,
    DenoiserSpatialBuresData sampleBures,
    DenoiserSpatialGeometry sampleGeometry, float geometryExponent,
    float kernelWeight,
    float lightSourceToleranceScale, float phiLuminance,
    float hitDistanceAlpha, vec3 centerVirtualPosition,
    vec3 centerVirtualNormal, float virtualRejectionScale,
    out float hitDistanceWeight) {
    hitDistanceWeight = kernelWeight * hitDistanceAlpha
        * exp(-geometryExponent);
    float signalExponent = geometryExponent
        + denoiserSpatialVirtualPlaneDepthExponent(
            centerVirtualPosition, centerVirtualNormal, sampleGeometry,
            sampleSignal.hitDistance, virtualRejectionScale);
    float distanceSq = denoiserSpatialBuresDistanceSq(
            centerSignal.maxEntY, centerBures,
            sampleSignal.maxEntY, sampleBures);
    float variance = centerSignal.variance + sampleSignal.variance;
    // Both variances may be exactly zero; without the floor, identical
    // signals produce 0/0 and poison the exponential with NaN.
    signalExponent += phiLuminance * distanceSq
            / max(lightSourceToleranceScale * variance, 1e-12);
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
    accum.maxEntY = center.maxEntY;
    accum.CoCg = center.CoCg;
    accum.varianceEnergy = vec2(center.variance);
    accum.weight = 1.0;
    accum.hitDistance = center.hitDistance;
    accum.hitWeight = 1.0;
    return accum;
}

void denoiserSpatialAccumulate(inout DenoiserSpatialAccumulator accum,
    DenoiserMaxEntSignal neighbor, float weight,
    float hitDistanceWeight) {
    accum.maxEntY += neighbor.maxEntY * weight;
    accum.CoCg += neighbor.CoCg * weight;
    float weightedVariance = weight * neighbor.variance;
    accum.varianceEnergy += vec2(weightedVariance,
            weight * weightedVariance);
    accum.weight += weight;
    accum.hitDistance += neighbor.hitDistance * hitDistanceWeight;
    accum.hitWeight += hitDistanceWeight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(
    DenoiserSpatialAccumulator accum, int stepRadius) {
    // Both sums start at one and only receive nonnegative exponential weights.
    float invWeight = 1.0 / accum.weight;
    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = accum.maxEntY * invWeight;
    outputSignal.CoCg = accum.CoCg * invWeight;
    float power = denoiserSpatialVariancePower(stepRadius);
    float varianceMix = 2.0 - exp2(2.0 - power);
    outputSignal.variance = mix(accum.varianceEnergy.x,
            accum.varianceEnergy.y, varianceMix) * pow(invWeight, power);
    outputSignal.hitDistance = accum.hitDistance
            / accum.hitWeight;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_SPATIAL_COMMON_GLSL
