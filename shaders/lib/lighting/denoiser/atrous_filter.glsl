#ifndef MAXENT_DENOISER_ATROUS_FILTER_GLSL
#define MAXENT_DENOISER_ATROUS_FILTER_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/atrous_policy.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/light_difference.glsl"
#include "/lib/lighting/denoiser/geometry.glsl"
#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/math/statistics.glsl"

// Trusted spatial weights are finite, nonnegative, and bounded by the
// kernel coefficient. They share the moment accumulator's total weight.
// Unknown uncertainty still owns squared weights and borrows only known sigma.
struct DenoiserSpatialVariance {
    float independentVariance;
    float weightedSigma;
    float knownWeight;
    float unknownSquaredWeight;
};

void denoiserSpatialAccumulateVariance(inout DenoiserSpatialVariance variance,
        float sigma, float weight) {
    // Preparation sanitizes sigma; invalid records are rejected before this
    // update. The remaining values are finite nonnegative FP16 or exactly -2.
    if (sigma >= 0.0) {
        float weightedSigma = weight * sigma;
        variance.independentVariance += weightedSigma * weightedSigma;
        variance.weightedSigma += weightedSigma;
        variance.knownWeight += weight;
    } else {
        variance.unknownSquaredWeight += weight * weight;
    }
}

struct DenoiserSpatialAccumulator {
    vec4 maxEntY;
    DenoiserSpatialVariance uncertainty;
    float weight;
    float virtualDistance;
    float virtualWeight;
};


// The current estimator owns the output chroma; only the proposal guides
// virtual geometry. No current virtual-distance statistic survives resolve.
struct DenoiserSpatialCurrentAccumulator {
    vec4 maxEntY;
    vec2 CoCg;
    DenoiserSpatialVariance uncertainty;
    float weight;
};

float denoiserSpatialEffectiveSampleCorrelationForStep(int stepRadius) {
    if (stepRadius <= 1) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_16;
    return MAXENT_SPATIAL_EFFECTIVE_SAMPLE_CORRELATION_STEP_32;
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

float denoiserSpatialWeight(MaxEntLightMetric centerMetric,
    DenoiserMaxEntSignal centerSignal,
    DenoiserMaxEntSignal sampleSignal,
    vec3 samplePrimaryRay, float surfaceGeometryExponent,
    float kernelWeight,
    float phiLuminance,
    float virtualDistanceAlpha, vec3 centerVirtualPosition,
    vec3 centerVirtualNormal, float virtualRejectionScale,
    out float virtualDistanceWeight) {

    float signalExponent = surfaceGeometryExponent;
    signalExponent += denoiserSpatialVirtualPlaneDepthExponent(
            centerVirtualPosition, centerVirtualNormal, samplePrimaryRay,
            sampleSignal.virtualDistance, virtualRejectionScale);
    if (centerSignal.standardDeviation >= 0.0
            && sampleSignal.standardDeviation >= 0.0) {
        float distanceSq = maxentLightSampleDistanceSq(
                centerMetric, maxentPrepareLightMetric(sampleSignal.maxEntY));
        // Endpoints already store estimator uncertainty at this kernel level.
        // Endpoint covariance is omitted (independent-difference approximation).
        float differenceEstimatorVariance =
            centerSignal.standardDeviation * centerSignal.standardDeviation
            + sampleSignal.standardDeviation * sampleSignal.standardDeviation;
        signalExponent += phiLuminance
            * sqrt(distanceSq / max(differenceEstimatorVariance, 1e-20));
    }
    float surfaceWeight = kernelWeight * exp(-signalExponent);
    virtualDistanceWeight = kernelWeight * virtualDistanceAlpha
        * exp(-surfaceGeometryExponent / max(virtualDistanceAlpha, 1e-5));
    return surfaceWeight;
}

DenoiserSpatialAccumulator denoiserSpatialBeginAccumulation(
    DenoiserMaxEntSignal center) {
    DenoiserSpatialAccumulator accum;
    accum.maxEntY = center.maxEntY;
    accum.uncertainty = DenoiserSpatialVariance(0.0, 0.0, 0.0, 0.0);
    denoiserSpatialAccumulateVariance(accum.uncertainty, center.standardDeviation, 1.0);
    accum.weight = 1.0;
    accum.virtualDistance = center.virtualDistance;
    accum.virtualWeight = 1.0;
    return accum;
}

void denoiserSpatialAccumulate(inout DenoiserSpatialAccumulator accum,
    DenoiserMaxEntSignal neighbor, float weight,
    float virtualDistanceWeight) {
    accum.maxEntY += neighbor.maxEntY * weight;
    denoiserSpatialAccumulateVariance(accum.uncertainty, neighbor.standardDeviation, weight);
    accum.weight += weight;
    accum.virtualDistance += neighbor.virtualDistance
        * virtualDistanceWeight;
    accum.virtualWeight += virtualDistanceWeight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(DenoiserSpatialAccumulator accum, int stepRadius) {
    // Both sums start at one and only receive nonnegative exponential weights.
    float invWeight = 1.0 / accum.weight;
    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = accum.maxEntY * invWeight;
    // Proposal chroma has no consumer: only the current estimator is resolved.
    outputSignal.CoCg = vec2(0.0);
    outputSignal.standardDeviation = denoiserResolveEstimatorSigma(DenoiserEstimatorVarianceAccumulator(
        accum.uncertainty.independentVariance, accum.uncertainty.weightedSigma,
        accum.uncertainty.knownWeight, accum.weight, accum.uncertainty.unknownSquaredWeight),
        denoiserSpatialEffectiveSampleCorrelationForStep(stepRadius));
    outputSignal.virtualDistance = accum.virtualDistance
        / accum.virtualWeight;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}


DenoiserSpatialCurrentAccumulator denoiserSpatialBeginCurrentAccumulation(
    DenoiserMaxEntSignal center) {
    DenoiserSpatialCurrentAccumulator accum;
    accum.maxEntY = center.maxEntY;
    accum.CoCg = center.CoCg;
    accum.uncertainty = DenoiserSpatialVariance(0.0, 0.0, 0.0, 0.0);
    denoiserSpatialAccumulateVariance(accum.uncertainty, center.standardDeviation, 1.0);
    accum.weight = 1.0;
    return accum;
}

void denoiserSpatialAccumulate(inout DenoiserSpatialCurrentAccumulator accum,
    DenoiserMaxEntSignal neighbor, float weight,
    float virtualDistanceWeight) {
    accum.maxEntY += neighbor.maxEntY * weight;
    accum.CoCg += neighbor.CoCg * weight;
    denoiserSpatialAccumulateVariance(accum.uncertainty, neighbor.standardDeviation, weight);
    accum.weight += weight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(DenoiserSpatialCurrentAccumulator accum, int stepRadius) {
    float invWeight = 1.0 / accum.weight;
    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = accum.maxEntY * invWeight;
    outputSignal.CoCg = accum.CoCg * invWeight;
    outputSignal.standardDeviation = denoiserResolveEstimatorSigma(DenoiserEstimatorVarianceAccumulator(
        accum.uncertainty.independentVariance, accum.uncertainty.weightedSigma,
        accum.uncertainty.knownWeight, accum.weight, accum.uncertainty.unknownSquaredWeight),
        denoiserSpatialEffectiveSampleCorrelationForStep(stepRadius));
    // Persistent outputs either replace this half with their history tag or
    // derive virtual distance from primary geometry and the committed hit.
    outputSignal.virtualDistance = 0.0;
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_DENOISER_ATROUS_FILTER_GLSL
