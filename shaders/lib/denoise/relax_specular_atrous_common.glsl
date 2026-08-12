#ifndef RELAX_SPECULAR_ATROUS_COMMON_GLSL
#define RELAX_SPECULAR_ATROUS_COMMON_GLSL

#include "/lib/denoise/relax_specular_common.glsl"
#include "/lib/lighting/alice.glsl"

uniform sampler2D colortex9;
#if RELAX_ATROUS_INPUT == 5
uniform usampler2D colortex5;
#else
uniform usampler2D colortex6;
#endif

#if RELAX_ATROUS_OUTPUT == 5
layout(rgba32ui) uniform writeonly uimage2D colorimg5;
#else
layout(rgba32ui) uniform writeonly uimage2D colorimg6;
#endif

struct RelaxAtrousBuresData {
    vec2 stddev;
    float trace;
    float anisotropy;
    float invMeanLength2;
};

uvec4 relaxFetchAtrousWords(ivec2 pixel) {
#if RELAX_ATROUS_INPUT == 5
    return texelFetch(colortex5, pixel, 0);
#else
    return texelFetch(colortex6, pixel, 0);
#endif
}

void relaxStoreAtrous(ivec2 pixel, RelaxSpatialSignal signal) {
#if RELAX_ATROUS_OUTPUT == 5
    imageStore(colorimg5, pixel, relaxPackSpatial(signal));
#else
    imageStore(colorimg6, pixel, relaxPackSpatial(signal));
#endif
}

void relaxFinishAtrous(ivec2 pixel, RelaxSpatialSignal signal) {
    relaxStoreAtrous(pixel, signal);
#if DEBUG_VIEW == RELAX_ATROUS_DEBUG_VIEW
    writeReflLight(uvec2(pixel), specularMaxEntTotalRgb(signal.signal),
        signal.hitDistance, 1.0);
#endif
#if defined(RELAX_ATROUS_RESOLVE)
#if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14 || \
        (DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30)
    // Preserve the diagnostic value written by its owning pass.
#else
    writeReflMaxEnt(uvec2(pixel), signal.signal, signal.hitDistance, 1.0);
#endif
#endif
}

bool relaxAtrousGeometryValid(vec4 geometry) {
    return any(notEqual(geometry.xyz, vec3(0.0)))
        || floatBitsToUint(geometry.w) != 0u;
}

RelaxAtrousBuresData relaxMakeAtrousBuresData(vec4 aliceY) {
    RelaxAtrousBuresData data;
    float meanLength2 = dot(aliceY.xyz, aliceY.xyz);
    float rho = sqrt(meanLength2) / max(aliceY.w, 1e-8);
    rho = clamp(rho, 0.0, 1.0 - 1e-6);
    float kappa = 3.0 * rho /
        (2.0 + sqrt(max(4.0 - 3.0 * rho * rho, 1e-12)));
    data.stddev = alice_eigen_std(max(aliceY.w, 0.0), kappa);
    vec2 variance = data.stddev * data.stddev;
    data.trace = 2.0 * variance.x + variance.y;
    data.anisotropy = variance.y - variance.x;
    data.invMeanLength2 = meanLength2 > 1e-16
        ? 1.0 / meanLength2 : 0.0;
    return data;
}

float relaxAtrousBuresDistanceSq(vec4 centerAliceY,
        RelaxAtrousBuresData centerData, vec4 sampleAliceY,
        RelaxAtrousBuresData sampleData) {
    float directionCosine2 = 0.0;
    if (centerData.invMeanLength2 > 0.0
            && sampleData.invMeanLength2 > 0.0) {
        float meanDot = dot(centerAliceY.xyz, sampleAliceY.xyz);
        directionCosine2 = min(1.0, meanDot * meanDot
            * centerData.invMeanLength2 * sampleData.invMeanLength2);
    }

    float crossAxes = centerData.stddev.x * sampleData.stddev.y
        + centerData.stddev.y * sampleData.stddev.x;
    float cross2d = sqrt(max(crossAxes * crossAxes
        + directionCosine2 * centerData.anisotropy
            * sampleData.anisotropy, 0.0));
    float crossTrace = centerData.stddev.x * sampleData.stddev.x
        + cross2d;
    vec3 meanDelta = centerAliceY.xyz - sampleAliceY.xyz;
    return max(dot(meanDelta, meanDelta) + centerData.trace
        + sampleData.trace - 2.0 * crossTrace, 0.0);
}

float relaxAtrousGeometryExponent(vec3 centerPosition, vec3 centerNormal,
        vec3 samplePosition) {
    return relaxSpatialPlaneExponent(centerPosition,
        centerNormal, samplePosition);
}

float relaxAtrousLightExponent(RelaxSpatialSignal center,
        RelaxAtrousBuresData centerData, RelaxSpatialSignal sampleSignal,
        RelaxAtrousBuresData sampleData) {
    float distanceSq = relaxAtrousBuresDistanceSq(center.signal.aliceY,
        centerData, sampleSignal.signal.aliceY, sampleData);
    float varianceScale = center.variance + sampleSignal.variance;
    float energyScale = max(center.signal.aliceY.w * center.signal.aliceY.w,
        sampleSignal.signal.aliceY.w * sampleSignal.signal.aliceY.w);
    varianceScale = max(varianceScale, energyScale * 1e-6 + 1e-12);
    // RELAX_SPEC_PHI_LUMINANCE is a tolerance, hence it divides the exponent.
    return distanceSq / max(RELAX_SPEC_PHI_LUMINANCE * varianceScale, 1e-12);
}

float relaxAtrousSpecularWeight(RelaxSpatialSignal center,
        RelaxAtrousBuresData centerData, vec3 centerPosition,
        vec3 centerNormal, float centerRoughness,
        RelaxSpatialSignal sampleSignal, RelaxAtrousBuresData sampleData,
        vec3 samplePosition, float sampleRoughness, float kernelWeight) {
    float exponent = relaxAtrousGeometryExponent(centerPosition,
        centerNormal, samplePosition);
    exponent += relaxAtrousLightExponent(center, centerData,
        sampleSignal, sampleData);

    vec2 roughnessParams = relaxRoughnessWeightParams(centerRoughness,
        RELAX_ROUGHNESS_FRACTION);
    float roughnessWeight = relaxExponentialWeight(sampleRoughness,
        roughnessParams);

    // Hit distance remains a separate mirror-GI constraint. It prevents
    // different virtual depths from mixing while retaining a nonzero floor.
    float hitWeight = relaxHitDistanceWeight(center.hitDistance,
        sampleSignal.hitDistance, centerRoughness);

    return kernelWeight * exp(-exponent) * roughnessWeight * hitWeight;
}

float relaxAtrousVariancePower() {
#if RELAX_ATROUS_STEP == 1
    const float coefficient = 0.3339015144;
#elif RELAX_ATROUS_STEP == 2
    const float coefficient = 0.4375201036;
#elif RELAX_ATROUS_STEP == 4
    const float coefficient = 0.4592464660;
#elif RELAX_ATROUS_STEP == 8
    const float coefficient = 0.4644479501;
#else
    const float coefficient = 0.4657344365;
#endif
    return max(1.0, 2.0 - ATROUS_GAMMA * coefficient);
}

float relaxAtrousFilteredVariance(vec2 varianceEnergy, float sumWeight) {
    float power = relaxAtrousVariancePower();
    float varianceMix = 2.0 - exp2(2.0 - power);
    return mix(varianceEnergy.x, varianceEnergy.y, varianceMix)
        * pow(1.0 / max(sumWeight, 1e-6), power);
}

#endif // RELAX_SPECULAR_ATROUS_COMMON_GLSL
