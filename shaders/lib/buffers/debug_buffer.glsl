#ifndef BUFFERS_DEBUG_BUFFER_GLSL
#define BUFFERS_DEBUG_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/debug/view_ids.glsl"

// Dedicated per-pixel diagnostic records. Debug producers never borrow a
// production buffer, so changing DEBUG_VIEW cannot change renderer data flow.
// COMMON: x=encoded reflection sample direction, y/z=diffuse/specular
// noise-only current weight V_H/(V_H+V_C), w=FP16x2 diffuse/specular prepared
// estimator standard deviation.
// SPECULAR_TEMPORAL: xyz=resolved temporal MaxEnt, w=FP16x2 temporal tracking
// hit distance/resolved history contribution (1-currentAlpha).
const uint DEBUG_N_COMMON = 0u;
const uint DEBUG_N_SPECULAR_TEMPORAL = 1u;

layout(std430, set = 3, binding = 6) buffer DebugBuffer {
    uvec4 data[];
} debugBuffer;

void debugStore(uint record, uvec2 xy, uvec4 value) {
    debugBuffer.data[addr(record, xy)] = value;
}

uvec4 debugLoad(uint record, uvec2 xy) {
    return debugBuffer.data[addr(record, xy)];
}

void debugWriteReflectionSampleDirection(uvec2 xy, vec3 direction) {
    // The reflection ray pass owns the start of the diagnostic frame and also
    // clears values that later compute passes may replace.
    debugStore(DEBUG_N_COMMON, xy, uvec4(encodeNormalU(direction),
        floatBitsToUint(-1.0), floatBitsToUint(-1.0), 0u));
}

vec3 debugReadReflectionSampleDirection(uvec2 xy) {
    return decodeNormalU(debugLoad(DEBUG_N_COMMON, xy).x);
}

void debugWriteDiffuseNoiseOnlyCurrentWeight(uvec2 xy, float currentWeight) {
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].y = floatBitsToUint(currentWeight);
}

float debugReadDiffuseNoiseOnlyCurrentWeight(uvec2 xy) {
    return uintBitsToFloat(debugLoad(DEBUG_N_COMMON, xy).y);
}

void debugWriteSpecularNoiseOnlyCurrentWeight(uvec2 xy, float currentWeight) {
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].z = floatBitsToUint(currentWeight);
}

float debugReadSpecularNoiseOnlyCurrentWeight(uvec2 xy) {
    return uintBitsToFloat(debugLoad(DEBUG_N_COMMON, xy).z);
}

void debugWriteDiffusePreparedEstimatorStandardDeviation(
        uvec2 xy, float standardDeviation) {
    uint index = addr(DEBUG_N_COMMON, xy);
    vec2 preparedEstimatorStandardDeviations = unpackHalf2x16(
        debugBuffer.data[index].w);
    preparedEstimatorStandardDeviations.x = clamp(
        standardDeviation, -1.0, 65504.0);
    debugBuffer.data[index].w = packHalf2x16(
        preparedEstimatorStandardDeviations);
}

void debugWriteSpecularPreparedEstimatorStandardDeviation(
        uvec2 xy, float standardDeviation) {
    uint index = addr(DEBUG_N_COMMON, xy);
    vec2 preparedEstimatorStandardDeviations = unpackHalf2x16(
        debugBuffer.data[index].w);
    preparedEstimatorStandardDeviations.y = clamp(
        standardDeviation, -1.0, 65504.0);
    debugBuffer.data[index].w = packHalf2x16(
        preparedEstimatorStandardDeviations);
}

float debugReadDiffusePreparedEstimatorStandardDeviation(uvec2 xy) {
    return unpackHalf2x16(debugLoad(DEBUG_N_COMMON, xy).w).x;
}

float debugReadSpecularPreparedEstimatorStandardDeviation(uvec2 xy) {
    return unpackHalf2x16(debugLoad(DEBUG_N_COMMON, xy).w).y;
}

void debugWriteSpecularTemporalState(uvec2 xy, uvec3 signalWords, float trackingHitDistance, float resolvedHistoryContribution) {
    debugStore(DEBUG_N_SPECULAR_TEMPORAL, xy, uvec4(signalWords,
        packHalf2x16(clamp(vec2(trackingHitDistance, resolvedHistoryContribution),
            vec2(-65504.0), vec2(65504.0)))));
}

void debugWriteSpecularTemporalStateInvalid(uvec2 xy) {
    debugStore(DEBUG_N_SPECULAR_TEMPORAL, xy, uvec4(
        0u, 0u, 0u, packHalf2x16(vec2(-1.0, 0.0))));
}

void debugReadSpecularTemporalState(uvec2 xy, out uvec3 signalWords, out float trackingHitDistance, out float resolvedHistoryContribution) {
    uvec4 words = debugLoad(DEBUG_N_SPECULAR_TEMPORAL, xy);
    signalWords = words.xyz;
    vec2 metadata = unpackHalf2x16(words.w);
    trackingHitDistance = metadata.x;
    resolvedHistoryContribution = metadata.y;
}

#endif
