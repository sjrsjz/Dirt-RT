#ifndef BUFFERS_DEBUG_BUFFER_GLSL
#define BUFFERS_DEBUG_BUFFER_GLSL

#include "/lib/settings.glsl"
#include "/lib/buffers/addr.glsl"
#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/debug/view_ids.glsl"

// Dedicated per-pixel diagnostic records. Debug producers never borrow a
// production buffer, so changing DEBUG_VIEW cannot change renderer data flow.
// COMMON: x=encoded reflection sample direction, y/z=diffuse/specular
// noise-only current weight V_H/(V_H+V_C), w=FP16x2 diffuse/specular selected
// MC standard deviation. Prepared and final-filtered views reuse w because only
// one compile-time DEBUG_VIEW can produce it in a frame.
// SPECULAR_TEMPORAL: xyz=resolved temporal MaxEnt, w=FP16x2 temporal tracking
// hit distance/resolved history contribution (1-currentAlpha).
// DEBUG_VIEW is a compile-time option. Produce only records consumed by that
// view; inactive fields are unspecified and must never feed renderer state.
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
#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_SAMPLED_DIRECTION
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].x = encodeNormalU(direction);
#endif
}

vec3 debugReadReflectionSampleDirection(uvec2 xy) {
    return decodeNormalU(debugLoad(DEBUG_N_COMMON, xy).x);
}

void debugWriteDiffuseNoiseOnlyCurrentWeight(uvec2 xy, float currentWeight) {
#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_NOISE_ONLY_CURRENT_WEIGHT
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].y = floatBitsToUint(currentWeight);
#endif
}

float debugReadDiffuseNoiseOnlyCurrentWeight(uvec2 xy) {
    return uintBitsToFloat(debugLoad(DEBUG_N_COMMON, xy).y);
}

void debugWriteSpecularNoiseOnlyCurrentWeight(uvec2 xy, float currentWeight) {
#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_NOISE_ONLY_CURRENT_WEIGHT
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].z = floatBitsToUint(currentWeight);
#endif
}

float debugReadSpecularNoiseOnlyCurrentWeight(uvec2 xy) {
    return uintBitsToFloat(debugLoad(DEBUG_N_COMMON, xy).z);
}

void debugWriteDiffuseMonteCarloStandardDeviation(uvec2 xy, float standardDeviation) {
#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_PREPARED_MONTE_CARLO_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_FILTERED_MONTE_CARLO_VARIANCE
    // A selected view consumes one half only. Writing zero to the unused half
    // avoids a read/modify/write dependency and any per-frame initialization.
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].w =
        packHalf2x16(vec2(clamp(standardDeviation, -2.0, 65504.0), 0.0));
#endif
}

void debugWriteSpecularMonteCarloStandardDeviation(uvec2 xy, float standardDeviation) {
#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_PREPARED_MONTE_CARLO_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_FILTERED_MONTE_CARLO_VARIANCE
    debugBuffer.data[addr(DEBUG_N_COMMON, xy)].w =
        packHalf2x16(vec2(0.0, clamp(standardDeviation, -2.0, 65504.0)));
#endif
}

void debugWriteDiffusePreparedMonteCarloStandardDeviation(uvec2 xy, float standardDeviation) { debugWriteDiffuseMonteCarloStandardDeviation(xy, standardDeviation); }
void debugWriteSpecularPreparedMonteCarloStandardDeviation(uvec2 xy, float standardDeviation) { debugWriteSpecularMonteCarloStandardDeviation(xy, standardDeviation); }
void debugWriteDiffuseFilteredMonteCarloStandardDeviation(uvec2 xy, float standardDeviation) { debugWriteDiffuseMonteCarloStandardDeviation(xy, standardDeviation); }
void debugWriteSpecularFilteredMonteCarloStandardDeviation(uvec2 xy, float standardDeviation) { debugWriteSpecularMonteCarloStandardDeviation(xy, standardDeviation); }

float debugReadDiffuseMonteCarloStandardDeviation(uvec2 xy) { return unpackHalf2x16(debugLoad(DEBUG_N_COMMON, xy).w).x; }
float debugReadSpecularMonteCarloStandardDeviation(uvec2 xy) { return unpackHalf2x16(debugLoad(DEBUG_N_COMMON, xy).w).y; }
float debugReadDiffusePreparedMonteCarloStandardDeviation(uvec2 xy) { return debugReadDiffuseMonteCarloStandardDeviation(xy); }
float debugReadSpecularPreparedMonteCarloStandardDeviation(uvec2 xy) { return debugReadSpecularMonteCarloStandardDeviation(xy); }
float debugReadDiffuseFilteredMonteCarloStandardDeviation(uvec2 xy) { return debugReadDiffuseMonteCarloStandardDeviation(xy); }
float debugReadSpecularFilteredMonteCarloStandardDeviation(uvec2 xy) { return debugReadSpecularMonteCarloStandardDeviation(xy); }

void debugWriteSpecularTemporalState(uvec2 xy, uvec3 signalWords, float trackingHitDistance, float resolvedHistoryContribution) {
#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_CURRENT_TRACKING_HIT_DISTANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_TEMPORAL_HISTORY_SIGNAL || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_RESOLVED_HISTORY_CONTRIBUTION
    debugStore(DEBUG_N_SPECULAR_TEMPORAL, xy, uvec4(signalWords,
        packHalf2x16(clamp(vec2(trackingHitDistance, resolvedHistoryContribution),
            vec2(-65504.0), vec2(65504.0)))));
#endif
}

void debugWriteSpecularTemporalStateInvalid(uvec2 xy) {
#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_CURRENT_TRACKING_HIT_DISTANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_TEMPORAL_HISTORY_SIGNAL || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_RESOLVED_HISTORY_CONTRIBUTION
    debugStore(DEBUG_N_SPECULAR_TEMPORAL, xy, uvec4(
        0u, 0u, 0u, packHalf2x16(vec2(-1.0, 0.0))));
#endif
}

void debugReadSpecularTemporalState(uvec2 xy, out uvec3 signalWords, out float trackingHitDistance, out float resolvedHistoryContribution) {
    uvec4 words = debugLoad(DEBUG_N_SPECULAR_TEMPORAL, xy);
    signalWords = words.xyz;
    vec2 metadata = unpackHalf2x16(words.w);
    trackingHitDistance = metadata.x;
    resolvedHistoryContribution = metadata.y;
}

#endif
