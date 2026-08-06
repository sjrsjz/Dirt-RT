#version 430 compatibility
#ifndef TEMPORAL_RADIANCE_CACHE_GLSL
#define TEMPORAL_RADIANCE_CACHE_GLSL

// In-place sparse temporal accumulation. World-keyed mappings eliminate the
// dense 128x128x256 intermediate image and camera-origin reprojection copy.
#include "/lib/buffers/radiance_cache.glsl"
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
const ivec3 workGroups = ivec3(16, 16, 16);

bool isFiniteRadianceCache(RadianceCache c) {
    return !any(isnan(c.alice.aliceR)) && !any(isinf(c.alice.aliceR))
        && !any(isnan(c.alice.aliceG)) && !any(isinf(c.alice.aliceG))
        && !any(isnan(c.alice.aliceB)) && !any(isinf(c.alice.aliceB))
        && !isnan(c.weight) && !isinf(c.weight);
}
bool isValidRadianceCache(RadianceCache c) {
    return c.weight > 0.0 && isFiniteRadianceCache(c);
}
RadianceCache accumulateTemporalRadiance(RadianceCache current, RadianceCache history) {
    bool currentValid = isValidRadianceCache(current);
    bool historyValid = isValidRadianceCache(history);
    if (!currentValid) return historyValid ? history : emptyCache();
    current.weight = min(current.weight, RADIANCE_CACHE_MAX_HIST);
    if (!historyValid) return current;
    history.weight = min(history.weight, RADIANCE_CACHE_MAX_HIST);
    float combinedWeight = history.weight + current.weight;
    float alpha = current.weight / max(combinedWeight, 1e-6);
    RadianceCache result;
    result.alice.aliceR = mix(history.alice.aliceR, current.alice.aliceR, alpha);
    result.alice.aliceG = mix(history.alice.aliceG, current.alice.aliceG, alpha);
    result.alice.aliceB = mix(history.alice.aliceB, current.alice.aliceB, alpha);
    result.weight = min(combinedWeight, RADIANCE_CACHE_MAX_HIST);
    return result;
}
void main() {
    uint linearIndex = gl_GlobalInvocationID.x
        + gl_GlobalInvocationID.y * 64u
        + gl_GlobalInvocationID.z * 4096u;
    ivec3 worldVoxel;
    RadianceCacheAddress address = radianceCacheAddressForPoolVoxel(
        linearIndex, camPos, worldVoxel);
    if (!validateRadianceCacheAddress(address)) return;
    RadianceCache current = loadRadianceCachePlanes(
        address, RC_PLANE_CURRENT_0, RC_PLANE_CURRENT_1);
    RadianceCache history = radianceCacheAddressHasHistory(
        address, uint(max(frame_id, 0)))
        ? loadRadianceCachePlanes(address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1)
        : emptyCache();
    storeRadianceCachePlanes(address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1,
        accumulateTemporalRadiance(current, history));
}
#endif
