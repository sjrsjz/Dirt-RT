#version 430 compatibility

#ifndef TEMPORAL_RADIANCE_CACHE_GLSL
#define TEMPORAL_RADIANCE_CACHE_GLSL

// ===========================================================================
// 世界格点对齐辐射率缓存的时域重投影
//
// RADIANCE_CACHE_SWAP: 本帧原始样本（只读）
// RADIANCE_CACHE_HIST: 上一帧累积历史（只读）
// radianceCacheTemporal: 本帧合成结果，由后续独立 pass 写回 HIST
//
// RC 原点按 VOXEL_SIZE 吸附到世界格点，因此帧间位移恒为整数 voxel。
// 每个输出只需一次历史读取，不需要连续相机网格的 8-tap 三线性重投影。
// ===========================================================================

#include "/lib/buffers/radiance_cache.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(rgba32ui) uniform writeonly uimage3D radianceCacheTemporal;

// Iris parses workGroups arguments with Integer.parseInt; use literal values.
const ivec3 workGroups = ivec3(32, 32, 32);

bool isFiniteRadianceCache(RadianceCache cache) {
    return !any(isnan(cache.alice.aliceR))
        && !any(isinf(cache.alice.aliceR))
        && !any(isnan(cache.alice.aliceG))
        && !any(isinf(cache.alice.aliceG))
        && !any(isnan(cache.alice.aliceB))
        && !any(isinf(cache.alice.aliceB))
        && !isnan(cache.weight)
        && !isinf(cache.weight);
}

bool isValidRadianceCache(RadianceCache cache) {
    return cache.weight > 0.0 && isFiniteRadianceCache(cache);
}

RadianceCache accumulateTemporalRadiance(RadianceCache current, RadianceCache history) {
    bool currentValid = isValidRadianceCache(current);
    bool historyValid = isValidRadianceCache(history);

    if (!currentValid) {
        if (historyValid) return history;
        return emptyCache();
    }

    current.weight = min(current.weight, RADIANCE_CACHE_MAX_HIST);
    if (!historyValid) return current;

    history.weight = min(history.weight, RADIANCE_CACHE_MAX_HIST);
    float combinedWeight = history.weight + current.weight;
    float currentAlpha = current.weight / max(combinedWeight, 1e-6);

    RadianceCache result;
    result.alice.aliceR = mix(history.alice.aliceR, current.alice.aliceR, currentAlpha);
    result.alice.aliceG = mix(history.alice.aliceG, current.alice.aliceG, currentAlpha);
    result.alice.aliceB = mix(history.alice.aliceB, current.alice.aliceB, currentAlpha);
    result.weight = min(combinedWeight, RADIANCE_CACHE_MAX_HIST);
    return result;
}

void main() {
    uvec3 voxelCoord = gl_GlobalInvocationID.xyz;
    RadianceCache current = loadRadianceCacheSwap(voxelCoord);
    RadianceCache history = emptyCache();

    vec3 currentOrigin = radianceCacheOrigin(camPos);
    vec3 previousOrigin = radianceCacheOrigin(prevRaytracingCamPos);
    ivec3 originShift = ivec3(round((currentOrigin - previousOrigin) / VOXEL_SIZE));
    ivec3 previousCoord = ivec3(voxelCoord) + originShift;

    if (isRadianceCacheCoordInBounds(previousCoord)) {
        history = loadRadianceCacheHist(uvec3(previousCoord));
    }

    RadianceCache result = accumulateTemporalRadiance(current, history);
    PackedRadianceCache packedCache = packRadianceCache(result);
    ivec3 word0Coord = ivec3(voxelCoord);
    ivec3 word1Coord = word0Coord + ivec3(0, 0, RADIANCE_CACHE_D);
    imageStore(
        radianceCacheTemporal,
        word0Coord,
        floatBitsToUint(packedCache.word0)
    );
    imageStore(
        radianceCacheTemporal,
        word1Coord,
        floatBitsToUint(packedCache.word1)
    );
}

#endif // TEMPORAL_RADIANCE_CACHE_GLSL
