#version 430 compatibility

#ifndef TEMPORAL_RADIANCE_CACHE_GLSL
#define TEMPORAL_RADIANCE_CACHE_GLSL

// ===========================================================================
// 摄像机相对辐射率缓存时域重投影
//
// RADIANCE_CACHE_SWAP: 本帧原始样本（只读）
// RADIANCE_CACHE_HIST: 上一帧累积历史（只读）
// radianceCacheTemporal: 本帧合成结果，由后续独立 pass 写回 HIST
// ===========================================================================

#include "/lib/buffers/radiance_cache.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(rgba32ui) uniform writeonly uimage3D radianceCacheTemporal;

const ivec3 workGroups = ivec3(
    RADIANCE_CACHE_W / 4,
    RADIANCE_CACHE_H / 4,
    RADIANCE_CACHE_D / 4
);

const uint RC_TILE_SIZE = 4u;
const int RC_HISTORY_WINDOW_SIZE = 5;
const uint RC_HISTORY_WINDOW_COUNT = 125u;

// 4³ 输出只需要 5³ 历史输入。64 个线程协作加载 125 项，
// 将每体素 8 次 HIST SSBO 读取摊销为约 1.95 次。
shared vec4 sharedHistoryY[RC_HISTORY_WINDOW_COUNT];
shared vec4 sharedHistoryCoCgWeight[RC_HISTORY_WINDOW_COUNT];

bool isFiniteRadianceCache(RadianceCache cache) {
    return !any(isnan(cache.alice.aliceY))
        && !any(isinf(cache.alice.aliceY))
        && !any(isnan(cache.alice.CoCg))
        && !any(isinf(cache.alice.CoCg))
        && !isnan(cache.weight)
        && !isinf(cache.weight);
}

bool isValidRadianceCache(RadianceCache cache) {
    return cache.weight > 0.0 && isFiniteRadianceCache(cache);
}

void accumulateHistoryTap(
    uint index,
    float trilinearWeight,
    inout AliceEncoding aliceAccum,
    inout float historyWeightAccum,
    inout float spatialWeightAccum
) {
    vec4 tapCoCgWeight = sharedHistoryCoCgWeight[index];
    if (tapCoCgWeight.w <= 0.0 || trilinearWeight <= 0.0) return;

    aliceAccum.aliceY += sharedHistoryY[index] * trilinearWeight;
    aliceAccum.CoCg += tapCoCgWeight.xy * trilinearWeight;
    historyWeightAccum += tapCoCgWeight.z * trilinearWeight;
    spatialWeightAccum += trilinearWeight;
}

RadianceCache sampleSharedHistory(uvec3 localCoord, vec3 fraction) {
    uint base = localCoord.x
        + localCoord.y * uint(RC_HISTORY_WINDOW_SIZE)
        + localCoord.z * uint(RC_HISTORY_WINDOW_SIZE * RC_HISTORY_WINDOW_SIZE);

    uint strideY = uint(RC_HISTORY_WINDOW_SIZE);
    uint strideZ = uint(RC_HISTORY_WINDOW_SIZE * RC_HISTORY_WINDOW_SIZE);

    vec3 inverseFraction = 1.0 - fraction;
    float w000 = inverseFraction.x * inverseFraction.y * inverseFraction.z;
    float w100 = fraction.x * inverseFraction.y * inverseFraction.z;
    float w010 = inverseFraction.x * fraction.y * inverseFraction.z;
    float w110 = fraction.x * fraction.y * inverseFraction.z;
    float w001 = inverseFraction.x * inverseFraction.y * fraction.z;
    float w101 = fraction.x * inverseFraction.y * fraction.z;
    float w011 = inverseFraction.x * fraction.y * fraction.z;
    float w111 = fraction.x * fraction.y * fraction.z;

    AliceEncoding aliceAccum = init_alice();
    float historyWeightAccum = 0.0;
    float spatialWeightAccum = 0.0;

    accumulateHistoryTap(base,                       w000, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + 1u,                  w100, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + strideY,             w010, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + strideY + 1u,        w110, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + strideZ,             w001, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + strideZ + 1u,        w101, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + strideZ + strideY,   w011, aliceAccum, historyWeightAccum, spatialWeightAccum);
    accumulateHistoryTap(base + strideZ + strideY + 1u, w111, aliceAccum, historyWeightAccum, spatialWeightAccum);

    if (spatialWeightAccum <= 1e-6) return emptyCache();

    float invSpatialWeight = 1.0 / spatialWeightAccum;
    RadianceCache result;
    result.alice.aliceY = aliceAccum.aliceY * invSpatialWeight;
    result.alice.CoCg = aliceAccum.CoCg * invSpatialWeight;
    result.weight = min(historyWeightAccum * invSpatialWeight, RADIANCE_CACHE_MAX_HIST);
    return result;
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
    result.alice.aliceY = mix(history.alice.aliceY, current.alice.aliceY, currentAlpha);
    result.alice.CoCg = mix(history.alice.CoCg, current.alice.CoCg, currentAlpha);
    result.weight = min(combinedWeight, RADIANCE_CACHE_MAX_HIST);
    return result;
}

void main() {
    uvec3 localCoord = gl_LocalInvocationID.xyz;
    uvec3 groupOrigin = gl_WorkGroupID.xyz * RC_TILE_SIZE;
    uvec3 voxelCoord = groupOrigin + localCoord;

    // 相同世界空间采样点在上一帧摄像机相对网格中的坐标。
    vec3 cameraDeltaVoxels = (camPos - prevRaytracingCamPos) / VOXEL_SIZE;
    ivec3 historyShift = ivec3(floor(cameraDeltaVoxels));
    vec3 historyFraction = fract(cameraDeltaVoxels);
    ivec3 historyWindowOrigin = ivec3(groupOrigin) + historyShift;

    // 首帧跳过所有历史 SSBO 读取。其余帧中，越界项写为无效共享样本。
    for (uint i = gl_LocalInvocationIndex; i < RC_HISTORY_WINDOW_COUNT; i += 64u) {
        ivec3 windowCoord = ivec3(
            int(i % uint(RC_HISTORY_WINDOW_SIZE)),
            int((i / uint(RC_HISTORY_WINDOW_SIZE)) % uint(RC_HISTORY_WINDOW_SIZE)),
            int(i / uint(RC_HISTORY_WINDOW_SIZE * RC_HISTORY_WINDOW_SIZE))
        );
        ivec3 historyCoord = historyWindowOrigin + windowCoord;

        RadianceCache history = emptyCache();
        bool inBounds = all(greaterThanEqual(historyCoord, ivec3(0)))
            && all(lessThan(historyCoord, ivec3(
                RADIANCE_CACHE_W,
                RADIANCE_CACHE_H,
                RADIANCE_CACHE_D
            )));
        if (frame_id > 0 && inBounds) {
            history = loadRadianceCacheHist(uvec3(historyCoord));
        }

        bool valid = isValidRadianceCache(history);
        sharedHistoryY[i] = valid ? history.alice.aliceY : vec4(0.0);
        sharedHistoryCoCgWeight[i] = valid
            ? vec4(history.alice.CoCg, min(history.weight, RADIANCE_CACHE_MAX_HIST), 1.0)
            : vec4(0.0);
    }

    memoryBarrierShared();
    barrier();

    RadianceCache current = loadRadianceCacheSwap(voxelCoord);
    RadianceCache history = emptyCache();

    // 仅要求具有非零三线性权重的角点位于缓存内。
    ivec3 historyMin = historyWindowOrigin + ivec3(localCoord);
    ivec3 historyMax = historyMin + ivec3(
        historyFraction.x > 0.0 ? 1 : 0,
        historyFraction.y > 0.0 ? 1 : 0,
        historyFraction.z > 0.0 ? 1 : 0
    );
    bool footprintInBounds = all(greaterThanEqual(historyMin, ivec3(0)))
        && all(lessThan(historyMax, ivec3(
            RADIANCE_CACHE_W,
            RADIANCE_CACHE_H,
            RADIANCE_CACHE_D
        )));

    if (frame_id > 0 && footprintInBounds) {
        history = sampleSharedHistory(localCoord, historyFraction);
    }

    RadianceCache result = accumulateTemporalRadiance(current, history);
    imageStore(
        radianceCacheTemporal,
        ivec3(voxelCoord),
        floatBitsToUint(packRadianceCache(result))
    );
}

#endif // TEMPORAL_RADIANCE_CACHE_GLSL
