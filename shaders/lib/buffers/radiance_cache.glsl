#ifndef RADIANCE_CACHE_GLSL
#define RADIANCE_CACHE_GLSL
#include "/lib/buffers/addr.glsl"
#include "/lib/lighting/alice_encode.glsl"

#define RADIANCE_CACHE_W 128
#define RADIANCE_CACHE_H 128
#define RADIANCE_CACHE_D 128
#define RADIANCE_CACHE_ABSTRACT_IMAGE_SIZE (RADIANCE_CACHE_W * RADIANCE_CACHE_H * RADIANCE_CACHE_D)
#define VOXEL_SIZE 1.0 // 每个体素的大小，单位为米
#define VOXEL_RANGE_HALF_W (RADIANCE_CACHE_W * VOXEL_SIZE * 0.5)
#define VOXEL_RANGE_HALF_H (RADIANCE_CACHE_H * VOXEL_SIZE * 0.5)
#define VOXEL_RANGE_HALF_D (RADIANCE_CACHE_D * VOXEL_SIZE * 0.5)
// 表面查询沿几何法线移向空气侧；限制为体素尺寸的 1%，且不超过现有 RT epsilon。
#define RADIANCE_CACHE_SURFACE_EPSILON min(0.001, VOXEL_SIZE * 0.01)

#define RADIANCE_CACHE_MAX_HIST 32.0 // 最大历史权重，用于归一化历史辐射率缓存的权重

// ===========================================================================
// 4x4x4 微块 (Tiled) 3D 体素存储寻址
// 适用于 128x64x128 空间，0 循环、0 分支、极致 L1 Cache 局部性
// ===========================================================================

// 1. 编码：(x, y, z) -> uint 1D Index
uint encodeVoxelTile4x4x4(uint x, uint y, uint z) {
    // 块内局部坐标 (0..3)
    uint lx = x & 3u; // x % 4
    uint ly = y & 3u; // y % 4
    uint lz = z & 3u; // z % 4
    // 块内 1D 偏移 (0..63)
    uint localIndex = lx | (ly << 2u) | (lz << 4u);

    // 宏观 Tile 块坐标
    uint tx = x >> 2u; // x / 4
    uint ty = y >> 2u; // y / 4
    uint tz = z >> 2u; // z / 4

    // 宏观 Tile 跨度 (W/4, H/4)
    const uint tilesPerRow = RADIANCE_CACHE_W >> 2u;
    const uint tilesPerSlice = (RADIANCE_CACHE_W >> 2u) * (RADIANCE_CACHE_H >> 2u);

    // 宏观 Tile 块的 1D 索引
    uint tileIndex = tx + ty * tilesPerRow + tz * tilesPerSlice;

    // 最终 1D 物理 SSBO 地址 = (Tile索引 * 64) + 块内局部索引
    return (tileIndex << 6u) | localIndex;
}

// 2. 解码：uint 1D Index -> (x, y, z)
uvec3 decodeVoxelTile4x4x4(uint index) {
    // 提取块内局部索引 (低 6 位)
    uint localIndex = index & 63u;
    uint lx = localIndex & 3u;
    uint ly = (localIndex >> 2u) & 3u;
    uint lz = (localIndex >> 4u) & 3u;

    // 提取宏观 Tile 索引 (高位)
    uint tileIndex = index >> 6u;
    const uint tilesPerRow = RADIANCE_CACHE_W >> 2u;
    const uint tilesPerSlice = (RADIANCE_CACHE_W >> 2u) * (RADIANCE_CACHE_H >> 2u);
    const uint log2TilesPerRow = uint(findMSB(tilesPerRow)); // 128->32->5
    const uint log2TilesPerSlice = uint(findMSB(tilesPerSlice)); // 1024->10

    uint tz = tileIndex >> log2TilesPerSlice;
    uint rem = tileIndex & (tilesPerSlice - 1u);
    uint ty = rem >> log2TilesPerRow;
    uint tx = rem & (tilesPerRow - 1u);

    return uvec3((tx << 2u) | lx, (ty << 2u) | ly, (tz << 2u) | lz);
}

struct RadianceCache {
    AliceEncoding alice;
    float weight;
};

RadianceCache emptyCache() {
    RadianceCache rc;
    rc.alice.aliceY = vec4(0.0);
    rc.alice.CoCg = vec2(0.0);
    rc.weight = 0.0;
    return rc;
}

vec4 packRadianceCache(RadianceCache rc) {
    return vec4(packAlice(rc.alice), rc.weight);
}

RadianceCache unpackRadianceCache(vec4 encoded) {
    RadianceCache rc;
    rc.alice = unpackAlice(encoded.x, encoded.y, encoded.z);
    rc.weight = encoded.w;
    return rc;
}

// 辐射率缓存存储结构
// 第一个抽象纹理存放交换缓存，第二个存放实际的累积后的辐射率缓存数据
#define RADIANCE_CACHE_SWAP (RADIANCE_CACHE_ABSTRACT_IMAGE_SIZE * 0)
#define RADIANCE_CACHE_HIST (RADIANCE_CACHE_ABSTRACT_IMAGE_SIZE * 1)

RadianceCache loadRadianceCacheSwap(uvec3 voxelCoord) {
    if (any(greaterThanEqual(voxelCoord, uvec3(RADIANCE_CACHE_W, RADIANCE_CACHE_H, RADIANCE_CACHE_D)))) {
        return emptyCache();
    }
    uint index = encodeVoxelTile4x4x4(voxelCoord.x, voxelCoord.y, voxelCoord.z);
    vec4 raw = radianceCacheBuffer.data[RADIANCE_CACHE_SWAP + index];
    return unpackRadianceCache(raw);
}

void storeRadianceCacheSwap(uvec3 voxelCoord, RadianceCache rc) {
    if (any(greaterThanEqual(voxelCoord, uvec3(RADIANCE_CACHE_W, RADIANCE_CACHE_H, RADIANCE_CACHE_D)))) {
        return;
    }
    uint index = encodeVoxelTile4x4x4(voxelCoord.x, voxelCoord.y, voxelCoord.z);
    radianceCacheBuffer.data[RADIANCE_CACHE_SWAP + index] = packRadianceCache(rc);
}

RadianceCache loadRadianceCacheHist(uvec3 voxelCoord) {
    if (any(greaterThanEqual(voxelCoord, uvec3(RADIANCE_CACHE_W, RADIANCE_CACHE_H, RADIANCE_CACHE_D)))) {
        return emptyCache();
    }
    uint index = encodeVoxelTile4x4x4(voxelCoord.x, voxelCoord.y, voxelCoord.z);
    vec4 raw = radianceCacheBuffer.data[RADIANCE_CACHE_HIST + index];
    return unpackRadianceCache(raw);
}

void storeRadianceCacheHist(uvec3 voxelCoord, RadianceCache rc) {
    if (any(greaterThanEqual(voxelCoord, uvec3(RADIANCE_CACHE_W, RADIANCE_CACHE_H, RADIANCE_CACHE_D)))) {
        return;
    }
    uint index = encodeVoxelTile4x4x4(voxelCoord.x, voxelCoord.y, voxelCoord.z);
    radianceCacheBuffer.data[RADIANCE_CACHE_HIST + index] = packRadianceCache(rc);
}

RadianceCache lerpRadianceCache(RadianceCache a, RadianceCache b, float t) {
    vec4 packA_0 = vec4(a.alice.aliceY);
    vec4 packA_1 = vec4(a.alice.CoCg, a.weight, 0.0);

    vec4 packB_0 = vec4(b.alice.aliceY);
    vec4 packB_1 = vec4(b.alice.CoCg, b.weight, 0.0);

    vec4 res_0 = mix(packA_0, packB_0, t);
    vec4 res_1 = mix(packA_1, packB_1, t);

    RadianceCache result;
    result.alice.aliceY = res_0;
    result.alice.CoCg = res_1.xy;
    result.weight = res_1.z;
    return result;
}

// 体素中心最近邻采样。调用方通过法线偏移选择几何表面的空气侧 probe；
// 不跨越八个相邻体素混合，避免墙体两侧辐射相互泄漏。
RadianceCache sampleRadianceCacheHist(vec3 voxelCoord) {
    ivec3 nearestCoord = ivec3(floor(voxelCoord + 0.5));
    bool inBounds = all(greaterThanEqual(nearestCoord, ivec3(0)))
        && all(lessThan(nearestCoord, ivec3(
            RADIANCE_CACHE_W,
            RADIANCE_CACHE_H,
            RADIANCE_CACHE_D
        )));
    if (!inBounds) return emptyCache();
    return loadRadianceCacheHist(uvec3(nearestCoord));
}

// ===========================================================================
// 摄像机相对、世界格点对齐的缓存坐标
//
// 探针位于缓存体素中心而不是格点角点，避免 VOXEL_SIZE == 1 时探针
// 与 Minecraft 方块表面精确重合。偶数尺寸下索引 W/2 对齐摄像机所在
// 体素中心，因此相对范围为 [-W/2, W/2-1]，正负方向相差一个体素。
// ===========================================================================

vec3 radianceCacheAnchor(vec3 cameraPosition) {
    return (floor(cameraPosition / VOXEL_SIZE) + 0.5) * VOXEL_SIZE;
}

// 返回索引 (0,0,0) 对应探针的世界坐标，而不是缓存包围盒的角点。
vec3 radianceCacheOrigin(vec3 cameraPosition) {
    const vec3 centerIndex = vec3(
        RADIANCE_CACHE_W / 2,
        RADIANCE_CACHE_H / 2,
        RADIANCE_CACHE_D / 2
    );
    return radianceCacheAnchor(cameraPosition) - centerIndex * VOXEL_SIZE;
}

vec3 radianceCacheVoxelWorldPos(uvec3 voxelCoord, vec3 cameraPosition) {
    return radianceCacheOrigin(cameraPosition) + vec3(voxelCoord) * VOXEL_SIZE;
}

vec3 radianceCacheWorldToVoxel(vec3 worldPos, vec3 cameraPosition) {
    return (worldPos - radianceCacheOrigin(cameraPosition)) / VOXEL_SIZE;
}

bool isRadianceCacheCoordInBounds(ivec3 voxelCoord) {
    return all(greaterThanEqual(voxelCoord, ivec3(0)))
        && all(lessThan(voxelCoord, ivec3(
            RADIANCE_CACHE_W,
            RADIANCE_CACHE_H,
            RADIANCE_CACHE_D
        )));
}

bool isRadianceCacheSampleInBounds(vec3 voxelCoord) {
    return isRadianceCacheCoordInBounds(ivec3(floor(voxelCoord + 0.5)));
}

#endif // RADIANCE_CACHE_GLSL
