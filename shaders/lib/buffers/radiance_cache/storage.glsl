#ifndef RADIANCE_CACHE_STORAGE_GLSL
#define RADIANCE_CACHE_STORAGE_GLSL

// Storage ABI, moment encoding, world addressing, request submission and hash lookup.
// Included through radiance_cache.glsl; declaration order is intentional.

#define RADIANCE_CACHE_W 256
#define RADIANCE_CACHE_H 128
#define RADIANCE_CACHE_D 256
#define RADIANCE_CACHE_BRICK_SIZE 4
#define RADIANCE_CACHE_POOL_CAPACITY 4096
#define RADIANCE_CACHE_MAPPING_TABLE_SIZE 16384
#define RADIANCE_CACHE_MAX_HASH_PROBES 32
#define RADIANCE_CACHE_DISTANCE_BUCKET_COUNT 64
#define RADIANCE_CACHE_SAFE_WORLD_LIMIT 100000000.0
#ifndef VOXEL_SIZE
#define VOXEL_SIZE 1.0
#endif
#ifndef RADIANCE_CACHE_MAX_HIST
#define RADIANCE_CACHE_MAX_HIST 32.0
#endif
#ifndef RADIANCE_CACHE_FILTER_MAX_HIST
#define RADIANCE_CACHE_FILTER_MAX_HIST 8.0
#endif
#define RADIANCE_CACHE_BRICK_WORLD_SIZE (float(RADIANCE_CACHE_BRICK_SIZE) * VOXEL_SIZE)
#define RADIANCE_CACHE_BRICKS_X (RADIANCE_CACHE_W / RADIANCE_CACHE_BRICK_SIZE)
#define RADIANCE_CACHE_BRICKS_Y (RADIANCE_CACHE_H / RADIANCE_CACHE_BRICK_SIZE)
#define RADIANCE_CACHE_BRICKS_Z (RADIANCE_CACHE_D / RADIANCE_CACHE_BRICK_SIZE)
#define RADIANCE_CACHE_BRICK_COUNT (RADIANCE_CACHE_BRICKS_X * RADIANCE_CACHE_BRICKS_Y * RADIANCE_CACHE_BRICKS_Z)
#define RADIANCE_CACHE_SURFACE_EPSILON min(0.001, VOXEL_SIZE * 0.01)

#if (RADIANCE_CACHE_BRICK_SIZE != 4)
#error RADIANCE_CACHE_BRICK_SIZE must be 4; local addressing uses six bits.
#endif
#if ((RADIANCE_CACHE_MAPPING_TABLE_SIZE & (RADIANCE_CACHE_MAPPING_TABLE_SIZE - 1)) != 0)
#error RADIANCE_CACHE_MAPPING_TABLE_SIZE must be a power of two.
#endif
#if (RADIANCE_CACHE_POOL_CAPACITY > 4096)
#error Mapping tokens reserve 12 bits for the pool slot.
#endif
#if ((RADIANCE_CACHE_POOL_CAPACITY & (RADIANCE_CACHE_POOL_CAPACITY - 1)) != 0)
#error RADIANCE_CACHE_POOL_CAPACITY must be a power of two for CLOCK eviction.
#endif
#if ((RADIANCE_CACHE_MARK_TILE_SIZE & (RADIANCE_CACHE_MARK_TILE_SIZE - 1)) != 0)
#error RADIANCE_CACHE_MARK_TILE_SIZE must be a power of two.
#endif
#if (RADIANCE_CACHE_DISTANCE_BUCKET_COUNT < 1)
#error RADIANCE_CACHE_DISTANCE_BUCKET_COUNT must be positive.
#endif
#if (RADIANCE_CACHE_MAX_ALLOCATIONS_PER_FRAME < 1)
#error RADIANCE_CACHE_MAX_ALLOCATIONS_PER_FRAME must be positive.
#endif
#if (RADIANCE_CACHE_UPDATE_PERIOD < 1) || ((RADIANCE_CACHE_UPDATE_PERIOD & (RADIANCE_CACHE_UPDATE_PERIOD - 1)) != 0)
#error RADIANCE_CACHE_UPDATE_PERIOD must be a positive power of two.
#endif

// One SSBO, separated into request, allocator metadata, and payload regions.
// Only request submission is atomic. Different Vulkanite execution groups put
// barriers around allocator and payload phases, so all other accesses are plain.
#define RC_HEADER_WORDS 16u
#define RC_HEADER_MAGIC_ADDR 0u
#define RC_HEADER_NEXT_UNUSED_ADDR 1u
#define RC_HEADER_EVICTION_CURSOR_ADDR 2u
#define RC_HEADER_RESERVED_ADDR 3u
#define RC_HEADER_FAILED_ALLOCATIONS_ADDR 4u
#define RC_MAP_STRIDE 4u
#define RC_MAP_OFFSET RC_HEADER_WORDS
#define RC_POOL_META_STRIDE 9u
#define RC_POOL_META_OFFSET (RC_MAP_OFFSET + uint(RADIANCE_CACHE_MAPPING_TABLE_SIZE) * RC_MAP_STRIDE)
#define RC_REQUEST_WORDS (uint(RADIANCE_CACHE_BRICK_COUNT) / 32u)
#define RC_REQUEST_OFFSET (RC_POOL_META_OFFSET + uint(RADIANCE_CACHE_POOL_CAPACITY) * RC_POOL_META_STRIDE)
#define RC_DISTANCE_BUCKET_OFFSET (RC_REQUEST_OFFSET + RC_REQUEST_WORDS)
#define RC_MISSING_BUCKET_OFFSET (RC_DISTANCE_BUCKET_OFFSET + uint(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT))
#define RC_VOXELS_PER_BRICK 64u
#define RC_VEC4_WORDS 4u
#define RC_PLANES_PER_VOXEL 6u
#define RC_PLANE_WORDS (RC_VOXELS_PER_BRICK * RC_VEC4_WORDS)
#define RC_POOL_SLOT_WORDS (RC_PLANES_PER_VOXEL * RC_PLANE_WORDS)
#define RC_POOL_DATA_OFFSET (RC_MISSING_BUCKET_OFFSET + uint(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT))
#define RC_TOTAL_WORDS (RC_POOL_DATA_OFFSET + uint(RADIANCE_CACHE_POOL_CAPACITY) * RC_POOL_SLOT_WORDS)

// P8 resets legacy tombstones and linked-LRU metadata before enabling
// backward-shift deletion and CLOCK eviction.
#define RC_MAGIC 0x52435038u
#define RC_INVALID_TOKEN 0xffffffffu
#define RC_TOMBSTONE_TOKEN 0xfffffffeu
#define RC_LOCKED_TOKEN 0xfffffffdu
#define RC_INVALID_SLOT 0xffffffffu
#define RC_SLOT_MASK 0x00000fffu
#define RC_GENERATION_MASK 0x0007ffffu
#define RC_META_KEY_X 0u
#define RC_META_KEY_Y 1u
#define RC_META_KEY_Z 2u
#define RC_META_MAP_INDEX 3u
#define RC_META_TOKEN 4u
#define RC_META_LRU_PREV 5u
#define RC_META_LRU_NEXT 6u
#define RC_META_BIRTH_FRAME 7u
#define RC_META_PIN_FRAME 8u
#define RC_PLANE_CURRENT_0 0u
#define RC_PLANE_CURRENT_1 1u
#define RC_PLANE_HISTORY_0 2u
#define RC_PLANE_HISTORY_1 3u
#define RC_PLANE_FILTERED_0 4u
#define RC_PLANE_FILTERED_1 5u

struct RGBMaxEntEncoding { vec4 maxentR; vec4 maxentG; vec4 maxentB; };
// History stores a temporal RIS reservoir: maxent is the raw selected sample,
// W is the reciprocal-proposal normalization, and M is the represented sample
// count. Current-frame probes are unit reservoirs (W=1, M=1).
struct RadianceCache { RGBMaxEntEncoding maxent; float W; float M; };
struct PackedRadianceCache { uvec4 word0; uvec4 word1; };
struct RadianceCacheAddress { uint token; uint slot; uint localIndex; };

RadianceCacheAddress invalidRadianceCacheAddress() {
    RadianceCacheAddress a;
    a.token = RC_INVALID_TOKEN; a.slot = 0u; a.localIndex = 0u;
    return a;
}
bool radianceCacheStorageAvailable() {
    return uint(radianceCacheBuffer.data.length()) >= RC_TOTAL_WORDS;
}
uint rcLoad(uint address) { return radianceCacheBuffer.data[address]; }
void rcStore(uint address, uint value) { radianceCacheBuffer.data[address] = value; }
uint rcMapAddress(uint index, uint component) { return RC_MAP_OFFSET + index * RC_MAP_STRIDE + component; }
uint rcMetaAddress(uint slot, uint component) { return RC_POOL_META_OFFSET + slot * RC_POOL_META_STRIDE + component; }
uint rcPayloadAddress(uint slot, uint plane, uint localIndex) {
    return RC_POOL_DATA_OFFSET + slot * RC_POOL_SLOT_WORDS + plane * RC_PLANE_WORDS + localIndex * RC_VEC4_WORDS;
}
uvec4 rcLoadVec4(uint a) { return uvec4(rcLoad(a), rcLoad(a + 1u), rcLoad(a + 2u), rcLoad(a + 3u)); }
void rcStoreVec4(uint a, uvec4 v) {
    rcStore(a, v.x); rcStore(a + 1u, v.y); rcStore(a + 2u, v.z); rcStore(a + 3u, v.w);
}

RGBMaxEntEncoding radiance_to_rgb_maxent(vec3 radiance, vec3 direction) {
    RGBMaxEntEncoding e;
    e.maxentR = vec4(direction * radiance.r, radiance.r);
    e.maxentG = vec4(direction * radiance.g, radiance.g);
    e.maxentB = vec4(direction * radiance.b, radiance.b);
    return e;
}
vec3 project_rgb_maxent_irradiance(RGBMaxEntEncoding e, vec3 n) {
    return vec3(maxent_irradiance(e.maxentR, n), maxent_irradiance(e.maxentG, n), maxent_irradiance(e.maxentB, n));
}
vec3 radianceCacheDiffuseIncident(RadianceCache cache, vec3 n) {
    // Cache rays estimate a uniform-sphere average. Multiplying the cosine
    // projection by four converts it to E/pi, ready for a diffuse albedo. The
    // reservoir stores its selected sample unscaled, so apply W exactly once.
    return (4.0 * cache.W) * project_rgb_maxent_irradiance(cache.maxent, n);
}
bool radianceCacheValueValid(RadianceCache cache) {
    return cache.W > 0.0 && cache.M > 0.0
        && !isnan(cache.W) && !isinf(cache.W)
        && !isnan(cache.M) && !isinf(cache.M)
        && !any(isnan(cache.maxent.maxentR)) && !any(isinf(cache.maxent.maxentR))
        && !any(isnan(cache.maxent.maxentG)) && !any(isinf(cache.maxent.maxentG))
        && !any(isnan(cache.maxent.maxentB)) && !any(isinf(cache.maxent.maxentB));
}
vec4 rgb_maxent_luminance(RGBMaxEntEncoding e) {
    return e.maxentR * 0.2126 + e.maxentG * 0.7152 + e.maxentB * 0.0722;
}
RadianceCache emptyCache() {
    RadianceCache rc;
    rc.maxent.maxentR = vec4(0.0); rc.maxent.maxentG = vec4(0.0); rc.maxent.maxentB = vec4(0.0);
    rc.W = 0.0;
    rc.M = 0.0;
    return rc;
}
PackedRadianceCache packRadianceCache(RadianceCache rc) {
    PackedRadianceCache p;
    p.word0 = uvec4(
        packHalf2x16(rc.maxent.maxentR.xy), packHalf2x16(rc.maxent.maxentR.zw),
        packHalf2x16(rc.maxent.maxentG.xy), packHalf2x16(rc.maxent.maxentG.zw));
    p.word1 = uvec4(packHalf2x16(rc.maxent.maxentB.xy),
        packHalf2x16(rc.maxent.maxentB.zw),
        floatBitsToUint(rc.W), floatBitsToUint(rc.M));
    return p;
}
RadianceCache unpackRadianceCache(uvec4 word0, uvec4 word1) {
    RadianceCache rc;
    rc.maxent.maxentR = vec4(unpackHalf2x16(word0.x), unpackHalf2x16(word0.y));
    rc.maxent.maxentG = vec4(unpackHalf2x16(word0.z), unpackHalf2x16(word0.w));
    rc.maxent.maxentB = vec4(unpackHalf2x16(word1.x), unpackHalf2x16(word1.y));
    rc.W = uintBitsToFloat(word1.z);
    rc.M = uintBitsToFloat(word1.w);
    return rc;
}

ivec3 radianceCacheWorldVoxel(vec3 worldPos) { return ivec3(floor(worldPos / VOXEL_SIZE)); }
ivec3 radianceCacheWorldBrick(ivec3 worldVoxel) {
    return ivec3(floor(vec3(worldVoxel) / float(RADIANCE_CACHE_BRICK_SIZE)));
}
ivec3 radianceCacheClipMinBrick(vec3 cameraPosition) {
    return ivec3(floor(cameraPosition / RADIANCE_CACHE_BRICK_WORLD_SIZE))
        - ivec3(RADIANCE_CACHE_BRICKS_X / 2, RADIANCE_CACHE_BRICKS_Y / 2, RADIANCE_CACHE_BRICKS_Z / 2);
}
uint radianceCacheLocalIndex(ivec3 worldVoxel, ivec3 worldBrick) {
    ivec3 local = worldVoxel - worldBrick * RADIANCE_CACHE_BRICK_SIZE;
    return uint(local.x) | (uint(local.y) << 2u) | (uint(local.z) << 4u);
}
vec3 radianceCacheOrigin(vec3 cameraPosition) {
    return vec3(radianceCacheClipMinBrick(cameraPosition) * RADIANCE_CACHE_BRICK_SIZE) * VOXEL_SIZE
        + vec3(0.5 * VOXEL_SIZE);
}
vec3 radianceCacheVoxelWorldPos(uvec3 voxelCoord, vec3 cameraPosition) {
    return radianceCacheOrigin(cameraPosition) + vec3(voxelCoord) * VOXEL_SIZE;
}
vec3 radianceCacheWorldToVoxel(vec3 worldPos, vec3 cameraPosition) {
    return (worldPos - radianceCacheOrigin(cameraPosition)) / VOXEL_SIZE;
}
bool isRadianceCacheCoordInBounds(ivec3 voxelCoord) {
    return all(greaterThanEqual(voxelCoord, ivec3(0)))
        && all(lessThan(voxelCoord, ivec3(RADIANCE_CACHE_W, RADIANCE_CACHE_H, RADIANCE_CACHE_D)));
}
bool isRadianceCacheSampleInBounds(vec3 voxelCoord) {
    return isRadianceCacheCoordInBounds(ivec3(floor(voxelCoord + 0.5)));
}
bool rcFiniteWorldPosition(vec3 p) {
    return !any(isnan(p)) && !any(isinf(p))
        && !any(greaterThan(abs(p), vec3(RADIANCE_CACHE_SAFE_WORLD_LIMIT)));
}
uint rcFrameTag(uint frameStamp) {
    uint tag = frameStamp + 1u;
    return tag == 0u ? 1u : tag;
}

// Geometry allocation request API. This is the only hot-path atomic operation:
// one bit per world brick in the current 64x32x64 brick clip volume.
bool markRadianceCacheBlockHasGeometry(ivec3 worldBrick, vec3 cameraPosition) {
    if (!radianceCacheStorageAvailable() || !rcFiniteWorldPosition(cameraPosition))
        return false;
    ivec3 minBrick = radianceCacheClipMinBrick(cameraPosition);
    ivec3 brickDimensions = ivec3(
        RADIANCE_CACHE_BRICKS_X, RADIANCE_CACHE_BRICKS_Y, RADIANCE_CACHE_BRICKS_Z);
    if (any(lessThan(worldBrick, minBrick))
            || any(greaterThanEqual(worldBrick, minBrick + brickDimensions)))
        return false;
    ivec3 relative = worldBrick - minBrick;
    uint requestIndex = uint(relative.x + relative.y * RADIANCE_CACHE_BRICKS_X
        + relative.z * RADIANCE_CACHE_BRICKS_X * RADIANCE_CACHE_BRICKS_Y);
    atomicOr(radianceCacheBuffer.data[RC_REQUEST_OFFSET + (requestIndex >> 5u)],
        1u << (requestIndex & 31u));
    return true;
}
bool markRadianceCacheBlockHasGeometry(vec3 worldPos, vec3 cameraPosition) {
    if (!rcFiniteWorldPosition(worldPos)) return false;
    return markRadianceCacheBlockHasGeometry(
        radianceCacheWorldBrick(radianceCacheWorldVoxel(worldPos)), cameraPosition);
}

uint radianceCacheHash(ivec3 key) {
    uvec3 v = uvec3(key);
    uint h = (v.x * 0x8da6b343u) ^ (v.y * 0xd8163841u) ^ (v.z * 0xcb1ab31fu);
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u;
    return h;
}
bool radianceCacheShouldUpdate(ivec3 worldVoxel, uint frameStamp) {
#if RADIANCE_CACHE_UPDATE_PERIOD == 1
    return true;
#else
    uint phaseMask = uint(RADIANCE_CACHE_UPDATE_PERIOD - 1);
    ivec3 worldBrick = radianceCacheWorldBrick(worldVoxel);
    uint localIndex = radianceCacheLocalIndex(worldVoxel, worldBrick);
    // A reversible bit mix distributes every axis across phases while keeping
    // exactly 64 / period updates in each brick. The brick hash rotates the
    // phase spatially, avoiding coherent world-space update planes.
    uint localPhase = localIndex ^ (localIndex >> 2u) ^ (localIndex >> 4u);
    uint phase = (localPhase + radianceCacheHash(worldBrick)) & phaseMask;
    return phase == (frameStamp & phaseMask);
#endif
}
bool rcMapKeyEquals(uint i, ivec3 k) {
    return rcLoad(rcMapAddress(i, 0u)) == uint(k.x)
        && rcLoad(rcMapAddress(i, 1u)) == uint(k.y)
        && rcLoad(rcMapAddress(i, 2u)) == uint(k.z);
}
bool radianceCacheReady() {
    return radianceCacheStorageAvailable() && rcLoad(RC_HEADER_MAGIC_ADDR) == RC_MAGIC;
}
uint findRadianceCacheMapping(ivec3 key) {
    if (!radianceCacheReady()) return RC_INVALID_TOKEN;
    uint mask = uint(RADIANCE_CACHE_MAPPING_TABLE_SIZE - 1);
    uint first = radianceCacheHash(key) & mask;
    for (uint probe = 0u; probe < uint(RADIANCE_CACHE_MAX_HASH_PROBES); ++probe) {
        uint mapIndex = (first + probe) & mask;
        uint token = rcLoad(rcMapAddress(mapIndex, 3u));
        if (token == RC_INVALID_TOKEN) return RC_INVALID_TOKEN;
        if (token >= RC_LOCKED_TOKEN || !rcMapKeyEquals(mapIndex, key)) continue;
        uint slot = token & RC_SLOT_MASK;
        if (slot >= uint(RADIANCE_CACHE_POOL_CAPACITY)) continue;
        // Mapping key + generation token are sufficient after the allocator
        // barrier. Re-reading the same XYZ key from pool metadata costs three
        // random SSBO loads per successful lookup without adding safety.
        if (rcLoad(rcMetaAddress(slot, RC_META_TOKEN)) == token) return token;
    }
    return RC_INVALID_TOKEN;
}
#endif // RADIANCE_CACHE_STORAGE_GLSL
