#ifndef RADIANCE_CACHE_GLSL
#define RADIANCE_CACHE_GLSL

#include "/lib/settings.glsl"
#include "/lib/buffers/addr.glsl"
#include "/lib/lighting/alice_encode.glsl"

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

struct RGBAliceEncoding { vec4 aliceR; vec4 aliceG; vec4 aliceB; };
// History stores a temporal RIS reservoir: alice is the raw selected sample,
// W is the reciprocal-proposal normalization, and M is the represented sample
// count. Current-frame probes are unit reservoirs (W=1, M=1).
struct RadianceCache { RGBAliceEncoding alice; float W; float M; };
struct PackedRadianceCache { vec4 word0; vec4 word1; };
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

RGBAliceEncoding radiance_to_rgb_alice(vec3 radiance, vec3 direction) {
    RGBAliceEncoding e;
    e.aliceR = vec4(direction * radiance.r, radiance.r);
    e.aliceG = vec4(direction * radiance.g, radiance.g);
    e.aliceB = vec4(direction * radiance.b, radiance.b);
    return e;
}
vec3 project_rgb_alice_irradiance(RGBAliceEncoding e, vec3 n) {
    return vec3(alice_irradiance(e.aliceR, n), alice_irradiance(e.aliceG, n), alice_irradiance(e.aliceB, n));
}
vec3 radianceCacheDiffuseIncident(RadianceCache cache, vec3 n) {
    // Cache rays estimate a uniform-sphere average. Multiplying the cosine
    // projection by four converts it to E/pi, ready for a diffuse albedo. The
    // reservoir stores its selected sample unscaled, so apply W exactly once.
    return (4.0 * cache.W) * project_rgb_alice_irradiance(cache.alice, n);
}
bool radianceCacheValueValid(RadianceCache cache) {
    return cache.W > 0.0 && cache.M > 0.0
        && !isnan(cache.W) && !isinf(cache.W)
        && !isnan(cache.M) && !isinf(cache.M)
        && !any(isnan(cache.alice.aliceR)) && !any(isinf(cache.alice.aliceR))
        && !any(isnan(cache.alice.aliceG)) && !any(isinf(cache.alice.aliceG))
        && !any(isnan(cache.alice.aliceB)) && !any(isinf(cache.alice.aliceB));
}
vec4 rgb_alice_luminance(RGBAliceEncoding e) {
    return e.aliceR * 0.2126 + e.aliceG * 0.7152 + e.aliceB * 0.0722;
}
RadianceCache emptyCache() {
    RadianceCache rc;
    rc.alice.aliceR = vec4(0.0); rc.alice.aliceG = vec4(0.0); rc.alice.aliceB = vec4(0.0);
    rc.W = 0.0;
    rc.M = 0.0;
    return rc;
}
PackedRadianceCache packRadianceCache(RadianceCache rc) {
    PackedRadianceCache p;
    p.word0 = uintBitsToFloat(uvec4(
        packHalf2x16(rc.alice.aliceR.xy), packHalf2x16(rc.alice.aliceR.zw),
        packHalf2x16(rc.alice.aliceG.xy), packHalf2x16(rc.alice.aliceG.zw)));
    p.word1 = vec4(uintBitsToFloat(packHalf2x16(rc.alice.aliceB.xy)),
        uintBitsToFloat(packHalf2x16(rc.alice.aliceB.zw)), rc.W, rc.M);
    return p;
}
RadianceCache unpackRadianceCache(vec4 word0, vec4 word1) {
    uvec4 p = floatBitsToUint(word0);
    RadianceCache rc;
    rc.alice.aliceR = vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
    rc.alice.aliceG = vec4(unpackHalf2x16(p.z), unpackHalf2x16(p.w));
    rc.alice.aliceB = vec4(unpackHalf2x16(floatBitsToUint(word1.x)), unpackHalf2x16(floatBitsToUint(word1.y)));
    rc.W = word1.z;
    rc.M = word1.w;
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
uint rcNextGeneration(uint oldToken) {
    uint generation = oldToken < RC_LOCKED_TOKEN ? (oldToken >> 12u) : 0u;
    generation = (generation + 1u) & RC_GENERATION_MASK;
    return max(generation, 1u);
}
void rcEraseRadianceCacheMapping(uint erasedIndex, uint erasedToken) {
    if (erasedIndex >= uint(RADIANCE_CACHE_MAPPING_TABLE_SIZE)
            || rcLoad(rcMapAddress(erasedIndex, 3u)) != erasedToken) return;

    // Backward-shift deletion preserves the early-empty lookup invariant and
    // prevents camera motion from filling the table with tombstones. The loop
    // is bounded; an exceptionally long cluster falls back to one tombstone
    // rather than risking an unbounded allocator pass.
    uint mask = uint(RADIANCE_CACHE_MAPPING_TABLE_SIZE - 1);
    uint hole = erasedIndex;
    for (uint step = 1u; step <= uint(RADIANCE_CACHE_MAX_HASH_PROBES); ++step) {
        uint scan = (erasedIndex + step) & mask;
        uint token = rcLoad(rcMapAddress(scan, 3u));
        if (token == RC_INVALID_TOKEN) {
            rcStore(rcMapAddress(hole, 3u), RC_INVALID_TOKEN);
            return;
        }
        if (token >= RC_LOCKED_TOKEN) {
            rcStore(rcMapAddress(hole, 3u), RC_TOMBSTONE_TOKEN);
            return;
        }

        ivec3 key = ivec3(
            rcLoad(rcMapAddress(scan, 0u)),
            rcLoad(rcMapAddress(scan, 1u)),
            rcLoad(rcMapAddress(scan, 2u)));
        uint home = radianceCacheHash(key) & mask;
        uint scanDistance = (scan - home) & mask;
        uint holeDistance = (hole - home) & mask;
        if (holeDistance < scanDistance) {
            rcStore(rcMapAddress(hole, 0u), uint(key.x));
            rcStore(rcMapAddress(hole, 1u), uint(key.y));
            rcStore(rcMapAddress(hole, 2u), uint(key.z));
            rcStore(rcMapAddress(hole, 3u), token);
            uint movedSlot = token & RC_SLOT_MASK;
            if (movedSlot < uint(RADIANCE_CACHE_POOL_CAPACITY)
                    && rcLoad(rcMetaAddress(movedSlot, RC_META_TOKEN)) == token)
                rcStore(rcMetaAddress(movedSlot, RC_META_MAP_INDEX), hole);
            hole = scan;
        }
    }
    rcStore(rcMapAddress(hole, 3u), RC_TOMBSTONE_TOKEN);
}
void rcPinPoolSlot(uint slot, uint frameStamp) {
    rcStore(rcMetaAddress(slot, RC_META_PIN_FRAME), frameStamp);
}
void initializeRadianceCacheAllocator() {
    rcStore(RC_HEADER_NEXT_UNUSED_ADDR, 0u);
    rcStore(RC_HEADER_EVICTION_CURSOR_ADDR, 0u);
    rcStore(RC_HEADER_RESERVED_ADDR, 0u);
    rcStore(RC_HEADER_FAILED_ALLOCATIONS_ADDR, 0u);
    for (uint i = 0u; i < uint(RADIANCE_CACHE_MAPPING_TABLE_SIZE); ++i)
        rcStore(rcMapAddress(i, 3u), RC_INVALID_TOKEN);
    for (uint i = 0u; i < uint(RADIANCE_CACHE_POOL_CAPACITY); ++i) {
        rcStore(rcMetaAddress(i, RC_META_TOKEN), RC_INVALID_TOKEN);
        rcStore(rcMetaAddress(i, RC_META_LRU_PREV), RC_INVALID_SLOT);
        rcStore(rcMetaAddress(i, RC_META_LRU_NEXT), RC_INVALID_SLOT);
        rcStore(rcMetaAddress(i, RC_META_BIRTH_FRAME), 0u);
        rcStore(rcMetaAddress(i, RC_META_PIN_FRAME), 0u);
    }
    rcStore(RC_HEADER_MAGIC_ADDR, RC_MAGIC);
}
bool allocateRadianceCacheBrick(ivec3 worldBrick, uint frameStamp) {
    uint mask = uint(RADIANCE_CACHE_MAPPING_TABLE_SIZE - 1);
    uint first = radianceCacheHash(worldBrick) & mask;
    uint insertionIndex = RC_INVALID_TOKEN;
    for (uint probe = 0u; probe < uint(RADIANCE_CACHE_MAX_HASH_PROBES); ++probe) {
        uint mapIndex = (first + probe) & mask;
        uint token = rcLoad(rcMapAddress(mapIndex, 3u));
        if (token == RC_TOMBSTONE_TOKEN && insertionIndex == RC_INVALID_TOKEN) insertionIndex = mapIndex;
        if (token == RC_INVALID_TOKEN) {
            if (insertionIndex == RC_INVALID_TOKEN) insertionIndex = mapIndex;
            break;
        }
    }
    if (insertionIndex == RC_INVALID_TOKEN) {
        rcStore(RC_HEADER_FAILED_ALLOCATIONS_ADDR, rcLoad(RC_HEADER_FAILED_ALLOCATIONS_ADDR) + 1u);
        return false;
    }
    uint slot;
    uint nextUnused = rcLoad(RC_HEADER_NEXT_UNUSED_ADDR);
    if (nextUnused < uint(RADIANCE_CACHE_POOL_CAPACITY)) {
        slot = nextUnused;
        rcStore(RC_HEADER_NEXT_UNUSED_ADDR, nextUnused + 1u);
    } else {
        // Nearest-first admission has already pinned every page selected for
        // this frame. A CLOCK scan therefore only needs to find an unpinned
        // page; maintaining an exact linked LRU would add several serialized,
        // random SSBO writes for every resident request.
        slot = RC_INVALID_SLOT;
        uint capacityMask = uint(RADIANCE_CACHE_POOL_CAPACITY - 1);
        uint cursor = rcLoad(RC_HEADER_EVICTION_CURSOR_ADDR) & capacityMask;
        for (uint scan = 0u; scan < uint(RADIANCE_CACHE_POOL_CAPACITY); ++scan) {
            uint candidate = (cursor + scan) & capacityMask;
            if (rcLoad(rcMetaAddress(candidate, RC_META_PIN_FRAME)) != frameStamp) {
                slot = candidate;
                rcStore(RC_HEADER_EVICTION_CURSOR_ADDR,
                    (candidate + 1u) & capacityMask);
                break;
            }
        }
        if (slot >= uint(RADIANCE_CACHE_POOL_CAPACITY)) {
            rcStore(RC_HEADER_FAILED_ALLOCATIONS_ADDR,
                rcLoad(RC_HEADER_FAILED_ALLOCATIONS_ADDR) + 1u);
            return false;
        }
        uint oldToken = rcLoad(rcMetaAddress(slot, RC_META_TOKEN));
        uint oldMapIndex = rcLoad(rcMetaAddress(slot, RC_META_MAP_INDEX));
        if (oldToken < RC_LOCKED_TOKEN)
            rcEraseRadianceCacheMapping(oldMapIndex, oldToken);
    }
    uint oldToken = rcLoad(rcMetaAddress(slot, RC_META_TOKEN));
    uint newToken = (rcNextGeneration(oldToken) << 12u) | slot;
    rcStore(rcMetaAddress(slot, RC_META_KEY_X), uint(worldBrick.x));
    rcStore(rcMetaAddress(slot, RC_META_KEY_Y), uint(worldBrick.y));
    rcStore(rcMetaAddress(slot, RC_META_KEY_Z), uint(worldBrick.z));
    rcStore(rcMetaAddress(slot, RC_META_MAP_INDEX), insertionIndex);
    rcStore(rcMetaAddress(slot, RC_META_TOKEN), newToken);
    // ray4 overwrites every current voxel. Temporal treats both RIS and
    // filtered history as empty on this birth frame, so recycling never needs
    // a full payload clear.
    rcStore(rcMetaAddress(slot, RC_META_BIRTH_FRAME), frameStamp);
    rcStore(rcMetaAddress(slot, RC_META_PIN_FRAME), frameStamp);
    rcStore(rcMapAddress(insertionIndex, 0u), uint(worldBrick.x));
    rcStore(rcMapAddress(insertionIndex, 1u), uint(worldBrick.y));
    rcStore(rcMapAddress(insertionIndex, 2u), uint(worldBrick.z));
    rcStore(rcMapAddress(insertionIndex, 3u), newToken);
    return true;
}
ivec3 rcRequestWorldBrick(uint requestIndex, ivec3 minBrick) {
    uint x = requestIndex % uint(RADIANCE_CACHE_BRICKS_X);
    uint yz = requestIndex / uint(RADIANCE_CACHE_BRICKS_X);
    uint y = yz % uint(RADIANCE_CACHE_BRICKS_Y);
    uint z = yz / uint(RADIANCE_CACHE_BRICKS_Y);
    return minBrick + ivec3(x, y, z);
}
uint rcDistanceBucket(ivec3 worldBrick, vec3 cameraPosition) {
    if (!rcFiniteWorldPosition(cameraPosition))
        return uint(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT - 1);
    vec3 brickCenter = (vec3(worldBrick) * float(RADIANCE_CACHE_BRICK_SIZE)
        + vec3(0.5 * float(RADIANCE_CACHE_BRICK_SIZE))) * VOXEL_SIZE;
    vec3 delta = abs(brickCenter - cameraPosition);
    float distanceInBricks = max(max(delta.x, delta.y), delta.z)
        / RADIANCE_CACHE_BRICK_WORLD_SIZE;
    float clampedBucket = min(floor(distanceInBricks),
        float(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT - 1));
    return uint(max(clampedBucket, 0.0));
}
// Called by the one-invocation ray3 allocator pass.
void processRadianceCacheAllocationRequests(vec3 cameraPosition, uint frameStamp) {
    if (!radianceCacheStorageAvailable()) return;
    if (!rcFiniteWorldPosition(cameraPosition)) {
        for (uint wordIndex = 0u; wordIndex < RC_REQUEST_WORDS; ++wordIndex)
            rcStore(RC_REQUEST_OFFSET + wordIndex, 0u);
        return;
    }
    if (rcLoad(RC_HEADER_MAGIC_ADDR) != RC_MAGIC) initializeRadianceCacheAllocator();
    uint currentFrame = rcFrameTag(frameStamp);
    ivec3 minBrick = radianceCacheClipMinBrick(cameraPosition);

    for (uint bucket = 0u; bucket < uint(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT); ++bucket) {
        rcStore(RC_DISTANCE_BUCKET_OFFSET + bucket, 0u);
        rcStore(RC_MISSING_BUCKET_OFFSET + bucket, 0u);
    }

    // Phase 1: count every request by camera distance. Resident pages receive
    // a small hysteresis advantage so motion near the capacity boundary does
    // not repeatedly swap equally useful pages.
    for (uint wordIndex = 0u; wordIndex < RC_REQUEST_WORDS; ++wordIndex) {
        uint bits = rcLoad(RC_REQUEST_OFFSET + wordIndex);
        while (bits != 0u) {
            uint bit = uint(findLSB(bits));
            uint requestIndex = (wordIndex << 5u) + bit;
            ivec3 worldBrick = rcRequestWorldBrick(requestIndex, minBrick);
            uint token = findRadianceCacheMapping(worldBrick);
            uint bucket = rcDistanceBucket(worldBrick, cameraPosition);
            if (token != RC_INVALID_TOKEN)
                bucket -= min(bucket, uint(RADIANCE_CACHE_RESIDENT_HYSTERESIS_BUCKETS));
            uint bucketAddress = RC_DISTANCE_BUCKET_OFFSET + bucket;
            rcStore(bucketAddress, rcLoad(bucketAddress) + 1u);
            bits &= bits - 1u;
        }
    }

    // Alpha-Piscium-style nearest-first admission across both resident and
    // missing pages. Overflow is rejected before it can perturb the LRU.
    uint residencyBudget = uint(RADIANCE_CACHE_POOL_CAPACITY);
    for (uint bucket = 0u; bucket < uint(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT); ++bucket) {
        uint bucketAddress = RC_DISTANCE_BUCKET_OFFSET + bucket;
        uint allowed = min(rcLoad(bucketAddress), residencyBudget);
        rcStore(bucketAddress, allowed);
        residencyBudget -= allowed;
    }

    // Phase 2: pin selected resident pages and compact selected misses back into
    // the request bitset. Once this finishes, every possible victim is stale.
    for (uint wordIndex = 0u; wordIndex < RC_REQUEST_WORDS; ++wordIndex) {
        uint bits = rcLoad(RC_REQUEST_OFFSET + wordIndex);
        uint selectedMissingBits = 0u;
        while (bits != 0u) {
            uint bit = uint(findLSB(bits));
            uint requestIndex = (wordIndex << 5u) + bit;
            ivec3 worldBrick = rcRequestWorldBrick(requestIndex, minBrick);
            uint token = findRadianceCacheMapping(worldBrick);
            uint bucket = rcDistanceBucket(worldBrick, cameraPosition);
            if (token != RC_INVALID_TOKEN)
                bucket -= min(bucket, uint(RADIANCE_CACHE_RESIDENT_HYSTERESIS_BUCKETS));
            uint bucketAddress = RC_DISTANCE_BUCKET_OFFSET + bucket;
            uint allowed = rcLoad(bucketAddress);
            if (allowed != 0u) {
                rcStore(bucketAddress, allowed - 1u);
                if (token != RC_INVALID_TOKEN) {
                    rcPinPoolSlot(token & RC_SLOT_MASK, currentFrame);
                } else {
                    selectedMissingBits |= 1u << bit;
                    uint missingAddress = RC_MISSING_BUCKET_OFFSET + bucket;
                    rcStore(missingAddress, rcLoad(missingAddress) + 1u);
                }
            }
            bits &= bits - 1u;
        }
        rcStore(RC_REQUEST_OFFSET + wordIndex, selectedMissingBits);
    }

    // Phase 3: rate-limit physical changes, again nearest-first. This bounds
    // both allocator work and the number of birth-frame payloads per frame.
    uint allocationBudget = uint(RADIANCE_CACHE_MAX_ALLOCATIONS_PER_FRAME);
    for (uint bucket = 0u; bucket < uint(RADIANCE_CACHE_DISTANCE_BUCKET_COUNT); ++bucket) {
        uint bucketAddress = RC_MISSING_BUCKET_OFFSET + bucket;
        uint allowed = min(rcLoad(bucketAddress), allocationBudget);
        rcStore(bucketAddress, allowed);
        allocationBudget -= allowed;
    }

    // Phase 4: allocate admitted misses. Selected residents are pinned at the
    // LRU tail, so allocation can only recycle an unselected page.
    for (uint wordIndex = 0u; wordIndex < RC_REQUEST_WORDS; ++wordIndex) {
        uint bits = rcLoad(RC_REQUEST_OFFSET + wordIndex);
        while (bits != 0u) {
            uint bit = uint(findLSB(bits));
            uint requestIndex = (wordIndex << 5u) + bit;
            ivec3 worldBrick = rcRequestWorldBrick(requestIndex, minBrick);
            uint bucket = rcDistanceBucket(worldBrick, cameraPosition);
            uint bucketAddress = RC_MISSING_BUCKET_OFFSET + bucket;
            uint allowed = rcLoad(bucketAddress);
            if (allowed != 0u && allocateRadianceCacheBrick(worldBrick, currentFrame))
                rcStore(bucketAddress, allowed - 1u);
            bits &= bits - 1u;
        }
        rcStore(RC_REQUEST_OFFSET + wordIndex, 0u);
    }
}

RadianceCacheAddress findRadianceCacheAddress(vec3 worldPos) {
    if (!rcFiniteWorldPosition(worldPos)) return invalidRadianceCacheAddress();
    ivec3 voxel = radianceCacheWorldVoxel(worldPos);
    ivec3 brick = radianceCacheWorldBrick(voxel);
    uint token = findRadianceCacheMapping(brick);
    if (token == RC_INVALID_TOKEN) return invalidRadianceCacheAddress();
    RadianceCacheAddress a;
    a.token = token; a.slot = token & RC_SLOT_MASK;
    a.localIndex = radianceCacheLocalIndex(voxel, brick);
    return a;
}
bool validateRadianceCacheAddress(RadianceCacheAddress a) {
    return a.token < RC_LOCKED_TOKEN && a.slot < uint(RADIANCE_CACHE_POOL_CAPACITY)
        && a.localIndex < RC_VOXELS_PER_BRICK
        && rcLoad(rcMetaAddress(a.slot, RC_META_TOKEN)) == a.token;
}
bool radianceCacheAddressHasHistory(RadianceCacheAddress a, uint frameStamp) {
    return validateRadianceCacheAddress(a)
        && rcLoad(rcMetaAddress(a.slot, RC_META_BIRTH_FRAME)) != rcFrameTag(frameStamp);
}
bool radianceCacheResolvedPoolAddressHasHistory(
        RadianceCacheAddress a, uint frameStamp) {
    // For addresses returned immediately by radianceCacheAddressForPoolVoxel.
    // That function has just read the slot token and execution-group barriers
    // prevent allocator mutation here, so repeating the token load is wasteful.
    // Structural bounds remain explicit, preserving defined SSBO access even
    // if an invalid address is passed accidentally.
    return a.token < RC_LOCKED_TOKEN
        && a.slot < uint(RADIANCE_CACHE_POOL_CAPACITY)
        && a.localIndex < RC_VOXELS_PER_BRICK
        && rcLoad(rcMetaAddress(a.slot, RC_META_BIRTH_FRAME)) != rcFrameTag(frameStamp);
}
RadianceCache loadRadianceCachePlanes(RadianceCacheAddress a, uint plane0, uint plane1) {
    if (!radianceCacheStorageAvailable() || !validateRadianceCacheAddress(a)) return emptyCache();
    return unpackRadianceCache(
        uintBitsToFloat(rcLoadVec4(rcPayloadAddress(a.slot, plane0, a.localIndex))),
        uintBitsToFloat(rcLoadVec4(rcPayloadAddress(a.slot, plane1, a.localIndex))));
}
void storeRadianceCachePlanes(RadianceCacheAddress a, uint plane0, uint plane1, RadianceCache rc) {
    if (!radianceCacheStorageAvailable() || !validateRadianceCacheAddress(a)) return;
    PackedRadianceCache p = packRadianceCache(rc);
    rcStoreVec4(rcPayloadAddress(a.slot, plane0, a.localIndex), floatBitsToUint(p.word0));
    rcStoreVec4(rcPayloadAddress(a.slot, plane1, a.localIndex), floatBitsToUint(p.word1));
}
RadianceCacheAddress radianceCacheAddressForVoxel(uvec3 voxelCoord, vec3 cameraPosition) {
    if (any(greaterThanEqual(voxelCoord, uvec3(RADIANCE_CACHE_W, RADIANCE_CACHE_H, RADIANCE_CACHE_D))))
        return invalidRadianceCacheAddress();
    return findRadianceCacheAddress(radianceCacheVoxelWorldPos(voxelCoord, cameraPosition));
}
RadianceCacheAddress radianceCacheAddressForPoolVoxel(
    uint linearIndex, vec3 cameraPosition, out ivec3 worldVoxel) {
    worldVoxel = ivec3(0);
    if (linearIndex >= uint(RADIANCE_CACHE_POOL_CAPACITY) * RC_VOXELS_PER_BRICK
            || !radianceCacheReady()) return invalidRadianceCacheAddress();
    uint slot = linearIndex >> 6u;
    uint localIndex = linearIndex & 63u;
    uint token = rcLoad(rcMetaAddress(slot, RC_META_TOKEN));
    if (token >= RC_LOCKED_TOKEN || (token & RC_SLOT_MASK) != slot)
        return invalidRadianceCacheAddress();
    ivec3 worldBrick = ivec3(
        rcLoad(rcMetaAddress(slot, RC_META_KEY_X)),
        rcLoad(rcMetaAddress(slot, RC_META_KEY_Y)),
        rcLoad(rcMetaAddress(slot, RC_META_KEY_Z)));
    ivec3 relative = worldBrick - radianceCacheClipMinBrick(cameraPosition);
    if (any(lessThan(relative, ivec3(0))) || any(greaterThanEqual(relative,
            ivec3(RADIANCE_CACHE_BRICKS_X, RADIANCE_CACHE_BRICKS_Y,
                RADIANCE_CACHE_BRICKS_Z)))) return invalidRadianceCacheAddress();
    ivec3 local = ivec3(
        int(localIndex & 3u),
        int((localIndex >> 2u) & 3u),
        int((localIndex >> 4u) & 3u));
    worldVoxel = worldBrick * RADIANCE_CACHE_BRICK_SIZE + local;
    RadianceCacheAddress a;
    a.token = token; a.slot = slot; a.localIndex = localIndex;
    return a;
}
RadianceCache loadRadianceCacheHistWorld(vec3 worldPos) {
    return loadRadianceCachePlanes(findRadianceCacheAddress(worldPos),
        RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
}
RadianceCache loadRadianceCacheHist(uvec3 voxelCoord, vec3 cameraPosition) {
    return loadRadianceCachePlanes(radianceCacheAddressForVoxel(voxelCoord, cameraPosition),
        RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
}
RadianceCache sampleRadianceCacheHist(vec3 voxelCoord, vec3 cameraPosition) {
    ivec3 nearest = ivec3(floor(voxelCoord + 0.5));
    if (!isRadianceCacheCoordInBounds(nearest)) return emptyCache();
    return loadRadianceCacheHist(uvec3(nearest), cameraPosition);
}

#endif // RADIANCE_CACHE_GLSL
