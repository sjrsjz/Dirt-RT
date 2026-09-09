#ifndef RADIANCE_CACHE_ALLOCATOR_GLSL
#define RADIANCE_CACHE_ALLOCATOR_GLSL

// Single-writer allocation, nearest-first admission, backward-shift deletion and CLOCK eviction.
// Included through radiance_cache.glsl; declaration order is intentional.

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
    // ray5 overwrites every current voxel. Temporal treats both RIS and
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
// Called by the one-invocation ray4 allocator pass.
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

    // Nearest-first admission across both resident and
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

#endif // RADIANCE_CACHE_ALLOCATOR_GLSL
