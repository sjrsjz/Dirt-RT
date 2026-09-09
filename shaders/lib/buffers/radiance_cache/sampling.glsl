#ifndef RADIANCE_CACHE_SAMPLING_GLSL
#define RADIANCE_CACHE_SAMPLING_GLSL

// Validated pool access and filtered cache sampling. Allocation must have finished before use.
// Included through radiance_cache.glsl; declaration order is intentional.

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
        rcLoadVec4(rcPayloadAddress(a.slot, plane0, a.localIndex)),
        rcLoadVec4(rcPayloadAddress(a.slot, plane1, a.localIndex)));
}
void storeRadianceCachePlanes(RadianceCacheAddress a, uint plane0, uint plane1, RadianceCache rc) {
    if (!radianceCacheStorageAvailable() || !validateRadianceCacheAddress(a)) return;
    PackedRadianceCache p = packRadianceCache(rc);
    rcStoreVec4(rcPayloadAddress(a.slot, plane0, a.localIndex), p.word0);
    rcStoreVec4(rcPayloadAddress(a.slot, plane1, a.localIndex), p.word1);
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

#endif // RADIANCE_CACHE_SAMPLING_GLSL
