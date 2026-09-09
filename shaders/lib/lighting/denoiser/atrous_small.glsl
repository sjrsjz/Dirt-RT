#ifndef MAXENT_DENOISER_ATROUS_SMALL_GLSL
#define MAXENT_DENOISER_ATROUS_SMALL_GLSL

// Compile-time policy interface required from the including pass:
//   ivec2 denoiserSpatialImageSize()
//   uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel)
//   uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel)
//   uvec4 denoiserSpatialLoadIndependentCurrentWords(ivec2 pixel)
//   DenoiserSpatialCenterGeometry denoiserSpatialDecodeCenterGeometry(words, pixel)
//   void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal,
//       DenoiserMaxEntSignal independentCurrent)
//   void denoiserSpatialStoreInvalid(ivec2 pixel)
// Required macros:
//   DENOISER_SPATIAL_STEP, DENOISER_SPATIAL_PHI_LUMINANCE

// Two bits per tap encode each offset component plus one. The diagonal mask
// preserves the existing eight-tap order without dynamically indexed arrays.
// Offsets: (-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1).
// Keep the original rounded weight literals; replacing them with fractions
// changes the existing kernel and its FP16 rounding.
const uint DENOISER_SPATIAL_GRID_X_BITS = 0x9224u;
const uint DENOISER_SPATIAL_GRID_Y_BITS = 0xa940u;
const uint DENOISER_SPATIAL_DIAGONAL_BITS = 0xa5u;

bool denoiserSpatialInBounds(ivec2 pixel, ivec2 size) {
    return all(greaterThanEqual(pixel, ivec2(0)))
        && all(lessThan(pixel, size));
}

vec3 denoiserSpatialLoadVirtualPosition(
    ivec2 pixel, ivec2 size, vec3 fallback) {
    if (!denoiserSpatialInBounds(pixel, size)) return fallback;
    return denoiserSpatialVirtualWorldPositionFromWords(
        pixel, denoiserSpatialLoadSignalWords(pixel), fallback);
}

#define DENOISER_RAY_TILE_SIZE (16 + 2 * DENOISER_SPATIAL_STEP)
#define DENOISER_RAY_TILE_AREA (DENOISER_RAY_TILE_SIZE * DENOISER_RAY_TILE_SIZE)
// Share the exact FP32 result, including projection jitter and world rotation.
// Each source ray serves several tap and virtual-normal queries; storing it
// avoids repeating projection reciprocals, normalization and matrix products.
// Split xyz into vec2 + float to avoid shared vec3 padding.
shared vec2 denoiserRayTileXY[DENOISER_RAY_TILE_AREA];
shared float denoiserRayTileZ[DENOISER_RAY_TILE_AREA];
vec3 denoiserSpatialPrimaryRay(ivec2 pixel) {
    ivec2 origin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(DENOISER_SPATIAL_STEP);
    ivec2 p = pixel - origin;
    uint index = uint(p.y * DENOISER_RAY_TILE_SIZE + p.x);
    return vec3(denoiserRayTileXY[index], denoiserRayTileZ[index]);
}

void main() {
    // The +/- step halo also contains the four +/- 1 virtual-normal samples.
    // All lanes, including out-of-image lanes, must reach the barrier.
    ivec2 rayOrigin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(DENOISER_SPATIAL_STEP);
    for (uint index = gl_LocalInvocationIndex; index < uint(DENOISER_RAY_TILE_AREA); index += 256u) {
        ivec2 p = rayOrigin + ivec2(index % uint(DENOISER_RAY_TILE_SIZE), index / uint(DENOISER_RAY_TILE_SIZE));
        // No output references out-of-image halo slots; avoid unsigned wrap
        // during their otherwise unused ray reconstruction.
        vec3 ray = reconstructPrimaryRay(uvec2(max(p, ivec2(0))));
        denoiserRayTileXY[index] = ray.xy;
        denoiserRayTileZ[index] = ray.z;
    }
    barrier();
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = denoiserSpatialImageSize();
    if (!denoiserSpatialInBounds(pixel, size)) return;

    uvec4 centerSignalWords = denoiserSpatialLoadSignalWords(pixel);
    uvec4 centerCurrentWords =
        denoiserSpatialLoadIndependentCurrentWords(pixel);
    if (!denoiserSpatialPreparedSignalWordsValid(centerSignalWords)
            || !denoiserSpatialPreparedSignalWordsValid(centerCurrentWords)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }

    uvec4 centerGeometryWords = denoiserSpatialLoadGeometryWords(pixel);
    if (!denoiserSpatialGeometryWordsValid(centerGeometryWords)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }
    DenoiserSpatialCenterGeometry centerGeometry =
        denoiserSpatialDecodeCenterGeometry(centerGeometryWords, pixel);
    DenoiserMaxEntSignal centerSignal = denoiserUnpackMaxEntSignalTrusted(centerSignalWords);
    DenoiserMaxEntSignal centerCurrent =
        denoiserUnpackMaxEntSignalTrusted(centerCurrentWords);
    if (!statisticsValidEffectiveSampleCount(centerGeometry.effectiveSamples)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }

    float surfaceRejectionScale = denoiserSpatialDistanceRejectionScale(
            centerGeometry.surfaceDistance, float(size.y));
    MaxEntLightMetric centerMetric = maxentPrepareLightMetric(centerSignal.maxEntY);
    float virtualDistanceAlpha = centerGeometry.ggxAlpha;
    vec3 centerVirtualPosition = denoiserSpatialVirtualWorldPosition(
            centerGeometry.primaryRay, centerSignal.virtualDistance);
    float virtualRejectionScale = denoiserSpatialVirtualRejectionScale(
            centerGeometry.ggxAlpha, centerSignal.virtualDistance);
    // Accumulate tangents in place so four decoded positions do not have to
    // remain live across the last neighbor fetch.
    vec3 virtualTangentX = -denoiserSpatialLoadVirtualPosition(pixel + ivec2(-1, 0), size, centerVirtualPosition);
    virtualTangentX += denoiserSpatialLoadVirtualPosition(pixel + ivec2(1, 0), size, centerVirtualPosition);
    vec3 virtualTangentY = -denoiserSpatialLoadVirtualPosition(pixel + ivec2(0, -1), size, centerVirtualPosition);
    virtualTangentY += denoiserSpatialLoadVirtualPosition(pixel + ivec2(0, 1), size, centerVirtualPosition);
    vec3 centerVirtualNormal = denoiserSpatialVirtualNormal(
            virtualTangentX, virtualTangentY, centerGeometry.primaryRay);
    float lightDifferenceScale = DENOISER_SPATIAL_PHI_LUMINANCE;
    DenoiserSpatialAccumulator accum = denoiserSpatialBeginAccumulation(centerSignal);
    DenoiserSpatialCurrentAccumulator currentAccum =
        denoiserSpatialBeginCurrentAccumulation(centerCurrent);

    for (int i = 0; i < 8; ++i) {
        int tapShift = i * 2;
        ivec2 tapOffset = ivec2(
            bitfieldExtract(DENOISER_SPATIAL_GRID_X_BITS, tapShift, 2),
            bitfieldExtract(DENOISER_SPATIAL_GRID_Y_BITS, tapShift, 2)) - 1;
        ivec2 samplePixel = pixel + tapOffset * DENOISER_SPATIAL_STEP;
        if (!denoiserSpatialInBounds(samplePixel, size)) continue;

        float kernelWeight = bitfieldExtract(DENOISER_SPATIAL_DIAGONAL_BITS, i, 1) != 0u
            ? 0.44445 : 0.66667;
#include "/lib/lighting/denoiser/atrous_tap.glsl"
    }

    denoiserSpatialStore(pixel, denoiserSpatialResolve(accum, DENOISER_SPATIAL_STEP), denoiserSpatialResolve(currentAccum, DENOISER_SPATIAL_STEP));
}

#endif // MAXENT_DENOISER_ATROUS_SMALL_GLSL
