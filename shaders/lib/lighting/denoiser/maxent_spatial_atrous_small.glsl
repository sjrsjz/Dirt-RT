#ifndef MAXENT_SPATIAL_ATROUS_SMALL_GLSL
#define MAXENT_SPATIAL_ATROUS_SMALL_GLSL

// Compile-time policy interface required from the including pass:
//   ivec2 denoiserSpatialImageSize()
//   uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel)
//   uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel)
//   DenoiserSpatialCenterGeometry denoiserSpatialDecodeCenterGeometry(words)
//   void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal)
//   void denoiserSpatialStoreInvalid(ivec2 pixel)
// Required macros:
//   DENOISER_SPATIAL_STEP, DENOISER_SPATIAL_PHI_LUMINANCE

#define DENOISER_SPATIAL_TILE_SIZE (16 + 2 * DENOISER_SPATIAL_STEP)
#define DENOISER_SPATIAL_TILE_AREA (DENOISER_SPATIAL_TILE_SIZE * DENOISER_SPATIAL_TILE_SIZE)

// Geometry stays on its native cached path. Share the expensive Bures
// eigendecomposition across overlapping taps. Packing both standard
// deviations into one uint halves the LDS footprint; the signal itself has
// already crossed an FP16 storage boundary before this pass.
shared uint denoiserSpatialTileStddev[DENOISER_SPATIAL_TILE_AREA];

const ivec2 DENOISER_SPATIAL_GRID_8[8] = ivec2[](
        ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0),
        ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));
const float DENOISER_SPATIAL_GRID_WEIGHT[8] = float[](
        0.44445, 0.66667, 0.44445, 0.66667,
        0.66667, 0.44445, 0.66667, 0.44445);

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

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    uvec2 localID = gl_LocalInvocationID.xy;
    ivec2 size = denoiserSpatialImageSize();
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(DENOISER_SPATIAL_STEP);

    for (uint tileIndex = gl_LocalInvocationIndex;
        tileIndex < uint(DENOISER_SPATIAL_TILE_AREA);
        tileIndex += 256u) {
        uint tileY = tileIndex / uint(DENOISER_SPATIAL_TILE_SIZE);
        uint tileX = tileIndex - tileY * uint(DENOISER_SPATIAL_TILE_SIZE);
        ivec2 sourcePixel = tileOrigin + ivec2(tileX, tileY);
        uvec4 signalWords = denoiserInvalidMaxEntSignalWords();
        vec2 stddev = vec2(0.0);
        if (denoiserSpatialInBounds(sourcePixel, size)) {
            signalWords = denoiserSpatialLoadSignalWords(sourcePixel);
            if (denoiserSpatialSignalWordsValid(signalWords)) {
                DenoiserMaxEntSignal signal = denoiserUnpackMaxEntSignalTrusted(signalWords);
                stddev = denoiserSpatialMakeBuresData(signal.maxEntY).stddev;
            }
        }
        denoiserSpatialTileStddev[tileIndex] = packHalf2x16(min(stddev, vec2(DENOISER_SPATIAL_FP16_MAX)));
    }

    barrier();
    if (!denoiserSpatialInBounds(pixel, size)) return;

    uint centerX = localID.x + uint(DENOISER_SPATIAL_STEP);
    uint centerY = localID.y + uint(DENOISER_SPATIAL_STEP);
    uint centerIndex = centerY * uint(DENOISER_SPATIAL_TILE_SIZE) + centerX;
    uvec4 centerSignalWords = denoiserSpatialLoadSignalWords(pixel);
    if (!denoiserSpatialSignalWordsValid(centerSignalWords)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }

    uvec4 centerGeometryWords = denoiserSpatialLoadGeometryWords(pixel);
    if (!denoiserSpatialGeometryWordsValid(centerGeometryWords)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }
    DenoiserSpatialCenterGeometry centerGeometry =
        denoiserSpatialDecodeCenterGeometry(centerGeometryWords);
    DenoiserMaxEntSignal centerSignal = denoiserUnpackMaxEntSignalTrusted(centerSignalWords);
    DenoiserSpatialBuresData centerBures =
        denoiserSpatialMakeBuresDataFromStddev(unpackHalf2x16(denoiserSpatialTileStddev[centerIndex]));

    float surfaceRejectionScale = denoiserSpatialSurfaceRejectionScale(
            centerGeometry.surfaceDistance, float(size.y));
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
    DenoiserSpatialAccumulator accum = denoiserSpatialBeginAccumulation(centerSignal);

    for (int i = 0; i < 8; ++i) {
        int sampleX = int(centerX) + DENOISER_SPATIAL_GRID_8[i].x * DENOISER_SPATIAL_STEP;
        int sampleY = int(centerY) + DENOISER_SPATIAL_GRID_8[i].y * DENOISER_SPATIAL_STEP;
        uint sampleIndex = uint(sampleY * DENOISER_SPATIAL_TILE_SIZE + sampleX);
        ivec2 samplePixel = pixel + DENOISER_SPATIAL_GRID_8[i] * DENOISER_SPATIAL_STEP;
        if (!denoiserSpatialInBounds(samplePixel, size)) continue;

        uvec4 sampleGeometryWords = denoiserSpatialLoadGeometryWords(samplePixel);
        uvec4 sampleSignalWords = denoiserSpatialLoadSignalWords(samplePixel);
        if (!denoiserSpatialGeometryWordsValid(sampleGeometryWords)
                || !denoiserSpatialSignalWordsValid(sampleSignalWords))
            continue;

        vec3 samplePrimaryRay;
        float sampleSurfaceDistance;
        denoiserSpatialDecodeSampleGeometry(sampleGeometryWords,
            samplePrimaryRay, sampleSurfaceDistance);
        vec3 sampleSurfacePosition = samplePrimaryRay
                * sampleSurfaceDistance;
        float surfaceGeometryExponent = surfaceRejectionScale
                * abs(dot(centerGeometry.surfaceNormal, sampleSurfacePosition)
                        - centerGeometry.surfacePlaneOffset);

        DenoiserMaxEntSignal sampleSignal = denoiserUnpackMaxEntSignalTrusted(sampleSignalWords);
        DenoiserSpatialBuresData sampleBures =
            denoiserSpatialMakeBuresDataFromStddev(unpackHalf2x16(
                    denoiserSpatialTileStddev[sampleIndex]));
        float virtualDistanceWeight;
        float weight = denoiserSpatialWeight(centerSignal, centerBures,
                sampleSignal, sampleBures, samplePrimaryRay,
                surfaceGeometryExponent,
                DENOISER_SPATIAL_GRID_WEIGHT[i],
                DENOISER_SPATIAL_PHI_LUMINANCE,
                virtualDistanceAlpha, centerVirtualPosition,
                centerVirtualNormal, virtualRejectionScale,
                virtualDistanceWeight);
        denoiserSpatialAccumulate(accum, sampleSignal, weight,
            virtualDistanceWeight);
    }

    denoiserSpatialStore(pixel, denoiserSpatialResolve(accum, DENOISER_SPATIAL_STEP));
}

#endif // MAXENT_SPATIAL_ATROUS_SMALL_GLSL
