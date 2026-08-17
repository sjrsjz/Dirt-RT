#ifndef MAXENT_SPATIAL_ATROUS_SMALL_GLSL
#define MAXENT_SPATIAL_ATROUS_SMALL_GLSL

// Compile-time policy interface required from the including pass:
//   ivec2 denoiserSpatialImageSize()
//   uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel)
//   uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel)
//   DenoiserSpatialGeometry denoiserSpatialDecodeGeometry(words, pixel)
//   void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal)
//   void denoiserSpatialStoreInvalid(ivec2 pixel)
// Required macros:
//   DENOISER_SPATIAL_STEP, DENOISER_SPATIAL_PHI_LUMINANCE

#define DENOISER_SPATIAL_TILE_SIZE (16 + 2 * DENOISER_SPATIAL_STEP)
#define DENOISER_SPATIAL_TILE_AREA (DENOISER_SPATIAL_TILE_SIZE * DENOISER_SPATIAL_TILE_SIZE)

// Geometry stays on its native cached path. Share the expensive Bures
// eigendecomposition across overlapping taps.
shared vec2 denoiserSpatialTileStddev[DENOISER_SPATIAL_TILE_AREA];

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

vec3 denoiserSpatialLoadVirtualPosition(ivec2 pixel, ivec2 size, uvec4 signalWords, vec3 fallback) {
    if (!denoiserSpatialInBounds(pixel, size)) return fallback;
    return denoiserSpatialVirtualWorldPositionFromWords(
        denoiserSpatialLoadGeometryWords(pixel), signalWords, fallback);
}

uvec4 denoiserSpatialLoadVirtualSignalWords(ivec2 pixel, ivec2 size) {
    return denoiserSpatialInBounds(pixel, size)
    ? denoiserSpatialLoadSignalWords(pixel) : denoiserInvalidMaxEntSignalWords();
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
        uint tileX = tileIndex
                - tileY * uint(DENOISER_SPATIAL_TILE_SIZE);
        ivec2 sourcePixel = tileOrigin + ivec2(tileX, tileY);
        uvec4 signalWords = denoiserInvalidMaxEntSignalWords();
        vec2 stddev = vec2(0.0);
        if (denoiserSpatialInBounds(sourcePixel, size)) {
            signalWords = denoiserSpatialLoadSignalWords(sourcePixel);
            if (denoiserSpatialSignalWordsValid(signalWords)) {
                DenoiserMaxEntSignal signal =
                    denoiserUnpackMaxEntSignalTrusted(signalWords);
                stddev = denoiserSpatialMakeBuresData(
                        signal.maxEntY).stddev;
            }
        }
        denoiserSpatialTileStddev[tileIndex] = stddev;
    }

    barrier();
    if (!denoiserSpatialInBounds(pixel, size)) return;

    uint centerX = localID.x + uint(DENOISER_SPATIAL_STEP);
    uint centerY = localID.y + uint(DENOISER_SPATIAL_STEP);
    uint centerIndex = centerY * uint(DENOISER_SPATIAL_TILE_SIZE)
            + centerX;
    uvec4 centerGeometryWords = denoiserSpatialLoadGeometryWords(pixel);
    uvec4 centerSignalWords = denoiserSpatialLoadSignalWords(pixel);
    if (!denoiserSpatialGeometryWordsValid(centerGeometryWords)
            || !denoiserSpatialSignalWordsValid(centerSignalWords)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }

    DenoiserSpatialGeometry centerGeometry = denoiserSpatialDecodeGeometry(centerGeometryWords, pixel);
    DenoiserMaxEntSignal centerSignal = denoiserUnpackMaxEntSignalTrusted(centerSignalWords);
    DenoiserSpatialBuresData centerBures = denoiserSpatialMakeBuresDataFromStddev(denoiserSpatialTileStddev[centerIndex]);

    float lightSourceToleranceScale = denoiserSpatialLightSourceToleranceScale(centerGeometry.roughness);
    float hitDistanceAlpha = centerGeometry.ggxAlpha;
    vec3 centerVirtualPosition = denoiserSpatialVirtualWorldPosition(centerGeometry, centerSignal.hitDistance);
    float virtualRejectionScale = denoiserSpatialVirtualRejectionScale(centerGeometry, centerSignal.hitDistance);
    // Accumulate tangents in place so four decoded positions do not have to
    // remain live across the last neighbor fetch.
    vec3 virtualTangentX = -denoiserSpatialLoadVirtualPosition(
            pixel + ivec2(-1, 0), size,
            denoiserSpatialLoadVirtualSignalWords(pixel + ivec2(-1, 0), size),
            centerVirtualPosition);
    virtualTangentX += denoiserSpatialLoadVirtualPosition(
            pixel + ivec2(1, 0), size,
            denoiserSpatialLoadVirtualSignalWords(pixel + ivec2(1, 0), size),
            centerVirtualPosition);
    vec3 virtualTangentY = -denoiserSpatialLoadVirtualPosition(
            pixel + ivec2(0, -1), size,
            denoiserSpatialLoadVirtualSignalWords(pixel + ivec2(0, -1), size),
            centerVirtualPosition);
    virtualTangentY += denoiserSpatialLoadVirtualPosition(
            pixel + ivec2(0, 1), size,
            denoiserSpatialLoadVirtualSignalWords(pixel + ivec2(0, 1), size),
            centerVirtualPosition);
    vec3 centerVirtualNormal = denoiserSpatialVirtualNormal(
            virtualTangentX, virtualTangentY, centerGeometry.pdfDirection);
    DenoiserSpatialAccumulator accum =
        denoiserSpatialBeginAccumulation(centerSignal);

    for (int i = 0; i < 8; ++i) {
        int sampleX = int(centerX) + DENOISER_SPATIAL_GRID_8[i].x * DENOISER_SPATIAL_STEP;
        int sampleY = int(centerY) + DENOISER_SPATIAL_GRID_8[i].y * DENOISER_SPATIAL_STEP;
        uint sampleIndex = uint(sampleY * DENOISER_SPATIAL_TILE_SIZE + sampleX);
        ivec2 samplePixel = pixel + DENOISER_SPATIAL_GRID_8[i] * DENOISER_SPATIAL_STEP;
        if (!denoiserSpatialInBounds(samplePixel, size)) continue;

        uvec4 sampleGeometryWords = denoiserSpatialLoadGeometryWords(samplePixel);
        uvec4 sampleSignalWords = denoiserSpatialLoadSignalWords(samplePixel);
        if (!denoiserSpatialGeometryWordsValid(sampleGeometryWords) || !denoiserSpatialSignalWordsValid(sampleSignalWords))
            continue;

        DenoiserSpatialGeometry sampleGeometry = denoiserSpatialDecodeGeometry(sampleGeometryWords, samplePixel);
        float geometryExponent = denoiserSpatialPdfDirectionExponent(centerGeometry.pdfDirection, sampleGeometry.pdfDirection);

        DenoiserMaxEntSignal sampleSignal = denoiserUnpackMaxEntSignalTrusted(sampleSignalWords);
        DenoiserSpatialBuresData sampleBures = denoiserSpatialMakeBuresDataFromStddev(denoiserSpatialTileStddev[sampleIndex]);
        float hitDistanceWeight;
        float weight = denoiserSpatialWeight(centerSignal, centerBures,
                sampleSignal, sampleBures, sampleGeometry, geometryExponent,
                DENOISER_SPATIAL_GRID_WEIGHT[i],
                lightSourceToleranceScale,
                DENOISER_SPATIAL_PHI_LUMINANCE,
                hitDistanceAlpha, centerVirtualPosition,
                centerVirtualNormal, virtualRejectionScale,
                hitDistanceWeight);
        denoiserSpatialAccumulate(accum, sampleSignal, weight, hitDistanceWeight);
    }

    denoiserSpatialStore(pixel, denoiserSpatialResolve(accum, DENOISER_SPATIAL_STEP));
}

#endif // MAXENT_SPATIAL_ATROUS_SMALL_GLSL
