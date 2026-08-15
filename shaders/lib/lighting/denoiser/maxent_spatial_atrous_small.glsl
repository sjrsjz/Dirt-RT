#ifndef MAXENT_SPATIAL_ATROUS_SMALL_GLSL
#define MAXENT_SPATIAL_ATROUS_SMALL_GLSL

// Compile-time policy interface required from the including pass:
//   ivec2 denoiserSpatialImageSize()
//   uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel)
//   uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel)
//   DenoiserSpatialGeometry denoiserSpatialDecodeGeometry(words, pixel)
//   bool denoiserSpatialGeometryCompatible(center, sample)
//   void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal)
//   void denoiserSpatialStoreInvalid(ivec2 pixel)
// Required macros:
//   DENOISER_SPATIAL_STEP, DENOISER_SPATIAL_PHI_LUMINANCE

#define DENOISER_SPATIAL_TILE_SIZE (16 + 2 * DENOISER_SPATIAL_STEP)
#define DENOISER_SPATIAL_TILE_AREA \
    (DENOISER_SPATIAL_TILE_SIZE * DENOISER_SPATIAL_TILE_SIZE)

shared uvec4 denoiserSpatialTileGeometry[DENOISER_SPATIAL_TILE_AREA];
shared uvec4 denoiserSpatialTileSignal[DENOISER_SPATIAL_TILE_AREA];
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

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    uvec2 localID = gl_LocalInvocationID.xy;
    ivec2 size = denoiserSpatialImageSize();
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u)
        - ivec2(DENOISER_SPATIAL_STEP);

    // Every invocation participates in the cooperative load and barrier.
    for (uint tileY = localID.y;
            tileY < uint(DENOISER_SPATIAL_TILE_SIZE); tileY += 16u) {
        for (uint tileX = localID.x;
                tileX < uint(DENOISER_SPATIAL_TILE_SIZE); tileX += 16u) {
            uint tileIndex = tileY
                * uint(DENOISER_SPATIAL_TILE_SIZE) + tileX;
            ivec2 sourcePixel = tileOrigin + ivec2(tileX, tileY);
            if (denoiserSpatialInBounds(sourcePixel, size)) {
                uvec4 geometryWords =
                    denoiserSpatialLoadGeometryWords(sourcePixel);
                uvec4 signalWords =
                    denoiserSpatialLoadSignalWords(sourcePixel);
                denoiserSpatialTileGeometry[tileIndex] = geometryWords;
                denoiserSpatialTileSignal[tileIndex] = signalWords;

                DenoiserSpatialGeometry geometry =
                    denoiserSpatialDecodeGeometry(geometryWords, sourcePixel);
                if (geometry.valid
                        && denoiserSpatialSignalWordsValid(signalWords)) {
                    DenoiserMaxEntSignal signal =
                        denoiserUnpackMaxEntSignal(signalWords);
                    vec2 stddev = denoiserSpatialMakeBuresData(
                        signal.maxEntY).stddev;
                    denoiserSpatialTileStddev[tileIndex] = packHalf2x16(
                        clamp(stddev, vec2(0.0),
                            vec2(DENOISER_SPATIAL_FP16_MAX)));
                } else {
                    denoiserSpatialTileStddev[tileIndex] = 0u;
                }
            } else {
                denoiserSpatialTileGeometry[tileIndex] = uvec4(0u);
                denoiserSpatialTileSignal[tileIndex] =
                    denoiserInvalidMaxEntSignalWords();
                denoiserSpatialTileStddev[tileIndex] = 0u;
            }
        }
    }

    barrier();
    if (!denoiserSpatialInBounds(pixel, size)) return;

    uint centerX = localID.x + uint(DENOISER_SPATIAL_STEP);
    uint centerY = localID.y + uint(DENOISER_SPATIAL_STEP);
    uint centerIndex = centerY * uint(DENOISER_SPATIAL_TILE_SIZE)
        + centerX;
    uvec4 centerGeometryWords =
        denoiserSpatialTileGeometry[centerIndex];
    uvec4 centerSignalWords = denoiserSpatialTileSignal[centerIndex];
    DenoiserSpatialGeometry centerGeometry =
        denoiserSpatialDecodeGeometry(centerGeometryWords, pixel);
    if (!centerGeometry.valid
            || !denoiserSpatialSignalWordsValid(centerSignalWords)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }

    DenoiserMaxEntSignal centerSignal =
        denoiserUnpackMaxEntSignal(centerSignalWords);
    DenoiserSpatialBuresData centerBures =
        denoiserSpatialMakeBuresDataFromStddev(centerSignal.maxEntY,
            unpackHalf2x16(denoiserSpatialTileStddev[centerIndex]));
    float lightSourceToleranceScale =
        denoiserSpatialLightSourceToleranceScale(centerGeometry.roughness);
    DenoiserSpatialAccumulator accum =
        denoiserSpatialBeginAccumulation(centerSignal);

    for (int i = 0; i < 8; ++i) {
        int sampleX = int(centerX) + DENOISER_SPATIAL_GRID_8[i].x
            * DENOISER_SPATIAL_STEP;
        int sampleY = int(centerY) + DENOISER_SPATIAL_GRID_8[i].y
            * DENOISER_SPATIAL_STEP;
        uint sampleIndex = uint(sampleY * DENOISER_SPATIAL_TILE_SIZE
            + sampleX);
        uvec4 sampleGeometryWords =
            denoiserSpatialTileGeometry[sampleIndex];
        uvec4 sampleSignalWords = denoiserSpatialTileSignal[sampleIndex];
        ivec2 samplePixel = pixel + DENOISER_SPATIAL_GRID_8[i]
            * DENOISER_SPATIAL_STEP;
        DenoiserSpatialGeometry sampleGeometry =
            denoiserSpatialDecodeGeometry(sampleGeometryWords, samplePixel);
        if (!sampleGeometry.valid
                || !denoiserSpatialSignalWordsValid(sampleSignalWords)
                || !denoiserSpatialGeometryCompatible(
                    centerGeometry, sampleGeometry))
            continue;

        DenoiserMaxEntSignal sampleSignal =
            denoiserUnpackMaxEntSignal(sampleSignalWords);
        DenoiserSpatialBuresData sampleBures =
            denoiserSpatialMakeBuresDataFromStddev(sampleSignal.maxEntY,
                unpackHalf2x16(denoiserSpatialTileStddev[sampleIndex]));
        float weight = denoiserSpatialWeight(centerSignal, centerBures,
            centerGeometry, sampleSignal, sampleBures, sampleGeometry,
            DENOISER_SPATIAL_GRID_WEIGHT[i],
            lightSourceToleranceScale,
            DENOISER_SPATIAL_PHI_LUMINANCE, float(size.y));
        if (weight <= 1e-6) continue;
        denoiserSpatialAccumulate(accum, sampleSignal, weight);
    }

    denoiserSpatialStore(pixel, denoiserSpatialResolve(accum,
        DENOISER_SPATIAL_STEP));
}

#endif // MAXENT_SPATIAL_ATROUS_SMALL_GLSL
