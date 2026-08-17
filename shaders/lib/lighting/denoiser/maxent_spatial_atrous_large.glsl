#ifndef MAXENT_SPATIAL_ATROUS_LARGE_GLSL
#define MAXENT_SPATIAL_ATROUS_LARGE_GLSL

// Same policy interface as maxent_spatial_atrous_small.glsl, except stores are
// owned by the including pass. This lets the shared Poisson implementation be
// used by both fragment and compute entry points.

#define DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE 8
#define DENOISER_SPATIAL_LARGE_TILE_SIZE (DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE + 2)
#define DENOISER_SPATIAL_LARGE_TILE_AREA (DENOISER_SPATIAL_LARGE_TILE_SIZE * DENOISER_SPATIAL_LARGE_TILE_SIZE)

// xyz is the decoded camera-relative virtual position; w is validity.
shared vec4 denoiserSpatialLargeTileVirtualPosition[DENOISER_SPATIAL_LARGE_TILE_AREA];

const vec4 DENOISER_SPATIAL_POISSON_8[8] = vec4[](
        vec4(-0.4706069, -0.4427112, 0.6461146, 0.81170),
        vec4(-0.9057375, 0.3003471, 0.9542373, 0.63422),
        vec4(-0.3487388, 0.4037880, 0.5335386, 0.86734),
        vec4(0.1023042, 0.6439373, 0.6520134, 0.80847),
        vec4(0.5699277, 0.3513750, 0.6695386, 0.79925),
        vec4(0.2939128, -0.1131226, 0.3149309, 0.95161),
        vec4(0.7836658, -0.4208784, 0.8895339, 0.67328),
        vec4(0.1564120, -0.8198990, 0.8346850, 0.70589));

bool denoiserSpatialLargeInBounds(ivec2 pixel, ivec2 size) {
    return all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, size));
}

vec4 denoiserSpatialLargePackVirtualPosition(uvec4 geometryWords, uvec4 signalWords) {
    vec3 position;
    if (!denoiserSpatialTryVirtualWorldPositionFromWords(geometryWords, signalWords, position))
        return vec4(0.0);
    return vec4(position, 1.0);
}

vec3 denoiserSpatialLargeReadVirtualPosition(uint tileIndex, vec3 fallback) {
    vec4 packedPosition = denoiserSpatialLargeTileVirtualPosition[tileIndex];
    return packedPosition.w > 0.0 ? packedPosition.xyz : fallback;
}

bool denoiserSpatialFilterLarge(ivec2 pixel, out DenoiserMaxEntSignal outputSignal) {
    ivec2 size = denoiserSpatialImageSize();
    uvec2 localID = gl_LocalInvocationID.xy;
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * uint(DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE)) - ivec2(1);

    uint centerIndex = (localID.y + 1u) * uint(DENOISER_SPATIAL_LARGE_TILE_SIZE) + localID.x + 1u;
    uvec4 centerGeometryWords = uvec4(floatBitsToUint(-1.0), 0u, 0u, 0u);
    uvec4 centerSignalWords = denoiserInvalidMaxEntSignalWords();
    if (denoiserSpatialLargeInBounds(pixel, size)) {
        centerGeometryWords = denoiserSpatialLoadGeometryWords(pixel);
        centerSignalWords = denoiserSpatialLoadSignalWords(pixel);
    }
    denoiserSpatialLargeTileVirtualPosition[centerIndex] = denoiserSpatialLargePackVirtualPosition(centerGeometryWords, centerSignalWords);

    // The 8x8 interior is written exactly once by its owning invocation.
    // Cooperatively fetch only the 36 one-pixel halo entries.
    uint workgroupArea = uint(DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE * DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE);
    for (uint tileIndex = gl_LocalInvocationIndex;
        tileIndex < uint(DENOISER_SPATIAL_LARGE_TILE_AREA);
        tileIndex += workgroupArea) {
        uint tileX = tileIndex % uint(DENOISER_SPATIAL_LARGE_TILE_SIZE);
        uint tileY = tileIndex / uint(DENOISER_SPATIAL_LARGE_TILE_SIZE);
        bool interior = tileX >= 1u
                && tileX <= uint(DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE)
                && tileY >= 1u
                && tileY <= uint(DENOISER_SPATIAL_LARGE_WORKGROUP_SIZE);
        if (interior) continue;

        ivec2 sourcePixel = tileOrigin + ivec2(tileX, tileY);
        vec4 packedPosition = vec4(0.0);
        if (denoiserSpatialLargeInBounds(sourcePixel, size)) {
            packedPosition = denoiserSpatialLargePackVirtualPosition(
                    denoiserSpatialLoadGeometryWords(sourcePixel),
                    denoiserSpatialLoadSignalWords(sourcePixel));
        }
        denoiserSpatialLargeTileVirtualPosition[tileIndex] = packedPosition;
    }

    barrier();
    outputSignal = denoiserEmptyMaxEntSignal();
    if (!denoiserSpatialLargeInBounds(pixel, size)) return false;

    if (!denoiserSpatialGeometryWordsValid(centerGeometryWords)
            || !denoiserSpatialSignalWordsValid(centerSignalWords))
        return false;
    DenoiserSpatialGeometry centerGeometry = denoiserSpatialDecodeGeometry(centerGeometryWords, pixel);

    DenoiserMaxEntSignal centerSignal = denoiserUnpackMaxEntSignalTrusted(centerSignalWords);
    DenoiserSpatialBuresData centerBures = denoiserSpatialMakeBuresData(centerSignal.maxEntY);
    float lightSourceToleranceScale = denoiserSpatialLightSourceToleranceScale(centerGeometry.roughness);

    float hitDistanceAlpha = centerGeometry.ggxAlpha;
    vec3 centerVirtualPosition = denoiserSpatialVirtualWorldPosition(centerGeometry, centerSignal.hitDistance);
    float virtualRejectionScale = denoiserSpatialVirtualRejectionScale(centerGeometry, centerSignal.hitDistance);
    uint rowStride = uint(DENOISER_SPATIAL_LARGE_TILE_SIZE);

    vec3 virtualTangentX = -denoiserSpatialLargeReadVirtualPosition(centerIndex - 1u, centerVirtualPosition);
    virtualTangentX += denoiserSpatialLargeReadVirtualPosition(centerIndex + 1u, centerVirtualPosition);
    vec3 virtualTangentY = -denoiserSpatialLargeReadVirtualPosition(centerIndex - rowStride, centerVirtualPosition);
    virtualTangentY += denoiserSpatialLargeReadVirtualPosition(centerIndex + rowStride, centerVirtualPosition);
    vec3 centerVirtualNormal = denoiserSpatialVirtualNormal(virtualTangentX, virtualTangentY, centerGeometry.pdfDirection);

    DenoiserSpatialAccumulator accum = denoiserSpatialBeginAccumulation(centerSignal);

    // Independent per-pixel rotation avoids exposing the 8x8 workgroup grid
    // through correlated Poisson directions. Use a scalar integer hash here:
    // the generic vec2 RNG needs extra VGPRs when its input is nonuniform.
    uint rotationSeed = uint(pixel.x) * 0x9e3779b9u
            ^ uint(pixel.y) * 0x85ebca6bu
            ^ uint(DENOISER_SPATIAL_STEP) * 0xc2b2ae35u;
    float rotationAngle = 2.0 * PI * float(whash(rotationSeed) >> 8u) * (1.0 / 16777216.0);
    float cs = cos(rotationAngle);
    float sn = sin(rotationAngle);
    mat2 rotation = mat2(cs, -sn, sn, cs) * (float(DENOISER_SPATIAL_STEP) * 1.75);

    for (int i = 0; i < 8; ++i) {
        ivec2 samplePixel = pixel + ivec2(round(rotation * DENOISER_SPATIAL_POISSON_8[i].xy));
        if (!denoiserSpatialLargeInBounds(samplePixel, size)) continue;

        uvec4 sampleGeometryWords = denoiserSpatialLoadGeometryWords(samplePixel);
        uvec4 sampleSignalWords = denoiserSpatialLoadSignalWords(samplePixel);
        if (!denoiserSpatialGeometryWordsValid(sampleGeometryWords)
                || !denoiserSpatialSignalWordsValid(sampleSignalWords))
            continue;
        DenoiserSpatialGeometry sampleGeometry = denoiserSpatialDecodeGeometry(sampleGeometryWords, samplePixel);
        float geometryExponent = denoiserSpatialPdfDirectionExponent(centerGeometry.pdfDirection, sampleGeometry.pdfDirection);

        DenoiserMaxEntSignal sampleSignal = denoiserUnpackMaxEntSignalTrusted(sampleSignalWords);
        DenoiserSpatialBuresData sampleBures = denoiserSpatialMakeBuresData(sampleSignal.maxEntY);

        float hitDistanceWeight;
        float weight = denoiserSpatialWeight(centerSignal, centerBures,
                sampleSignal, sampleBures, sampleGeometry, geometryExponent,
                DENOISER_SPATIAL_POISSON_8[i].w,
                lightSourceToleranceScale,
                DENOISER_SPATIAL_PHI_LUMINANCE,
                hitDistanceAlpha, centerVirtualPosition,
                centerVirtualNormal, virtualRejectionScale,
                hitDistanceWeight);
        denoiserSpatialAccumulate(accum, sampleSignal, weight, hitDistanceWeight);
    }

    outputSignal = denoiserSpatialResolve(accum, DENOISER_SPATIAL_STEP);
    return true;
}

#endif // MAXENT_SPATIAL_ATROUS_LARGE_GLSL
