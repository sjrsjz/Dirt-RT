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
    ivec2 size = denoiserSpatialImageSize();
    if (!denoiserSpatialInBounds(pixel, size)) return;

    uvec4 centerSignalWords = denoiserSpatialLoadSignalWords(pixel);
    uvec4 centerCurrentWords =
        denoiserSpatialLoadIndependentCurrentWords(pixel);
    if (!denoiserSpatialSignalWordsValid(centerSignalWords)
            || !denoiserSpatialSignalWordsValid(centerCurrentWords)) {
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
    float virtualDistanceAlpha = centerGeometry.ggxAlpha;
    vec3 centerVirtualPosition = denoiserSpatialVirtualWorldPosition(
            centerGeometry.primaryRay, centerSignal.virtualDistance);
    float virtualRejectionScale = denoiserSpatialVirtualRejectionScale(
            centerGeometry.ggxAlpha, centerSignal.virtualDistance);
    float lightDifferenceScale = DENOISER_SPATIAL_PHI_LUMINANCE;
    // Accumulate tangents in place so four decoded positions do not have to
    // remain live across the last neighbor fetch.
    vec3 virtualTangentX = -denoiserSpatialLoadVirtualPosition(pixel + ivec2(-1, 0), size, centerVirtualPosition);
    virtualTangentX += denoiserSpatialLoadVirtualPosition(pixel + ivec2(1, 0), size, centerVirtualPosition);
    vec3 virtualTangentY = -denoiserSpatialLoadVirtualPosition(pixel + ivec2(0, -1), size, centerVirtualPosition);
    virtualTangentY += denoiserSpatialLoadVirtualPosition(pixel + ivec2(0, 1), size, centerVirtualPosition);
    vec3 centerVirtualNormal = denoiserSpatialVirtualNormal(
            virtualTangentX, virtualTangentY, centerGeometry.primaryRay);
    DenoiserSpatialAccumulator accum = denoiserSpatialBeginAccumulation(centerSignal);
    DenoiserSpatialAccumulator currentAccum =
        denoiserSpatialBeginAccumulation(centerCurrent);

    for (int i = 0; i < 8; ++i) {
        ivec2 samplePixel = pixel + DENOISER_SPATIAL_GRID_8[i] * DENOISER_SPATIAL_STEP;
        if (!denoiserSpatialInBounds(samplePixel, size)) continue;

        uvec4 sampleGeometryWords = denoiserSpatialLoadGeometryWords(samplePixel);
        uvec4 sampleSignalWords = denoiserSpatialLoadSignalWords(samplePixel);
        uvec4 sampleCurrentWords =
            denoiserSpatialLoadIndependentCurrentWords(samplePixel);
        if (!denoiserSpatialGeometryWordsValid(sampleGeometryWords)
                || !denoiserSpatialSignalWordsValid(sampleSignalWords)
                || !denoiserSpatialSignalWordsValid(sampleCurrentWords))
            continue;

        vec3 samplePrimaryRay;
        float sampleSurfaceDistance;
        float sampleEffectiveSamples;
        denoiserSpatialDecodeSampleGeometry(sampleGeometryWords, samplePixel,
            samplePrimaryRay, sampleSurfaceDistance,
            sampleEffectiveSamples);
        if (!statisticsValidEffectiveSampleCount(sampleEffectiveSamples))
            continue;
        vec3 sampleSurfacePosition = samplePrimaryRay
                * sampleSurfaceDistance;
        float surfaceGeometryExponent = surfaceRejectionScale
                * abs(dot(centerGeometry.geometryNormal, sampleSurfacePosition)
                        - centerGeometry.surfacePlaneOffset);

        DenoiserMaxEntSignal sampleSignal = denoiserUnpackMaxEntSignalTrusted(sampleSignalWords);
        DenoiserMaxEntSignal sampleCurrent =
            denoiserUnpackMaxEntSignalTrusted(sampleCurrentWords);
        float virtualDistanceWeight;
        float weight = denoiserSpatialWeight(centerSignal,
                sampleSignal, samplePrimaryRay,
                surfaceGeometryExponent,
                DENOISER_SPATIAL_GRID_WEIGHT[i],
                lightDifferenceScale,
                virtualDistanceAlpha, centerVirtualPosition,
                centerVirtualNormal, virtualRejectionScale,
                virtualDistanceWeight);
        denoiserSpatialAccumulate(accum, sampleSignal, weight,
            virtualDistanceWeight);
        denoiserSpatialAccumulate(currentAccum, sampleCurrent, weight,
            virtualDistanceWeight);
    }

    denoiserSpatialStore(pixel, denoiserSpatialResolve(accum, DENOISER_SPATIAL_STEP), denoiserSpatialResolve(currentAccum, DENOISER_SPATIAL_STEP));
}

#endif // MAXENT_DENOISER_ATROUS_SMALL_GLSL
