#version 430 core

// The final A-trous dispatch cannot read neighboring values that it is still
// producing. This resolve runs after that dispatch, estimates a robust current
// moment from the completed 5x5 final-output neighborhood, and only then
// replaces SPEC_N_LIGHT's transient reprojection data with production output.
layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"
#include "/lib/lighting/denoiser/maxent_moment_statistics.glsl"

uniform usampler2D colortex4;
#include "/lib/lighting/denoiser/maxent_temporal_robust_tile.glsl"

vec3 maxentSpecularNeighborhoodVirtualPosition(
        vec3 primaryRay, float virtualDistance) {
    return primaryRay * virtualDistance;
}

float maxentSpecularNeighborhoodVirtualRejectionScale(
        float ggxAlpha, float virtualDistance) {
    if (virtualDistance <= 0.0) return 0.0;
    return (1.0 - ggxAlpha) / max(
        float(MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE) * virtualDistance,
        1e-5);
}

vec3 maxentSpecularNeighborhoodVirtualNormal(
        vec3 tangentX, vec3 tangentY, vec3 fallback) {
    return denoiserSpatialSafeDirection(cross(tangentX, tangentY), fallback);
}

float maxentSpecularNeighborhoodVirtualPlaneExponent(
        vec3 centerPosition, vec3 centerNormal, vec3 samplePrimaryRay,
        float sampleVirtualDistance, float rejectionScale) {
    vec3 samplePosition = maxentSpecularNeighborhoodVirtualPosition(
        samplePrimaryRay, sampleVirtualDistance);
    return rejectionScale
        * abs(dot(centerNormal, samplePosition - centerPosition));
}

bool maxentSpecularFinalSignal(ivec2 offset,
        out DenoiserMaxEntSignal signal) {
    signal = denoiserEmptyMaxEntSignal();
    uvec4 words = maxentTemporalRobustTileSignalWords(offset);
    if (!denoiserSpatialSignalWordsValid(words)) return false;
    signal = denoiserUnpackMaxEntSignal(words);
    return true;
}

vec3 maxentSpecularFinalVirtualPosition(ivec2 centerPixel, ivec2 offset,
        vec3 fallback) {
    DenoiserMaxEntSignal signal;
    if (!maxentSpecularFinalSignal(offset, signal))
        return fallback;
    return maxentSpecularNeighborhoodVirtualPosition(
        reconstructPrimaryRay(uvec2(centerPixel + offset)),
        signal.virtualDistance);
}

MaxEntTemporalRobustEstimate maxentSpecularCurrentRobustMomentEstimate(
        ivec2 centerPixel,
        DenoiserMaxEntSignal centerSignal, MaxEntGeometry centerGeometry) {
    uint acceptedMask = maxentTemporalRobustNeighborhoodBit(ivec2(0));
    int sampleCount = 1;
    vec4 momentSum = centerSignal.maxEntY;

    ivec2 imageSize = textureSize(colortex4, 0);
    vec3 centerPrimaryRay = reconstructPrimaryRay(uvec2(centerPixel));
    float centerPlaneOffset = centerGeometry.distance
        * dot(centerGeometry.normal, centerPrimaryRay);
    float surfaceRejectionScale = denoiserSpatialSurfaceRejectionScale(
        centerGeometry.distance, float(imageSize.y));

    vec3 centerVirtualPosition = maxentSpecularNeighborhoodVirtualPosition(
        centerPrimaryRay, centerSignal.virtualDistance);
    vec3 virtualTangentX = -maxentSpecularFinalVirtualPosition(
        centerPixel, ivec2(-1, 0), centerVirtualPosition);
    virtualTangentX += maxentSpecularFinalVirtualPosition(
        centerPixel, ivec2(1, 0), centerVirtualPosition);
    vec3 virtualTangentY = -maxentSpecularFinalVirtualPosition(
        centerPixel, ivec2(0, -1), centerVirtualPosition);
    virtualTangentY += maxentSpecularFinalVirtualPosition(
        centerPixel, ivec2(0, 1), centerVirtualPosition);
    vec3 centerVirtualNormal = maxentSpecularNeighborhoodVirtualNormal(
        virtualTangentX, virtualTangentY, centerPrimaryRay);
    float ggxAlpha = centerGeometry.roughness * centerGeometry.roughness;
    float virtualRejectionScale =
        maxentSpecularNeighborhoodVirtualRejectionScale(
        ggxAlpha, centerSignal.virtualDistance);

    for (int offsetY = -MAXENT_TEMPORAL_ROBUST_RADIUS;
            offsetY <= MAXENT_TEMPORAL_ROBUST_RADIUS; ++offsetY) {
        for (int offsetX = -MAXENT_TEMPORAL_ROBUST_RADIUS;
                offsetX <= MAXENT_TEMPORAL_ROBUST_RADIUS; ++offsetX) {
            if (offsetX == 0 && offsetY == 0) continue;
            ivec2 offset = ivec2(offsetX, offsetY);
            ivec2 samplePixel = centerPixel + offset;
            DenoiserMaxEntSignal sampleSignal;
            if (!maxentSpecularFinalSignal(offset, sampleSignal))
                continue;

            MaxEntGeometry sampleGeometry = maxentDecodeGeometry(
                maxentTemporalRobustTileGeometryWords(offset),
                uvec2(samplePixel));
            if (!sampleGeometry.valid
                    || sampleGeometry.materialID != centerGeometry.materialID
                    || dot(centerGeometry.normal, sampleGeometry.normal) <= 0.0)
                continue;

            vec3 samplePrimaryRay = reconstructPrimaryRay(
                uvec2(samplePixel));
            float surfaceExponent = denoiserSpatialSurfacePlaneDepthExponent(
                centerPlaneOffset, centerGeometry.normal, samplePrimaryRay,
                sampleGeometry.distance, surfaceRejectionScale);
            if (surfaceExponent
                    > MAXENT_TEMPORAL_ROBUST_MAX_PLANE_EXPONENT)
                continue;
            float virtualExponent =
                maxentSpecularNeighborhoodVirtualPlaneExponent(
                centerVirtualPosition, centerVirtualNormal, samplePrimaryRay,
                sampleSignal.virtualDistance, virtualRejectionScale);
            if (virtualExponent
                    > MAXENT_TEMPORAL_ROBUST_MAX_PLANE_EXPONENT)
                continue;

            acceptedMask |= maxentTemporalRobustNeighborhoodBit(offset);
            momentSum += sampleSignal.maxEntY;
            ++sampleCount;
        }
    }

    return maxentTemporalGaussianReweightedTileEstimate(
        acceptedMask, sampleCount, momentSum, centerSignal.maxEntY,
        centerSignal.standardDeviation);
}

void main() {
    maxentTemporalRobustLoadSharedTile();
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    DenoiserMaxEntSignal currentSignal;
    MaxEntGeometry currentGeometry = maxentDecodeGeometry(
        maxentTemporalRobustTileGeometryWords(ivec2(0)), pixel);
    if (!currentGeometry.valid
            || !maxentSpecularFinalSignal(ivec2(0), currentSignal)) {
        debugWriteSpecularDenoisedDifference(pixel, -1.0);
        writeReflMaxEnt(pixel, emptySpecularMaxEnt(), 0.0, 0.0);
        return;
    }

    float normalizedDistance = -1.0;
    uvec4 historyWords = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)];
    vec2 momentHistory = unpackHalf2x16(historyWords.w);
    if (momentHistory.y >= 1.0 && !any(isnan(momentHistory))
            && !any(isinf(momentHistory))) {
        vec4 historyMaxEntY;
        float historyStddev, historySamples;
        float validWeight, temporalCurrentWeight;
        if (readMaxEntSpecularDenoisedReprojection(pixel, historyMaxEntY,
                historyStddev, historySamples, validWeight,
                temporalCurrentWeight)) {
            MaxEntTemporalRobustEstimate robustCurrent =
                maxentSpecularCurrentRobustMomentEstimate(
                ivec2(pixel), currentSignal, currentGeometry);
            momentHistory.y = maxentClampHistoryWeightByMomentDifference(
                momentHistory.y,
                robustCurrent.moment, robustCurrent.standardDeviation,
                historyMaxEntY,
                historyStddev, historySamples, validWeight,
                temporalCurrentWeight,
                float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY),
                MAXENT_SPECULAR_TEMPORAL_DIFFERENCE_TOLERANCE,
                normalizedDistance);
            historyWords.w = pack2HalfClampedU(
                momentHistory.x, momentHistory.y);
            reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = historyWords;
        }
    }
    debugWriteSpecularDenoisedDifference(pixel, normalizedDistance);

    SpecularMaxEnt specular;
    specular.maxEntY = currentSignal.maxEntY;
    specular.CoCg = currentSignal.CoCg;
    writeReflMaxEnt(pixel, specular, currentSignal.virtualDistance, 1.0);
}
