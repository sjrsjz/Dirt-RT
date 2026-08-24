#version 430 core

// Final temporal resolve. The A-trous chain supplies the stable response proxy
// and its propagated variance; this pass obtains alpha from the completed 5x5
// robust estimate and only then commits one Raw RT observation to history.
layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"
#include "/lib/lighting/denoiser/maxent_moment_statistics.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex6;

uvec4 maxentTemporalRobustLoadSignalWords(ivec2 pixel) {
    return readDiffuseIndependentCurrentA(uvec2(pixel));
}

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

bool maxentSpecularRobustCurrentSignal(ivec2 offset,
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
    if (!maxentSpecularRobustCurrentSignal(offset, signal))
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
            if (!maxentSpecularRobustCurrentSignal(offset, sampleSignal))
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

    uvec4 currentSignalWords = texelFetch(colortex4, ivec2(pixel), 0);
    DenoiserMaxEntSignal currentSignal =
        denoiserUnpackMaxEntSignal(currentSignalWords);
    DenoiserMaxEntSignal independentCurrent =
        denoiserEmptyMaxEntSignal();
    MaxEntGeometry currentGeometry = maxentDecodeGeometry(
        maxentTemporalRobustTileGeometryWords(ivec2(0)), pixel);
    if (!currentGeometry.valid
            || !denoiserSpatialSignalWordsValid(currentSignalWords)) {
        debugWriteSpecularVarianceOptimalAlpha(pixel, -1.0);
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        writeReflMaxEnt(pixel, emptySpecularMaxEnt(), 0.0, 0.0);
        return;
    }

    MaxEntSpecularInput noisy = maxentUnpackSpecularInput(
        texelFetch(colortex6, ivec2(pixel), 0));
    MaxEntSpecularHistory reprojected =
        readMaxEntSpecularHistory(pixel);

    float varianceOptimalAlpha = -1.0;
    SpecularMaxEnt historyDenoisedSignal;
    float historyStddev, reprojectionAlphaFloor;
    float historyHitDistance;
    bool hasHistory = statisticsValidEffectiveSampleCount(
            reprojected.historyLength)
        && readMaxEntSpecularDenoisedReprojection(pixel,
            historyDenoisedSignal, historyStddev,
            reprojectionAlphaFloor, historyHitDistance);

    float actualAlpha = 1.0;
    float proposalAlpha = 1.0;
    bool independentCurrentValid = false;
    if (hasHistory) {
        proposalAlpha = clamp(
            float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
        actualAlpha = proposalAlpha;
        independentCurrentValid = maxentSpecularRobustCurrentSignal(
            ivec2(0), independentCurrent);
        if (independentCurrentValid) {
            MaxEntTemporalRobustEstimate robustCurrent =
                maxentSpecularCurrentRobustMomentEstimate(
                    ivec2(pixel), independentCurrent, currentGeometry);
            float windowAlphaFloor =
                statisticsKishFiniteWindowCurrentWeightFloor(
                    reprojected.historyLength,
                    float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY));
            float alphaFloor = max(reprojectionAlphaFloor,
                windowAlphaFloor);
            float unusedAdaptiveAlpha = maxentTemporalMinimumMseAlpha(
                alphaFloor, robustCurrent.moment,
                robustCurrent.standardDeviation
                    * robustCurrent.standardDeviation,
                historyDenoisedSignal.maxEntY, historyStddev,
                reprojected.historyLength,
                varianceOptimalAlpha);
        }
    }
    debugWriteSpecularVarianceOptimalAlpha(pixel, varianceOptimalAlpha);

    MaxEntSpecularHistory committed;
    committed.surfacePosition = currentGeometry.position;
    committed.geometryNormal = currentGeometry.normal;
    committed.roughness = currentGeometry.roughness;
    committed.materialID = currentGeometry.materialID;
    if (hasHistory) {
        committed.signal = maxentMixMaxEnt(reprojected.signal,
            noisy.signal, actualAlpha);
        committed.rootMeanY2 = sqrt(max(mix(
            reprojected.rootMeanY2 * reprojected.rootMeanY2,
            noisy.signal.maxEntY.w * noisy.signal.maxEntY.w,
            actualAlpha), 0.0));
        committed.hitDistance = mix(reprojected.hitDistance,
            noisy.hitDistance, actualAlpha);
        committed.historyLength =
            statisticsKishUpdateEffectiveSampleCount(
                reprojected.historyLength, actualAlpha);
    } else {
        committed.signal = noisy.signal;
        committed.rootMeanY2 = max(noisy.signal.maxEntY.w, 0.0);
        committed.hitDistance = noisy.hitDistance;
        committed.historyLength = 1.0;
    }
    writeMaxEntSpecularTemporalHistory(pixel, committed);
    debugWriteSpecularTemporal(pixel, packSpecularMaxEnt(committed.signal),
        committed.hitDistance, 1.0 - actualAlpha);

    SpecularMaxEnt filtered;
    float resolvedStandardDeviation = sqrt(
        maxentTemporalCurrentEstimatorVariance(
            currentSignal.standardDeviation));
    if (hasHistory && independentCurrentValid) {
        float correctionAlpha = clamp((actualAlpha - proposalAlpha)
            / max(1.0 - proposalAlpha, 1e-6), 0.0, 1.0);
        filtered.maxEntY = mix(currentSignal.maxEntY,
            independentCurrent.maxEntY, correctionAlpha);
        filtered.CoCg = mix(currentSignal.CoCg,
            independentCurrent.CoCg, correctionAlpha);
        resolvedStandardDeviation =
            maxentTemporalProposalCorrectedStandardDeviation(
                currentSignal.standardDeviation,
                independentCurrent.standardDeviation,
                correctionAlpha);
    } else if (hasHistory) {
        filtered.maxEntY = currentSignal.maxEntY;
        filtered.CoCg = currentSignal.CoCg;
    } else {
        filtered.maxEntY = currentSignal.maxEntY;
        filtered.CoCg = currentSignal.CoCg;
    }
    writeMaxEntSpecularDenoisedHistory(pixel, filtered,
        resolvedStandardDeviation);
    vec3 primaryRay = reconstructPrimaryRay(pixel);
    float virtualScale = denoiserSpatialSpecularVirtualScale(primaryRay,
        currentGeometry.normal, currentGeometry.roughness);
    float resolvedVirtualDistance = currentGeometry.distance
        + virtualScale * committed.hitDistance;
    writeReflMaxEnt(pixel, filtered, resolvedVirtualDistance, 1.0);
}
