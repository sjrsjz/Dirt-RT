#version 430 core

// Purpose: choose reflection currentAlpha from denoised estimators and commit raw/filtered histories.
// Dispatch: 16x16.
// Reads: colortex5 final A-Trous signal, colortex6 Raw reflection, shared independent-current scratch,
//        staged raw reprojection, and filtered-history reprojection scratch.
// Writes: reflection temporal history, filtered history, and public reflection output.
// Persistent side effects: this is the only reflection history commit point.
// Invalid representation: invalid signal metadata clears every owned persistent record.
layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/post/denoiser/reflection/common.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/lighting/denoiser/temporal_response.glsl"

uniform usampler2D colortex5;
uniform usampler2D colortex6;

#include "/lib/lighting/denoiser/scratch_io.glsl"

uvec4 maxentTemporalRobustLoadSignalWords(ivec2 pixel) {
    return denoiserScratchLoadA(pixel);
}

ivec2 maxentTemporalRobustImageSize() { return textureSize(colortex5, 0); }
uvec4 maxentTemporalRobustLoadGeometryWords(ivec2 pixel) { return readPrimaryGeometryWords(uvec2(pixel)); }

#include "/lib/lighting/denoiser/robust_mean.glsl"

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

    ivec2 imageSize = textureSize(colortex5, 0);
    vec3 centerPrimaryRay = reconstructPrimaryRay(uvec2(centerPixel));
    float centerPlaneOffset = centerGeometry.distance
        * dot(centerGeometry.normal, centerPrimaryRay);
    float surfaceRejectionScale = denoiserSpatialDistanceRejectionScale(
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
            float surfaceExponent = denoiserSpatialAxialDistanceExponent(
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
        centerSignal.estimatorStdDev);
}

void main() {
    maxentTemporalRobustLoadSharedTile();
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    uvec4 currentSignalWords = texelFetch(colortex5, ivec2(pixel), 0);
    DenoiserMaxEntSignal currentSignal =
        denoiserUnpackMaxEntSignal(currentSignalWords);
    DenoiserMaxEntSignal independentCurrent =
        denoiserEmptyMaxEntSignal();
    MaxEntGeometry currentGeometry = maxentDecodeGeometry(
        maxentTemporalRobustTileGeometryWords(ivec2(0)), pixel);
    if (!currentGeometry.valid
            || !denoiserSpatialSignalWordsValid(currentSignalWords)) {
        debugWriteSpecularNoiseOnlyCurrentWeight(pixel, -1.0);
        debugWriteSpecularTemporalStateInvalid(pixel);
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        writeMaxEntSpecularDenoisedHistoryInvalid(pixel);
        writeReflMaxEnt(pixel, emptySpecularMaxEnt(), 0.0, 0.0);
        return;
    }

    MaxEntSpecularInput noisy = maxentUnpackSpecularInput(
        texelFetch(colortex6, ivec2(pixel), 0));
    MaxEntSpecularHistory reprojected =
        readMaxEntSpecularHistory(pixel);

    float noiseOnlyCurrentWeight = -1.0;
    SpecularMaxEnt historyDenoisedSignal;
    float historyEstimatorStdDev, reprojectionAlphaFloor, currentTrackingHitDistance;
    bool denoisedReprojectionValid = readMaxEntSpecularDenoisedReprojection(pixel, historyDenoisedSignal,
        historyEstimatorStdDev, reprojectionAlphaFloor, currentTrackingHitDistance);
    bool hasHistory = statisticsValidEffectiveSampleCount(reprojected.historyEffectiveSamples)
        && denoisedReprojectionValid;

    float currentAlpha = 1.0;
    float proposalAlpha = 1.0;
    bool independentCurrentValid = false;
    if (hasHistory) {
        proposalAlpha = clamp(
            float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
        currentAlpha = max(proposalAlpha, reprojectionAlphaFloor);
        independentCurrentValid = maxentSpecularRobustCurrentSignal(
            ivec2(0), independentCurrent);
        if (independentCurrentValid) {
            MaxEntTemporalRobustEstimate robustCurrent =
                maxentSpecularCurrentRobustMomentEstimate(
                    ivec2(pixel), independentCurrent, currentGeometry);
            float responseAlpha = maxentTemporalResponseAlpha(
                reprojectionAlphaFloor, robustCurrent.moment,
                robustCurrent.estimatorStdDev * robustCurrent.estimatorStdDev,
                historyDenoisedSignal.maxEntY, historyEstimatorStdDev,
                reprojected.historyEffectiveSamples,
                noiseOnlyCurrentWeight);
            currentAlpha = responseAlpha;
        }
    }
    debugWriteSpecularNoiseOnlyCurrentWeight(pixel, noiseOnlyCurrentWeight);

    MaxEntSpecularHistory committed;
    committed.surfacePosition = currentGeometry.position;
    committed.geometryNormal = currentGeometry.normal;
    committed.roughness = currentGeometry.roughness;
    committed.materialID = currentGeometry.materialID;
    committed.hitDistance = currentTrackingHitDistance;
    if (hasHistory) {
        committed.signal = maxentMixMaxEnt(reprojected.signal,
            noisy.signal, currentAlpha);
        committed.rootMeanY2 = sqrt(mix(
            reprojected.rootMeanY2 * reprojected.rootMeanY2,
            noisy.signal.maxEntY.w * noisy.signal.maxEntY.w,
            currentAlpha));
        committed.historyEffectiveSamples =
            statisticsKishUpdateEffectiveSampleCount(
                reprojected.historyEffectiveSamples, currentAlpha);
    } else {
        committed.signal = noisy.signal;
        committed.rootMeanY2 = abs(noisy.signal.maxEntY.w);
        committed.historyEffectiveSamples = 1.0;
    }
    writeMaxEntSpecularTemporalHistory(pixel, committed);
    debugWriteSpecularTemporalState(pixel, packSpecularMaxEnt(committed.signal),
        committed.hitDistance, 1.0 - currentAlpha);

    SpecularMaxEnt filtered;
    float resolvedEstimatorStdDev = sqrt(maxentTemporalCurrentEstimatorVariance(currentSignal.estimatorStdDev));
    if (hasHistory && independentCurrentValid) {
        float correctionCurrentWeight = clamp((currentAlpha - proposalAlpha)
            / max(1.0 - proposalAlpha, 1e-6), 0.0, 1.0);
        filtered.maxEntY = mix(currentSignal.maxEntY,
            independentCurrent.maxEntY, correctionCurrentWeight);
        filtered.CoCg = mix(currentSignal.CoCg,
            independentCurrent.CoCg, correctionCurrentWeight);
        resolvedEstimatorStdDev = maxentTemporalProposalCorrectedStandardDeviation(
            currentSignal.estimatorStdDev, independentCurrent.estimatorStdDev, correctionCurrentWeight);
    } else {
        filtered.maxEntY = currentSignal.maxEntY;
        filtered.CoCg = currentSignal.CoCg;
    }
    writeMaxEntSpecularDenoisedHistory(pixel, filtered, resolvedEstimatorStdDev);
    vec3 primaryRay = reconstructPrimaryRay(pixel);
    float virtualScale = denoiserSpatialSpecularVirtualScale(primaryRay,
        currentGeometry.normal, currentGeometry.roughness);
    float resolvedVirtualDistance = currentGeometry.distance
        + virtualScale * committed.hitDistance;
    writeReflMaxEnt(pixel, filtered, resolvedVirtualDistance, 1.0);
}
