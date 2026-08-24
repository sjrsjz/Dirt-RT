#version 430 core
// ===========================================================================
// MaxEnt diffuse final temporal resolve and path-guide SSBO publication.
// ===========================================================================
layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"
#include "/lib/lighting/denoiser/maxent_moment_statistics.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex5;
uniform usampler2D colortex6;

uvec4 maxentTemporalRobustLoadSignalWords(ivec2 pixel) {
    return readDiffuseIndependentCurrentA(uvec2(pixel));
}

#include "/lib/lighting/denoiser/maxent_temporal_robust_tile.glsl"

MaxEntTemporalRobustEstimate maxentDiffuseCurrentRobustMomentEstimate(
        ivec2 centerPixel, vec4 centerMoment,
        float centerStandardDeviation) {
    uint acceptedMask = maxentTemporalRobustNeighborhoodBit(ivec2(0));
    int sampleCount = 1;
    vec4 momentSum = centerMoment;

    ivec2 imageSize = textureSize(colortex4, 0);
    uvec4 centerGeometryWords =
        maxentTemporalRobustTileGeometryWords(ivec2(0));
    float centerSurfaceDistance = uintBitsToFloat(centerGeometryWords.w);
    if (!(centerSurfaceDistance >= 0.0) || isnan(centerSurfaceDistance)
            || isinf(centerSurfaceDistance)) {
        MaxEntTemporalRobustEstimate centerEstimate;
        centerEstimate.moment = centerMoment;
        centerEstimate.standardDeviation = centerStandardDeviation;
        return centerEstimate;
    }

    uint centerMaterial = centerGeometryWords.y >> 16u;
    vec3 centerSurfaceNormal = decodeNormalU(centerGeometryWords.x);
    vec3 centerPrimaryRay = reconstructPrimaryRay(uvec2(centerPixel));
    float centerPlaneOffset = centerSurfaceDistance
        * dot(centerSurfaceNormal, centerPrimaryRay);
    float surfaceRejectionScale = denoiserSpatialSurfaceRejectionScale(
        centerSurfaceDistance, float(imageSize.y));

    for (int offsetY = -MAXENT_TEMPORAL_ROBUST_RADIUS;
            offsetY <= MAXENT_TEMPORAL_ROBUST_RADIUS; ++offsetY) {
        for (int offsetX = -MAXENT_TEMPORAL_ROBUST_RADIUS;
                offsetX <= MAXENT_TEMPORAL_ROBUST_RADIUS; ++offsetX) {
            if (offsetX == 0 && offsetY == 0) continue;
            ivec2 offset = ivec2(offsetX, offsetY);
            ivec2 samplePixel = centerPixel + offset;
            if (any(lessThan(samplePixel, ivec2(0)))
                    || any(greaterThanEqual(samplePixel, imageSize)))
                continue;

            uvec4 sampleGeometryWords =
                maxentTemporalRobustTileGeometryWords(offset);
            float sampleSurfaceDistance = uintBitsToFloat(
                sampleGeometryWords.w);
            if (!(sampleSurfaceDistance >= 0.0)
                    || isnan(sampleSurfaceDistance)
                    || isinf(sampleSurfaceDistance)
                    || (sampleGeometryWords.y >> 16u) != centerMaterial)
                continue;

            vec3 sampleSurfaceNormal = decodeNormalU(sampleGeometryWords.x);
            if (dot(centerSurfaceNormal, sampleSurfaceNormal) <= 0.0)
                continue;
            vec3 samplePrimaryRay = reconstructPrimaryRay(uvec2(samplePixel));
            float planeExponent = denoiserSpatialSurfacePlaneDepthExponent(
                centerPlaneOffset, centerSurfaceNormal, samplePrimaryRay,
                sampleSurfaceDistance, surfaceRejectionScale);
            if (planeExponent
                    > MAXENT_TEMPORAL_ROBUST_MAX_PLANE_EXPONENT)
                continue;

            uvec4 sampleWords = maxentTemporalRobustTileSignalWords(offset);
            if (!denoiserSpatialSignalWordsValid(sampleWords)) continue;
            vec4 sampleMoment = maxentTemporalRobustTileMoment(offset);
            if (any(isnan(sampleMoment)) || any(isinf(sampleMoment))) continue;
            acceptedMask |= maxentTemporalRobustNeighborhoodBit(offset);
            momentSum += sampleMoment;
            ++sampleCount;
        }
    }

    return maxentTemporalGaussianReweightedTileEstimate(
        acceptedMask, sampleCount, momentSum, centerMoment,
        centerStandardDeviation);
}

void main() {
    maxentTemporalRobustLoadSharedTile();
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(gid, uvec2(resolution_global)))) return;
    ivec2 pix = ivec2(gid);
    uvec2 gxy = uvec2(pix);

    // Temporal used this parity as its reprojected-denoised scratch. Consume it
    // before replacing it with the exact current spatial result.
    uvec4 packedLight = texelFetch(colortex4, pix, 0);
    if (unpackHalf2x16(packedLight.w).x < 0.0) {
        writeDiffuseDenoisedCurrentRaw(gxy, packedLight);
        // No surface: invalidate every temporal consumer with four raw stores
        // instead of decoding two light textures and previous histories.
        diffuseBuffer.data[addr(DIF_N_HIST, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_SWAP, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = uvec4(0u);
        writeDiffuseHistGeoInvalid(gxy);
        debugWriteDiffuseVarianceOptimalAlpha(gxy, -1.0);
        return;
    }

    DenoiserMaxEntSignal filteredSignal =
        denoiserUnpackMaxEntSignal(packedLight);
    MaxEntEncoding currentRaw;
    float currentRootMeanY2;
    readDiffuseLightRT(gxy, currentRaw, currentRootMeanY2);

    MaxEntEncoding reprojectedRaw;
    float historySamples, historyRootMeanY2;
    unpackDiffuseTemporalState(texelFetch(colortex6, pix, 0),
        reprojectedRaw, historySamples, historyRootMeanY2);

    vec4 historyDenoisedMoment;
    vec2 historyDenoisedCoCg;
    float historyDenoisedStddev;
    float validWeight;
    bool hasHistory = readDiffuseDenoisedReprojection(gxy,
        historyDenoisedMoment, historyDenoisedCoCg,
        historyDenoisedStddev, validWeight)
        && statisticsValidEffectiveSampleCount(historySamples)
        && !any(isnan(reprojectedRaw.maxEntY))
        && !any(isinf(reprojectedRaw.maxEntY))
        && !any(isnan(reprojectedRaw.CoCg))
        && !any(isinf(reprojectedRaw.CoCg));

    float actualAlpha = 1.0;
    float proposalAlpha = 1.0;
    float varianceOptimalAlpha = -1.0;
    DenoiserMaxEntSignal independentCurrent =
        denoiserEmptyMaxEntSignal();
    bool independentCurrentValid = false;
    if (hasHistory) {
        proposalAlpha = clamp(
            float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
        actualAlpha = proposalAlpha;
        uvec4 independentCurrentWords =
            maxentTemporalRobustTileSignalWords(ivec2(0));
        independentCurrentValid =
            denoiserSpatialSignalWordsValid(independentCurrentWords);
        if (independentCurrentValid) {
            independentCurrent =
                denoiserUnpackMaxEntSignal(independentCurrentWords);
            MaxEntTemporalRobustEstimate robustCurrent =
                maxentDiffuseCurrentRobustMomentEstimate(pix,
                    independentCurrent.maxEntY,
                    independentCurrent.standardDeviation);
            float reprojectionAlphaFloor =
                1.0 - clamp(validWeight, 0.0, 1.0);
            float windowAlphaFloor =
                statisticsKishFiniteWindowCurrentWeightFloor(
                    historySamples,
                    float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY));
            float alphaFloor = max(reprojectionAlphaFloor,
                windowAlphaFloor);
            float unusedAdaptiveAlpha = maxentTemporalMinimumMseAlpha(
                alphaFloor, robustCurrent.moment,
                robustCurrent.standardDeviation
                    * robustCurrent.standardDeviation,
                historyDenoisedMoment, historyDenoisedStddev,
                historySamples,
                varianceOptimalAlpha);
            actualAlpha = unusedAdaptiveAlpha;
        }
    }
    debugWriteDiffuseVarianceOptimalAlpha(gxy, varianceOptimalAlpha);

    MaxEntEncoding committedRaw;
    float committedRootMeanY2;
    float committedSamples;
    if (hasHistory) {
        committedRaw = mix_maxent(reprojectedRaw, currentRaw, actualAlpha);
        committedRootMeanY2 = sqrt(max(mix(
            historyRootMeanY2 * historyRootMeanY2,
            currentRootMeanY2 * currentRootMeanY2, actualAlpha), 0.0));
        committedSamples = statisticsKishUpdateEffectiveSampleCount(
            historySamples, actualAlpha);
    } else {
        committedRaw = currentRaw;
        committedRootMeanY2 = currentRootMeanY2;
        committedSamples = 1.0;
    }

    DenoiserMaxEntSignal resolvedSignal = filteredSignal;
    MaxEntEncoding filteredEncoding;
    if (hasHistory && independentCurrentValid) {
        // actualAlpha is never below proposalAlpha.  Correct a provisional
        // P=(1-beta)H+beta C toward the independently filtered current C with
        // gamma=(alpha-beta)/(1-beta). Raw RT updates only the moment history;
        // alpha=1 must still publish a spatially denoised current estimator.
        float correctionAlpha = clamp((actualAlpha - proposalAlpha)
            / max(1.0 - proposalAlpha, 1e-6), 0.0, 1.0);
        resolvedSignal.maxEntY = mix(filteredSignal.maxEntY,
            independentCurrent.maxEntY, correctionAlpha);
        filteredEncoding.CoCg = mix(filteredSignal.CoCg,
            independentCurrent.CoCg, correctionAlpha);
        resolvedSignal.standardDeviation =
            maxentTemporalProposalCorrectedStandardDeviation(
                filteredSignal.standardDeviation,
                independentCurrent.standardDeviation,
                correctionAlpha);
    } else if (hasHistory) {
        filteredEncoding.CoCg = filteredSignal.CoCg;
    } else {
        filteredEncoding.CoCg = filteredSignal.CoCg;
        resolvedSignal.standardDeviation = sqrt(
            maxentTemporalCurrentEstimatorVariance(
                filteredSignal.standardDeviation));
    }
    filteredEncoding.maxEntY = resolvedSignal.maxEntY;
    // The ping-ponged denoised history must contain the exact same resolved
    // six-component estimator as DIF_N_SWAP.
    resolvedSignal.CoCg = filteredEncoding.CoCg;
    packedLight = denoiserPackMaxEntSignal(resolvedSignal);
    writeDiffuseDenoisedCurrentRaw(gxy, packedLight);

    writeDiffuseHist(gxy, committedRaw, committedSamples,
        committedRootMeanY2);
    vec3 currentPosition;
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, currentPosition, primaryDistance);
    writeDiffuseHistGeo(gxy, currentPosition,
        readPrimaryGeometryNormal(gxy), committedSamples);
    writeDiffuseSwap(gxy, filteredEncoding, committedSamples,
        committedRootMeanY2);

    // Publish the path-guide reservoir after consuming temporal scratch.
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex5, pix, 0);
}
