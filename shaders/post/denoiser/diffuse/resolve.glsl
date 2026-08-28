#version 430 core
// Purpose: choose diffuse currentAlpha from denoised estimators and commit raw/filtered histories.
// Dispatch: 16x16.
// Reads: colortex4 final A-Trous signal, colortex5 path guide, colortex6 raw reprojection,
//        shared independent-current scratch, Raw RT diffuse signal, filtered-history reprojection.
// Writes: diffuse raw history, geometry history, filtered history, swap state, and path guide.
// Persistent side effects: this is the only diffuse history commit point.
// Invalid representation: invalid signal metadata clears every owned persistent record.
layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/lighting/denoiser/temporal_response.glsl"
#include "/lib/lighting/denoiser/scratch_io.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex5;
uniform usampler2D colortex6;

uvec4 maxentTemporalRobustLoadSignalWords(ivec2 pixel) {
    return denoiserScratchLoadA(pixel);
}

ivec2 maxentTemporalRobustImageSize() { return textureSize(colortex4, 0); }
uvec4 maxentTemporalRobustLoadGeometryWords(ivec2 pixel) { return readPrimaryGeometryWords(uvec2(pixel)); }

#include "/lib/lighting/denoiser/robust_mean.glsl"

MaxEntTemporalRobustEstimate maxentDiffuseCurrentRobustMomentEstimate(ivec2 centerPixel, vec4 centerMoment,
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

    vec3 centerSurfaceNormal = decodeNormalU(centerGeometryWords.x);
    vec3 centerPrimaryRay = reconstructPrimaryRay(uvec2(centerPixel));
    float centerPlaneOffset = centerSurfaceDistance
        * dot(centerSurfaceNormal, centerPrimaryRay);
    float surfaceRejectionScale = denoiserSpatialDistanceRejectionScale(
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
                    || isinf(sampleSurfaceDistance))
                continue;

            vec3 sampleSurfaceNormal = decodeNormalU(sampleGeometryWords.x);
            if (dot(centerSurfaceNormal, sampleSurfaceNormal) <= 0.0)
                continue;
            vec3 samplePrimaryRay = reconstructPrimaryRay(uvec2(samplePixel));
            float planeExponent = denoiserSpatialAxialDistanceExponent(
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
        debugWriteDiffuseNoiseOnlyCurrentWeight(gxy, -1.0);
        return;
    }

    DenoiserMaxEntSignal filteredSignal =
        denoiserUnpackMaxEntSignal(packedLight);
    MaxEntEncoding currentRaw;
    float currentRootMeanY2;
    readDiffuseLightRT(gxy, currentRaw, currentRootMeanY2);

    MaxEntEncoding reprojectedRaw;
    float historyEffectiveSamples, historyRootMeanY2;
    unpackDiffuseTemporalState(texelFetch(colortex6, pix, 0),
        reprojectedRaw, historyEffectiveSamples, historyRootMeanY2);

    vec4 historyDenoisedMoment;
    vec2 historyDenoisedCoCg;
    float historyDenoisedPropagatedStandardDeviation;
    float validWeight;
    bool hasHistory = readDiffuseDenoisedReprojection(gxy,
        historyDenoisedMoment, historyDenoisedCoCg,
        historyDenoisedPropagatedStandardDeviation, validWeight)
        && statisticsValidEffectiveSampleCount(historyEffectiveSamples)
        && !any(isnan(reprojectedRaw.maxEntY))
        && !any(isinf(reprojectedRaw.maxEntY))
        && !any(isnan(reprojectedRaw.CoCg))
        && !any(isinf(reprojectedRaw.CoCg));

    float currentAlpha = 1.0;
    float proposalAlpha = 1.0;
    float noiseOnlyCurrentWeight = -1.0;
    DenoiserMaxEntSignal independentCurrent =
        denoiserEmptyMaxEntSignal();
    bool independentCurrentValid = false;
    if (hasHistory) {
        proposalAlpha = clamp(
            float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
        currentAlpha = proposalAlpha;
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
            float responseAlpha = maxentTemporalResponseAlpha(
                reprojectionAlphaFloor, robustCurrent.moment,
                robustCurrent.standardDeviation * robustCurrent.standardDeviation,
                historyDenoisedMoment, historyDenoisedPropagatedStandardDeviation,
                historyEffectiveSamples,
                noiseOnlyCurrentWeight);
            currentAlpha = responseAlpha;
        }
    }
    debugWriteDiffuseNoiseOnlyCurrentWeight(gxy, noiseOnlyCurrentWeight);

    MaxEntEncoding committedRaw;
    float committedRootMeanY2;
    float committedEffectiveSamples;
    if (hasHistory) {
        committedRaw = mix_maxent(reprojectedRaw, currentRaw, currentAlpha);
        committedRootMeanY2 = sqrt(mix(
            historyRootMeanY2 * historyRootMeanY2,
            currentRootMeanY2 * currentRootMeanY2, currentAlpha));
        committedEffectiveSamples = statisticsKishUpdateEffectiveSampleCount(historyEffectiveSamples, currentAlpha);
    } else {
        committedRaw = currentRaw;
        committedRootMeanY2 = currentRootMeanY2;
        committedEffectiveSamples = 1.0;
    }

    DenoiserMaxEntSignal resolvedSignal = filteredSignal;
    MaxEntEncoding filteredEncoding;
    if (hasHistory && independentCurrentValid) {
        // When currentAlpha exceeds proposalAlpha, correct the provisional
        // P=(1-beta)H+beta C toward the independently filtered current C with
        // correction=(currentAlpha-proposalAlpha)/(1-proposalAlpha). Raw RT updates only the moment history;
        // currentAlpha=1 must still publish a spatially denoised current estimator.
        float correctionCurrentWeight = clamp((currentAlpha - proposalAlpha)
            / max(1.0 - proposalAlpha, 1e-6), 0.0, 1.0);
        resolvedSignal.maxEntY = mix(filteredSignal.maxEntY,
            independentCurrent.maxEntY, correctionCurrentWeight);
        filteredEncoding.CoCg = mix(filteredSignal.CoCg,
            independentCurrent.CoCg, correctionCurrentWeight);
        resolvedSignal.standardDeviation =
            maxentTemporalProposalCorrectedStandardDeviation(
                filteredSignal.standardDeviation,
                independentCurrent.standardDeviation,
                correctionCurrentWeight);
    } else {
        filteredEncoding.CoCg = filteredSignal.CoCg;
        if (!hasHistory) resolvedSignal.standardDeviation = sqrt(maxentTemporalCurrentVariance(filteredSignal.standardDeviation));
    }
    filteredEncoding.maxEntY = resolvedSignal.maxEntY;
    // The ping-ponged denoised history must contain the exact same resolved
    // six-component estimator as DIF_N_SWAP.
    resolvedSignal.CoCg = filteredEncoding.CoCg;
    packedLight = denoiserPackMaxEntSignal(resolvedSignal);
    writeDiffuseDenoisedCurrentRaw(gxy, packedLight);

    writeDiffuseHist(gxy, committedRaw, committedEffectiveSamples,
        committedRootMeanY2);
    vec3 currentPosition;
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, currentPosition, primaryDistance);
    writeDiffuseHistGeo(gxy, currentPosition,
        readPrimaryGeometryNormal(gxy), committedEffectiveSamples);
    writeDiffuseSwap(gxy, filteredEncoding, committedEffectiveSamples,
        committedRootMeanY2);

    // Publish the path-guide reservoir after consuming temporal scratch.
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex5, pix, 0);
}
