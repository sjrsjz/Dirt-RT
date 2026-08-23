#version 430 core
// ===========================================================================
// MaxEnt diffuse history resolve and path-guide SSBO publication.
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
#include "/lib/lighting/denoiser/maxent_temporal_robust_tile.glsl"

MaxEntEncoding unpackLightSample(uvec4 d1) {
    MaxEntEncoding encoded;
    encoded.maxEntY = vec4(unpackHalf2x16(d1.x), unpackHalf2x16(d1.y));
    encoded.CoCg = unpackHalf2x16(d1.z);
    return encoded;
}

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
    uvec4 reprojectedWords = readDiffuseDenoisedCurrentRaw(gxy);
    uvec4 packedLight = maxentTemporalRobustTileSignalWords(ivec2(0));
    writeDiffuseDenoisedCurrentRaw(gxy, packedLight);
    if (unpackHalf2x16(packedLight.w).x < 0.0) {
        // No surface: invalidate every temporal consumer with four raw stores
        // instead of decoding two light textures and previous histories.
        diffuseBuffer.data[addr(DIF_N_HIST, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_SWAP, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = uvec4(0u);
        writeDiffuseHistGeoInvalid(gxy);
        debugWriteDiffuseDenoisedDifference(gxy, -1.0);
        return;
    }
    // Phase 1: swap3
    diffuseIlluminationData tmp = fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.maxEntY))) tmp.data_swap.maxEntY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);
    vec2 historyMeta = unpackHalf2x16(reprojectedWords.z);
    vec2 historyDeviationAlpha = unpackHalf2x16(reprojectedWords.w);
    vec4 currentMaxEntY = vec4(unpackHalf2x16(packedLight.x), unpackHalf2x16(packedLight.y));
    vec4 historyMaxEntY = vec4(unpackHalf2x16(reprojectedWords.x), unpackHalf2x16(reprojectedWords.y));
    float currentStandardDeviation = max(
        unpackHalf2x16(packedLight.w).x, 0.0);
    MaxEntTemporalRobustEstimate robustCurrent =
        maxentDiffuseCurrentRobustMomentEstimate(
            pix, currentMaxEntY, currentStandardDeviation);
    float normalizedDistance;
    tmp.weight = maxentClampHistoryWeightByMomentDifference(tmp.weight,
            robustCurrent.moment, robustCurrent.standardDeviation,
            historyMaxEntY, historyDeviationAlpha.x,
            historyMeta.x, historyMeta.y, historyDeviationAlpha.y,
            float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY),
            MAXENT_DIFFUSE_TEMPORAL_DIFFERENCE_TOLERANCE, normalizedDistance);
    debugWriteDiffuseDenoisedDifference(gxy, normalizedDistance);
    tmp.prev_weight = tmp.weight;
    tmp.data = tmp.data_swap;
    tmp.prevRootMeanY2 = tmp.rootMeanY2;

    MaxEntEncoding encoded = unpackLightSample(packedLight);
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, tmp.pos, primaryDistance);
    tmp.histNormal = readPrimaryGeometryNormal(gxy);
    tmp.data_swap = encoded;

    writeDiffuse(tmp, pix);

    // Phase 2: publish the path-guide reservoir from temporal scratch.
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex5, pix, 0);
}
