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
#include "/lib/lighting/denoiser/maxent_bures.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex6;

MaxEntEncoding unpackLightSample(uvec4 d1) {
    MaxEntEncoding encoded;
    encoded.maxEntY = vec4(unpackHalf2x16(d1.x), unpackHalf2x16(d1.y));
    encoded.CoCg = unpackHalf2x16(d1.z);
    return encoded;
}

float clampDiffuseHistoryWeightByDenoisedDifference(float historyWeight, uvec4 currentWords,
    uvec4 reprojectedWords, out float normalizedDistance) {
    normalizedDistance = -1.0;
    historyWeight = isnan(historyWeight) || isinf(historyWeight) ? 1.0 : max(historyWeight, 1.0);
    if (!denoiserSpatialSignalWordsValid(currentWords)
            || !denoiserSpatialSignalWordsValid(reprojectedWords)) return historyWeight;

    vec2 historyMeta = unpackHalf2x16(reprojectedWords.z);
    float temporalCurrentWeight = unpackHalf2x16(reprojectedWords.w).y;
    if (!(historyMeta.x >= 1.0) || !(historyMeta.y > 0.0)
            || !(temporalCurrentWeight > 0.0) || any(isnan(historyMeta))
            || any(isinf(historyMeta)) || isnan(temporalCurrentWeight)
            || isinf(temporalCurrentWeight)) return historyWeight;

    vec4 currentMaxEntY = vec4(unpackHalf2x16(currentWords.x), unpackHalf2x16(currentWords.y));
    vec4 historyMaxEntY = vec4(unpackHalf2x16(reprojectedWords.x), unpackHalf2x16(reprojectedWords.y));
    DenoiserSpatialBuresData currentBures = denoiserSpatialMakeBuresData(currentMaxEntY);
    DenoiserSpatialBuresData historyBures = denoiserSpatialMakeBuresData(historyMaxEntY);
    float distanceSq = denoiserSpatialBuresDistanceSq(currentMaxEntY, currentBures, historyMaxEntY, historyBures);
    float currentStddev = unpackHalf2x16(currentWords.w).x;
    float historyStddev = unpackHalf2x16(reprojectedWords.w).x;
    // T = (1-alpha)H + alpha*X, so D(T,H)/alpha estimates the unattenuated
    // innovation distance. Normalize that distance by the combined filtered
    // standard deviation; this is a z-score, not a squared denoising exponent.
    float combinedStddev = sqrt(currentStddev * currentStddev + historyStddev * historyStddev);
    normalizedDistance = sqrt(clamp(historyMeta.y, 0.0, 1.0) * distanceSq) / max(temporalCurrentWeight * combinedStddev, 1e-6);
    if (isnan(normalizedDistance) || isinf(normalizedDistance)) {
        normalizedDistance = -1.0;
        return historyWeight;
    }

    // Convert statistical agreement into an upper bound on reusable history.
    // One effective sample is always retained, so a large change responds in
    // the next frame without turning the estimator into an invalid N_eff < 1.
    float agreement = exp(-min(normalizedDistance, 80.0) * (1.0 / MAXENT_DIFFUSE_TEMPORAL_DIFFERENCE_TOLERANCE));
    float maximumHistory = max(float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY), 1.0);
    float historyCap = 1.0 + (maximumHistory - 1.0) * agreement;
    return min(historyWeight, historyCap);
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(gid, uvec2(resolution_global)))) return;
    ivec2 pix = ivec2(gid);
    uvec2 gxy = uvec2(pix);

    // Temporal used this parity as its reprojected-denoised scratch. Consume it
    // before replacing it with the exact current spatial result.
    uvec4 reprojectedWords = readDiffuseDenoisedCurrentRaw(gxy);
    uvec4 packedLight = texelFetch(colortex4, pix, 0);
    writeDiffuseDenoisedCurrentRaw(gxy, packedLight);
    if (unpackHalf2x16(packedLight.w).x < 0.0) {
        // No surface: invalidate every temporal consumer with four raw stores
        // instead of decoding two light textures and previous histories.
        diffuseBuffer.data[addr(DIF_N_HIST, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_SWAP, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = uvec4(0u);
        writeDiffuseHistGeoInvalid(gxy);
        writeDiffuseDenoisedDifference(gxy, -1.0);
        return;
    }
    // Phase 1: swap3
    diffuseIlluminationData tmp = fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.maxEntY))) tmp.data_swap.maxEntY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);
    float normalizedDistance;
    tmp.weight = clampDiffuseHistoryWeightByDenoisedDifference(
            tmp.weight, packedLight, reprojectedWords, normalizedDistance);
    writeDiffuseDenoisedDifference(gxy, normalizedDistance);
    tmp.prev_weight = tmp.weight;
    tmp.data = tmp.data_swap;
    tmp.prev_meanY2 = tmp.meanY2;

    MaxEntEncoding encoded = unpackLightSample(packedLight);
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, tmp.pos, primaryDistance);
    tmp.histNormal = readPrimaryGeometryNormal(gxy);
    tmp.data_swap = encoded;

    writeDiffuse(tmp, pix);

    // Phase 2: publish the path-guide reservoir from temporal scratch.
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex6, pix, 0);
}
