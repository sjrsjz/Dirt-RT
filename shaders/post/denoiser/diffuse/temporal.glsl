#version 430 core

// Purpose: build the provisional diffuse temporal proposal from the raw and
// final-denoised histories jointly reprojected by ray1.
// Dispatch: 16x8.
// Reads: Raw RT diffuse signal, DIF_N_SWAP prepared raw history, and the
// current-parity prepared final-denoised history.
// Writes: diffuse swap proposal and colorimg6 raw reprojection.
// Persistent side effects: none; resolve.glsl owns accepted history updates.

layout(local_size_x = 16, local_size_y = 8) in;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

#define DIFFUSE_BUFFER_MIN

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/math/statistics.glsl"

uniform vec2 resolution;

MaxEntEncoding outputMaxEnt;
float outputMeanY2;
float outputEffectiveSamples = 0.0;

uvec4 currentLightPacked(uvec2 pixel) {
    return readDiffuseLightRTRaw(pixel);
}

void unpackCurrentLight(uvec2 pixel, out MaxEntEncoding maxEnt,
        out float meanY2) {
    uvec4 packedLight = currentLightPacked(pixel);
    maxEnt.maxEntY = vec4(unpackHalf2x16(packedLight.x),
        unpackHalf2x16(packedLight.y));
    maxEnt.CoCg = unpackHalf2x16(packedLight.z);
    float rootMeanY2 = unpackHalf2x16(packedLight.w).y;
    meanY2 = rootMeanY2 * rootMeanY2;
}

void resetToCurrentSample(uvec2 pixel) {
    outputEffectiveSamples = 1.0;
    unpackCurrentLight(pixel, outputMaxEnt, outputMeanY2);
}

void buildDiffuseTemporalProposal(uvec2 pixel) {
    MaxEntEncoding historyMaxEnt;
    float historyEffectiveSamples, historyRootMeanY2;
    readDiffuseSwap(pixel, historyMaxEnt, historyEffectiveSamples,
        historyRootMeanY2);

    vec4 historyDenoisedMoment;
    vec2 historyDenoisedChroma;
    float historyDenoisedSigma, validCoverage;
    bool hasDenoisedHistory = readDiffuseDenoisedReprojection(pixel,
        historyDenoisedMoment, historyDenoisedChroma,
        historyDenoisedSigma, validCoverage);
    bool hasRawHistory = statisticsValidEffectiveSampleCount(
            historyEffectiveSamples)
        && historyRootMeanY2 >= 0.0
        && !isnan(historyRootMeanY2) && !isinf(historyRootMeanY2)
        && !any(isnan(historyMaxEnt.maxEntY))
        && !any(isinf(historyMaxEnt.maxEntY))
        && !any(isnan(historyMaxEnt.CoCg))
        && !any(isinf(historyMaxEnt.CoCg));

    if (!hasRawHistory || !hasDenoisedHistory) {
        writeDiffuseDenoisedReprojectionInvalid(pixel);
        resetToCurrentSample(pixel);
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    MaxEntEncoding currentMaxEnt;
    float currentMeanY2;
    unpackCurrentLight(pixel, currentMaxEnt, currentMeanY2);
    float historyMeanY2 = historyRootMeanY2 * historyRootMeanY2;
    float proposalAlpha = clamp(
        float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
    outputEffectiveSamples = statisticsKishUpdateEffectiveSampleCount(
        historyEffectiveSamples, proposalAlpha);
    outputMeanY2 = mix(historyMeanY2, currentMeanY2, proposalAlpha);
    outputMaxEnt = mix_maxent(
        historyMaxEnt, currentMaxEnt, proposalAlpha);

    imageStore(colorimg6, ivec2(pixel),
        packDiffuseTemporalState(historyMaxEnt,
            historyEffectiveSamples, historyRootMeanY2));
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, uvec2(resolution)))) return;

    float surfaceDistance;
    vec3 surfacePosition;
    readDiffusePrimaryGeometry(pixel, surfacePosition, surfaceDistance);
    if (surfaceDistance < -0.5) {
        diffuseBuffer.data[addr(DIF_N_SWAP, pixel)] = uvec4(0u);
        writeDiffuseDenoisedReprojectionInvalid(pixel);
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    buildDiffuseTemporalProposal(pixel);
    writeDiffuseSwap(pixel, outputMaxEnt, outputEffectiveSamples,
        sqrt(outputMeanY2));
}
