#version 430 core
// Purpose: choose diffuse currentAlpha from denoised estimators and commit raw/filtered histories.
// Dispatch: 16x16.
// Reads: colortex4 final A-Trous signal, colortex6 raw reprojection,
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
uniform usampler2D colortex6;

#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 1
#include "/lib/lighting/denoiser/temporal_confidence.glsl"
void maxentConfidenceLoadRaw(ivec2 pixel, out vec4 moment, out vec2 chroma) {
    MaxEntEncoding raw;
    float rms;
    readDiffuseLightRT(uvec2(pixel), raw, rms);
    moment = raw.maxEntY;
    chroma = raw.CoCg;
}
#endif

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(gid, uvec2(resolution_global)))) return;
    ivec2 pix = ivec2(gid);
    uvec2 gxy = uvec2(pix);

    // Temporal used this parity as its reprojected-denoised scratch. Consume it
    // before replacing it with the exact current spatial result.
    uvec4 packedLight = texelFetch(colortex4, pix, 0);
    if (!denoiserSpatialSignalWordsValid(packedLight)) {
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

    vec4 historyDenoisedMoment = vec4(0.0);
    vec2 historyDenoisedCoCg = vec2(0.0);
    float historyDenoisedMonteCarloStandardDeviation = 0.0, historyDenoisedEffectiveSamples = 1.0;
    float validWeight = 0.0;
    bool hasHistory = readDiffuseDenoisedReprojection(gxy,
        historyDenoisedMoment, historyDenoisedCoCg,
        historyDenoisedMonteCarloStandardDeviation, historyDenoisedEffectiveSamples, validWeight)
        && statisticsValidEffectiveSampleCount(historyEffectiveSamples)
        && !any(isnan(reprojectedRaw.maxEntY))
        && !any(isinf(reprojectedRaw.maxEntY))
        && !any(isnan(reprojectedRaw.CoCg))
        && !any(isinf(reprojectedRaw.CoCg));

    float currentAlpha = 1.0;
    float noiseOnlyCurrentWeight = -1.0;
    uvec4 independentCurrentWords = denoiserScratchLoadA(pix);
    bool independentCurrentValid = denoiserSpatialSignalWordsValid(independentCurrentWords);
    DenoiserMaxEntSignal independentCurrent = independentCurrentValid
        ? denoiserUnpackMaxEntSignal(independentCurrentWords) : denoiserEmptyMaxEntSignal();
    if (hasHistory) {
        currentAlpha = clamp(float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 0
        if (independentCurrentValid) {
            float reprojectionAlphaFloor = 1.0 - clamp(validWeight, 0.0, 1.0);
            currentAlpha = maxentTemporalResponseAlpha(reprojectionAlphaFloor, independentCurrent.maxEntY,
                independentCurrent.standardDeviation, historyDenoisedMoment,
                historyDenoisedMonteCarloStandardDeviation,
                historyEffectiveSamples, noiseOnlyCurrentWeight);
        }
#endif
    }
    debugWriteDiffuseNoiseOnlyCurrentWeight(gxy, noiseOnlyCurrentWeight);

#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 1
    // Raw population moments have a predictable gain. No confidence decision
    // or raw-current brightness selects their temporal write weight.
    hasHistory = hasHistory && denoiserSigmaKnown(historyDenoisedMonteCarloStandardDeviation);
    currentAlpha = hasHistory ? max(float(MAXENT_TEMPORAL_FIXED_ALPHA),
        1.0 - clamp(validWeight, 0.0, 1.0)) : 1.0;
#endif

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
    float resolvedDenoisedEffectiveSamples = 1.0; // Reserved history ABI; sigma is estimator uncertainty.
#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 0
    if (independentCurrentValid) {
        // Temporal response and filtered history consume the exact same final A-Trous center estimator.
        if (hasHistory) {
            resolvedSignal.maxEntY = mix(historyDenoisedMoment, independentCurrent.maxEntY, currentAlpha);
            filteredEncoding.CoCg = mix(historyDenoisedCoCg, independentCurrent.CoCg, currentAlpha);
            resolvedSignal.standardDeviation = maxentTemporalMixEstimatorStandardDeviation(
                historyDenoisedMonteCarloStandardDeviation, independentCurrent.standardDeviation, currentAlpha);
        } else {
            resolvedSignal = independentCurrent;
            filteredEncoding.CoCg = independentCurrent.CoCg;
        }
    } else {
        filteredEncoding.CoCg = filteredSignal.CoCg;
    }
#endif
#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 1
    MaxentConfidenceGroup pilot, checkA, checkB, splitCurrent;
    maxentConfidenceGather(pix, pilot, checkA, checkB, splitCurrent);
    if (splitCurrent.valid) {
        float priorVariance = hasHistory ? maxentConfidencePriorObservationVariance(
            reprojectedRaw.maxEntY, historyRootMeanY2, historyEffectiveSamples) : 0.0;
        float estimatorVariance, confidenceGain;
        maxentConfidenceResolve(pilot, checkA, checkB, splitCurrent, historyDenoisedMoment,
            historyDenoisedCoCg, historyDenoisedMonteCarloStandardDeviation
                * historyDenoisedMonteCarloStandardDeviation, hasHistory, priorVariance, historyEffectiveSamples,
            hasHistory ? 1.0 - clamp(validWeight, 0.0, 1.0) : 1.0,
            resolvedSignal.maxEntY, filteredEncoding.CoCg, estimatorVariance, confidenceGain);
        resolvedSignal.standardDeviation = sqrt(estimatorVariance);
        debugWriteDiffuseNoiseOnlyCurrentWeight(gxy, confidenceGain);
    } else {
        // Valid center normally guarantees splitCurrent. Invalid raw packets
        // fail closed and carry deliberately broad linear-moment uncertainty.
        resolvedSignal.maxEntY = currentRaw.maxEntY;
        filteredEncoding.CoCg = currentRaw.CoCg;
        resolvedSignal.standardDeviation = DENOISER_UNKNOWN_UNCERTAINTY;
    }
    resolvedDenoisedEffectiveSamples = 1.0;
#endif
    filteredEncoding.maxEntY = resolvedSignal.maxEntY;
    // The ping-ponged denoised history must contain the exact same resolved
    // six-component estimator as DIF_N_SWAP.
    resolvedSignal.CoCg = filteredEncoding.CoCg;
    packedLight = denoiserPackMaxEntSignal(resolvedSignal);
    writeDiffuseDenoisedCurrentRaw(gxy, packedLight, resolvedDenoisedEffectiveSamples);

    writeDiffuseHist(gxy, committedRaw, committedEffectiveSamples,
        committedRootMeanY2);
    vec3 currentPosition;
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, currentPosition, primaryDistance);
    writeDiffuseHistGeo(gxy, currentPosition,
        readPrimaryGeometryNormal(gxy), committedEffectiveSamples);
    writeDiffuseSwap(gxy, filteredEncoding, committedEffectiveSamples,
        committedRootMeanY2);

    // Publish the exact packed final denoised moments for next-frame guiding.
    // The two fixed validity tags preserve the direct-guide cache layout.
    // The next frame's sampler and NEE both read this same guide cache.
    bool guideValid = unpackHalf2x16(packedLight.y).y > 1e-8;
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = guideValid
        ? uvec4(packedLight.xy, floatBitsToUint(1.0), floatBitsToUint(1.0))
        : uvec4(0u);
}
