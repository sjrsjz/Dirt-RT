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

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    uvec4 currentSignalWords = texelFetch(colortex5, ivec2(pixel), 0);
    DenoiserMaxEntSignal currentSignal =
        denoiserUnpackMaxEntSignal(currentSignalWords);
    MaxEntGeometry currentGeometry = maxentDecodeGeometry(
        readPrimaryGeometryWords(pixel), pixel);
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
    float historyMonteCarloStandardDeviation, historyDenoisedEffectiveSamples;
    float reprojectionAlphaFloor, currentTrackingHitDistance;
    bool denoisedReprojectionValid = readMaxEntSpecularDenoisedReprojection(pixel, historyDenoisedSignal,
        historyMonteCarloStandardDeviation, historyDenoisedEffectiveSamples,
        reprojectionAlphaFloor, currentTrackingHitDistance);
    bool hasHistory = statisticsValidEffectiveSampleCount(reprojected.historyEffectiveSamples)
        && denoisedReprojectionValid;

    float currentAlpha = 1.0;
    uvec4 independentCurrentWords = denoiserScratchLoadA(ivec2(pixel));
    float independentCurrentEffectiveSamples = denoiserScratchLoadEffectiveSamplesA(ivec2(pixel));
    bool independentCurrentValid = denoiserSpatialSignalWordsValid(independentCurrentWords)
        && statisticsValidEffectiveSampleCount(independentCurrentEffectiveSamples);
    DenoiserMaxEntSignal independentCurrent = independentCurrentValid
        ? denoiserUnpackMaxEntSignal(independentCurrentWords) : denoiserEmptyMaxEntSignal();
    if (hasHistory) {
        currentAlpha = max(clamp(float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0), reprojectionAlphaFloor);
        if (independentCurrentValid) {
            float responseAlpha = maxentTemporalResponseAlpha(
                reprojectionAlphaFloor, independentCurrent.maxEntY,
                independentCurrent.standardDeviation, independentCurrentEffectiveSamples,
                historyDenoisedSignal.maxEntY, historyMonteCarloStandardDeviation,
                historyDenoisedEffectiveSamples, reprojected.historyEffectiveSamples, noiseOnlyCurrentWeight);
            currentAlpha = responseAlpha;
        }
    }
    debugWriteSpecularNoiseOnlyCurrentWeight(pixel, noiseOnlyCurrentWeight);

    MaxEntSpecularHistory committed;
    committed.surfacePosition = currentGeometry.position;
    committed.geometryNormal = currentGeometry.normal;
    committed.roughness = currentGeometry.roughness;
    committed.materialID = maxentReflectionHistoryMaterialID(
        currentGeometry.materialID);
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
    float resolvedStandardDeviation = currentSignal.standardDeviation;
    float resolvedDenoisedEffectiveSamples = hasHistory ? historyDenoisedEffectiveSamples : max(reprojected.historyEffectiveSamples, 1.0);
    if (independentCurrentValid) {
        // Commit the exact final A-Trous center estimator used to choose currentAlpha.
        if (hasHistory) {
            filtered.maxEntY = mix(historyDenoisedSignal.maxEntY, independentCurrent.maxEntY, currentAlpha);
            filtered.CoCg = mix(historyDenoisedSignal.CoCg, independentCurrent.CoCg, currentAlpha);
            resolvedStandardDeviation = maxentTemporalMixMonteCarloStandardDeviation(
                historyMonteCarloStandardDeviation, independentCurrent.standardDeviation, currentAlpha);
            resolvedDenoisedEffectiveSamples = statisticsKishBlendEffectiveSampleCounts(
                historyDenoisedEffectiveSamples, independentCurrentEffectiveSamples, currentAlpha);
        } else {
            filtered.maxEntY = independentCurrent.maxEntY;
            filtered.CoCg = independentCurrent.CoCg;
            resolvedStandardDeviation = independentCurrent.standardDeviation;
            resolvedDenoisedEffectiveSamples = independentCurrentEffectiveSamples;
        }
    } else {
        filtered.maxEntY = currentSignal.maxEntY;
        filtered.CoCg = currentSignal.CoCg;
    }
    writeMaxEntSpecularDenoisedHistory(pixel, filtered, resolvedStandardDeviation, resolvedDenoisedEffectiveSamples);
    vec3 primaryRay = reconstructPrimaryRay(pixel);
    float virtualScale = denoiserSpatialSpecularVirtualScale(primaryRay,
        currentGeometry.normal, currentGeometry.roughness);
    float resolvedVirtualDistance = currentGeometry.distance
        + virtualScale * committed.hitDistance;
    writeReflMaxEnt(pixel, filtered, resolvedVirtualDistance, 1.0);
}
