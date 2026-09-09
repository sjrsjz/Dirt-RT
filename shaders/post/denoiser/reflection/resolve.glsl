#version 430 core

// Purpose: choose reflection currentAlpha from denoised estimators and commit raw/filtered histories.
// Dispatch: 16x16.
// Reads: final independent-current scratch and colortex6 Raw reflection,
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

uniform usampler2D colortex6;

#include "/lib/lighting/denoiser/scratch_io.glsl"

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    // Both spatial estimators share the same acceptance/invalid decision.
    // The final independent-current record supplies the resolved estimator and
    // its validity, so the final proposal needs no image store or reload.
    uvec4 independentCurrentWords = denoiserScratchLoadA(ivec2(pixel));
    MaxEntGeometry currentGeometry = maxentDecodeGeometry(
        readPrimaryGeometryWords(pixel), pixel);
    if (!currentGeometry.valid
            || !denoiserSpatialSignalWordsValid(independentCurrentWords)) {
        debugWriteSpecularNoiseOnlyCurrentWeight(pixel, -1.0);
        debugWriteSpecularTemporalStateInvalid(pixel);
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        writeMaxEntSpecularDenoisedHistoryInvalid(pixel);
        writeReflMaxEnt(pixel, emptySpecularMaxEnt(), 0.0, 0.0);
        return;
    }

    DenoiserMaxEntSignal independentCurrent =
        denoiserUnpackMaxEntSignal(independentCurrentWords);
    MaxEntSpecularInput noisy = maxentUnpackSpecularInput(
        texelFetch(colortex6, ivec2(pixel), 0));
    MaxEntSpecularHistory reprojected =
        readMaxEntSpecularHistory(pixel);

    float noiseOnlyCurrentWeight = -1.0;
    SpecularMaxEnt historyDenoisedSignal = emptySpecularMaxEnt();
    float historyMonteCarloStandardDeviation = 0.0, historyDenoisedEffectiveSamples = 1.0;
    float reprojectionAlphaFloor = 1.0, currentTrackingHitDistance = noisy.hitDistance;
    bool denoisedReprojectionValid = readMaxEntSpecularDenoisedReprojection(pixel, historyDenoisedSignal,
        historyMonteCarloStandardDeviation, historyDenoisedEffectiveSamples,
        reprojectionAlphaFloor, currentTrackingHitDistance);
    bool hasHistory = statisticsValidEffectiveSampleCount(reprojected.historyEffectiveSamples)
        && denoisedReprojectionValid;

    float currentAlpha = 1.0;
    if (hasHistory) {
        currentAlpha = maxentTemporalResponseAlpha(
            reprojectionAlphaFloor, independentCurrent.maxEntY,
            independentCurrent.standardDeviation,
            historyDenoisedSignal.maxEntY, historyMonteCarloStandardDeviation,
            reprojected.historyEffectiveSamples, noiseOnlyCurrentWeight);
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
    float resolvedStandardDeviation = independentCurrent.standardDeviation;
    float resolvedDenoisedEffectiveSamples = 1.0; // Reserved history ABI; sigma is estimator uncertainty.
    // Commit the exact final estimator used to choose currentAlpha.
    if (hasHistory) {
        filtered.maxEntY = mix(historyDenoisedSignal.maxEntY, independentCurrent.maxEntY, currentAlpha);
        filtered.CoCg = mix(historyDenoisedSignal.CoCg, independentCurrent.CoCg, currentAlpha);
        resolvedStandardDeviation = maxentTemporalMixEstimatorStandardDeviation(
            historyMonteCarloStandardDeviation, independentCurrent.standardDeviation, currentAlpha);
    } else {
        filtered.maxEntY = independentCurrent.maxEntY;
        filtered.CoCg = independentCurrent.CoCg;
    }
    writeMaxEntSpecularDenoisedHistory(pixel, filtered, resolvedStandardDeviation, resolvedDenoisedEffectiveSamples);
    vec3 primaryRay = reconstructPrimaryRay(pixel);
    float virtualScale = denoiserSpatialSpecularVirtualScale(primaryRay,
        currentGeometry.normal, currentGeometry.roughness);
    float resolvedVirtualDistance = currentGeometry.distance
        + virtualScale * committed.hitDistance;
    writeReflMaxEnt(pixel, filtered, resolvedVirtualDistance, 1.0);
}
