#version 430 core

// Purpose: reproject reflection raw/filtered histories and build the provisional temporal proposal.
// Dispatch: 8x8.
// Reads: colortex6 raw reflection, previous raw/virtual history, compact
// geometry, motion, and ray0's shared surface-denoised reprojection.
// Writes: colorimg4 proposal, colorimg5 raw reprojection, filtered-history reprojection scratch.
// Persistent side effects: none; resolve.glsl owns the accepted history update.
// Invalid representation: zero image words plus invalid filtered reprojection metadata.

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/post/denoiser/reflection/common.glsl"
#include "/lib/math/statistics.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

#include "/post/denoiser/reflection/reprojection.glsl"

float maxentSmoothWeight(float x) {
    x = clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

float maxentAngularConfidence(vec3 a, vec3 b, float maxAngle) {
    float angle = acos(clamp(dot(a, b), -1.0, 1.0));
    return maxentSmoothWeight(1.0 - angle / max(maxAngle, 1e-5));
}

float maxentDominantFactor(float NoV, float roughness) {
    float a = 0.298475 * log(max(39.4115 - 39.0029 * roughness, 1e-5));
    return clamp(pow(clamp(1.0 - NoV, 0.0, 1.0), 10.8649)
        * (1.0 - a) + a, 0.0, 1.0);
}

float maxentSpecMagicCurve(float roughness) {
    return 1.0 - exp2(-200.0 * roughness * roughness);
}

void maxentPublishDenoisedReprojection(MaxEntReprojectedHistory history, float alphaFloor,
        float currentTrackingHitDistance) {
    if (!history.found) {
        writeMaxEntSpecularDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        return;
    }
    SpecularMaxEnt denoised = unpackSpecularMaxEnt(
        history.denoisedWords.xyz);
    float standardDeviation = uintBitsToFloat(history.denoisedWords.w);
    writeMaxEntSpecularDenoisedReprojection(gl_GlobalInvocationID.xy,
        denoised, standardDeviation, history.denoisedEffectiveSamples, currentTrackingHitDistance,
        alphaFloor);
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    MaxEntSpecularInput noisy = maxentUnpackSpecularInput(
        texelFetch(colortex6, ivec2(pixel), 0));
    float noisyY = noisy.signal.maxEntY.w;
    float noisyM2 = noisyY * noisyY;

    MaxEntGeometry currentGeometry = maxentLoadGeometry(pixel);
    if (!currentGeometry.valid) {
        imageStore(colorimg4, ivec2(pixel), uvec4(0u));
        imageStore(colorimg5, ivec2(pixel), uvec4(0u));
        writeMaxEntSpecularDenoisedReprojectionInvalid(pixel);
        debugWriteSpecularTemporalStateInvalid(pixel);
        return;
    }

    vec3 currentPos = currentGeometry.position;
    vec3 currentNormal = currentGeometry.normal;
    uint currentMaterial = maxentReflectionHistoryMaterialID(
        currentGeometry.materialID);
    float currentRoughness = currentGeometry.roughness;

    vec3 cameraDelta = camPos - prevRaytracingCamPos;
    vec3 surfaceMotion;
    float motionValid;
    readSurfaceMotion(pixel, surfaceMotion, motionValid);
    cameraDelta -= surfaceMotion;

    vec2 surfaceUv = maxentProjectPrevious(currentPos, cameraDelta);
    MaxEntReprojectedHistory surface = maxentLoadHistory(surfaceUv, pixel,
        currentPos, currentNormal, currentMaterial, cameraDelta,
        false, true, false);
    SpecularMaxEnt preparedSurfaceDenoised;
    float preparedSurfaceSigma, preparedSurfaceCoverage;
    bool preparedSurfaceValid = readMaxEntSpecularPreparedSurfaceDenoised(
        pixel, preparedSurfaceDenoised, preparedSurfaceSigma,
        preparedSurfaceCoverage);
    if (surface.found && preparedSurfaceValid) {
        surface.denoisedWords = uvec4(
            packSpecularMaxEnt(preparedSurfaceDenoised),
            floatBitsToUint(preparedSurfaceSigma));
        surface.denoisedEffectiveSamples = 1.0;
        surface.footprintQuality = min(surface.footprintQuality,
            preparedSurfaceCoverage);
    } else {
        surface.found = false;
    }

    // Keep the tracking guide per-pixel. Spatial minimum reconstruction changes its statistical meaning and creates
    // a persistent current/history mismatch even for a static delta mirror on a smooth hit-distance gradient.
    float currentTrackingHitDistance = noisy.hitDistance > 0.0 ? noisy.hitDistance : 0.0;

    float surfaceDistance = length(currentPos);
    vec3 primaryRay = maxentSafeNormalize(currentPos, vec3(0.0, 0.0, 1.0));
    vec3 V = -primaryRay;
    float NoV = abs(dot(currentNormal, V));
    float virtualScale = maxentDominantFactor(NoV, currentRoughness);
    vec3 virtualPoint = primaryRay *
        (surfaceDistance + currentTrackingHitDistance * virtualScale);
    vec2 virtualUv = maxentProjectPrevious(virtualPoint, cameraDelta);
    MaxEntReprojectedHistory virtualHistory = maxentLoadHistory(virtualUv,
        pixel, currentPos, currentNormal, currentMaterial, cameraDelta,
        true, false, true);
    if (motionValid < 0.5) {
        surface.found = false;
        virtualHistory.found = false;
    }

    float lobeHalfAngle = max(atan(maxentSpecLobeTanHalfAngle(
        currentRoughness, MAXENT_SPECULAR_TEMPORAL_LOBE_FRACTION)), 1.5 / 255.0);
    vec3 Vprev = -maxentSafeNormalize(currentPos + cameraDelta, currentPos);
    float smbConfidence = surface.found
        ? maxentAngularConfidence(V, Vprev,
            lobeHalfAngle * max(NoV, 0.01)) : 0.0;

    float virtualRoughnessWeight = virtualHistory.found
        ? exp(-8.0 * abs(virtualHistory.roughness - currentRoughness)) : 0.0;
    float vmbConfidence = virtualHistory.found
        ? 0.1 + 0.9 * virtualRoughnessWeight : 0.0;
    if (virtualHistory.found)
        vmbConfidence *= float(dot(decodeNormalU(virtualHistory.normalWord), currentNormal) > 0.0);

    float magicCurve = maxentSpecMagicCurve(currentRoughness);
    float hitDistanceCenter = mix(noisy.hitDistance,
        surface.found ? surface.hitDistance : noisy.hitDistance, magicCurve);
    float maxHitDistance = max(hitDistanceCenter,
        virtualHistory.found ? virtualHistory.hitDistance : hitDistanceCenter);
    float relativeHitError = abs(hitDistanceCenter -
        (virtualHistory.found ? virtualHistory.hitDistance : hitDistanceCenter))
        / max(surfaceDistance + maxHitDistance, 1e-5);
    float hitConfidence = mix(1.0 - clamp(mix(20.0, 0.0, magicCurve)
        * relativeHitError, 0.0, 1.0), 1.0, magicCurve);

    if (virtualHistory.found) {
        vec3 trackedVirtualPoint = primaryRay * (surfaceDistance
            + virtualHistory.hitDistance * virtualScale);
        vec2 trackedUv = maxentProjectPrevious(trackedVirtualPoint, cameraDelta);
        float uvErrorPixels = length((trackedUv - virtualUv)
            * vec2(resolution_global));
        float lobeTan = max(maxentSpecLobeTanHalfAngle(currentRoughness, 0.6),
            0.5 / max(float(resolution_global.x), 1.0));
        float lobeRadiusPixels = min(currentTrackingHitDistance,
            virtualHistory.hitDistance) * lobeTan
            * float(resolution_global.y) / max(surfaceDistance, 1e-4);
        hitConfidence *= 1.0 - smoothstep(0.0,
            lobeRadiusPixels + 0.25, uvErrorPixels);
    }

    MaxEntReprojectedHistory history;
    float historyConfidence;
    if (currentRoughness == 0.0) {
        // A delta mirror has no surface-motion lobe. Invalid virtual motion rejects history instead of falling back
        // to a long surface-space history, whose reflected signal is not attached to the primary surface.
        history = virtualHistory;
        historyConfidence = virtualHistory.found ? vmbConfidence * hitConfidence : 0.0;
    } else {
        float virtualAmount = virtualHistory.found ? virtualScale * virtualHistory.footprintQuality : 0.0;
        virtualAmount *= clamp(vmbConfidence / max(smbConfidence, 1e-6), 0.0, 1.0) * hitConfidence;
        virtualAmount = clamp(virtualAmount, 0.0, 1.0);
        history = maxentCombineReprojectedHistories(surface, virtualHistory, virtualAmount);

        float surfaceBranchWeight = surface.found ? 1.0 - virtualAmount : 0.0;
        float virtualBranchWeight = virtualHistory.found ? virtualAmount : 0.0;
        if (!surface.found && virtualHistory.found) virtualBranchWeight = 1.0;
        if (surface.found && !virtualHistory.found) surfaceBranchWeight = 1.0;
        float branchWeightSum = surfaceBranchWeight + virtualBranchWeight;
        historyConfidence = branchWeightSum > 1e-5 ? (surfaceBranchWeight * smbConfidence
            + virtualBranchWeight * vmbConfidence * hitConfidence) / branchWeightSum : 0.0;
    }

    MaxEntTemporalSignal temporal;
    if (history.found) {
        float alphaFloor = 1.0 - clamp(historyConfidence, 0.0, 1.0);
        // Build the fixed-alpha provisional estimator consumed by variance preparation and A-Trous.
        // Final resolve evaluates the tuned response and corrects it with the independent-current branch.
        float proposalAlpha = clamp(
            float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
        SpecularMaxEnt historySignal = unpackSpecularMaxEnt(
            history.signalWords);
        temporal.signal = maxentMixMaxEnt(historySignal, noisy.signal,
            proposalAlpha);
        temporal.rootMeanY2 = sqrt(mix(history.meanY2, noisyM2, proposalAlpha));
        temporal.historyEffectiveSamples = statisticsKishUpdateEffectiveSampleCount(
            history.historyEffectiveSamples, proposalAlpha);
        maxentPublishDenoisedReprojection(history, alphaFloor, currentTrackingHitDistance);
    } else {
        temporal.signal = noisy.signal;
        temporal.rootMeanY2 = abs(noisyY);
        temporal.historyEffectiveSamples = 1.0;
        writeMaxEntSpecularDenoisedReprojectionInvalid(pixel, currentTrackingHitDistance);
    }

    MaxEntTemporalSignal reprojectedTemporal;
    reprojectedTemporal.signal = history.found
        ? unpackSpecularMaxEnt(history.signalWords)
        : emptySpecularMaxEnt();
    reprojectedTemporal.rootMeanY2 = history.found ? sqrt(history.meanY2) : 0.0;
    reprojectedTemporal.historyEffectiveSamples = history.found ? history.historyEffectiveSamples : 0.0;
    imageStore(colorimg4, ivec2(pixel), maxentPackTemporal(temporal));
    imageStore(colorimg5, ivec2(pixel), history.found
        ? maxentPackTemporal(reprojectedTemporal) : uvec4(0u));
    debugWriteSpecularTemporalState(pixel, packSpecularMaxEnt(temporal.signal),
        noisy.hitDistance, 0.0);
}
