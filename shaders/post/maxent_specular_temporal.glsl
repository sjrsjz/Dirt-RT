#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"
#include "/lib/math/statistics.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

struct MaxEntReprojectedHistory {
    uvec3 signalWords;
    float secondMoment;
    uvec4 denoisedWords;
    float hitDistance;
    uint normalWord;
    float roughness;
    float historyLength;
    float footprintQuality;
    bool found;
};

MaxEntReprojectedHistory maxentEmptyHistory() {
    MaxEntReprojectedHistory h;
    h.signalWords = uvec3(0u);
    h.secondMoment = 0.0;
    h.denoisedWords = uvec4(0u);
    h.hitDistance = 0.0;
    h.normalWord = encodeNormalU(vec3(0.0, 1.0, 0.0));
    h.roughness = 1.0;
    h.historyLength = 0.0;
    h.footprintQuality = 0.0;
    h.found = false;
    return h;
}

bool maxentSurfaceFootprintContains(uvec2 currentPixel,
        vec3 historyPositionCurrentSpace) {
    vec4 clip = rtViewProjection * vec4(historyPositionCurrentSpace, 1.0);
    if (clip.w <= 1e-7 || any(isnan(clip)) || any(isinf(clip))) return false;
    vec2 projectedPixel = (clip.xy / clip.w * 0.5 + 0.5) *
        vec2(resolution_global);
    return all(lessThanEqual(abs(projectedPixel - vec2(currentPixel)),
        vec2(MAXENT_SPECULAR_TEMPORAL_REPROJECTION_RADIUS + 1e-5)));
}

MaxEntReprojectedHistory maxentLoadHistory(
    vec2 uv,
    uvec2 currentPixel,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    vec3 cameraDelta,
    bool requireFullFootprint,
    bool requireSurfaceFootprint
) {
    MaxEntReprojectedHistory outHistory = maxentEmptyHistory();
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
        return outHistory;

    ivec2 size = ivec2(resolution_global);
    vec2 pixelPosition = uv * vec2(size);
    ivec2 origin = ivec2(floor(pixelPosition));
    vec2 f = fract(pixelPosition);
    vec4 bilinear = vec4((1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    float sumWeight = 0.0;
    float weightOverRootSamples = 0.0;
    float weightedEstimatorStddev = 0.0;
    float validBilinearWeight = 0.0;
    int validTapCount = 0;
    SpecularMaxEnt signalSum = emptySpecularMaxEnt();
    SpecularMaxEnt denoisedSum = emptySpecularMaxEnt();
    float denoisedSquaredWeightVariance = 0.0;
    float denoisedWeightedStddev = 0.0;
    vec3 normalSum = vec3(0.0);
    float depthThreshold = MAXENT_SPECULAR_TEMPORAL_DISOCCLUSION_THRESHOLD *
        max(length(currentSurfacePosition), 1.0);

    for (int i = 0; i < 4; ++i) {
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (!maxentInBounds(p, size)) continue;
        uvec2 historyPixel = uvec2(p);
        uvec4 geometryWords = reflectBuffer.data[addr(SPEC_N_HISTGEO, historyPixel)];
        uvec4 signalWords = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, historyPixel)];
        uvec4 denoisedWords = reflectBuffer.data[addr(SPEC_N_HISTMETA, historyPixel)];
        float surfaceDistance = uintBitsToFloat(geometryWords.x);
        vec2 momentHistory = unpackHalf2x16(signalWords.w);
        vec2 denoisedMetadata = unpackHalf2x16(denoisedWords.w);
        if (!(surfaceDistance >= 0.0) || isnan(surfaceDistance) || isinf(surfaceDistance)
                || !(momentHistory.x >= 0.0)
                || !(momentHistory.y >= 1.0) || denoisedMetadata.y != -2.0
                || !(denoisedMetadata.x >= 0.0) || any(isnan(momentHistory))
                || any(isinf(momentHistory)) || any(isnan(denoisedMetadata))
                || any(isinf(denoisedMetadata))) continue;

        vec3 historyNormal;
        uint historyMaterial;
        unpackMaxEntHistoryNormalMaterial(geometryWords.z, historyNormal, historyMaterial);
        if (historyMaterial != currentMaterial) continue;

        vec3 previousSurfaceCurrent = decodeNormalU(geometryWords.y) * surfaceDistance - cameraDelta;
        if (abs(dot(previousSurfaceCurrent - currentSurfacePosition,
                currentNormal)) > depthThreshold)
            continue;
        // Geometry normals only provide a hard topology/silhouette guard.
        // They never attenuate a valid history sample.
        if (dot(currentNormal, historyNormal) <= 0.0) continue;
        if (requireSurfaceFootprint && !maxentSurfaceFootprintContains(
                currentPixel, previousSurfaceCurrent))
            continue;

        float w = bilinear[i];
        validBilinearWeight += w;
        ++validTapCount;
        if (w <= 0.0) continue;
        vec4 tapMaxEntY = vec4(unpackHalf2x16(signalWords.x),
            unpackHalf2x16(signalWords.y));
        signalSum.maxEntY += tapMaxEntY * w;
        signalSum.CoCg += unpackHalf2x16(signalWords.z) * w;
        denoisedSum.maxEntY += vec4(unpackHalf2x16(denoisedWords.x), unpackHalf2x16(denoisedWords.y)) * w;
        denoisedSum.CoCg += unpackHalf2x16(denoisedWords.z) * w;
        denoisedSquaredWeightVariance += denoisedMetadata.x
            * denoisedMetadata.x * w * w;
        denoisedWeightedStddev += denoisedMetadata.x * w;
        vec2 hitRoughness = unpackHalf2x16(geometryWords.w);
        outHistory.hitDistance += hitRoughness.x * w;
        outHistory.roughness += (hitRoughness.y - 1.0) * w;
        float samples = momentHistory.y;
        float tapSecondMoment = momentHistory.x * momentHistory.x;
        float tapEstimatorVariance =
            statisticsEstimatorVarianceFromMoments(
                2.0 * tapSecondMoment, tapMaxEntY, samples);
        weightedEstimatorStddev += w * sqrt(tapEstimatorVariance);
        weightOverRootSamples += w * inversesqrt(samples);
        normalSum += historyNormal * w;
        sumWeight += w;
    }

    bool accepted = requireFullFootprint
        ? (validTapCount == 4 && validBilinearWeight > 0.999)
        : (sumWeight > 1e-5);
    if (!accepted || sumWeight <= 1e-5) return maxentEmptyHistory();

    float invWeight = 1.0 / sumWeight;
    signalSum = maxentScaleMaxEnt(signalSum, invWeight);
    outHistory.signalWords = packSpecularMaxEnt(signalSum);
    denoisedSum = maxentScaleMaxEnt(denoisedSum, invWeight);
    float denoisedStandardDeviation =
        statisticsWeightedMeanStandardDeviation(
        denoisedSquaredWeightVariance, denoisedWeightedStddev, invWeight,
        MAXENT_TEMPORAL_REPROJECTION_CORRELATION);
    outHistory.denoisedWords = uvec4(packSpecularMaxEnt(denoisedSum),
        floatBitsToUint(denoisedStandardDeviation));
    outHistory.hitDistance *= invWeight;
    outHistory.roughness = clamp(1.0 +
        (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyLength = statisticsReconstructedEffectiveSampleCount(
        sumWeight, weightOverRootSamples);
    float historyEstimatorVariance = statisticsWeightedMeanVariance(
        0.0, weightedEstimatorStddev, invWeight, 1.0);
    outHistory.secondMoment = 0.5
        * statisticsExpectedSquaredNormFromEstimatorVariance(
            signalSum.maxEntY, historyEstimatorVariance,
            outHistory.historyLength);
    outHistory.secondMoment = max(outHistory.secondMoment,
        signalSum.maxEntY.w * signalSum.maxEntY.w);
    outHistory.normalWord = encodeNormalU(maxentSafeNormalize(normalSum * invWeight, currentNormal));
    outHistory.footprintQuality = clamp(validBilinearWeight, 0.0, 1.0);
    outHistory.found = true;
    return outHistory;
}

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

MaxEntReprojectedHistory maxentCombineReprojectedHistories(
        MaxEntReprojectedHistory surface,
        MaxEntReprojectedHistory virtualHistory, float virtualAmount) {
    float surfaceWeight = surface.found ? 1.0 - virtualAmount : 0.0;
    float virtualWeight = virtualHistory.found ? virtualAmount : 0.0;
    if (!surface.found && virtualHistory.found) virtualWeight = 1.0;
    if (surface.found && !virtualHistory.found) surfaceWeight = 1.0;
    float historyWeight = surfaceWeight + virtualWeight;
    if (historyWeight <= 1e-5) return maxentEmptyHistory();

    float inverseHistoryWeight = 1.0 / historyWeight;
    MaxEntReprojectedHistory combined = maxentEmptyHistory();
    SpecularMaxEnt surfaceSignal = unpackSpecularMaxEnt(surface.signalWords);
    SpecularMaxEnt virtualSignal = unpackSpecularMaxEnt(
        virtualHistory.signalWords);
    SpecularMaxEnt combinedSignal = maxentScaleMaxEnt(
        maxentWeightedMaxEnt(surfaceSignal, surfaceWeight,
            virtualSignal, virtualWeight), inverseHistoryWeight);
    combined.signalWords = packSpecularMaxEnt(combinedSignal);

    SpecularMaxEnt surfaceDenoised = unpackSpecularMaxEnt(
        surface.denoisedWords.xyz);
    SpecularMaxEnt virtualDenoised = unpackSpecularMaxEnt(
        virtualHistory.denoisedWords.xyz);
    SpecularMaxEnt denoised = maxentScaleMaxEnt(maxentWeightedMaxEnt(
        surfaceDenoised, surfaceWeight, virtualDenoised, virtualWeight),
        inverseHistoryWeight);
    float surfaceStddev = surface.found
        ? uintBitsToFloat(surface.denoisedWords.w) : 0.0;
    float virtualStddev = virtualHistory.found
        ? uintBitsToFloat(virtualHistory.denoisedWords.w) : 0.0;
    float denoisedStddev = statisticsWeightedMeanStandardDeviation(
        surfaceWeight * surfaceWeight * surfaceStddev * surfaceStddev
            + virtualWeight * virtualWeight * virtualStddev * virtualStddev,
        surfaceWeight * surfaceStddev + virtualWeight * virtualStddev,
        inverseHistoryWeight, MAXENT_SPECULAR_BRANCH_CORRELATION);
    combined.denoisedWords = uvec4(packSpecularMaxEnt(denoised),
        floatBitsToUint(denoisedStddev));
    combined.hitDistance = (surfaceWeight * surface.hitDistance
        + virtualWeight * virtualHistory.hitDistance)
        * inverseHistoryWeight;
    combined.roughness = (surfaceWeight * surface.roughness
        + virtualWeight * virtualHistory.roughness)
        * inverseHistoryWeight;
    float surfaceSamples = max(surface.historyLength, 1.0);
    float virtualSamples = max(virtualHistory.historyLength, 1.0);
    combined.historyLength = statisticsCorrelatedEffectiveSampleCount(
        historyWeight,
        surfaceWeight * surfaceWeight / surfaceSamples
            + virtualWeight * virtualWeight / virtualSamples,
        surfaceWeight * inversesqrt(surfaceSamples)
            + virtualWeight * inversesqrt(virtualSamples),
        MAXENT_SPECULAR_BRANCH_CORRELATION);
    float surfaceEstimatorVariance =
        statisticsEstimatorVarianceFromMoments(
            2.0 * surface.secondMoment, surfaceSignal.maxEntY,
            surfaceSamples);
    float virtualEstimatorVariance =
        statisticsEstimatorVarianceFromMoments(
            2.0 * virtualHistory.secondMoment, virtualSignal.maxEntY,
            virtualSamples);
    float combinedEstimatorVariance = statisticsWeightedMeanVariance(
        surfaceWeight * surfaceWeight * surfaceEstimatorVariance
            + virtualWeight * virtualWeight * virtualEstimatorVariance,
        surfaceWeight * sqrt(surfaceEstimatorVariance)
            + virtualWeight * sqrt(virtualEstimatorVariance),
        inverseHistoryWeight, MAXENT_SPECULAR_BRANCH_CORRELATION);
    combined.secondMoment = 0.5
        * statisticsExpectedSquaredNormFromEstimatorVariance(
            combinedSignal.maxEntY, combinedEstimatorVariance,
            combined.historyLength);
    combined.secondMoment = max(combined.secondMoment,
        combinedSignal.maxEntY.w * combinedSignal.maxEntY.w);
    combined.footprintQuality = (surfaceWeight * surface.footprintQuality
        + virtualWeight * virtualHistory.footprintQuality)
        * inverseHistoryWeight;
    combined.found = statisticsValidEffectiveSampleCount(
        combined.historyLength);
    return combined;
}

void maxentPublishDenoisedReprojection(MaxEntReprojectedHistory history,
        float alphaFloor) {
    if (!history.found) {
        writeMaxEntSpecularDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        return;
    }
    SpecularMaxEnt denoised = unpackSpecularMaxEnt(
        history.denoisedWords.xyz);
    float standardDeviation = uintBitsToFloat(history.denoisedWords.w);
    writeMaxEntSpecularDenoisedReprojection(gl_GlobalInvocationID.xy,
        denoised, standardDeviation, history.hitDistance,
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
        debugWriteSpecularTemporalInvalid(pixel);
        return;
    }

    vec3 currentPos = currentGeometry.position;
    vec3 currentNormal = currentGeometry.normal;
    uint currentMaterial = currentGeometry.materialID;
    float currentRoughness = currentGeometry.roughness;

    vec3 cameraDelta = camPos - prevRaytracingCamPos;
    vec3 surfaceMotion;
    float motionValid;
    readSurfaceMotion(pixel, surfaceMotion, motionValid);
    cameraDelta -= surfaceMotion;

    vec2 surfaceUv = maxentProjectPrevious(currentPos, cameraDelta);
    MaxEntReprojectedHistory surface = maxentLoadHistory(surfaceUv, pixel,
        currentPos, currentNormal, currentMaterial, cameraDelta,
        false, true);

    // Use a stable 3x3 hit-distance choice for virtual reprojection.
    float focusedHitDistance = noisy.hitDistance > 0.0
        ? noisy.hitDistance : 1e30;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 q = ivec2(pixel) + ivec2(x, y);
            if (!maxentInBounds(q, ivec2(resolution_global))) continue;
            float qHit = maxentUnpackSpecularInput(
                texelFetch(colortex6, q, 0)).hitDistance;
            if (qHit > 0.0) focusedHitDistance = min(focusedHitDistance, qHit);
        }
    }
    if (focusedHitDistance == 1e30) focusedHitDistance = 0.0;

    float surfaceDistance = length(currentPos);
    vec3 primaryRay = maxentSafeNormalize(currentPos, vec3(0.0, 0.0, 1.0));
    vec3 V = -primaryRay;
    float NoV = abs(dot(currentNormal, V));
    float virtualScale = maxentDominantFactor(NoV, currentRoughness);
    vec3 virtualPoint = primaryRay *
        (surfaceDistance + focusedHitDistance * virtualScale);
    vec2 virtualUv = maxentProjectPrevious(virtualPoint, cameraDelta);
    MaxEntReprojectedHistory virtualHistory = maxentLoadHistory(virtualUv,
        pixel, currentPos, currentNormal, currentMaterial, cameraDelta,
        true, false);
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
        float lobeRadiusPixels = min(focusedHitDistance,
            virtualHistory.hitDistance) * lobeTan
            * float(resolution_global.y) / max(surfaceDistance, 1e-4);
        hitConfidence *= 1.0 - smoothstep(0.0,
            lobeRadiusPixels + 0.25, uvErrorPixels);
    }

    float virtualAmount = virtualHistory.found
        ? virtualScale * virtualHistory.footprintQuality : 0.0;
    virtualAmount *= clamp(vmbConfidence / max(smbConfidence, 1e-6), 0.0, 1.0);
    virtualAmount *= hitConfidence;
    virtualAmount = clamp(virtualAmount, 0.0, 1.0);

    MaxEntReprojectedHistory history = maxentCombineReprojectedHistories(
        surface, virtualHistory, virtualAmount);
    float surfaceBranchWeight = surface.found ? 1.0 - virtualAmount : 0.0;
    float virtualBranchWeight = virtualHistory.found ? virtualAmount : 0.0;
    if (!surface.found && virtualHistory.found) virtualBranchWeight = 1.0;
    if (surface.found && !virtualHistory.found) surfaceBranchWeight = 1.0;
    float branchWeightSum = surfaceBranchWeight + virtualBranchWeight;
    float historyConfidence = branchWeightSum > 1e-5
        ? (surfaceBranchWeight * smbConfidence
            + virtualBranchWeight * vmbConfidence * hitConfidence)
            / branchWeightSum
        : 0.0;

    MaxEntTemporalSignal temporal;
    if (history.found) {
        float alphaFloor = 1.0 - clamp(historyConfidence, 0.0, 1.0);
        // Keep the provisional estimator identical to the fixed-alpha update
        // performed by final resolve.
        float proposalAlpha = clamp(
            float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
        SpecularMaxEnt historySignal = unpackSpecularMaxEnt(
            history.signalWords);
        temporal.signal = maxentMixMaxEnt(historySignal, noisy.signal,
            proposalAlpha);
        temporal.rootMeanY2 = sqrt(max(mix(history.secondMoment, noisyM2,
            proposalAlpha), 0.0));
        temporal.historyLength = statisticsKishUpdateEffectiveSampleCount(
            history.historyLength, proposalAlpha);
        maxentPublishDenoisedReprojection(history, alphaFloor);
    } else {
        temporal.signal = noisy.signal;
        temporal.rootMeanY2 = sqrt(max(noisyM2, 0.0));
        temporal.historyLength = 1.0;
        writeMaxEntSpecularDenoisedReprojectionInvalid(pixel);
    }

    MaxEntTemporalSignal reprojectedTemporal;
    reprojectedTemporal.signal = history.found
        ? unpackSpecularMaxEnt(history.signalWords)
        : emptySpecularMaxEnt();
    reprojectedTemporal.rootMeanY2 = history.found
        ? sqrt(max(history.secondMoment, 0.0)) : 0.0;
    reprojectedTemporal.historyLength = history.found
        ? history.historyLength : 0.0;
    imageStore(colorimg4, ivec2(pixel), maxentPackTemporal(temporal));
    imageStore(colorimg5, ivec2(pixel), history.found
        ? maxentPackTemporal(reprojectedTemporal) : uvec4(0u));
    debugWriteSpecularTemporal(pixel, packSpecularMaxEnt(temporal.signal),
        noisy.hitDistance, 0.0);
}
