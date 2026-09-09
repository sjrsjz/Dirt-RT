#ifndef REFLECTION_HISTORY_REPROJECTION_GLSL
#define REFLECTION_HISTORY_REPROJECTION_GLSL

// Reconstruct raw and filtered history from the same accepted bilinear taps.
// The including temporal pass owns bindings and commits no persistent history.
struct MaxEntReprojectedHistory {
    uvec3 signalWords;
    float meanY2;
    uvec4 denoisedWords;
    float hitDistance;
    uint normalWord;
    float roughness;
    float historyEffectiveSamples;
    float denoisedEffectiveSamples;
    float footprintQuality;
    bool found;
};

MaxEntReprojectedHistory maxentEmptyHistory() {
    MaxEntReprojectedHistory h;
    h.signalWords = uvec3(0u);
    h.meanY2 = 0.0;
    h.denoisedWords = uvec4(0u);
    h.hitDistance = 0.0;
    h.normalWord = encodeNormalU(vec3(0.0, 1.0, 0.0));
    h.roughness = 1.0;
    h.historyEffectiveSamples = 0.0;
    h.denoisedEffectiveSamples = 0.0;
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
    bool requireSurfaceFootprint,
    bool loadDenoisedHistory
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
    float weightedMeanY2 = 0.0;
    float validBilinearWeight = 0.0;
    SpecularMaxEnt signalSum = emptySpecularMaxEnt();
    SpecularMaxEnt denoisedSum = emptySpecularMaxEnt();
    DenoiserEstimatorVarianceAccumulator denoisedUncertainty = denoiserBeginEstimatorVariance();
    vec3 normalSum = vec3(0.0);
    float depthThreshold = MAXENT_SPECULAR_TEMPORAL_DISOCCLUSION_THRESHOLD *
        max(length(currentSurfacePosition), 1.0);

    for (int i = 0; i < 4; ++i) {
        float w = bilinear[i];
        // Zero bilinear mass cannot affect either coverage or the estimator.
        if (w <= 0.0) continue;
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (!maxentInBounds(p, size)) continue;
        uvec2 historyPixel = uvec2(p);
        uvec4 geometryWords = reflectBuffer.data[addr(SPEC_N_HISTGEO, historyPixel)];
        uvec4 signalWords = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, historyPixel)];
        uvec4 denoisedWords = loadDenoisedHistory
            ? reflectBuffer.data[addr(SPEC_N_HISTMETA, historyPixel)]
            : uvec4(0u);
        float surfaceDistance = uintBitsToFloat(geometryWords.x);
        vec2 momentHistory = unpackHalf2x16(signalWords.w);
        vec2 denoisedMetadata = unpackHalf2x16(denoisedWords.w);
        bool denoisedTapValid = !loadDenoisedHistory
            || ((denoisedMetadata.y >= 1.0)
                && denoiserSigmaUsable(denoisedMetadata.x)
                && !any(isnan(denoisedMetadata))
                && !any(isinf(denoisedMetadata)));
        if (!(surfaceDistance >= 0.0) || isnan(surfaceDistance) || isinf(surfaceDistance)
                || !(momentHistory.x >= 0.0)
                || !(momentHistory.y >= 1.0) || !denoisedTapValid
                || any(isnan(momentHistory)) || any(isinf(momentHistory)))
            continue;

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

        vec4 rawTap = vec4(unpackHalf2x16(signalWords.x), unpackHalf2x16(signalWords.y));
        vec2 rawChroma = unpackHalf2x16(signalWords.z);
        vec4 filteredTap = vec4(unpackHalf2x16(denoisedWords.x), unpackHalf2x16(denoisedWords.y));
        vec2 filteredChroma = unpackHalf2x16(denoisedWords.z);
        bool filteredFinite = !loadDenoisedHistory
            || (!any(isnan(filteredTap)) && !any(isinf(filteredTap))
                && !any(isnan(filteredChroma))
                && !any(isinf(filteredChroma)));
        if (!denoiserTemporalMomentsFinite(rawTap, rawChroma,
                momentHistory.x) || !filteredFinite)
            continue;
        validBilinearWeight += w;
        signalSum.maxEntY += rawTap * w;
        signalSum.CoCg += rawChroma * w;
        if (loadDenoisedHistory) {
            denoisedSum.maxEntY += filteredTap * w;
            denoisedSum.CoCg += filteredChroma * w;
            denoiserAccumulateEstimatorVariance(
                denoisedUncertainty, denoisedMetadata.x, w);
        }
        vec2 hitRoughness = unpackHalf2x16(geometryWords.w);
        outHistory.hitDistance += hitRoughness.x * w;
        outHistory.roughness += (hitRoughness.y - 1.0) * w;
        float samples = momentHistory.y;
        float tapSecondMoment = momentHistory.x * momentHistory.x;
        weightedMeanY2 += w * tapSecondMoment;
        weightOverRootSamples += w * inversesqrt(samples);
        normalSum += historyNormal * w;
        sumWeight += w;
    }

    // Full-footprint validation concerns contributing bilinear mass, not the
    // four array entries. At an exact texel coordinate three taps have zero
    // weight and must not invalidate a valid mirror history at an edge.
    bool accepted = requireFullFootprint ? validBilinearWeight > 0.999 : sumWeight > 1e-5;
    if (!accepted || sumWeight <= 1e-5) return maxentEmptyHistory();

    float invWeight = 1.0 / sumWeight;
    signalSum = maxentScaleMaxEnt(signalSum, invWeight);
    outHistory.signalWords = packSpecularMaxEnt(signalSum);
    if (loadDenoisedHistory) {
        denoisedSum = maxentScaleMaxEnt(denoisedSum, invWeight);
        float denoisedMonteCarloStandardDeviation =
            denoiserResolveEstimatorSigma(denoisedUncertainty, 1.0);
        outHistory.denoisedWords = uvec4(packSpecularMaxEnt(denoisedSum),
            floatBitsToUint(denoisedMonteCarloStandardDeviation));
    }
    outHistory.hitDistance *= invWeight;
    outHistory.roughness = clamp(1.0 +
        (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyEffectiveSamples = statisticsReconstructedEffectiveSampleCount(
        sumWeight, weightOverRootSamples);
    outHistory.denoisedEffectiveSamples = 1.0; // Reserved legacy metadata.
    outHistory.meanY2 = weightedMeanY2 * invWeight;
    outHistory.normalWord = encodeNormalU(maxentSafeNormalize(normalSum * invWeight, currentNormal));
    outHistory.footprintQuality = clamp(validBilinearWeight, 0.0, 1.0);
    outHistory.found = statisticsValidEffectiveSampleCount(outHistory.historyEffectiveSamples)
        && statisticsValidEffectiveSampleCount(outHistory.denoisedEffectiveSamples);
    return outHistory;
}

MaxEntReprojectedHistory maxentCombineReprojectedHistories(
        MaxEntReprojectedHistory surface,
        MaxEntReprojectedHistory virtualHistory, float virtualAmount) {
    float surfaceWeight = surface.found ? 1.0 - virtualAmount : 0.0;
    float virtualWeight = virtualHistory.found ? virtualAmount : 0.0;
    if (!surface.found && virtualHistory.found) virtualWeight = 1.0;
    if (surface.found && !virtualHistory.found) surfaceWeight = 1.0;
    float historyMass = surfaceWeight + virtualWeight;
    if (historyMass <= 1e-5) return maxentEmptyHistory();

    float inverseHistoryWeight = 1.0 / historyMass;
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
    float surfaceMonteCarloStandardDeviation = surface.found
        ? uintBitsToFloat(surface.denoisedWords.w) : 0.0;
    float virtualMonteCarloStandardDeviation = virtualHistory.found
        ? uintBitsToFloat(virtualHistory.denoisedWords.w) : 0.0;
    float combinedSigma = denoiserMixEstimatorSigma(surfaceMonteCarloStandardDeviation,
        virtualMonteCarloStandardDeviation, virtualWeight * inverseHistoryWeight,
        MAXENT_SPECULAR_BRANCH_CORRELATION);
    combined.denoisedWords = uvec4(packSpecularMaxEnt(denoised),
        floatBitsToUint(combinedSigma));
    combined.hitDistance = (surfaceWeight * surface.hitDistance
        + virtualWeight * virtualHistory.hitDistance)
        * inverseHistoryWeight;
    combined.roughness = (surfaceWeight * surface.roughness
        + virtualWeight * virtualHistory.roughness)
        * inverseHistoryWeight;
    float surfaceEffectiveSamples = max(surface.historyEffectiveSamples, 1.0);
    float virtualEffectiveSamples = max(virtualHistory.historyEffectiveSamples, 1.0);
    combined.historyEffectiveSamples = statisticsCorrelatedEffectiveSampleCount(
        historyMass,
        surfaceWeight * surfaceWeight / surfaceEffectiveSamples
            + virtualWeight * virtualWeight / virtualEffectiveSamples,
        surfaceWeight * inversesqrt(surfaceEffectiveSamples)
            + virtualWeight * inversesqrt(virtualEffectiveSamples),
        MAXENT_SPECULAR_BRANCH_CORRELATION);
    combined.denoisedEffectiveSamples = 1.0;
    combined.meanY2 = (surfaceWeight * surface.meanY2 + virtualWeight * virtualHistory.meanY2) * inverseHistoryWeight;
    combined.footprintQuality = (surfaceWeight * surface.footprintQuality
        + virtualWeight * virtualHistory.footprintQuality)
        * inverseHistoryWeight;
    combined.found = statisticsValidEffectiveSampleCount(combined.historyEffectiveSamples)
        && statisticsValidEffectiveSampleCount(combined.denoisedEffectiveSamples);
    return combined;
}

#endif
