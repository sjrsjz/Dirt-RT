#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"
#include "/lib/lighting/denoiser/maxent_temporal_statistics.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

struct MaxEntReprojectedHistory {
    SpecularMaxEnt slowSignal;
    float secondMoment;
    SpecularMaxEnt responsiveSignal;
    float hitDistance;
    vec3 normal;
    float roughness;
    float historyLength;
    float responsiveHistoryLength;
    float historyEvidence;
    float responsiveHistoryEvidence;
    float footprintQuality;
    bool found;
};

MaxEntReprojectedHistory maxentEmptyHistory() {
    MaxEntReprojectedHistory h;
    h.slowSignal = emptySpecularMaxEnt();
    h.secondMoment = 0.0;
    h.responsiveSignal = emptySpecularMaxEnt();
    h.hitDistance = 0.0;
    h.normal = vec3(0.0, 1.0, 0.0);
    h.roughness = 1.0;
    h.historyLength = 0.0;
    h.responsiveHistoryLength = 0.0;
    h.historyEvidence = 0.0;
    h.responsiveHistoryEvidence = 0.0;
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
        vec2(MAXENT_TEMPORAL_REPROJECTION_RADIUS + 1e-5)));
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
    float slowWeightOverSamples = 0.0;
    float responsiveWeightOverSamples = 0.0;
    float slowSampleMass = 0.0;
    float responsiveSampleMass = 0.0;
    float validBilinearWeight = 0.0;
    int validTapCount = 0;
    vec3 normalSum = vec3(0.0);
    float depthThreshold = MAXENT_SPECULAR_TEMPORAL_DISOCCLUSION_THRESHOLD *
        max(length(currentSurfacePosition), 1.0);

    for (int i = 0; i < 4; ++i) {
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (!maxentInBounds(p, size)) continue;
        MaxEntSpecularHistory h = readMaxEntSpecularHistory(uvec2(p));
        if (h.historyLength < 0.5 || h.materialID != currentMaterial) continue;

        vec3 previousSurfaceCurrent = h.surfacePosition - cameraDelta;
        if (abs(dot(previousSurfaceCurrent - currentSurfacePosition,
                currentNormal)) > depthThreshold)
            continue;
        // Geometry normals only provide a hard topology/silhouette guard.
        // They never attenuate a valid history sample.
        if (dot(currentNormal, h.geometryNormal) <= 0.0) continue;
        if (requireSurfaceFootprint && !maxentSurfaceFootprintContains(
                currentPixel, previousSurfaceCurrent))
            continue;

        float w = bilinear[i];
        validBilinearWeight += w;
        ++validTapCount;
        if (w <= 0.0) continue;
        outHistory.slowSignal.maxEntY += h.slowSignal.maxEntY * w;
        outHistory.slowSignal.CoCg += h.slowSignal.CoCg * w;
        outHistory.secondMoment += h.secondMoment * w;
        outHistory.responsiveSignal.maxEntY +=
            h.responsiveSignal.maxEntY * w;
        outHistory.responsiveSignal.CoCg +=
            h.responsiveSignal.CoCg * w;
        outHistory.hitDistance += h.hitDistance * w;
        outHistory.roughness += (h.roughness - 1.0) * w;
        float slowSamples = max(h.historyLength, 1.0);
        float responsiveSamples = max(h.responsiveHistoryLength, 1.0);
        slowWeightOverSamples += w / slowSamples;
        responsiveWeightOverSamples += w / responsiveSamples;
        slowSampleMass += w * slowSamples;
        responsiveSampleMass += w * responsiveSamples;
        normalSum += h.geometryNormal * w;
        sumWeight += w;
    }

    bool accepted = requireFullFootprint
        ? (validTapCount == 4 && validBilinearWeight > 0.999)
        : (sumWeight > 1e-5);
    if (!accepted || sumWeight <= 1e-5) return maxentEmptyHistory();

    float invWeight = 1.0 / sumWeight;
    outHistory.slowSignal.maxEntY *= invWeight;
    outHistory.slowSignal.CoCg *= invWeight;
    outHistory.slowSignal = sanitizeSpecularMaxEnt(outHistory.slowSignal);
    outHistory.secondMoment *= invWeight;
    outHistory.responsiveSignal.maxEntY *= invWeight;
    outHistory.responsiveSignal.CoCg *= invWeight;
    outHistory.responsiveSignal = sanitizeSpecularMaxEnt(
        outHistory.responsiveSignal);
    outHistory.hitDistance *= invWeight;
    outHistory.roughness = clamp(1.0 +
        (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyLength = maxentTemporalReprojectedEffectiveSamples(
        sumWeight, slowWeightOverSamples,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY));
    outHistory.responsiveHistoryLength =
        maxentTemporalReprojectedEffectiveSamples(sumWeight,
            responsiveWeightOverSamples,
            float(MAXENT_SPECULAR_TEMPORAL_MAX_FAST_HISTORY));
    // History evidence controls temporal reuse.  Unlike the statistical ESS,
    // it is an arithmetic reconstruction of the contributing histories and
    // therefore cannot acquire a texel-phase-dependent 2x/4x gain.
    outHistory.historyEvidence = clamp(slowSampleMass * invWeight,
        0.0, float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY));
    outHistory.responsiveHistoryEvidence = clamp(
        responsiveSampleMass * invWeight,
        0.0, float(MAXENT_SPECULAR_TEMPORAL_MAX_FAST_HISTORY));
    outHistory.normal = maxentSafeNormalize(normalSum * invWeight, currentNormal);
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

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    MaxEntPrepassSignal noisy = maxentUnpackPrepass(
        texelFetch(colortex6, ivec2(pixel), 0));
    float noisyY = noisy.signal.maxEntY.w;
    float noisyM2 = noisyY * noisyY;

    MaxEntGeometry currentGeometry = maxentLoadGeometry(pixel);
    if (!currentGeometry.valid) {
        imageStore(colorimg4, ivec2(pixel), uvec4(0u));
        imageStore(colorimg5, ivec2(pixel), uvec4(0u));
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
            float qHit = maxentUnpackPrepass(texelFetch(colortex6, q, 0)).hitDistance;
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

    float surfaceSlowEvidence = min(surface.historyEvidence,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY));
    float virtualSlowEvidence = min(virtualHistory.historyEvidence,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY));
    float surfaceResponsiveEvidence = min(
        surface.responsiveHistoryEvidence,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_FAST_HISTORY));
    float virtualResponsiveEvidence = min(
        virtualHistory.responsiveHistoryEvidence,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_FAST_HISTORY));

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
        vmbConfidence *= float(dot(virtualHistory.normal, currentNormal) > 0.0);

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

    float smbSlowAlpha = surface.found
        ? max(1.0 - smbConfidence,
            1.0 / (1.0 + surfaceSlowEvidence)) : 1.0;
    float smbFastAlpha = surface.found
        ? max(smbSlowAlpha,
            1.0 / (1.0 + surfaceResponsiveEvidence)) : 1.0;
    float vmbSlowAlpha = virtualHistory.found
        ? max(1.0 - vmbConfidence,
            1.0 / (1.0 + virtualSlowEvidence)) : 1.0;
    float vmbFastAlpha = virtualHistory.found
        ? max(1.0 - vmbConfidence * hitConfidence,
            1.0 / (1.0 + virtualResponsiveEvidence)) : 1.0;
    float vmbHitAlpha = virtualHistory.found
        ? max(1.0 - vmbConfidence * hitConfidence,
            max(0.1, 1.0 / (1.0 + virtualSlowEvidence))) : 1.0;

    SpecularMaxEnt slowSMB = surface.found
        ? maxentMixMaxEnt(surface.slowSignal, noisy.signal, smbSlowAlpha)
        : noisy.signal;
    SpecularMaxEnt slowVMB = virtualHistory.found
        ? maxentMixMaxEnt(virtualHistory.slowSignal, noisy.signal, vmbSlowAlpha)
        : noisy.signal;
    SpecularMaxEnt fastSMB = surface.found
        ? maxentMixMaxEnt(surface.responsiveSignal, noisy.signal, smbFastAlpha)
        : noisy.signal;
    SpecularMaxEnt fastVMB = virtualHistory.found
        ? maxentMixMaxEnt(virtualHistory.responsiveSignal, noisy.signal,
            vmbFastAlpha) : noisy.signal;
    float m2SMB = surface.found ? mix(surface.secondMoment,
        noisyM2, smbSlowAlpha) : noisyM2;
    float m2VMB = virtualHistory.found ? mix(virtualHistory.secondMoment,
        noisyM2, vmbSlowAlpha) : noisyM2;
    float hitSMB = surface.found ? mix(surface.hitDistance,
        noisy.hitDistance, max(smbSlowAlpha, 0.1)) : noisy.hitDistance;
    float hitVMB = virtualHistory.found ? mix(virtualHistory.hitDistance,
        noisy.hitDistance, vmbHitAlpha) : noisy.hitDistance;

    float virtualAmount = virtualHistory.found
        ? virtualScale * virtualHistory.footprintQuality : 0.0;
    virtualAmount *= clamp(vmbConfidence / max(smbConfidence, 1e-6), 0.0, 1.0);
    virtualAmount = clamp(virtualAmount, 0.0, 1.0);

    MaxEntSlowSignal slow;
    slow.signal = maxentMixMaxEnt(slowSMB, slowVMB, virtualAmount);
    slow.secondMoment = mix(m2SMB, m2VMB, virtualAmount);
    slow.historyLength = maxentTemporalSharedCurrentEffectiveSamples(
        surface.historyLength, smbSlowAlpha, surface.found,
        virtualHistory.historyLength, vmbSlowAlpha,
        virtualHistory.found, virtualAmount,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY));

    MaxEntFastSignal fast;
    fast.signal = maxentMixMaxEnt(fastSMB, fastVMB, virtualAmount);
    fast.hitDistance = mix(hitSMB, hitVMB, virtualAmount);
    fast.historyLength = maxentTemporalSharedCurrentEffectiveSamples(
        surface.responsiveHistoryLength, smbFastAlpha, surface.found,
        virtualHistory.responsiveHistoryLength, vmbFastAlpha,
        virtualHistory.found, virtualAmount,
        float(MAXENT_SPECULAR_TEMPORAL_MAX_FAST_HISTORY));

    imageStore(colorimg4, ivec2(pixel), maxentPackSlow(slow));
    imageStore(colorimg5, ivec2(pixel), maxentPackFast(fast));

#if DEBUG_VIEW == 12
    writeReflLight(pixel, specularMaxEntTotalRgb(slow.signal),
        fast.hitDistance, 1.0 - mix(smbSlowAlpha, vmbSlowAlpha,
            virtualAmount));
#elif DEBUG_VIEW == 14
    writeReflLight(pixel, vec3(1.0 - mix(smbSlowAlpha, vmbSlowAlpha,
        virtualAmount)),
        fast.hitDistance, 0.0);
#endif
}
