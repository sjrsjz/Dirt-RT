#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

struct RelaxReprojectedHistory {
    SpecularMaxEnt slowSignal;
    float secondMoment;
    vec3 fastYCoCg;
    float hitDistance;
    vec3 normal;
    float roughness;
    float historyLength;
    float confidence;
    float footprintQuality;
    bool found;
};

RelaxReprojectedHistory relaxEmptyHistory() {
    RelaxReprojectedHistory h;
    h.slowSignal = emptySpecularMaxEnt();
    h.secondMoment = 0.0;
    h.fastYCoCg = vec3(0.0);
    h.hitDistance = 0.0;
    h.normal = vec3(0.0, 1.0, 0.0);
    h.roughness = 1.0;
    h.historyLength = 0.0;
    h.confidence = 0.0;
    h.footprintQuality = 0.0;
    h.found = false;
    return h;
}

bool relaxSurfaceFootprintContains(uvec2 currentPixel,
        vec3 historyPositionCurrentSpace) {
    vec4 clip = rtViewProjection * vec4(historyPositionCurrentSpace, 1.0);
    if (clip.w <= 1e-7 || any(isnan(clip)) || any(isinf(clip))) return false;
    vec2 projectedPixel = (clip.xy / clip.w * 0.5 + 0.5) *
        vec2(resolution_global);
    return all(lessThanEqual(abs(projectedPixel - vec2(currentPixel)),
        vec2(TEMPORAL_CLIP_PIXEL_RADIUS + 1e-5)));
}

RelaxReprojectedHistory relaxLoadHistory(
    vec2 uv,
    uvec2 currentPixel,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    vec3 cameraDelta,
    bool requireFullFootprint,
    bool requireSurfaceFootprint
) {
    RelaxReprojectedHistory outHistory = relaxEmptyHistory();
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))))
        return outHistory;

    ivec2 size = ivec2(resolution_global);
    vec2 pixelPosition = uv * vec2(size);
    ivec2 origin = ivec2(floor(pixelPosition));
    vec2 f = fract(pixelPosition);
    vec4 bilinear = vec4((1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    float sumWeight = 0.0;
    float validBilinearWeight = 0.0;
    int validTapCount = 0;
    vec3 normalSum = vec3(0.0);
    float depthThreshold = RELAX_DISOCCLUSION_THRESHOLD *
        max(length(currentSurfacePosition), 1.0);

    for (int i = 0; i < 4; ++i) {
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (!relaxInBounds(p, size)) continue;
        RelaxSpecularHistory h = readRelaxSpecularHistory(uvec2(p));
        if (h.historyLength < 0.5 || h.materialID != currentMaterial) continue;

        vec3 previousSurfaceCurrent = h.surfacePosition - cameraDelta;
        if (abs(dot(previousSurfaceCurrent - currentSurfacePosition,
                currentNormal)) > depthThreshold)
            continue;
        // Geometry normals only provide a hard topology/silhouette guard.
        // They never attenuate a valid history sample.
        if (dot(currentNormal, h.geometryNormal) <= 0.0) continue;
        if (requireSurfaceFootprint && !relaxSurfaceFootprintContains(
                currentPixel, previousSurfaceCurrent))
            continue;

        float w = bilinear[i];
        validBilinearWeight += w;
        ++validTapCount;
        if (w <= 0.0) continue;
        outHistory.slowSignal.aliceY += h.slowSignal.aliceY * w;
        outHistory.slowSignal.CoCg += h.slowSignal.CoCg * w;
        outHistory.secondMoment += h.secondMoment * w;
        outHistory.fastYCoCg += h.responsiveYCoCg * w;
        outHistory.hitDistance += h.hitDistance * w;
        outHistory.roughness += (h.roughness - 1.0) * w;
        outHistory.historyLength += h.historyLength * w;
        outHistory.confidence += h.reprojectionConfidence * w;
        normalSum += h.geometryNormal * w;
        sumWeight += w;
    }

    bool accepted = requireFullFootprint
        ? (validTapCount == 4 && validBilinearWeight > 0.999)
        : (sumWeight > 1e-5);
    if (!accepted || sumWeight <= 1e-5) return relaxEmptyHistory();

    float invWeight = 1.0 / sumWeight;
    outHistory.slowSignal.aliceY *= invWeight;
    outHistory.slowSignal.CoCg *= invWeight;
    outHistory.slowSignal = sanitizeSpecularMaxEnt(outHistory.slowSignal);
    outHistory.secondMoment *= invWeight;
    outHistory.fastYCoCg *= invWeight;
    outHistory.hitDistance *= invWeight;
    outHistory.roughness = clamp(1.0 +
        (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyLength *= invWeight;
    outHistory.confidence *= invWeight;
    outHistory.normal = relaxSafeNormalize(normalSum * invWeight, currentNormal);
    outHistory.footprintQuality = clamp(validBilinearWeight, 0.0, 1.0);
    outHistory.found = true;
    return outHistory;
}

RelaxReprojectedHistory relaxMixHistory(RelaxReprojectedHistory a,
        RelaxReprojectedHistory b, float t) {
    if (!a.found) return b;
    if (!b.found) return a;
    RelaxReprojectedHistory h = a;
    h.slowSignal = relaxMixMaxEnt(a.slowSignal, b.slowSignal, t);
    h.secondMoment = mix(a.secondMoment, b.secondMoment, t);
    h.fastYCoCg = mix(a.fastYCoCg, b.fastYCoCg, t);
    h.hitDistance = mix(a.hitDistance, b.hitDistance, t);
    h.roughness = mix(a.roughness, b.roughness, t);
    h.historyLength = mix(a.historyLength, b.historyLength, t);
    h.confidence = mix(a.confidence, b.confidence, t);
    h.footprintQuality = mix(a.footprintQuality, b.footprintQuality, t);
    h.found = true;
    return h;
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    RelaxPrepassSignal noisy = relaxUnpackPrepass(
        texelFetch(colortex6, ivec2(pixel), 0));
    float noisyY = noisy.signal.aliceY.w;
    float noisyM2 = noisyY * noisyY;

    vec3 currentPos;
    float primaryDistance;
    readGeo0(GEO_N_GEO, pixel, currentPos, primaryDistance);
    if (primaryDistance < -0.5) {
        imageStore(colorimg4, ivec2(pixel), uvec4(0u));
        imageStore(colorimg5, ivec2(pixel), uvec4(0u));
        return;
    }

    vec3 currentNormal;
    float currentAlpha, currentPathRoughness;
    int currentMaterialInt;
    readGeo1(GEO_N_NORMALS, pixel, currentNormal, currentAlpha,
        currentMaterialInt, currentPathRoughness);
    uint currentMaterial = uint(max(currentMaterialInt, 0));
    float currentRoughness = relaxPerceptualRoughness(currentAlpha);

    vec3 cameraDelta = camPos - prevRaytracingCamPos;
    vec3 surfaceMotion;
    float motionValid;
    readSurfaceMotion(pixel, surfaceMotion, motionValid);
    cameraDelta -= surfaceMotion;

    vec2 surfaceUv = relaxProjectPrevious(currentPos, cameraDelta);
    RelaxReprojectedHistory surface = relaxLoadHistory(surfaceUv, pixel,
        currentPos, currentNormal, currentMaterial, cameraDelta,
        false, true);

    // Classical hit-distance virtual reprojection, constrained to the primary
    // view ray. Under a pure camera rotation every positive scalar multiple
    // projects to the same UV, so roughness and sampled normals cannot twist
    // the history address.
    float surfaceDistance = length(currentPos);
    vec3 primaryRay = relaxSafeNormalize(currentPos, vec3(0.0, 0.0, 1.0));
    float virtualScale = 1.0 - currentRoughness;
    virtualScale *= virtualScale;
    vec3 virtualPoint = primaryRay *
        (surfaceDistance + noisy.hitDistance * virtualScale);
    vec2 virtualUv = relaxProjectPrevious(virtualPoint, cameraDelta);
    RelaxReprojectedHistory virtualHistory = relaxLoadHistory(virtualUv,
        pixel, currentPos, currentNormal, currentMaterial, cameraDelta,
        true, false);

    if (motionValid < 0.5) {
        surface.found = false;
        virtualHistory.found = false;
    }

    float roughnessAgreement = 1.0;
    if (virtualHistory.found)
        roughnessAgreement = exp(-8.0 * abs(
            virtualHistory.roughness - currentRoughness));
    float virtualAmount = virtualHistory.found
        ? virtualScale * virtualHistory.footprintQuality * roughnessAgreement
        : 0.0;
    RelaxReprojectedHistory history = relaxMixHistory(
        surface, virtualHistory, clamp(virtualAmount, 0.0, 1.0));

    float historyLength = history.found ? history.historyLength : 0.0;
    float confidence = history.found
        ? history.footprintQuality * mix(1.0, history.confidence, 0.5)
        : 0.0;
    if (history.found)
        confidence *= exp(-8.0 * abs(history.roughness - currentRoughness));
    confidence = clamp(confidence, 0.0, 1.0);

    float slowFrames = min(historyLength, float(RELAX_SPEC_MAX_HISTORY));
    float fastFrames = min(historyLength, float(RELAX_SPEC_MAX_FAST_HISTORY));
    float slowAlpha = history.found
        ? max(1.0 - confidence, 1.0 / (1.0 + slowFrames)) : 1.0;
    float fastAlpha = history.found
        ? max(1.0 - confidence, 1.0 / (1.0 + fastFrames)) : 1.0;

    RelaxSlowSignal slow;
    slow.signal = history.found
        ? relaxMixMaxEnt(history.slowSignal, noisy.signal, slowAlpha)
        : noisy.signal;
    slow.secondMoment = history.found
        ? mix(history.secondMoment, noisyM2, slowAlpha) : noisyM2;

    RelaxFastSignal fast;
    vec3 noisyYCoCg = relaxMaxEntYCoCg(noisy.signal);
    fast.YCoCg = history.found
        ? mix(history.fastYCoCg, noisyYCoCg, fastAlpha) : noisyYCoCg;
    fast.hitDistance = history.found
        ? mix(history.hitDistance, noisy.hitDistance, slowAlpha)
        : noisy.hitDistance;
    fast.historyLength = min(historyLength + 1.0,
        float(RELAX_SPEC_MAX_HISTORY));
    fast.confidence = confidence;
    fast.materialID = currentMaterial;

    imageStore(colorimg4, ivec2(pixel), relaxPackSlow(slow));
    imageStore(colorimg5, ivec2(pixel), relaxPackFast(fast));

#if DEBUG_VIEW == 12
    writeReflLight(pixel, specularMaxEntTotalRgb(slow.signal),
        fast.hitDistance, history.found ? 1.0 - slowAlpha : 0.0);
#elif DEBUG_VIEW == 14
    writeReflLight(pixel, vec3(history.found ? 1.0 - slowAlpha : 0.0),
        fast.hitDistance, 0.0);
#endif
}
