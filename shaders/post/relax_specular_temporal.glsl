#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

struct RelaxReprojectedHistory {
    vec3 slowRadiance;
    float secondMoment;
    vec3 fastRadiance;
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
    h.slowRadiance = vec3(0.0);
    h.secondMoment = 0.0;
    h.fastRadiance = vec3(0.0);
    h.hitDistance = 0.0;
    h.normal = vec3(0.0, 1.0, 0.0);
    h.roughness = 1.0;
    h.historyLength = 0.0;
    h.confidence = 0.0;
    h.footprintQuality = 0.0;
    h.found = false;
    return h;
}

RelaxReprojectedHistory relaxLoadHistory(
    vec2 uv,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    vec3 cameraDelta,
    bool requireFullFootprint
) {
    RelaxReprojectedHistory outHistory = relaxEmptyHistory();
    ivec2 size = ivec2(resolution_global);
    vec2 pixelPosition = uv * vec2(size);
    ivec2 origin = ivec2(floor(pixelPosition));
    vec2 f = fract(pixelPosition);
    vec4 bilinear = vec4(
        (1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y, f.x * f.y);

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

        vec3 previousSurfaceCurrentSpace = h.surfacePosition - cameraDelta;
        float planeDistance = abs(dot(
            previousSurfaceCurrentSpace - currentSurfacePosition,
            currentNormal));
        if (planeDistance > depthThreshold) continue;
        if (dot(currentNormal, h.geometryNormal) <= 0.0) continue;

        float w = bilinear[i];
        validBilinearWeight += w;
        ++validTapCount;
        if (w <= 0.0) continue;

        outHistory.slowRadiance += h.slowRadiance * w;
        outHistory.secondMoment += h.secondMoment * w;
        outHistory.fastRadiance += h.responsiveRadiance * w;
        outHistory.hitDistance += h.hitDistance * w;
        outHistory.roughness += (h.roughness - 1.0) * w;
        outHistory.historyLength += h.historyLength * w;
        outHistory.confidence += h.reprojectionConfidence * w;
        normalSum += h.geometryNormal * w;
        sumWeight += w;
    }

    outHistory.footprintQuality = clamp(validBilinearWeight, 0.0, 1.0);
    bool footprintAccepted = requireFullFootprint
        ? (validTapCount == 4 && validBilinearWeight > 0.999)
        : (sumWeight > 1e-5);
    if (!footprintAccepted || sumWeight <= 1e-5) return relaxEmptyHistory();

    float invWeight = 1.0 / sumWeight;
    outHistory.slowRadiance *= invWeight;
    outHistory.secondMoment *= invWeight;
    outHistory.fastRadiance *= invWeight;
    outHistory.hitDistance *= invWeight;
    outHistory.roughness = clamp(1.0 + (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyLength *= invWeight;
    outHistory.confidence *= invWeight;
    outHistory.normal = relaxSafeNormalize(normalSum * invWeight, currentNormal);
    outHistory.found = true;
    return outHistory;
}

float relaxEstimateCurvature(
    uvec2 pixel, vec3 centerPos, vec3 centerNormal,
    int centerMaterial, vec3 cameraDelta
) {
    vec2 currentUv = relaxCurrentUv(pixel);
    vec2 previousUv = relaxProjectPrevious(centerPos, cameraDelta);
    vec2 motion = (previousUv - currentUv) * vec2(resolution_global);
    float motionLength = length(motion);
    if (motionLength < 0.25 || RELAX_CURVATURE_STRENGTH <= 0.0) return 0.0;

    ivec2 axis = abs(motion.x) >= abs(motion.y) ? ivec2(1, 0) : ivec2(0, 1);
    ivec2 p0 = ivec2(pixel) - axis;
    ivec2 p1 = ivec2(pixel) + axis;
    ivec2 size = ivec2(resolution_global);
    if (!relaxInBounds(p0, size) || !relaxInBounds(p1, size)) return 0.0;

    vec3 x0, x1, n0, n1;
    float d0, d1, a0, a1, pr0, pr1;
    int m0, m1;
    readGeo0(GEO_N_GEO, uvec2(p0), x0, d0);
    readGeo0(GEO_N_GEO, uvec2(p1), x1, d1);
    readGeo1(GEO_N_NORMALS, uvec2(p0), n0, a0, m0, pr0);
    readGeo1(GEO_N_NORMALS, uvec2(p1), n1, a1, m1, pr1);
    if (d0 < -0.5 || d1 < -0.5 || m0 != centerMaterial || m1 != centerMaterial)
        return 0.0;

    float threshold = RELAX_DEPTH_THRESHOLD * max(length(centerPos), 1.0);
    if (abs(dot(x0 - centerPos, centerNormal)) > threshold ||
        abs(dot(x1 - centerPos, centerNormal)) > threshold) return 0.0;

    vec3 edge = x1 - x0;
    float edgeLengthSquared = dot(edge, edge);
    if (edgeLengthSquared < 1e-8) return 0.0;
    float curvature = dot(n1 - n0, edge) / edgeLengthSquared;
    curvature *= RELAX_CURVATURE_STRENGTH;

    // Same acceleration guard used by NRD: an inconsistent curvature estimate
    // must not generate virtual motion much larger than surface parallax.
    RelaxFastSignal noisy = relaxUnpackFast(texelFetch(colortex6, ivec2(pixel), 0));
    if (noisy.hitDistance > 0.0 && noisy.hitDistance < 0.5 * VPROJDIST_SKY) {
        float focused = relaxApplyThinLens(noisy.hitDistance, curvature);
        vec3 previousSurface = centerPos + cameraDelta;
        vec3 virtualPrevious = previousSurface + relaxSafeNormalize(centerPos, vec3(0.0, 0.0, 1.0)) * focused;
        vec2 virtualUv = relaxProjectPreviousRelative(virtualPrevious);
        float acceleration = length((virtualUv - previousUv) * vec2(resolution_global));
        if (acceleration > RELAX_MAX_VIRTUAL_MOTION_ACCELERATION *
            max(motionLength, 1.0)) return 0.0;
    }
    return curvature;
}

void relaxAccumulatePath(
    RelaxReprojectedHistory h,
    vec3 noisyRadiance,
    float noisyM2,
    float noisyHitDistance,
    float pathConfidence,
    out RelaxSlowSignal slow,
    out RelaxFastSignal fast
) {
    float historyLength = h.found ? max(h.historyLength, 0.0) : 0.0;
    float slowFrames = min(historyLength, float(RELAX_SPEC_MAX_HISTORY));
    float fastFrames = min(historyLength, float(RELAX_SPEC_MAX_FAST_HISTORY));
    float slowAlpha = max(1.0 - pathConfidence, 1.0 / (1.0 + slowFrames));
    float fastAlpha = max(1.0 - pathConfidence, 1.0 / (1.0 + fastFrames));

    slow.radiance = mix(h.slowRadiance, noisyRadiance, slowAlpha);
    slow.secondMoment = mix(h.secondMoment, noisyM2, slowAlpha);
    slow.hitDistance = mix(h.hitDistance, noisyHitDistance, max(slowAlpha, 0.1));
    slow.historyLength = min(historyLength + 1.0, float(RELAX_SPEC_MAX_HISTORY));
    slow.confidence = pathConfidence;

    fast.radiance = mix(h.fastRadiance, noisyRadiance, fastAlpha);
    fast.hitDistance = mix(h.hitDistance, noisyHitDistance, max(fastAlpha, 0.1));
    fast.historyLength = slow.historyLength;
    fast.confidence = pathConfidence;
    fast.materialID = 0u;
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    RelaxFastSignal noisy = relaxUnpackFast(texelFetch(colortex6, ivec2(pixel), 0));
    noisy.radiance = relaxFiniteColor(noisy.radiance);
    float noisyLuminance = relaxLuma(noisy.radiance);
    float noisyM2 = noisyLuminance * noisyLuminance;

    vec3 currentPos;
    float primaryDistance;
    readGeo0(GEO_N_GEO, pixel, currentPos, primaryDistance);
    if (primaryDistance < -0.5) {
        RelaxSlowSignal slow;
        slow.radiance = noisy.radiance;
        slow.secondMoment = noisyM2;
        slow.hitDistance = noisy.hitDistance;
        slow.historyLength = 0.0;
        slow.confidence = 0.0;
        noisy.historyLength = 0.0;
        noisy.confidence = 0.0;
        imageStore(colorimg4, ivec2(pixel), relaxPackSlow(slow));
        imageStore(colorimg5, ivec2(pixel), relaxPackFast(noisy));
#if DEBUG_VIEW == 12
        writeReflLight(pixel, slow.radiance, slow.hitDistance,
            slow.historyLength);
#endif
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
    vec3 viewDirection = relaxSafeNormalize(currentPos, vec3(0.0, 0.0, 1.0));
    vec3 V = -viewDirection;
    float NoV = abs(dot(currentNormal, V));

    vec2 surfaceUv = relaxProjectPrevious(currentPos, cameraDelta);
    RelaxReprojectedHistory surface = relaxLoadHistory(
        surfaceUv, currentPos, currentNormal, currentMaterial,
        cameraDelta, false);
    float lobeAngle = max(atan(relaxSpecLobeTanHalfAngle(currentRoughness, 0.75)),
        1.5 / 255.0);
    float surfaceViewWeight = 0.0;
    if (surface.found) {
        vec3 previousV = -relaxSafeNormalize(currentPos + cameraDelta, -V);
        float angle = acos(clamp(dot(V, previousV), -1.0, 1.0));
        surfaceViewWeight = clamp(1.0 - angle / max(lobeAngle * max(NoV, 0.05), 1e-4), 0.0, 1.0);
        surfaceViewWeight *= surface.footprintQuality;
    }

    float curvature = relaxEstimateCurvature(pixel, currentPos, currentNormal,
        currentMaterialInt, cameraDelta);
    float minHitDistance = noisy.hitDistance;
    ivec2 size = ivec2(resolution_global);
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        ivec2 p = ivec2(pixel) + ivec2(x, y);
        if (!relaxInBounds(p, size)) continue;
        float h = relaxUnpackFast(texelFetch(colortex6, p, 0)).hitDistance;
        if (h > 0.0) minHitDistance = min(minHitDistance, h);
    }
    float focusedHitDistance = relaxApplyThinLens(minHitDistance, curvature);
    vec3 previousSurfacePosition = currentPos + cameraDelta;
    vec3 previousVirtualPosition = previousSurfacePosition +
        viewDirection * focusedHitDistance;
    vec2 virtualUv = relaxProjectPreviousRelative(previousVirtualPosition);
    RelaxReprojectedHistory virtualHistory = relaxLoadHistory(
        virtualUv, currentPos, currentNormal, currentMaterial,
        cameraDelta, true);

    float dominantFactor = relaxDominantFactor(NoV, currentRoughness);
    float virtualAmount = virtualHistory.found ? dominantFactor : 0.0;
    float virtualConfidence = 0.0;
    if (virtualHistory.found) {
        vec2 normalParams = relaxNormalWeightParams(currentRoughness, 5.0, 1.0);
        vec3 previousVirtualV = -relaxSafeNormalize(previousVirtualPosition, -V);
        float normalWeight = relaxSpecularNormalWeight(normalParams,
            currentNormal, virtualHistory.normal, V, previousVirtualV);
        vec2 roughnessParams = relaxRoughnessWeightParams(
            currentRoughness * currentRoughness, RELAX_ROUGHNESS_FRACTION);
        float roughnessWeight = relaxExponentialWeight(
            virtualHistory.roughness * virtualHistory.roughness,
            roughnessParams);

        float smc = relaxSpecMagicCurve(currentRoughness);
        float currentFocused = relaxApplyThinLens(minHitDistance, curvature);
        float previousFocused = relaxApplyThinLens(
            virtualHistory.hitDistance, curvature);
        float hitDifference = abs(currentFocused - previousFocused);
        float hitScale = max(length(currentPos) +
            max(abs(currentFocused), abs(previousFocused)), 1e-3);
        float hitConfidence = 1.0 - clamp(
            mix(20.0, 0.0, smc) * hitDifference / hitScale, 0.0, 1.0);
        hitConfidence = mix(hitConfidence, 1.0, smc);

        virtualConfidence = normalWeight * (0.1 + 0.9 * roughnessWeight) *
            hitConfidence * virtualHistory.footprintQuality;
        virtualAmount *= virtualConfidence;
        if (surfaceViewWeight > 1e-5)
            virtualAmount *= clamp(virtualConfidence / surfaceViewWeight, 0.0, 1.0);
    }
    virtualAmount = clamp(virtualAmount, 0.0, 1.0);

    RelaxSlowSignal surfaceSlow, virtualSlow;
    RelaxFastSignal surfaceFast, virtualFast;
    relaxAccumulatePath(surface, noisy.radiance, noisyM2, noisy.hitDistance,
        surfaceViewWeight, surfaceSlow, surfaceFast);
    relaxAccumulatePath(virtualHistory, noisy.radiance, noisyM2,
        noisy.hitDistance, virtualConfidence, virtualSlow, virtualFast);

    RelaxSlowSignal outputSlow;
    RelaxFastSignal outputFast;
    outputSlow.radiance = mix(surfaceSlow.radiance, virtualSlow.radiance, virtualAmount);
    outputSlow.secondMoment = mix(surfaceSlow.secondMoment, virtualSlow.secondMoment, virtualAmount);
    outputSlow.hitDistance = mix(surfaceSlow.hitDistance, virtualSlow.hitDistance, virtualAmount);
    outputSlow.historyLength = mix(surfaceSlow.historyLength, virtualSlow.historyLength, virtualAmount);
    outputSlow.historyLength *= sqrt(max(surface.footprintQuality, 1.0 / max(outputSlow.historyLength, 1.0)));
    outputSlow.historyLength = clamp(outputSlow.historyLength, 1.0, float(RELAX_SPEC_MAX_HISTORY));
    outputSlow.confidence = mix(surfaceViewWeight, virtualConfidence, virtualAmount);
    outputFast.radiance = mix(surfaceFast.radiance, virtualFast.radiance, virtualAmount);
    outputFast.hitDistance = mix(surfaceFast.hitDistance, virtualFast.hitDistance, virtualAmount);
    outputFast.historyLength = outputSlow.historyLength;
    outputFast.confidence = outputSlow.confidence;
    outputFast.materialID = currentMaterial;

    if (outputSlow.secondMoment == 0.0)
        outputSlow.secondMoment = RELAX_SPEC_VARIANCE_BOOST *
            (1.0 - outputSlow.confidence);
    imageStore(colorimg4, ivec2(pixel), relaxPackSlow(outputSlow));
    imageStore(colorimg5, ivec2(pixel), relaxPackFast(outputFast));
#if DEBUG_VIEW == 12
    // Preserve the temporal-only result in ReflectBuffer N=1.  The later
    // RELAX passes may still execute, but resolve leaves this value untouched
    // in debug view 12, so no spatial stage contributes to the visualization.
    writeReflLight(pixel, relaxFiniteColor(outputSlow.radiance),
        outputSlow.hitDistance, outputSlow.historyLength);
#endif
}
