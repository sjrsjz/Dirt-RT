#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex3;
uniform usampler2D colortex5;
uniform usampler2D colortex6;
layout(rgba32f) uniform writeonly image2D colorimg9;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

void relaxStoreClampedHistory(uvec2 pixel, vec4 slow, RelaxFastSignal fast) {
    vec3 surfacePosition, geometryNormal;
    float primaryDistance, ggxAlpha, pathRoughness;
    int materialID;
    readGeo0(GEO_N_GEO, pixel, surfacePosition, primaryDistance);
    readGeo1(GEO_N_NORMALS, pixel, geometryNormal, ggxAlpha,
        materialID, pathRoughness);

    RelaxSpecularHistory history;
    history.surfacePosition = surfacePosition;
    history.geometryNormal = geometryNormal;
    history.slowRadiance = relaxFiniteColor(slow.rgb);
    history.secondMoment = max(slow.a, 0.0);
    history.responsiveRadiance = relaxFiniteColor(fast.radiance);
    history.hitDistance = max(fast.hitDistance, 0.0);
    history.roughness = relaxPerceptualRoughness(ggxAlpha);
    history.historyLength = primaryDistance > -0.5 ? fast.historyLength : 0.0;
    history.materialID = uint(max(materialID, 0));
    history.reprojectionConfidence = primaryDistance > -0.5
        ? fast.confidence : 0.0;
    writeRelaxSpecularHistory(pixel, history);
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec4 slow = texelFetch(colortex3, ivec2(pixel), 0);
    RelaxFastSignal fast = relaxUnpackFast(texelFetch(colortex5, ivec2(pixel), 0));
    RelaxFastSignal noisyCenter = relaxUnpackFast(texelFetch(colortex6, ivec2(pixel), 0));
    if (noisyCenter.confidence <= 0.0) {
        relaxStoreClampedHistory(pixel, slow, fast);
        imageStore(colorimg9, ivec2(pixel), slow);
        imageStore(colorimg4, ivec2(pixel), relaxPackFast(fast));
#if DEBUG_VIEW == 23
        writeReflLight(pixel, relaxFiniteColor(slow.rgb), fast.hitDistance,
            fast.historyLength);
#endif
        return;
    }

    vec3 fastM1 = vec3(0.0), fastM2 = vec3(0.0);
    vec3 noisyM1 = vec3(0.0);
    float noisyLumaM2 = 0.0;
    float sampleCount = 0.0;
    for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        ivec2 q = ivec2(pixel) + ivec2(x, y);
        if (!relaxInBounds(q, size)) continue;
        RelaxFastSignal qFast = relaxUnpackFast(texelFetch(colortex5, q, 0));
        RelaxFastSignal qNoisy = relaxUnpackFast(texelFetch(colortex6, q, 0));
        if (qNoisy.confidence <= 0.0 || qFast.materialID != fast.materialID) continue;
        vec3 ycocg = relaxRgbToYCoCg(qFast.radiance);
        fastM1 += ycocg;
        fastM2 += ycocg * ycocg;
        noisyM1 += qNoisy.radiance;
        float noisyLuma = relaxLuma(qNoisy.radiance);
        noisyLumaM2 += noisyLuma * noisyLuma;
        sampleCount += 1.0;
    }

    if (sampleCount > 0.0) {
        fastM1 /= sampleCount;
        fastM2 /= sampleCount;
        noisyM1 /= sampleCount;
        noisyLumaM2 /= sampleCount;
        vec3 sigma = sqrt(max(fastM2 - fastM1 * fastM1, vec3(0.0)));
        vec3 boxMin = fastM1 - RELAX_COLOR_BOX_SIGMA * sigma;
        vec3 boxMax = fastM1 + RELAX_COLOR_BOX_SIGMA * sigma;
        vec3 fastCenterYCoCg = relaxRgbToYCoCg(fast.radiance);
        boxMin = min(boxMin, fastCenterYCoCg);
        boxMax = max(boxMax, fastCenterYCoCg);

        vec3 slowYCoCg = relaxRgbToYCoCg(slow.rgb);
        vec3 clampedYCoCg = slowYCoCg;
        if (RELAX_SPEC_MAX_FAST_HISTORY < RELAX_SPEC_MAX_HISTORY)
            clampedYCoCg = clamp(slowYCoCg, boxMin, boxMax);
        vec3 clampedSlow = relaxYCoCgToRgb(clampedYCoCg);

        float clampingFactor = abs(fastCenterYCoCg.x - slowYCoCg.x) > 1e-6
            ? clamp((clampedYCoCg.x - slowYCoCg.x) /
                (fastCenterYCoCg.x - slowYCoCg.x), 0.0, 1.0)
            : 0.0;
        if (fast.historyLength <= RELAX_HISTORY_FIX_FRAMES) {
            clampedSlow = fast.radiance;
            clampingFactor = 1.0;
        }

        float historyDifference = 0.33 * RELAX_HISTORY_ACCELERATION *
            relaxLuma(abs(fast.radiance - slow.rgb)) * clampingFactor;
        if (fast.historyLength <= RELAX_HISTORY_FIX_FRAMES) historyDifference = 0.0;
        vec3 distanceToNoisy = noisyM1 - fast.radiance;
        float distanceLuma = relaxLuma(abs(distanceToNoisy));
        vec3 acceleration = distanceLuma > 1e-6
            ? distanceToNoisy * min(historyDifference / distanceLuma, 1.0)
            : vec3(0.0);
        clampedSlow += acceleration;
        fast.radiance += acceleration;

        float slowLumaBefore = relaxLuma(slow.rgb);
        float noisyLuma = relaxLuma(noisyM1);
        float temporalSigma = RELAX_HISTORY_RESET_TEMPORAL_SIGMA * sqrt(max(
            noisyLumaM2 - noisyLuma * noisyLuma, 0.0));
        float spatialSigma = RELAX_HISTORY_RESET_SPATIAL_SIGMA * sigma.x;
        float reset = 0.5 * RELAX_HISTORY_RESET_AMOUNT * max(0.0,
            abs(slowLumaBefore - noisyLuma) - spatialSigma - temporalSigma) /
            max(max(slowLumaBefore, noisyLuma) + spatialSigma + temporalSigma, 1e-6);
        reset = clamp(reset, 0.0, 1.0);
        clampedSlow = mix(clampedSlow, noisyCenter.radiance, reset);
        fast.radiance = mix(fast.radiance, noisyCenter.radiance, reset);

        float slowLumaAfter = relaxLuma(clampedSlow);
        slow.a = max(slow.a + slowLumaAfter * slowLumaAfter -
            slowLumaBefore * slowLumaBefore, 0.0);
        slow.rgb = relaxFiniteColor(clampedSlow);
        fast.radiance = relaxFiniteColor(fast.radiance);
    }

    relaxStoreClampedHistory(pixel, slow, fast);
    imageStore(colorimg9, ivec2(pixel), slow);
    imageStore(colorimg4, ivec2(pixel), relaxPackFast(fast));
#if DEBUG_VIEW == 23
    writeReflLight(pixel, relaxFiniteColor(slow.rgb), fast.hitDistance,
        fast.historyLength);
#endif
}
