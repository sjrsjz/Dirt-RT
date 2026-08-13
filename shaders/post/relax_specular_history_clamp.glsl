#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex3;
uniform usampler2D colortex4;
uniform usampler2D colortex5;
uniform usampler2D colortex6;
layout(rgba32f) uniform writeonly image2D colorimg9;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec3 position;
    float primaryDistance;
    readGeo0(GEO_N_GEO, pixel, position, primaryDistance);
    if (primaryDistance < -0.5) {
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTMETA, pixel)] = uvec4(0u);
        imageStore(colorimg9, ivec2(pixel), vec4(0.0));
        imageStore(colorimg4, ivec2(pixel), uvec4(0u));
        return;
    }

    RelaxSlowSignal slow = relaxUnpackSlow(
        texelFetch(colortex4, ivec2(pixel), 0));
    slow.signal.aliceY = texelFetch(colortex3, ivec2(pixel), 0);
    slow.signal = sanitizeSpecularMaxEnt(slow.signal);
    RelaxFastSignal fast = relaxUnpackFast(
        texelFetch(colortex5, ivec2(pixel), 0));
    RelaxPrepassSignal noisyCenter = relaxUnpackPrepass(
        texelFetch(colortex6, ivec2(pixel), 0));

    vec3 geometryNormal;
    float ggxAlpha, pathRoughness;
    int materialID;
    readGeo1(GEO_N_NORMALS, pixel, geometryNormal, ggxAlpha,
        materialID, pathRoughness);

    vec3 fastM1 = vec3(0.0);
    vec3 fastM2 = vec3(0.0);
    vec3 noisyM1 = vec3(0.0);
    vec4 noisyAliceY = vec4(0.0);
    vec2 noisyCoCg = vec2(0.0);
    float noisyY2 = 0.0;
    float sampleCount = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 q = ivec2(pixel) + ivec2(x, y);
            if (!relaxInBounds(q, size)) continue;
            RelaxFastSignal qFast = relaxUnpackFast(
                texelFetch(colortex5, q, 0));
            vec3 qNormal;
            float qAlpha, qPathRoughness;
            int qMaterial;
            readGeo1(GEO_N_NORMALS, uvec2(q), qNormal, qAlpha,
                qMaterial, qPathRoughness);
            if (qMaterial != materialID) continue;
            RelaxPrepassSignal qNoisy = relaxUnpackPrepass(
                texelFetch(colortex6, q, 0));
            vec3 qNoisyYCoCg = relaxMaxEntYCoCg(qNoisy.signal);
            vec3 qFastYCoCg = relaxMaxEntYCoCg(qFast.signal);
            fastM1 += qFastYCoCg;
            fastM2 += qFastYCoCg * qFastYCoCg;
            noisyM1 += qNoisyYCoCg;
            noisyAliceY += qNoisy.signal.aliceY;
            noisyCoCg += qNoisy.signal.CoCg;
            noisyY2 += qNoisyYCoCg.x * qNoisyYCoCg.x;
            sampleCount += 1.0;
        }
    }

    vec3 slowYCoCg = relaxMaxEntYCoCg(slow.signal);
    float oldSlowY = slowYCoCg.x;
    if (sampleCount > 0.0) {
        float invCount = 1.0 / sampleCount;
        fastM1 *= invCount;
        fastM2 *= invCount;
        noisyM1 *= invCount;
        noisyAliceY *= invCount;
        noisyCoCg *= invCount;
        noisyY2 *= invCount;
        vec3 sigma = sqrt(max(fastM2 - fastM1 * fastM1, vec3(0.0)));
        vec3 boxMin = fastM1 - RELAX_COLOR_BOX_SIGMA * sigma;
        vec3 boxMax = fastM1 + RELAX_COLOR_BOX_SIGMA * sigma;
        vec3 fastYCoCg = relaxMaxEntYCoCg(fast.signal);
        boxMin = min(boxMin, fastYCoCg);
        boxMax = max(boxMax, fastYCoCg);

        vec3 clampedYCoCg = slowYCoCg;
        if (RELAX_SPEC_MAX_FAST_HISTORY < RELAX_SPEC_MAX_HISTORY)
            clampedYCoCg = clamp(slowYCoCg, boxMin, boxMax);
        if (fast.historyLength <= RELAX_HISTORY_FIX_FRAMES)
            clampedYCoCg = fastYCoCg;

        float clampDenominator = fastYCoCg.x - slowYCoCg.x;
        float clampingFactor = abs(clampedYCoCg.x - slowYCoCg.x) > 1e-8
                && abs(clampDenominator) > 1e-8
            ? clamp((clampedYCoCg.x - slowYCoCg.x) /
                clampDenominator, 0.0, 1.0) : 0.0;
        if (fast.historyLength <= RELAX_HISTORY_FIX_FRAMES)
            clampingFactor = 1.0;

        slow.signal = relaxMixMaxEnt(slow.signal, fast.signal,
            clampingFactor);
        slow.signal = relaxSetMaxEntYCoCg(slow.signal, clampedYCoCg);

        SpecularMaxEnt noisyMean;
        noisyMean.aliceY = noisyAliceY;
        noisyMean.CoCg = noisyCoCg;
        noisyMean = sanitizeSpecularMaxEnt(noisyMean);
        float historyDifference = abs(fastYCoCg.x - slowYCoCg.x);
        float distanceToNoisy = abs(noisyM1.x - fastYCoCg.x);
        float accelerationDistance = 0.33 * 10.0
            * RELAX_HISTORY_ACCELERATION * historyDifference
            * clampingFactor;
        float acceleration = distanceToNoisy > 1e-8
            ? min(accelerationDistance / distanceToNoisy, 1.0) : 0.0;
        if (fast.historyLength <= RELAX_HISTORY_FIX_FRAMES)
            acceleration = 0.0;
        // NRD adds the same responsive-to-noisy correction to both histories.
        // Apply that correction to every MaxEnt component, including direction.
        vec4 accelerationAliceY =
            (noisyMean.aliceY - fast.signal.aliceY) * acceleration;
        vec2 accelerationCoCg =
            (noisyMean.CoCg - fast.signal.CoCg) * acceleration;
        slow.signal.aliceY += accelerationAliceY;
        slow.signal.CoCg += accelerationCoCg;
        fast.signal.aliceY += accelerationAliceY;
        fast.signal.CoCg += accelerationCoCg;
        slow.signal = sanitizeSpecularMaxEnt(slow.signal);
        fast.signal = sanitizeSpecularMaxEnt(fast.signal);

        float temporalSigma = RELAX_HISTORY_RESET_TEMPORAL_SIGMA *
            sqrt(max(noisyY2 - noisyM1.x * noisyM1.x, 0.0));
        float spatialSigma = RELAX_HISTORY_RESET_SPATIAL_SIGMA * sigma.x;
        float reset = 0.5 * RELAX_HISTORY_RESET_AMOUNT * max(0.0,
            abs(oldSlowY - noisyM1.x) - spatialSigma - temporalSigma) /
            max(max(oldSlowY, noisyM1.x) + spatialSigma + temporalSigma,
                1e-6);
        reset = clamp(reset, 0.0, 1.0);
        slow.signal = relaxMixMaxEnt(slow.signal, noisyCenter.signal, reset);
        fast.signal = relaxMixMaxEnt(fast.signal, noisyCenter.signal, reset);
        slowYCoCg = relaxMaxEntYCoCg(slow.signal);
    }

    slow.signal = sanitizeSpecularMaxEnt(slow.signal);
    slow.secondMoment = max(slow.secondMoment +
        slowYCoCg.x * slowYCoCg.x - oldSlowY * oldSlowY, 0.0);

    RelaxSpecularHistory history;
    history.surfacePosition = position;
    history.geometryNormal = geometryNormal;
    history.slowSignal = slow.signal;
    history.secondMoment = slow.secondMoment;
    history.responsiveSignal = fast.signal;
    history.hitDistance = fast.hitDistance;
    history.roughness = relaxPerceptualRoughness(ggxAlpha);
    history.historyLength = fast.historyLength;
    history.materialID = uint(max(materialID, 0));
    history.reprojectionConfidence = slow.confidence;
    writeRelaxSpecularHistory(pixel, history);

    RelaxPostSignal postSignal;
    postSignal.CoCg = slow.signal.CoCg;
    postSignal.secondMoment = slow.secondMoment;
    postSignal.hitDistance = fast.hitDistance;
    postSignal.historyLength = fast.historyLength;
    postSignal.confidence = slow.confidence;
    postSignal.materialID = uint(max(materialID, 0));
    imageStore(colorimg9, ivec2(pixel), slow.signal.aliceY);
    imageStore(colorimg4, ivec2(pixel), relaxPackPost(postSignal));

#if DEBUG_VIEW == 23
    writeReflLight(pixel, specularMaxEntTotalRgb(slow.signal),
        fast.hitDistance, fast.historyLength);
#endif
}
