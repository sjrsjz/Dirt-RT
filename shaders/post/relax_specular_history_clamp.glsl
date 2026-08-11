#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex3;
uniform usampler2D colortex5;
uniform usampler2D colortex6;
layout(rgba32f) uniform writeonly image2D colorimg9;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

const uint CLAMP_GROUP_SIZE = 8u;
const uint CLAMP_HALO = 2u;
const uint CLAMP_TILE_SIZE = CLAMP_GROUP_SIZE + 2u * CLAMP_HALO;
const uint CLAMP_TILE_AREA = CLAMP_TILE_SIZE * CLAMP_TILE_SIZE;

// The 5x5 clamp only consumes fast RGB/material, noisy RGB and geometry
// validity. Keep packed source words in LDS and decode per consumer.
shared uvec4 clampFastTile[CLAMP_TILE_AREA];
shared uvec2 clampNoisyTile[CLAMP_TILE_AREA];
shared uint clampValidTile[CLAMP_TILE_AREA];

vec3 relaxUnpackClampRadiance(uvec2 packed_) {
    vec2 rg = unpackHalf2x16(packed_.x);
    vec2 bUnused = unpackHalf2x16(packed_.y);
    return relaxFiniteColor(vec3(rg, bUnused.x));
}

void relaxStoreClampedHistory(
    uvec2 pixel,
    vec4 slow,
    RelaxFastSignal fast,
    vec3 surfacePosition,
    float primaryDistance,
    vec3 geometryNormal,
    float ggxAlpha,
    int materialID
) {
    RelaxSpecularHistory history;
    history.surfacePosition = surfacePosition;
    history.geometryNormal = geometryNormal;
    history.slowRadiance = relaxFiniteColor(slow.rgb);
    history.secondMoment = max(slow.a, 0.0);
    history.responsiveRadiance = relaxFiniteColor(fast.radiance);
    history.endpoint = primaryDistance > -0.5
        ? readReflEndpointMoments(pixel) : emptyRelaxEndpointMoments();
    history.roughness = relaxPerceptualRoughness(ggxAlpha);
    history.historyLength = primaryDistance > -0.5 ? fast.historyLength : 0.0;
    history.materialID = uint(max(materialID, 0));
    history.reprojectionConfidence = primaryDistance > -0.5
        ? fast.confidence : 0.0;
    writeRelaxSpecularHistory(pixel, history);
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    uvec2 localPixel = gl_LocalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * CLAMP_GROUP_SIZE) -
        ivec2(CLAMP_HALO);

    // Each lane owns its center so the geometry needed for history output is
    // retained instead of being fetched once for the tile and again later.
    uint centerX = localPixel.x + CLAMP_HALO;
    uint centerY = localPixel.y + CLAMP_HALO;
    uint centerIndex = centerY * CLAMP_TILE_SIZE + centerX;
    ivec2 centerCoord = ivec2(pixel);
    bool centerInBounds = relaxInBounds(centerCoord, size);
    ivec2 centerClamped = clamp(centerCoord, ivec2(0), size - 1);
    vec3 centerPosition;
    float centerPrimaryDistance;
    readGeo0(GEO_N_GEO, uvec2(centerClamped), centerPosition,
        centerPrimaryDistance);
    clampFastTile[centerIndex] = centerInBounds
        ? texelFetch(colortex5, centerClamped, 0) : uvec4(0u);
    uvec4 centerNoisyPacked = centerInBounds
        ? texelFetch(colortex6, centerClamped, 0) : uvec4(0u);
    clampNoisyTile[centerIndex] = centerNoisyPacked.xy;
    clampValidTile[centerIndex] = centerInBounds &&
        centerPrimaryDistance > -0.5 ? 1u : 0u;

    for (uint i = gl_LocalInvocationIndex; i < CLAMP_TILE_AREA; i += 64u) {
        uint tx = i % CLAMP_TILE_SIZE;
        uint ty = i / CLAMP_TILE_SIZE;
        bool interior = tx >= CLAMP_HALO &&
            tx < CLAMP_HALO + CLAMP_GROUP_SIZE && ty >= CLAMP_HALO &&
            ty < CLAMP_HALO + CLAMP_GROUP_SIZE;
        if (interior) continue;
        ivec2 q = tileOrigin + ivec2(tx, ty);
        bool inBounds = relaxInBounds(q, size);
        ivec2 qc = clamp(q, ivec2(0), size - 1);

        clampFastTile[i] = inBounds
            ? texelFetch(colortex5, qc, 0) : uvec4(0u);
        uvec4 noisyPacked = inBounds
            ? texelFetch(colortex6, qc, 0) : uvec4(0u);
        clampNoisyTile[i] = noisyPacked.xy;

        vec3 positionUnused;
        float primaryDistance;
        readGeo0(GEO_N_GEO, uvec2(qc), positionUnused, primaryDistance);
        clampValidTile[i] = inBounds && primaryDistance > -0.5 ? 1u : 0u;
    }
    barrier();

    if (any(greaterThanEqual(pixel, resolution_global))) return;

    uvec4 centerFastPacked = clampFastTile[centerIndex];

    vec4 slow = texelFetch(colortex3, ivec2(pixel), 0);
    RelaxFastSignal fast = relaxUnpackFast(centerFastPacked);
    vec3 noisyCenter = relaxUnpackClampRadiance(
        clampNoisyTile[centerIndex]);

    vec3 geometryNormal;
    float ggxAlpha, pathRoughnessUnused;
    int materialID;
    readGeo1(GEO_N_NORMALS, pixel, geometryNormal, ggxAlpha,
        materialID, pathRoughnessUnused);

    if (centerPrimaryDistance < -0.5) {
        relaxStoreClampedHistory(pixel, slow, fast, centerPosition,
            centerPrimaryDistance, geometryNormal, ggxAlpha, materialID);
        imageStore(colorimg9, ivec2(pixel), slow);
        imageStore(colorimg4, ivec2(pixel), relaxPackFast(fast));
#if DEBUG_VIEW == 23
        writeReflLight(pixel, relaxFiniteColor(slow.rgb), fast.endpointDistance,
            fast.historyLength);
#endif
        return;
    }

    vec3 fastM1 = vec3(0.0), fastM2 = vec3(0.0);
    vec3 noisyM1 = vec3(0.0);
    float noisyLumaM2 = 0.0;
    float sampleCount = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            uint sampleIndex = uint(int(centerY) + y) * CLAMP_TILE_SIZE +
                uint(int(centerX) + x);
            if (clampValidTile[sampleIndex] == 0u) continue;

            uvec4 qFastPacked = clampFastTile[sampleIndex];
            if (qFastPacked.w != fast.materialID) continue;
            vec3 fastRadiance = relaxUnpackClampRadiance(qFastPacked.xy);
            vec3 noisyRadiance = relaxUnpackClampRadiance(
                clampNoisyTile[sampleIndex]);
            vec3 ycocg = relaxRgbToYCoCg(fastRadiance);
            fastM1 += ycocg;
            fastM2 += ycocg * ycocg;
            noisyM1 += noisyRadiance;
            float noisyLuma = relaxLuma(noisyRadiance);
            noisyLumaM2 += noisyLuma * noisyLuma;
            sampleCount += 1.0;
        }
    }

    if (sampleCount > 0.0) {
        float inverseSampleCount = 1.0 / sampleCount;
        fastM1 *= inverseSampleCount;
        fastM2 *= inverseSampleCount;
        noisyM1 *= inverseSampleCount;
        noisyLumaM2 *= inverseSampleCount;
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
        if (fast.historyLength <= RELAX_HISTORY_FIX_FRAMES)
            historyDifference = 0.0;
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
            max(max(slowLumaBefore, noisyLuma) + spatialSigma + temporalSigma,
                1e-6);
        reset = clamp(reset, 0.0, 1.0);
        clampedSlow = mix(clampedSlow, noisyCenter, reset);
        fast.radiance = mix(fast.radiance, noisyCenter, reset);

        float slowLumaAfter = relaxLuma(clampedSlow);
        slow.a = max(slow.a + slowLumaAfter * slowLumaAfter -
            slowLumaBefore * slowLumaBefore, 0.0);
        slow.rgb = relaxFiniteColor(clampedSlow);
        fast.radiance = relaxFiniteColor(fast.radiance);
    }

    relaxStoreClampedHistory(pixel, slow, fast, centerPosition,
        centerPrimaryDistance, geometryNormal, ggxAlpha, materialID);
    imageStore(colorimg9, ivec2(pixel), slow);
    imageStore(colorimg4, ivec2(pixel), relaxPackFast(fast));
#if DEBUG_VIEW == 23
    writeReflLight(pixel, relaxFiniteColor(slow.rgb), fast.endpointDistance,
        fast.historyLength);
#endif
}
