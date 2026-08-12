#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex3;
uniform usampler2D colortex4;
layout(rgba32f) uniform writeonly image2D colorimg9;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec3 centerPos;
    float centerDepth;
    readGeo0(GEO_N_GEO, pixel, centerPos, centerDepth);
    if (centerDepth < -0.5) {
        imageStore(colorimg9, ivec2(pixel), vec4(0.0));
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    vec3 centerNormal;
    float centerAlpha, centerPath;
    int centerMaterialInt;
    readGeo1(GEO_N_NORMALS, pixel, centerNormal, centerAlpha,
        centerMaterialInt, centerPath);
    uint centerMaterial = uint(max(centerMaterialInt, 0));
    RelaxPostSignal centerMeta = relaxUnpackPost(
        texelFetch(colortex4, ivec2(pixel), 0));
    SpecularMaxEnt filtered;
    filtered.aliceY = texelFetch(colortex3, ivec2(pixel), 0);
    filtered.CoCg = centerMeta.CoCg;
    float filteredM2 = centerMeta.secondMoment;
    float filteredHit = centerMeta.hitDistance;

    if (centerMeta.historyLength < RELAX_HISTORY_THRESHOLD) {
        vec4 sumY = vec4(0.0);
        vec2 sumCoCg = vec2(0.0);
        float sumM2 = 0.0;
        float sumHit = 0.0;
        float sumWeight = 0.0;
        float depthThreshold = RELAX_DEPTH_THRESHOLD *
            max(length(centerPos), 1.0);
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x) {
                ivec2 q = ivec2(pixel) + ivec2(x, y);
                if (!relaxInBounds(q, size)) continue;
                vec3 qPos;
                float qDepth;
                readGeo0(GEO_N_GEO, uvec2(q), qPos, qDepth);
                if (qDepth < -0.5) continue;
                RelaxPostSignal qMeta = relaxUnpackPost(
                    texelFetch(colortex4, q, 0));
                if (qMeta.materialID != centerMaterial) continue;
                float w = relaxPlaneWeight(centerPos, centerNormal, qPos,
                    depthThreshold);
                w *= exp(-0.125 * float(x * x + y * y));
                if (w <= 1e-5) continue;
                sumY += texelFetch(colortex3, q, 0) * w;
                sumCoCg += qMeta.CoCg * w;
                sumM2 += qMeta.secondMoment * w;
                sumHit += qMeta.hitDistance * w;
                sumWeight += w;
            }
        }
        if (sumWeight > 1e-5) {
            float invWeight = 1.0 / sumWeight;
            filtered.aliceY = sumY * invWeight;
            filtered.CoCg = sumCoCg * invWeight;
            filteredM2 = sumM2 * invWeight;
            filteredHit = sumHit * invWeight;
        }
    }
    filtered = sanitizeSpecularMaxEnt(filtered);

    float variance = max(filteredM2 -
        filtered.aliceY.w * filtered.aliceY.w, 0.0);
    if (centerMeta.historyLength < RELAX_HISTORY_THRESHOLD)
        variance *= max(1.0, RELAX_HISTORY_THRESHOLD /
            max(centerMeta.historyLength, 1.0));

    uint geometryWord = relaxPackNormalMaterial(centerNormal, centerMaterial);
    imageStore(colorimg9, ivec2(pixel), vec4(centerPos,
        uintBitsToFloat(geometryWord)));

    RelaxSpatialSignal spatial;
    spatial.signal = filtered;
    spatial.variance = variance;
    spatial.hitDistance = filteredHit;
    imageStore(colorimg6, ivec2(pixel), relaxPackSpatial(spatial));

#if DEBUG_VIEW == 25
    writeReflLight(pixel, specularMaxEntTotalRgb(spatial.signal),
        spatial.hitDistance, centerMeta.historyLength);
#endif
}
