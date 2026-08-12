#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

const ivec2 RELAX_PREPASS_OFFSETS[8] = ivec2[](
    ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1),
    ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec3 centerPos;
    float primaryDistance;
    readGeo0(GEO_N_GEO, pixel, centerPos, primaryDistance);
    if (primaryDistance < -0.5) {
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    RelaxPrepassSignal center = relaxUnpackPrepass(
        texelFetch(colortex6, ivec2(pixel), 0));
    if (RELAX_PREPASS_RADIUS <= 0.0) {
        imageStore(colorimg6, ivec2(pixel), relaxPackPrepass(center));
        return;
    }

    vec3 centerNormal;
    float centerAlpha, centerPathRoughness;
    int centerMaterial;
    readGeo1(GEO_N_NORMALS, pixel, centerNormal, centerAlpha,
        centerMaterial, centerPathRoughness);
    float centerRoughness = relaxPerceptualRoughness(centerAlpha);
    vec2 roughnessParams = relaxRoughnessWeightParams(
        centerRoughness, RELAX_ROUGHNESS_FRACTION);
    float depthThreshold = RELAX_DEPTH_THRESHOLD * max(length(centerPos), 1.0);
    int stride = max(1, int(floor(RELAX_PREPASS_RADIUS *
        mix(0.25, 1.0, centerRoughness) + 0.5)));

    vec4 sumY = center.signal.aliceY;
    vec2 sumCoCg = center.signal.CoCg;
    float sumHit = center.hitDistance;
    float sumWeight = 1.0;

    for (int i = 0; i < 8; ++i) {
        ivec2 q = ivec2(pixel) + RELAX_PREPASS_OFFSETS[i] * stride;
        if (!relaxInBounds(q, size)) continue;
        vec3 qPos, qNormal;
        float qDepth, qAlpha, qPath;
        int qMaterial;
        readGeo0(GEO_N_GEO, uvec2(q), qPos, qDepth);
        readGeo1(GEO_N_NORMALS, uvec2(q), qNormal, qAlpha,
            qMaterial, qPath);
        if (qDepth < -0.5 || qMaterial != centerMaterial) continue;

        float qRoughness = relaxPerceptualRoughness(qAlpha);
        float w = relaxPlaneWeight(centerPos, centerNormal, qPos,
            depthThreshold);
        w *= relaxExponentialWeight(qRoughness, roughnessParams);
        if (w <= 1e-4) continue;

        RelaxPrepassSignal sampleSignal = relaxUnpackPrepass(
            texelFetch(colortex6, q, 0));
        sumY += sampleSignal.signal.aliceY * w;
        sumCoCg += sampleSignal.signal.CoCg * w;
        sumHit += sampleSignal.hitDistance * w;
        sumWeight += w;
    }

    float invWeight = 1.0 / max(sumWeight, 1e-6);
    RelaxPrepassSignal outputSignal;
    outputSignal.signal.aliceY = sumY * invWeight;
    outputSignal.signal.CoCg = sumCoCg * invWeight;
    outputSignal.signal = sanitizeSpecularMaxEnt(outputSignal.signal);
    outputSignal.hitDistance = sumHit * invWeight;
    imageStore(colorimg6, ivec2(pixel), relaxPackPrepass(outputSignal));
}
