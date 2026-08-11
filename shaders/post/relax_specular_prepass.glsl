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

    // The reflection continuation is empty for primary sky pixels.  Keep this
    // check ahead of the reflection/endpoint reads: sky is the common case in
    // the pathological trace and the packed empty prepass value is all zero.
    if (primaryDistance < -0.5) {
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    vec3 raw;
    float unusedDistance, unusedWeight;
    readReflLight(pixel, raw, unusedDistance, unusedWeight);
    raw = relaxFiniteColor(raw);

    RelaxPrepassSignal outputSignal;
    outputSignal.radiance = raw;
    outputSignal.endpoint = relaxUnpackEndpointMoments(
        texelFetch(colortex6, ivec2(pixel), 0).xy);

    vec3 centerNormal;
    float centerAlpha, centerPathRoughness;
    int centerMaterial;
    readGeo1(GEO_N_NORMALS, pixel, centerNormal, centerAlpha,
        centerMaterial, centerPathRoughness);
    float centerRoughness = relaxPerceptualRoughness(centerAlpha);
    if (RELAX_PREPASS_RADIUS <= 0.0) {
        imageStore(colorimg6, ivec2(pixel), relaxPackPrepass(outputSignal));
        return;
    }
    vec3 sumRadiance = raw;
    float sumWeight = 1.0;
    vec2 roughnessParams = relaxRoughnessWeightParams(
        centerRoughness, RELAX_ROUGHNESS_FRACTION);
    float normalParam = 1.0 / max(
        atan(relaxSpecLobeTanHalfAngle(centerRoughness,
            0.5 * RELAX_LOBE_ANGLE_FRACTION)), 1.5 / 255.0);

    int stride = max(1, int(floor(RELAX_PREPASS_RADIUS *
        mix(0.25, 1.0, centerRoughness) + 0.5)));
    float depthThreshold = RELAX_DEPTH_THRESHOLD * max(length(centerPos), 1.0);

    for (int i = 0; i < 8; ++i) {
        ivec2 q = ivec2(pixel) + RELAX_PREPASS_OFFSETS[i] * stride;
        if (!relaxInBounds(q, size)) continue;

        vec3 samplePos;
        float samplePrimaryDistance;
        readGeo0(GEO_N_GEO, uvec2(q), samplePos, samplePrimaryDistance);
        if (samplePrimaryDistance < -0.5) continue;

        vec3 sampleNormal;
        float sampleAlpha, samplePathRoughness;
        int sampleMaterial;
        readGeo1(GEO_N_NORMALS, uvec2(q), sampleNormal, sampleAlpha,
            sampleMaterial, samplePathRoughness);
        if (sampleMaterial != centerMaterial) continue;

        float sampleRoughness = relaxPerceptualRoughness(sampleAlpha);
        float w = relaxPlaneWeight(centerPos, centerNormal, samplePos,
            depthThreshold);
        float angle = acos(clamp(dot(centerNormal, sampleNormal), -1.0, 1.0));
        w *= clamp(1.0 - angle * normalParam, 0.0, 1.0);
        w *= relaxExponentialWeight(sampleRoughness, roughnessParams);
        if (w <= 1e-4) continue;

        vec3 sampleRadiance;
        float sampleUnusedDistance, sampleWeight;
        readReflLight(uvec2(q), sampleRadiance, sampleUnusedDistance, sampleWeight);
        sampleRadiance = relaxFiniteColor(sampleRadiance);

        sumRadiance += sampleRadiance * w;
        sumWeight += w;
    }

    outputSignal.radiance = relaxFiniteColor(sumRadiance / max(sumWeight, 1e-6));
    imageStore(colorimg6, ivec2(pixel), relaxPackPrepass(outputSignal));
}
