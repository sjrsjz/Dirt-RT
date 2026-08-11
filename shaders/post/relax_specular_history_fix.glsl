#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex5;
layout(rgba32f) uniform writeonly image2D colorimg3;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec3 centerPos;
    float centerDepth;
    readGeo0(GEO_N_GEO, pixel, centerPos, centerDepth);

    // All upstream reflection signals are explicitly empty for sky.  Avoid
    // unpacking two signals and decoding the unused normal/material record.
    if (centerDepth < -0.5) {
        imageStore(colorimg3, ivec2(pixel), vec4(0.0));
        return;
    }

    RelaxSlowSignal packedCenter = relaxUnpackSlow(
        texelFetch(colortex4, ivec2(pixel), 0));
    vec4 centerSignal = vec4(packedCenter.radiance, packedCenter.secondMoment);
    RelaxFastSignal centerFast = relaxUnpackFast(
        texelFetch(colortex5, ivec2(pixel), 0));
    vec3 centerNormal;
    float centerAlpha, centerPath;
    int centerMaterial;
    readGeo1(GEO_N_NORMALS, pixel, centerNormal, centerAlpha,
        centerMaterial, centerPath);

    if (centerFast.historyLength > RELAX_HISTORY_FIX_FRAMES ||
        RELAX_HISTORY_FIX_FRAMES <= 1.0) {
        imageStore(colorimg3, ivec2(pixel), centerSignal);
        return;
    }

    float roughness = relaxPerceptualRoughness(centerAlpha);
    vec3 centerV = -relaxSafeNormalize(centerPos, vec3(0.0, 0.0, 1.0));
    vec2 normalParams = relaxNormalWeightParams(roughness, 5.0, 1.0);
    float depthThreshold = RELAX_DEPTH_THRESHOLD * max(length(centerPos), 1.0);
    int stride = max(1, int(floor(RELAX_HISTORY_FIX_BASE_STRIDE /
        (1.0 + centerFast.historyLength) + 0.5)));
    vec4 sumSignal = centerSignal;
    float sumWeight = 1.0;

    for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        if (x == 0 && y == 0) continue;
        ivec2 q = ivec2(pixel) + ivec2(x, y) * stride;
        if (!relaxInBounds(q, size)) continue;
        vec3 qPos, qNormal;
        float qDepth, qAlpha, qPath;
        int qMaterial;
        readGeo0(GEO_N_GEO, uvec2(q), qPos, qDepth);
        readGeo1(GEO_N_NORMALS, uvec2(q), qNormal, qAlpha, qMaterial, qPath);
        if (qDepth < -0.5 || qMaterial != centerMaterial) continue;

        float w = relaxPlaneWeight(centerPos, centerNormal, qPos, depthThreshold);
        vec3 qV = -relaxSafeNormalize(qPos + RELAX_ROUGHNESS_EDGE_RELAXATION * centerPos, -centerV);
        w *= relaxSpecularNormalWeight(normalParams,
            centerNormal, qNormal, centerV, qV);
        if (w <= 1e-4) continue;
        RelaxSlowSignal qSignal = relaxUnpackSlow(texelFetch(colortex4, q, 0));
        sumSignal += vec4(qSignal.radiance, qSignal.secondMoment) * w;
        sumWeight += w;
    }
    imageStore(colorimg3, ivec2(pixel), sumSignal / max(sumWeight, 1e-6));
}
