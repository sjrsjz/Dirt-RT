#version 430 compatibility

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

    vec4 slowRaw = texelFetch(colortex3, ivec2(pixel), 0);
    RelaxFastSignal fast = relaxUnpackFast(texelFetch(colortex4, ivec2(pixel), 0));
    vec3 centerPos, centerNormal;
    float centerDepth, centerAlpha, centerPath;
    int centerMaterialInt;
    readGeo0(GEO_N_GEO, pixel, centerPos, centerDepth);
    readGeo1(GEO_N_NORMALS, pixel, centerNormal, centerAlpha,
        centerMaterialInt, centerPath);
    uint centerMaterial = uint(max(centerMaterialInt, 0));
    float centerRoughness = relaxPerceptualRoughness(centerAlpha);

    uint geometryPack = relaxPackNormalMaterial(centerNormal, centerMaterial);
    imageStore(colorimg9, ivec2(pixel), vec4(centerPos,
        uintBitsToFloat(geometryPack)));

    if (centerDepth < -0.5) {
        RelaxSpatialSignal sky;
        sky.radiance = relaxFiniteColor(slowRaw.rgb);
        sky.roughness = 1.0;
        sky.variance = 0.0;
        sky.endpointDistance = 0.0;
        sky.historyLength = 0.0;
        sky.confidence = 0.0;
        imageStore(colorimg6, ivec2(pixel), relaxPackSpatial(sky));
#if DEBUG_VIEW == 25
        writeReflLight(pixel, sky.radiance, sky.endpointDistance,
            sky.historyLength);
#endif
        return;
    }

    vec4 filteredMoments = slowRaw;
    if (fast.historyLength < RELAX_HISTORY_THRESHOLD) {
        vec4 sum = vec4(0.0);
        float sumWeight = 0.0;
        float depthThreshold = RELAX_DEPTH_THRESHOLD * max(length(centerPos), 1.0);
        vec3 centerV = -relaxSafeNormalize(centerPos, vec3(0.0, 0.0, 1.0));
        vec2 normalParams = relaxNormalWeightParams(
            centerRoughness, fast.historyLength, fast.confidence);
        for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
            ivec2 q = ivec2(pixel) + ivec2(x, y);
            if (!relaxInBounds(q, size)) continue;
            vec3 qPos, qNormal;
            float qDepth, qAlpha, qPath;
            int qMaterial;
            readGeo0(GEO_N_GEO, uvec2(q), qPos, qDepth);
            readGeo1(GEO_N_NORMALS, uvec2(q), qNormal, qAlpha, qMaterial, qPath);
            if (qDepth < -0.5 || qMaterial != centerMaterialInt) continue;
            vec3 qV = -relaxSafeNormalize(qPos + RELAX_ROUGHNESS_EDGE_RELAXATION * centerPos, -centerV);
            float w = relaxPlaneWeight(centerPos, centerNormal, qPos, depthThreshold);
            w *= relaxSpecularNormalWeight(normalParams,
                centerNormal, qNormal, centerV, qV);
            float gaussian = exp(-0.5 * float(x * x + y * y) / 4.0);
            w *= gaussian;
            sum += texelFetch(colortex3, q, 0) * w;
            sumWeight += w;
        }
        if (sumWeight > 1e-5) filteredMoments = sum / sumWeight;
    }

    float firstMoment = relaxLuma(filteredMoments.rgb);
    float variance = max(filteredMoments.a - firstMoment * firstMoment, 0.0);
    if (fast.historyLength < RELAX_HISTORY_THRESHOLD)
        variance *= max(1.0, RELAX_HISTORY_THRESHOLD /
            max(fast.historyLength, 1.0));

    RelaxSpatialSignal spatial;
    spatial.radiance = relaxFiniteColor(filteredMoments.rgb);
    spatial.roughness = centerRoughness;
    spatial.variance = variance;
    spatial.endpointDistance = fast.endpointDistance;
    spatial.historyLength = fast.historyLength;
    spatial.confidence = fast.confidence;
    imageStore(colorimg6, ivec2(pixel), relaxPackSpatial(spatial));
#if DEBUG_VIEW == 25
    // Spatial preparation output before the first A-trous pass.
    writeReflLight(pixel, relaxFiniteColor(spatial.radiance),
        spatial.endpointDistance, spatial.historyLength);
#endif
}
