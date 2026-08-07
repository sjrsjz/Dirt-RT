#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex9;
#if RELAX_ATROUS_INPUT == 5
uniform usampler2D colortex5;
#else
uniform usampler2D colortex6;
#endif

#if RELAX_ATROUS_OUTPUT == 5
layout(rgba32ui) uniform writeonly uimage2D colorimg5;
#else
layout(rgba32ui) uniform writeonly uimage2D colorimg6;
#endif

RelaxSpatialSignal relaxLoadAtrous(ivec2 p) {
#if RELAX_ATROUS_INPUT == 5
    return relaxUnpackSpatial(texelFetch(colortex5, p, 0));
#else
    return relaxUnpackSpatial(texelFetch(colortex6, p, 0));
#endif
}

void relaxStoreAtrous(ivec2 p, RelaxSpatialSignal s) {
#if RELAX_ATROUS_OUTPUT == 5
    imageStore(colorimg5, p, relaxPackSpatial(s));
#else
    imageStore(colorimg6, p, relaxPackSpatial(s));
#endif
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    ivec2 centerPixel = ivec2(pixel);
    RelaxSpatialSignal center = relaxLoadAtrous(centerPixel);
    vec4 centerGeometry = texelFetch(colortex9, centerPixel, 0);
    vec3 centerNormal;
    uint centerMaterial;
    relaxUnpackNormalMaterial(floatBitsToUint(centerGeometry.w),
        centerNormal, centerMaterial);
    if (center.historyLength <= 0.0) {
        relaxStoreAtrous(centerPixel, center);
        return;
    }

    vec3 centerPos = centerGeometry.xyz;
    vec3 centerV = -relaxSafeNormalize(centerPos, vec3(0.0, 0.0, 1.0));
    float centerLuminance = relaxLuma(center.radiance);
    float phiInv = 1.0 / max(RELAX_SPEC_PHI_LUMINANCE *
        sqrt(center.variance), 1e-4);
    float luminanceRelaxation = RELAX_ATROUS_STEP <= 4
        ? mix(1.0, center.confidence, RELAX_LUMINANCE_RELAXATION)
        : 1.0;
    vec2 roughnessParams = relaxRoughnessWeightParams(
        center.roughness, RELAX_ROUGHNESS_FRACTION);
    vec2 normalParams = relaxNormalWeightParams(
        center.roughness, center.historyLength, center.confidence);
    float depthThreshold = RELAX_DEPTH_THRESHOLD * max(length(centerPos), 1.0);
    ivec2 stochasticOffset = ivec2(0);
    if (RELAX_ATROUS_STEP > 4) {
        vec2 jitter = relaxHash2(pixel, uint(max(frame_id, 0))) - 0.5;
        stochasticOffset = ivec2(vec2(RELAX_ATROUS_STEP) * 0.5 * jitter);
    }

    const float kernelWeight[2] = float[](0.44198, 0.27901);
    float centerWeight = kernelWeight[0] * kernelWeight[0];
    vec3 sumRadiance = center.radiance * centerWeight;
    float sumVariance = center.variance * centerWeight * centerWeight;
    float sumWeight = centerWeight;

    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        if (x == 0 && y == 0) continue;
        ivec2 q = centerPixel + stochasticOffset +
            ivec2(x, y) * RELAX_ATROUS_STEP;
        if (!relaxInBounds(q, size)) continue;
        vec4 sampleGeometry = texelFetch(colortex9, q, 0);
        vec3 sampleNormal;
        uint sampleMaterial;
        relaxUnpackNormalMaterial(floatBitsToUint(sampleGeometry.w),
            sampleNormal, sampleMaterial);
        if (sampleMaterial != centerMaterial) continue;

        RelaxSpatialSignal sampleSignal = relaxLoadAtrous(q);
        vec3 samplePos = sampleGeometry.xyz;
        vec3 sampleV = -relaxSafeNormalize(samplePos +
            RELAX_ROUGHNESS_EDGE_RELAXATION * centerPos, -centerV);
        float w = kernelWeight[abs(x)] * kernelWeight[abs(y)];
        w *= relaxPlaneWeight(centerPos, centerNormal, samplePos, depthThreshold);
        w *= relaxSpecularNormalWeight(normalParams,
            centerNormal, sampleNormal, centerV, sampleV);
        w *= relaxExponentialWeight(sampleSignal.roughness, roughnessParams);
        float luminanceDifference = min(RELAX_MAX_LUMINANCE_DIFFERENCE,
            abs(centerLuminance - relaxLuma(sampleSignal.radiance)) * phiInv);
        w *= exp(-luminanceDifference * luminanceRelaxation);
        if (w <= 1e-5) continue;
        sumRadiance += sampleSignal.radiance * w;
        sumVariance += sampleSignal.variance * w * w;
        sumWeight += w;
    }

    RelaxSpatialSignal outputSignal = center;
    outputSignal.radiance = relaxFiniteColor(sumRadiance / max(sumWeight, 1e-6));
    outputSignal.variance = sumVariance / max(sumWeight * sumWeight, 1e-8);
    relaxStoreAtrous(centerPixel, outputSignal);
}
