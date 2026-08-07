#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

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

    vec3 raw;
    float centerHitDistance, unusedWeight;
    readReflLight(pixel, raw, centerHitDistance, unusedWeight);
    raw = relaxFiniteColor(raw);

    RelaxFastSignal outputSignal;
    outputSignal.radiance = raw;
    outputSignal.hitDistance = max(centerHitDistance, 0.0);
    outputSignal.historyLength = 0.0;
    outputSignal.confidence = primaryDistance > -0.5 ? 1.0 : 0.0;
    outputSignal.materialID = 0u;

    if (primaryDistance < -0.5) {
        imageStore(colorimg6, ivec2(pixel), relaxPackFast(outputSignal));
        return;
    }

    vec3 centerNormal;
    float centerAlpha, centerPathRoughness;
    int centerMaterial;
    readGeo1(GEO_N_NORMALS, pixel, centerNormal, centerAlpha,
        centerMaterial, centerPathRoughness);
    float centerRoughness = relaxPerceptualRoughness(centerAlpha);
    outputSignal.materialID = uint(max(centerMaterial, 0));
    if (RELAX_PREPASS_RADIUS <= 0.0) {
        imageStore(colorimg6, ivec2(pixel), relaxPackFast(outputSignal));
        return;
    }
    vec3 sumRadiance = raw;
    float sumHitDistance = max(centerHitDistance, 0.0);
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
        float sampleHitDistance, sampleWeight;
        readReflLight(uvec2(q), sampleRadiance, sampleHitDistance, sampleWeight);
        sampleRadiance = relaxFiniteColor(sampleRadiance);

        float hitScale = max(max(centerHitDistance, sampleHitDistance), 1.0);
        float hitWeight = exp(-abs(sampleHitDistance - centerHitDistance) /
            (hitScale * mix(0.02, 0.5, centerRoughness) + 1e-5));
        w *= mix(RELAX_MIN_HIT_DISTANCE_WEIGHT, 1.0, hitWeight);

        sumRadiance += sampleRadiance * w;
        sumHitDistance += max(sampleHitDistance, 0.0) * w;
        sumWeight += w;
    }

    outputSignal.radiance = relaxFiniteColor(sumRadiance / max(sumWeight, 1e-6));
    outputSignal.hitDistance = sumHitDistance / max(sumWeight, 1e-6);
    imageStore(colorimg6, ivec2(pixel), relaxPackFast(outputSignal));
}
