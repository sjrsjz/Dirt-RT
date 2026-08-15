#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

const ivec2 MAXENT_SPECULAR_PREPASS_OFFSETS[8] = ivec2[](
    ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1),
    ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    MaxEntGeometry centerGeometry = maxentLoadGeometry(pixel);
    if (!centerGeometry.valid) {
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    MaxEntPrepassSignal center = maxentUnpackPrepass(
        texelFetch(colortex6, ivec2(pixel), 0));
    if (MAXENT_SPECULAR_PREPASS_RADIUS <= 0.0) {
        imageStore(colorimg6, ivec2(pixel), maxentPackPrepass(center));
        return;
    }

    vec2 roughnessParams = maxentRoughnessWeightParams(
        centerGeometry.roughness, MAXENT_SPECULAR_PREPASS_ROUGHNESS_TOLERANCE);
    int stride = max(1, int(floor(MAXENT_SPECULAR_PREPASS_RADIUS *
        mix(0.25, 1.0, centerGeometry.roughness) + 0.5)));

    vec4 sumY = center.signal.maxEntY;
    vec2 sumCoCg = center.signal.CoCg;
    float sumWeight = 1.0;

    for (int i = 0; i < 8; ++i) {
        ivec2 q = ivec2(pixel) + MAXENT_SPECULAR_PREPASS_OFFSETS[i] * stride;
        if (!maxentInBounds(q, size)) continue;
        MaxEntGeometry qGeometry = maxentLoadGeometry(uvec2(q));
        if (!qGeometry.valid
                || qGeometry.materialID != centerGeometry.materialID)
            continue;

        float w = maxentSpatialPlaneWeight(centerGeometry.position,
            centerGeometry.normal, qGeometry.position);
        w *= maxentExponentialWeight(qGeometry.roughness, roughnessParams);
        MaxEntPrepassSignal sampleSignal = maxentUnpackPrepass(
            texelFetch(colortex6, q, 0));
        if (w <= 1e-4) continue;

        sumY += sampleSignal.signal.maxEntY * w;
        sumCoCg += sampleSignal.signal.CoCg * w;
        sumWeight += w;
    }

    float invWeight = 1.0 / max(sumWeight, 1e-6);
    MaxEntPrepassSignal outputSignal;
    outputSignal.signal.maxEntY = sumY * invWeight;
    outputSignal.signal.CoCg = sumCoCg * invWeight;
    outputSignal.signal = sanitizeSpecularMaxEnt(outputSignal.signal);
    // Temporal virtual reprojection still consumes the raw center distance;
    // spatial filtering must neither reject by nor blur this descriptor.
    outputSignal.hitDistance = center.hitDistance;
    imageStore(colorimg6, ivec2(pixel), maxentPackPrepass(outputSignal));
}
