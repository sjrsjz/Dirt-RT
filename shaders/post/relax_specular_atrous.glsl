#version 430 compatibility

#if defined(RELAX_ATROUS_SHARED)
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 8, local_size_y = 8) in;
#endif

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

uvec4 relaxFetchAtrousPacked(ivec2 p) {
#if RELAX_ATROUS_INPUT == 5
    return texelFetch(colortex5, p, 0);
#else
    return texelFetch(colortex6, p, 0);
#endif
}

RelaxSpatialSignal relaxLoadAtrous(ivec2 p) {
    return relaxUnpackSpatial(relaxFetchAtrousPacked(p));
}

void relaxStoreAtrous(ivec2 p, RelaxSpatialSignal s) {
#if RELAX_ATROUS_OUTPUT == 5
    imageStore(colorimg5, p, relaxPackSpatial(s));
#else
    imageStore(colorimg6, p, relaxPackSpatial(s));
#endif
}

void relaxStoreAtrousDebug(ivec2 p, RelaxSpatialSignal s) {
#if DEBUG_VIEW == RELAX_ATROUS_DEBUG_VIEW
    writeReflLight(uvec2(p), relaxFiniteColor(s.radiance),
        s.endpointDistance, s.historyLength);
#endif
}

void relaxResolveAtrous(ivec2 p, RelaxSpatialSignal s) {
#if defined(RELAX_ATROUS_RESOLVE)
#if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14 || \
        (DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30) || \
        (DEBUG_VIEW >= 38 && DEBUG_VIEW <= 40)
    // Preserve a diagnostic result written by its owning pass.
#else
    writeReflLight(uvec2(p), relaxFiniteColor(s.radiance),
        s.endpointDistance, s.historyLength);
#endif
#endif
}

#if defined(RELAX_ATROUS_SHARED)

#define RELAX_ATROUS_GROUP_SIZE 16
#define RELAX_ATROUS_HALO RELAX_ATROUS_STEP
#define RELAX_ATROUS_TILE_SIZE \
    (RELAX_ATROUS_GROUP_SIZE + 2 * RELAX_ATROUS_HALO)
#define RELAX_ATROUS_TILE_AREA \
    (RELAX_ATROUS_TILE_SIZE * RELAX_ATROUS_TILE_SIZE)

shared vec4 relaxSharedGeometry[RELAX_ATROUS_TILE_AREA];
shared uvec4 relaxSharedSignal[RELAX_ATROUS_TILE_AREA];

#else

// Same eight-point Poisson disk used by the diffuse large-radius stages.
// xy: normalized offset, z: radius, w: exp(-radius^2 / 2).
const vec4 RELAX_POISSON_8[8] = vec4[](
    vec4(-0.4706069, -0.4427112, 0.6461146, 0.81170),
    vec4(-0.9057375,  0.3003471, 0.9542373, 0.63422),
    vec4(-0.3487388,  0.4037880, 0.5335386, 0.86734),
    vec4( 0.1023042,  0.6439373, 0.6520134, 0.80847),
    vec4( 0.5699277,  0.3513750, 0.6695386, 0.79925),
    vec4( 0.2939128, -0.1131226, 0.3149309, 0.95161),
    vec4( 0.7836658, -0.4208784, 0.8895339, 0.67328),
    vec4( 0.1564120, -0.8198990, 0.8346850, 0.70589)
);

#endif

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(resolution_global);

#if defined(RELAX_ATROUS_SHARED)
    uint localIndex = gl_LocalInvocationIndex;
    ivec2 tileOrigin =
        ivec2(gl_WorkGroupID.xy) * RELAX_ATROUS_GROUP_SIZE -
        ivec2(RELAX_ATROUS_HALO);

    // Every invocation reaches the barrier, including workgroups partially
    // outside the image. Invalid halo entries carry zero history and are never
    // consumed as valid samples.
    for (uint i = localIndex; i < uint(RELAX_ATROUS_TILE_AREA); i += 256u) {
        int tx = int(i % uint(RELAX_ATROUS_TILE_SIZE));
        int ty = int(i / uint(RELAX_ATROUS_TILE_SIZE));
        ivec2 q = tileOrigin + ivec2(tx, ty);
        if (relaxInBounds(q, size)) {
            relaxSharedGeometry[i] = texelFetch(colortex9, q, 0);
            relaxSharedSignal[i] = relaxFetchAtrousPacked(q);
        } else {
            relaxSharedGeometry[i] = vec4(0.0);
            relaxSharedSignal[i] = uvec4(0u);
        }
    }
    memoryBarrierShared();
    barrier();
#endif

    if (!relaxInBounds(pixel, size)) return;

    vec4 centerGeometry;
    RelaxSpatialSignal center;
#if defined(RELAX_ATROUS_SHARED)
    ivec2 centerTile = ivec2(gl_LocalInvocationID.xy) +
        ivec2(RELAX_ATROUS_HALO);
    uint centerIndex = uint(centerTile.y * RELAX_ATROUS_TILE_SIZE +
        centerTile.x);
    centerGeometry = relaxSharedGeometry[centerIndex];
    center = relaxUnpackSpatial(relaxSharedSignal[centerIndex]);
#else
    centerGeometry = texelFetch(colortex9, pixel, 0);
    center = relaxLoadAtrous(pixel);
#endif

    if (center.historyLength <= 0.0) {
        relaxStoreAtrous(pixel, center);
        relaxResolveAtrous(pixel, center);
        relaxStoreAtrousDebug(pixel, center);
        return;
    }

    vec3 centerNormal;
    uint centerMaterial;
    relaxUnpackNormalMaterial(floatBitsToUint(centerGeometry.w),
        centerNormal, centerMaterial);
    vec3 centerPos = centerGeometry.xyz;
    vec3 centerV = -relaxSafeNormalize(centerPos,
        vec3(0.0, 0.0, 1.0));
    float centerLuminance = relaxLuma(center.radiance);
    float phiInv = 1.0 / max(RELAX_SPEC_PHI_LUMINANCE *
        sqrt(center.variance), 1e-4);
#if defined(RELAX_ATROUS_SHARED)
    float luminanceRelaxation = mix(
        1.0, center.confidence, RELAX_LUMINANCE_RELAXATION);
#else
    float luminanceRelaxation = 1.0;
#endif
    vec2 roughnessParams = relaxRoughnessWeightParams(
        center.roughness, RELAX_ROUGHNESS_FRACTION);
    vec2 normalParams = relaxNormalWeightParams(
        center.roughness, center.historyLength, center.confidence);
    float depthThreshold = RELAX_DEPTH_THRESHOLD *
        max(length(centerPos), 1.0);

#if defined(RELAX_ATROUS_SHARED)
    const ivec2 gridOffset[8] = ivec2[](
        ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
        ivec2(-1,  0),                 ivec2(1,  0),
        ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1)
    );
    const float gridWeight[8] = float[](
        0.07785, 0.12331, 0.07785,
        0.12331,          0.12331,
        0.07785, 0.12331, 0.07785
    );
    const float centerWeight = 0.44198 * 0.44198;
#else
    // A pass-specific but frame-stable rotation prevents the two Poisson
    // stages from sharing a visible sampling pattern without temporal shimmer.
    vec2 rotationSeed = relaxHash2(uvec2(pixel),
        uint(RELAX_ATROUS_STEP));
    float theta = 2.0 * PI * rotationSeed.x;
    float cs = cos(theta);
    float sn = sin(theta);
    mat2 rotation = mat2(cs, -sn, sn, cs) *
        (float(RELAX_ATROUS_STEP) * 1.75);
    const float centerWeight = 1.0;
#endif

    vec3 sumRadiance = center.radiance * centerWeight;
    float sumVariance = center.variance * centerWeight * centerWeight;
    float sumWeight = centerWeight;

    for (int i = 0; i < 8; ++i) {
        ivec2 q;
        float kernelWeight;
        vec4 sampleGeometry;
        RelaxSpatialSignal sampleSignal;

#if defined(RELAX_ATROUS_SHARED)
        ivec2 tileOffset = gridOffset[i] * RELAX_ATROUS_STEP;
        ivec2 sampleTile = centerTile + tileOffset;
        uint sampleIndex = uint(
            sampleTile.y * RELAX_ATROUS_TILE_SIZE + sampleTile.x);
        q = pixel + tileOffset;
        kernelWeight = gridWeight[i];
        sampleGeometry = relaxSharedGeometry[sampleIndex];
        sampleSignal = relaxUnpackSpatial(relaxSharedSignal[sampleIndex]);
#else
        vec4 poisson = RELAX_POISSON_8[i];
        q = pixel + ivec2(round(rotation * poisson.xy));
        if (!relaxInBounds(q, size)) continue;
        kernelWeight = poisson.w;
        sampleGeometry = texelFetch(colortex9, q, 0);
        sampleSignal = relaxLoadAtrous(q);
#endif

        if (sampleSignal.historyLength <= 0.0) continue;

        vec3 sampleNormal;
        uint sampleMaterial;
        relaxUnpackNormalMaterial(floatBitsToUint(sampleGeometry.w),
            sampleNormal, sampleMaterial);
        if (sampleMaterial != centerMaterial) continue;

        vec3 samplePos = sampleGeometry.xyz;
        vec3 sampleV = -relaxSafeNormalize(samplePos +
            RELAX_ROUGHNESS_EDGE_RELAXATION * centerPos, -centerV);
        float w = kernelWeight;
        w *= relaxPlaneWeight(centerPos, centerNormal,
            samplePos, depthThreshold);
        w *= relaxSpecularNormalWeight(normalParams,
            centerNormal, sampleNormal, centerV, sampleV);
        w *= relaxExponentialWeight(
            sampleSignal.roughness, roughnessParams);

        // length(E[X]) from the temporally filtered four-moment endpoint
        // state replaces ReLAX's separately accumulated hit distance.
        float hitScale = max(max(center.endpointDistance,
            sampleSignal.endpointDistance), 1.0);
        float hitWeight = exp(-abs(sampleSignal.endpointDistance -
            center.endpointDistance) / (hitScale * mix(0.02, 0.5,
                center.roughness) + 1e-5));
        w *= mix(RELAX_MIN_HIT_DISTANCE_WEIGHT, 1.0, hitWeight);

#if defined(RELAX_ATROUS_SHARED)
        float luminanceDifference = min(
            RELAX_MAX_LUMINANCE_DIFFERENCE,
            abs(centerLuminance - relaxLuma(sampleSignal.radiance)) *
                phiInv);
#else
        // A capped luminance rejection leaves exp(-2) ≈ 13.5% of every
        // high-contrast sample alive.  That is tolerable for the dense small
        // kernels, but is a visible leak for the sparse 8/16-pixel passes.
        float luminanceDifference = abs(centerLuminance -
            relaxLuma(sampleSignal.radiance)) * phiInv;
#endif
        w *= exp(-luminanceDifference * luminanceRelaxation);
        if (w <= 1e-5) continue;

        sumRadiance += sampleSignal.radiance * w;
        sumVariance += sampleSignal.variance * w * w;
        sumWeight += w;
    }

    RelaxSpatialSignal outputSignal = center;
    outputSignal.radiance = relaxFiniteColor(
        sumRadiance / max(sumWeight, 1e-6));
    outputSignal.variance = sumVariance /
        max(sumWeight * sumWeight, 1e-8);
    relaxStoreAtrous(pixel, outputSignal);
    relaxResolveAtrous(pixel, outputSignal);
    relaxStoreAtrousDebug(pixel, outputSignal);
}
