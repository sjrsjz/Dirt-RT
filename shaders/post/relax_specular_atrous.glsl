#version 430 core

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

const ivec2 RELAX_GRID_8[8] = ivec2[](
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0),
    ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));
const float RELAX_GRID_WEIGHT[8] = float[](
    0.07785, 0.12331, 0.07785, 0.12331,
    0.12331, 0.07785, 0.12331, 0.07785);
const vec4 RELAX_POISSON_8[8] = vec4[](
    vec4(-0.4706069, -0.4427112, 0.6461146, 0.81170),
    vec4(-0.9057375,  0.3003471, 0.9542373, 0.63422),
    vec4(-0.3487388,  0.4037880, 0.5335386, 0.86734),
    vec4( 0.1023042,  0.6439373, 0.6520134, 0.80847),
    vec4( 0.5699277,  0.3513750, 0.6695386, 0.79925),
    vec4( 0.2939128, -0.1131226, 0.3149309, 0.95161),
    vec4( 0.7836658, -0.4208784, 0.8895339, 0.67328),
    vec4( 0.1564120, -0.8198990, 0.8346850, 0.70589));

uvec4 relaxFetchAtrousWords(ivec2 p) {
#if RELAX_ATROUS_INPUT == 5
    return texelFetch(colortex5, p, 0);
#else
    return texelFetch(colortex6, p, 0);
#endif
}

void relaxStoreAtrous(ivec2 p, RelaxSpatialSignal s) {
#if RELAX_ATROUS_OUTPUT == 5
    imageStore(colorimg5, p, relaxPackSpatial(s));
#else
    imageStore(colorimg6, p, relaxPackSpatial(s));
#endif
}

void relaxFinishAtrous(ivec2 p, RelaxSpatialSignal s) {
    relaxStoreAtrous(p, s);
#if DEBUG_VIEW == RELAX_ATROUS_DEBUG_VIEW
    writeReflLight(uvec2(p), specularMaxEntTotalRgb(s.signal),
        s.hitDistance, 1.0);
#endif
#if defined(RELAX_ATROUS_RESOLVE)
#if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14 || \
        (DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30)
    // Preserve the diagnostic value written by its owning pass.
#else
    writeReflMaxEnt(uvec2(p), s.signal, s.hitDistance, 1.0);
#endif
#endif
}

bool relaxGeometryValid(vec4 g) {
    return any(notEqual(g.xyz, vec3(0.0))) || floatBitsToUint(g.w) != 0u;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(resolution_global);
    if (!relaxInBounds(pixel, size)) return;

    vec4 centerGeometry = texelFetch(colortex9, pixel, 0);
    RelaxSpatialSignal center = relaxUnpackSpatial(
        relaxFetchAtrousWords(pixel));
    if (!relaxGeometryValid(centerGeometry)) {
        relaxStoreAtrous(pixel, center);
#if defined(RELAX_ATROUS_RESOLVE)
        writeReflMaxEnt(uvec2(pixel), emptySpecularMaxEnt(), 0.0, 0.0);
#endif
        return;
    }

    vec3 centerNormal;
    uint centerMaterial;
    relaxUnpackNormalMaterial(floatBitsToUint(centerGeometry.w),
        centerNormal, centerMaterial);
    vec3 centerPos = centerGeometry.xyz;
    float centerAlpha, centerPath;
    int centerMaterialGBuffer;
    vec3 centerNormalGBuffer;
    readGeo1(GEO_N_NORMALS, uvec2(pixel), centerNormalGBuffer,
        centerAlpha, centerMaterialGBuffer, centerPath);
    float centerRoughness = relaxPerceptualRoughness(centerAlpha);
    float phiInv = 1.0 / max(RELAX_SPEC_PHI_LUMINANCE *
        sqrt(center.variance), 1e-4);
    float depthThreshold = RELAX_DEPTH_THRESHOLD *
        max(length(centerPos), 1.0);

#if defined(RELAX_ATROUS_POISSON)
    vec2 seed = relaxHash2(uvec2(pixel), uint(RELAX_ATROUS_STEP));
    float theta = 2.0 * PI * seed.x;
    float cs = cos(theta);
    float sn = sin(theta);
    mat2 rotation = mat2(cs, -sn, sn, cs) *
        (float(RELAX_ATROUS_STEP) * 1.75);
    float centerWeight = 1.0;
#else
    float centerWeight = 0.44198 * 0.44198;
#endif

    vec4 sumY = center.signal.aliceY * centerWeight;
    vec2 sumCoCg = center.signal.CoCg * centerWeight;
    float sumVariance = center.variance * centerWeight * centerWeight;
    float sumWeight = centerWeight;

    for (int i = 0; i < 8; ++i) {
        ivec2 q;
        float kernelWeight;
#if defined(RELAX_ATROUS_POISSON)
        q = pixel + ivec2(round(rotation * RELAX_POISSON_8[i].xy));
        kernelWeight = RELAX_POISSON_8[i].w;
#else
        q = pixel + RELAX_GRID_8[i] * RELAX_ATROUS_STEP;
        kernelWeight = RELAX_GRID_WEIGHT[i];
#endif
        if (!relaxInBounds(q, size)) continue;
        vec4 qGeometry = texelFetch(colortex9, q, 0);
        if (!relaxGeometryValid(qGeometry)) continue;
        vec3 qNormal;
        uint qMaterial;
        relaxUnpackNormalMaterial(floatBitsToUint(qGeometry.w),
            qNormal, qMaterial);
        if (qMaterial != centerMaterial) continue;

        RelaxSpatialSignal qSignal = relaxUnpackSpatial(
            relaxFetchAtrousWords(q));
        float w = kernelWeight;
        w *= relaxPlaneWeight(centerPos, centerNormal,
            qGeometry.xyz, depthThreshold);

        float hitScale = max(max(center.hitDistance,
            qSignal.hitDistance), 1.0);
        float hitSigma = hitScale * mix(0.02, 0.5,
            centerRoughness) + 1e-5;
        float hitWeight = exp(-abs(qSignal.hitDistance -
            center.hitDistance) / hitSigma);
        w *= mix(RELAX_MIN_HIT_DISTANCE_WEIGHT, 1.0, hitWeight);

        float luminanceDifference = abs(center.signal.aliceY.w -
            qSignal.signal.aliceY.w) * phiInv;
#if !defined(RELAX_ATROUS_POISSON)
        luminanceDifference = min(RELAX_MAX_LUMINANCE_DIFFERENCE,
            luminanceDifference);
#endif
        w *= exp(-luminanceDifference);
        if (w <= 1e-5) continue;

        sumY += qSignal.signal.aliceY * w;
        sumCoCg += qSignal.signal.CoCg * w;
        sumVariance += qSignal.variance * w * w;
        sumWeight += w;
    }

    float invWeight = 1.0 / max(sumWeight, 1e-6);
    RelaxSpatialSignal outputSignal = center;
    outputSignal.signal.aliceY = sumY * invWeight;
    outputSignal.signal.CoCg = sumCoCg * invWeight;
    outputSignal.signal = sanitizeSpecularMaxEnt(outputSignal.signal);
    outputSignal.variance = sumVariance /
        max(sumWeight * sumWeight, 1e-8);
    relaxFinishAtrous(pixel, outputSignal);
}
