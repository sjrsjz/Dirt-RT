#ifndef RT_DIFFUSE_IO_GLSL
#define RT_DIFFUSE_IO_GLSL

#include "/lib/buffers/diffuse_buffer.glsl"
#include "/lib/lighting/maxent_encode.glsl"
#include "/lib/math/statistics.glsl"

// In-memory wrappers preserve the storage ABI's root second moment. Any
// interpolation squares the roots, blends E[Y^2] linearly, then takes the one
// required root for the resulting stored value.
struct diffuseIlluminationData {
    MaxEntEncoding data;
    MaxEntEncoding data_swap;
    vec3 pos;
    float surfaceMask;
    vec3 histNormal;
    float weight;
    float prev_weight;
    float rootMeanY2;
    float prevRootMeanY2;
};

struct DiffuseIlluminationWriteData {
    MaxEntEncoding data_swap;
    vec3 pos;
    float surfaceMask;
    float weight;
    float rootMeanY2;
};

DiffuseIlluminationWriteData loadDiffuseInput(ivec2 p) {
    uvec2 xy = uvec2(p);
    DiffuseIlluminationWriteData t;
    MaxEntEncoding maxent;
    readDiffuseLightRT(xy, maxent, t.rootMeanY2);
    t.data_swap = maxent;
    float distance;
    readDiffusePrimaryGeometry(xy, t.pos, distance);
    t.surfaceMask = distance >= 0.0 ? 1.0 : 0.0;
    t.weight = 1.0;
    return t;
}

diffuseIlluminationData fetchDiffuse(ivec2 p) {
    uvec2 xy = uvec2(p);
    diffuseIlluminationData tmp;

    MaxEntEncoding maxent;
    float weight;
    readDiffuseSwap(xy, maxent, weight, tmp.rootMeanY2);
    tmp.data_swap = maxent;
    tmp.weight = weight;

#ifndef DIFFUSE_BUFFER_MIN2
    readDiffuseHist(xy, maxent, weight, tmp.prevRootMeanY2);
    tmp.data = maxent;
    tmp.prev_weight = weight;
    readDiffuseHistGeo(xy, tmp.pos, tmp.histNormal);
#endif
    return tmp;
}

float blendDiffuseRootMeanY2(float rootA, float rootB, float x) {
    return sqrt(max(mix(rootA * rootA, rootB * rootB, x), 0.0));
}

diffuseIlluminationData blendDiffuse(diffuseIlluminationData A,
        diffuseIlluminationData B, float x) {
    diffuseIlluminationData t;
    t.data_swap = mix_maxent(A.data_swap, B.data_swap, x);
    t.weight = statisticsKishBlendEffectiveSampleCounts(
        A.weight, B.weight, x);
    t.rootMeanY2 = blendDiffuseRootMeanY2(
        A.rootMeanY2, B.rootMeanY2, x);
#ifndef DIFFUSE_BUFFER_MIN2
    t.data = mix_maxent(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    vec3 blendedNormal = mix(A.histNormal, B.histNormal, x);
    float blendedNormalLen2 = dot(blendedNormal, blendedNormal);
    t.histNormal = blendedNormalLen2 > 1e-8
        ? blendedNormal * inversesqrt(blendedNormalLen2)
        : vec3(0.0, 1.0, 0.0);
    t.prev_weight = statisticsKishBlendEffectiveSampleCounts(
        A.prev_weight, B.prev_weight, x);
    t.prevRootMeanY2 = blendDiffuseRootMeanY2(
        A.prevRootMeanY2, B.prevRootMeanY2, x);
#endif
    return t;
}

diffuseIlluminationData sampleDiffuse(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    diffuseIlluminationData A = fetchDiffuse(p1);
    diffuseIlluminationData B = fetchDiffuse(p1 + ivec2(1, 0));
    diffuseIlluminationData C = fetchDiffuse(p1 + ivec2(0, 1));
    diffuseIlluminationData D = fetchDiffuse(p1 + ivec2(1, 1));
    return blendDiffuse(blendDiffuse(A, B, p2.x),
        blendDiffuse(C, D, p2.x), p2.y);
}

vec3 sampleDiffusePos(vec2 p) {
    uvec2 xy = uvec2(ivec2(floor(p) + round(fract(p))));
    vec3 pos, normal;
    readDiffuseHistGeo(xy, pos, normal);
    return pos;
}

void writeDiffuse(diffuseIlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    writeDiffuseSwap(xy, data.data_swap, data.weight, data.rootMeanY2);

#if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
    writeDiffuseHist(xy, data.data, data.prev_weight,
        data.prevRootMeanY2);
    writeDiffuseHistGeo(xy, data.pos, data.histNormal, data.prev_weight);
#endif
}

#if defined(PREV_DIFFUSE_BUFFER)

DiffuseIlluminationWriteData fetchPrevDiffuse(ivec2 p) {
    uvec2 xy = uvec2(p);
    DiffuseIlluminationWriteData t;
    MaxEntEncoding maxent;
    float weight;
    readDiffuseSwap(xy, maxent, weight, t.rootMeanY2);
    t.data_swap = maxent;
    t.weight = weight;

    float distance;
    readDiffusePrimaryGeometry(xy, t.pos, distance);
    t.surfaceMask = distance >= 0.0 ? 1.0 : 0.0;
    return t;
}

void writePrevDiffuse(DiffuseIlluminationWriteData data, ivec2 p) {
    writeDiffuseSwap(uvec2(p), data.data_swap, data.weight,
        data.rootMeanY2);
}

#endif

#endif // RT_DIFFUSE_IO_GLSL
