#ifndef RT_DIFFUSE_IO_GLSL
#define RT_DIFFUSE_IO_GLSL

#include "/lib/buffers/diffuse_buffer.glsl"
#include "/lib/lighting/alice_encode.glsl"

// ===========================================================================
// Diffuse data structs — in-memory unpacked representation (storage-agnostic)
// ===========================================================================

struct diffuseIlluminationData {
    AliceEncoding data;
    AliceEncoding data_swap;
    vec3 pos;
    float surfaceMask;
    vec3 histNormal;
    float weight;
    float prev_weight;
    float meanY2;       // second moment E[Y²] for swap (current accumulated)
    float prev_meanY2;  // second moment E[Y²] for history
};

struct DiffuseIlluminationWriteData {
    AliceEncoding data_swap;
    vec3 pos;
    float surfaceMask;
    float weight;
    float meanY2;       // second moment E[Y²]
};

// ===========================================================================
// Diffuse load/fetch/write — compatible with temporal_diffuse and composite
// ===========================================================================

// Load diffuse input from current-frame RT output (used by temporal_diffuse)
DiffuseIlluminationWriteData loadDiffuseInput(ivec2 p) {
    uvec2 xy = uvec2(p);
    DiffuseIlluminationWriteData t;
    AliceEncoding alice;
    float meanY2;
    readDiffuseLightRT(xy, alice, meanY2);
    t.data_swap = alice;
    t.meanY2 = meanY2;
    float mask;
    readDiffuseGeo(xy, t.pos, mask);
    t.surfaceMask = mask;
    t.weight = 1.0;
    return t;
}

// Read history diffuse illumination (used by temporal_diffuse)
diffuseIlluminationData fetchDiffuse(ivec2 p) {
    uvec2 xy = uvec2(p);
    diffuseIlluminationData tmp;

    // swap = current frame accumulated (N=4)
    AliceEncoding alice;
    float weight, meanY2;
    readDiffuseSwap(xy, alice, weight, meanY2);
    tmp.data_swap = alice;
    tmp.weight = weight;
    tmp.meanY2 = meanY2;

#ifndef DIFFUSE_BUFFER_MIN2
    // hist = previous frame history (N=2)
    readDiffuseHist(xy, alice, weight, meanY2);
    tmp.data = alice;
    tmp.prev_weight = weight;
    tmp.prev_meanY2 = meanY2;

    // History geometry (N=3): world position plus oct-encoded geometry normal.
    readDiffuseHistGeo(xy, tmp.pos, tmp.histNormal);
#endif
    return tmp;
}

diffuseIlluminationData blendDiffuse(diffuseIlluminationData A, diffuseIlluminationData B, float x) {
    diffuseIlluminationData t;
    t.data_swap = mix_alice(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
    t.meanY2 = (B.meanY2 - A.meanY2) * x + A.meanY2;
#ifndef DIFFUSE_BUFFER_MIN2
    t.data = mix_alice(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    vec3 blendedNormal = mix(A.histNormal, B.histNormal, x);
    float blendedNormalLen2 = dot(blendedNormal, blendedNormal);
    t.histNormal = blendedNormalLen2 > 1e-8
        ? blendedNormal * inversesqrt(blendedNormalLen2)
        : vec3(0.0, 1.0, 0.0);
    t.prev_weight = (B.prev_weight - A.prev_weight) * x + A.prev_weight;
    t.prev_meanY2 = (B.prev_meanY2 - A.prev_meanY2) * x + A.prev_meanY2;
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
    return blendDiffuse(blendDiffuse(A, B, p2.x), blendDiffuse(C, D, p2.x), p2.y);
}

vec3 sampleDiffusePos(vec2 p) {
    uvec2 xy = uvec2(ivec2(floor(p) + round(fract(p))));
    vec3 pos, normal;
    readDiffuseHistGeo(xy, pos, normal);
    return pos;
}

void writeDiffuse(diffuseIlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);

    // Always write swap (N=4)
    writeDiffuseSwap(xy, data.data_swap, data.weight, data.meanY2);

#if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
    // Full write: also update hist (N=2) + hist geometry (N=3)
    writeDiffuseHist(xy, data.data, data.prev_weight, data.prev_meanY2);
    writeDiffuseHistGeo(xy, data.pos, data.histNormal);
#endif
}

// ===========================================================================
// Diffuse prev-frame (ray1.rgen guiding)
// ===========================================================================

#if defined(PREV_DIFFUSE_BUFFER)

DiffuseIlluminationWriteData fetchPrevDiffuse(ivec2 p) {
    uvec2 xy = uvec2(p);
    DiffuseIlluminationWriteData t;

    // swap = previous frame final denoised result (used by ray1.rgen for guiding)
    AliceEncoding alice;
    float weight, meanY2;
    readDiffuseSwap(xy, alice, weight, meanY2);
    t.data_swap = alice;
    t.weight = weight;
    t.meanY2 = meanY2;

    // Read position+mask from the primary G-buffer published by ray0.
    float mask;
    readDiffuseGeo(xy, t.pos, mask);
    t.surfaceMask = mask;

    return t;
}

void writePrevDiffuse(DiffuseIlluminationWriteData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    writeDiffuseSwap(xy, data.data_swap, data.weight, data.meanY2);
}

#endif

#endif // RT_DIFFUSE_IO_GLSL
