#ifndef RT_SPECULAR_IO_GLSL
#define RT_SPECULAR_IO_GLSL

#include "/lib/buffers/specular_buffer.glsl"
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/common/pack_half.glsl"

// ===========================================================================
// Shared specular data struct
// ===========================================================================

struct vec3IlluminationData {
    vec3 data;
    vec3 data_swap;
    vec3 pos;
    vec3 normal;
    float weight;
    float prev_weight;
};

// ===========================================================================
// PackedLightSample — for colortex3/4 texture I/O (non-SSBO)
// ===========================================================================

struct PackedLightSample {
    vec4 data0;
    uvec4 data1;
};

PackedLightSample packSpecularSample(vec3 pos, vec3 R, vec3 radiance,
    float roughness, float variance, float virtualProjDist, vec3 H) {
    PackedLightSample s;
    s.data0 = vec4(pos, encodeNormal(R));
    s.data1 = uvec4(
        packHalf2x16(vec2(clamp(radiance.r, -65504.0, 65504.0), clamp(radiance.g, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(radiance.b, -65504.0, 65504.0), clamp(roughness, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(variance, -65504.0, 65504.0), clamp(virtualProjDist, -65504.0, 65504.0))),
        encodeNormalU(H)
    );
    return s;
}

void unpackSpecularSample(PackedLightSample s,
    out vec3 pos, out vec3 R, out vec3 radiance,
    out float roughness, out float variance, out float virtualProjDist, out vec3 H) {
    pos = s.data0.xyz;
    R = decodeNormal(s.data0.w);
    vec2 rg = unpackHalf2x16(s.data1.x);
    vec2 br = unpackHalf2x16(s.data1.y);
    vec2 vv = unpackHalf2x16(s.data1.z);
    radiance  = vec3(rg.x, rg.y, br.x);
    roughness = br.y;
    variance  = vv.x;
    virtualProjDist = vv.y;
    H = decodeNormalU(s.data1.w);
}

// A-trous only filters radiance/variance and avoids constructing data0 again
// when the pass writes data1.
void unpackSpecularFilterSample(vec4 geometry, uvec4 light,
    out vec3 pos, out vec3 radiance, out float roughness,
    out float variance, out float virtualProjDist, out vec3 H) {
    pos = geometry.xyz;
    vec2 rg = unpackHalf2x16(light.x);
    vec2 br = unpackHalf2x16(light.y);
    vec2 vv = unpackHalf2x16(light.z);
    radiance = vec3(rg, br.x);
    roughness = br.y;
    variance = vv.x;
    virtualProjDist = vv.y;
    H = decodeNormalU(light.w);
}

uvec4 packSpecularFilterLight(vec3 radiance, float roughness,
    float variance, float virtualProjDist, vec3 H) {
    return uvec4(
        packHalf2x16(clamp(radiance.rg, -65504.0, 65504.0)),
        packHalf2x16(clamp(vec2(radiance.b, roughness), -65504.0, 65504.0)),
        packHalf2x16(clamp(vec2(variance, virtualProjDist), -65504.0, 65504.0)),
        encodeNormalU(H));
}

// ===========================================================================
// SpecularRT pack/unpack — compatible with raytrace_rgen and temporal interfaces
// ===========================================================================

struct SpecularRTWriteData {
    vec3 pos;
    vec3 R;
    float virtualProjDist;
    vec3 color;
};

SpecularRTWriteData packSpecularRT(vec3 pos, vec3 R, float virtualProjDist, vec3 color) {
    SpecularRTWriteData e;
    e.pos = pos;
    e.R = R;
    e.virtualProjDist = virtualProjDist;
    e.color = color;
    return e;
}

void unpackSpecularRT_Refl(uvec2 xy, out vec3 pos, out vec3 R, out vec3 color, out float virtualProjDist) {
    float distance = readPrimaryDistance(xy);
    pos = reconstructPrimaryRelativePosition(xy, distance);
    R = readReflSampleDirection(xy);
    float accumW;
    readReflLight(xy, color, virtualProjDist, accumW);
}

void unpackSpecularRT_Refr(uvec2 xy, out vec3 pos, out vec3 R, out vec3 color, out float virtualProjDist) {
    readRefrGeo (xy, pos, R);
    float accumW;
    readRefrLight(xy, color, virtualProjDist, accumW);
}

// ===========================================================================
// Reflect fetch/write — compatible with temporal_reflect + composite
// ===========================================================================

#if defined(REFLECT_BUFFER) || defined(REFLECT_BUFFER_MIN) || defined(REFLECT_BUFFER_MIN2)

vec3IlluminationData fetchReflect(ivec2 p) {
    uvec2 xy = uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1));
    vec3IlluminationData tmp;

    // Current frame accumulated (N=1)
    float vproj;
    readReflLight(xy, tmp.data_swap, vproj, tmp.weight);

#ifndef REFLECT_BUFFER_MIN2
    MaxEntSpecularHistory history = readMaxEntSpecularHistory(xy);
    tmp.data = specularMaxEntTotalRgb(history.signal);
    tmp.prev_weight = history.historyLength;
    tmp.pos = history.surfacePosition;
    tmp.normal = history.geometryNormal * history.hitDistance;
#endif
    return tmp;
}

vec3IlluminationData blendReflect(vec3IlluminationData A, vec3IlluminationData B, float x) {
    vec3IlluminationData t;
    t.data_swap = mix(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
#ifndef REFLECT_BUFFER_MIN2
    t.prev_weight = (B.prev_weight - A.prev_weight) * x + A.prev_weight;
    t.data = mix(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = mix(A.normal, B.normal, x);
#endif
    return t;
}

bool fetchReflectHistoryGeometry(ivec2 p, out vec3 pos, out vec3 normal) {
    uvec2 xy = uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1));
    MaxEntSpecularHistory history = readMaxEntSpecularHistory(xy);
    pos = history.surfacePosition;
    normal = history.geometryNormal;
    return history.historyLength >= 0.5;
}

vec3IlluminationData sampleReflect(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    vec3IlluminationData A = fetchReflect(p1);
    vec3IlluminationData B = fetchReflect(p1 + ivec2(1, 0));
    vec3IlluminationData C = fetchReflect(p1 + ivec2(0, 1));
    vec3IlluminationData D = fetchReflect(p1 + ivec2(1, 1));
    return blendReflect(blendReflect(A, B, p2.x), blendReflect(C, D, p2.x), p2.y);
}

void writeReflect(vec3IlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    // Read vprojDist to preserve, only update color + weight
    vec3 color; float vproj, accumW;
    readReflLight(xy, color, vproj, accumW);
    writeReflLight(xy, data.data_swap, vproj, data.weight);
}

#endif

// ===========================================================================
// Refract fetch/write — compatible with temporal_refract + composite
// ===========================================================================

#if defined(REFRACT_BUFFER) || defined(REFRACT_BUFFER_MIN) || defined(REFRACT_BUFFER_MIN2)

vec3IlluminationData fetchRefract(ivec2 p) {
    uvec2 xy = uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1));
    vec3IlluminationData tmp;

    float vproj;
    readRefrLight(xy, tmp.data_swap, vproj, tmp.weight);

#ifndef REFRACT_BUFFER_MIN2
    readRefrHistLight(xy, tmp.data, vproj, tmp.prev_weight);
    vec3 T;
    readRefrHistGeo(xy, tmp.pos, T);
    tmp.normal = T * vproj;
#endif
    return tmp;
}

void writeRefract(vec3IlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    vec3 color; float vproj, accumW;
    readRefrLight(xy, color, vproj, accumW);
    writeRefrLight(xy, data.data_swap, vproj, data.weight);
}

void writeRefractHistory(vec3 preDenoiseColor, float prevWeight, vec3 pos, vec3 R, float virtualProjDist, ivec2 p) {
    uvec2 xy = uvec2(p);
    writeRefrHistGeo(xy, pos, R);
    writeRefrHistLight(xy, preDenoiseColor, virtualProjDist, prevWeight);
}
#endif

#endif // RT_SPECULAR_IO_GLSL
