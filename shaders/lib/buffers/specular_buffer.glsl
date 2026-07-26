#ifndef BUFFERS_SPECULAR_BUFFER_GLSL
#define BUFFERS_SPECULAR_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/common/oct_encode.glsl"

// ===========================================================================
// Binding 3/4 — SpecularBuffer (Reflect/Refract) pack/unpack
// ===========================================================================
// N=0: Current Geo+Dir — vec4(worldPos.xyz, oct(dir))
// N=1: Current Light   — vec4(packHalf(cR,cG), packHalf(cB,vprojDist), accum_weight, pad)
// N=2: History Geo+Dir — vec4(hist_pos.xyz, oct(hist_dir))
// N=3: History Light   — vec4(packHalf(hcR,hcG), packHalf(hcB,h_vproj), hist_weight, pad)

#define SPEC_N_GEO       0u
#define SPEC_N_LIGHT     1u
#define SPEC_N_HISTGEO   2u
#define SPEC_N_HISTLIGHT 3u

// --- N=0: Current Geometry + Direction ---
void writeReflGeo(uvec2 xy, vec3 pos, vec3 R) {
    reflectBuffer.data[addr(SPEC_N_GEO, xy)] = vec4(pos, encodeNormal(R));
}
void readReflGeo(uvec2 xy, out vec3 pos, out vec3 R) {
    vec4 v = reflectBuffer.data[addr(SPEC_N_GEO, xy)];
    pos = v.xyz; R = decodeNormal(v.w);
}
void writeRefrGeo(uvec2 xy, vec3 pos, vec3 T) {
    refractBuffer.data[addr(SPEC_N_GEO, xy)] = vec4(pos, encodeNormal(T));
}
void readRefrGeo(uvec2 xy, out vec3 pos, out vec3 T) {
    vec4 v = refractBuffer.data[addr(SPEC_N_GEO, xy)];
    pos = v.xyz; T = decodeNormal(v.w);
}

// --- N=1: Current Light ---
void writeReflLight(uvec2 xy, vec3 color, float vprojDist, float accumWeight) {
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = vec4(
        pack2HalfClamped(color.r, color.g),
        pack2HalfClamped(color.b, vprojDist),
        accumWeight,
        0.0
    );
}
void readReflLight(uvec2 xy, out vec3 color, out float vprojDist, out float accumWeight) {
    vec4 v = reflectBuffer.data[addr(SPEC_N_LIGHT, xy)];
    vec2 rg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 bv = unpackHalf2x16(floatBitsToUint(v.y));
    color      = vec3(rg.x, rg.y, bv.x);
    vprojDist  = bv.y;
    accumWeight = v.z;
}
void writeRefrLight(uvec2 xy, vec3 color, float vprojDist, float accumWeight) {
    refractBuffer.data[addr(SPEC_N_LIGHT, xy)] = vec4(
        pack2HalfClamped(color.r, color.g),
        pack2HalfClamped(color.b, vprojDist),
        accumWeight,
        0.0
    );
}
void readRefrLight(uvec2 xy, out vec3 color, out float vprojDist, out float accumWeight) {
    vec4 v = refractBuffer.data[addr(SPEC_N_LIGHT, xy)];
    vec2 rg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 bv = unpackHalf2x16(floatBitsToUint(v.y));
    color       = vec3(rg.x, rg.y, bv.x);
    vprojDist   = bv.y;
    accumWeight = v.z;
}

// --- N=2: History Geometry + Direction ---
void writeReflHistGeo(uvec2 xy, vec3 pos, vec3 R) {
    reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)] = vec4(pos, encodeNormal(R));
}
void readReflHistGeo(uvec2 xy, out vec3 pos, out vec3 R) {
    vec4 v = reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    pos = v.xyz; R = decodeNormal(v.w);
}
void writeRefrHistGeo(uvec2 xy, vec3 pos, vec3 T) {
    refractBuffer.data[addr(SPEC_N_HISTGEO, xy)] = vec4(pos, encodeNormal(T));
}
void readRefrHistGeo(uvec2 xy, out vec3 pos, out vec3 T) {
    vec4 v = refractBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    pos = v.xyz; T = decodeNormal(v.w);
}

// --- N=3: History Light ---
void writeReflHistLight(uvec2 xy, vec3 color, float vprojDist, float weight) {
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = vec4(
        pack2HalfClamped(color.r, color.g),
        pack2HalfClamped(color.b, vprojDist),
        weight,
        0.0
    );
}
void readReflHistLight(uvec2 xy, out vec3 color, out float vprojDist, out float weight) {
    vec4 v = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    vec2 rg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 bv = unpackHalf2x16(floatBitsToUint(v.y));
    color     = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    weight    = v.z;
}
void writeRefrHistLight(uvec2 xy, vec3 color, float vprojDist, float weight) {
    refractBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = vec4(
        pack2HalfClamped(color.r, color.g),
        pack2HalfClamped(color.b, vprojDist),
        weight,
        0.0
    );
}
void readRefrHistLight(uvec2 xy, out vec3 color, out float vprojDist, out float weight) {
    vec4 v = refractBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    vec2 rg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 bv = unpackHalf2x16(floatBitsToUint(v.y));
    color     = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    weight    = v.z;
}

#endif // BUFFERS_SPECULAR_BUFFER_GLSL
