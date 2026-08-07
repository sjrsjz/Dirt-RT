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
#define SPEC_N_HISTMETA  4u

struct RelaxSpecularHistory {
    vec3 surfacePosition;
    vec3 geometryNormal;
    vec3 slowRadiance;
    float secondMoment;
    vec3 responsiveRadiance;
    float hitDistance;
    float roughness;
    float historyLength;
    uint materialID;
    float reprojectionConfidence;
};

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

// Reflection-only RELAX history. Five tiled virtual images share binding 3;
// refraction continues to use the four legacy images above.
void writeRelaxSpecularHistory(uvec2 xy, RelaxSpecularHistory h) {
    reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)] = vec4(
        h.surfacePosition, encodeNormal(h.geometryNormal));
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = vec4(
        pack2HalfClamped(h.slowRadiance.r, h.slowRadiance.g),
        pack2HalfClamped(h.slowRadiance.b,
            encodeSqrtMomentFP16(h.secondMoment)),
        pack2HalfClamped(h.responsiveRadiance.r, h.responsiveRadiance.g),
        pack2HalfClamped(h.responsiveRadiance.b, h.hitDistance));
    reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)] = vec4(
        pack2HalfClamped(h.roughness, h.historyLength),
        float(h.materialID), h.reprojectionConfidence, 0.0);
}

RelaxSpecularHistory readRelaxSpecularHistory(uvec2 xy) {
    RelaxSpecularHistory h;
    vec4 g = reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    vec4 s = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    vec4 m = reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)];
    vec2 slowRG = unpackHalf2x16(floatBitsToUint(s.x));
    vec2 slowBM2 = unpackHalf2x16(floatBitsToUint(s.y));
    vec2 fastRG = unpackHalf2x16(floatBitsToUint(s.z));
    vec2 fastBHit = unpackHalf2x16(floatBitsToUint(s.w));
    vec2 roughHistory = unpackHalf2x16(floatBitsToUint(m.x));
    bool valid = !any(isnan(g.xyz)) && !any(isinf(g.xyz)) &&
        !any(isnan(slowRG)) && !any(isinf(slowRG)) &&
        !any(isnan(slowBM2)) && !any(isinf(slowBM2)) &&
        !any(isnan(fastRG)) && !any(isinf(fastRG)) &&
        !any(isnan(fastBHit)) && !any(isinf(fastBHit)) &&
        !any(isnan(roughHistory)) && !any(isinf(roughHistory)) &&
        !isnan(m.y) && !isinf(m.y) && !isnan(m.z) && !isinf(m.z) &&
        m.y >= 0.0 && m.y <= 16777215.0 &&
        roughHistory.y >= 0.0 && roughHistory.y <= 255.0;
    h.surfacePosition = g.xyz;
    h.geometryNormal = decodeNormal(g.w);
    h.slowRadiance = vec3(slowRG, slowBM2.x);
    h.secondMoment = decodeSqrtMomentFP16(slowBM2.y);
    h.responsiveRadiance = vec3(fastRG, fastBHit.x);
    h.hitDistance = fastBHit.y;
    h.roughness = roughHistory.x;
    h.historyLength = roughHistory.y;
    h.materialID = 0u;
    if (valid)
        h.materialID = uint(m.y + 0.5);
    h.reprojectionConfidence = clamp(m.z, 0.0, 1.0);
    if (!valid) {
        h.surfacePosition = vec3(0.0);
        h.geometryNormal = vec3(0.0, 1.0, 0.0);
        h.slowRadiance = vec3(0.0);
        h.secondMoment = 0.0;
        h.responsiveRadiance = vec3(0.0);
        h.hitDistance = 0.0;
        h.roughness = 1.0;
        h.historyLength = 0.0;
        h.reprojectionConfidence = 0.0;
    }
    return h;
}

#endif // BUFFERS_SPECULAR_BUFFER_GLSL
