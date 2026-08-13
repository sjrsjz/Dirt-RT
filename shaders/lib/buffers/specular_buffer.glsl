#ifndef BUFFERS_SPECULAR_BUFFER_GLSL
#define BUFFERS_SPECULAR_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/common/oct_encode.glsl"

#define SPEC_N_GEO       0u
#define SPEC_N_LIGHT     1u
#define SPEC_N_HISTGEO   2u
#define SPEC_N_HISTLIGHT 3u
#define SPEC_N_HISTMETA  4u

// MaxEnt-4 stores the energy weighted first directional moment in xyz and
// total luminance energy in w. CoCg is deliberately scalar/angularly shared.
struct SpecularMaxEnt {
    vec4 maxEntY;
    vec2 CoCg;
};

struct RelaxSpecularHistory {
    vec3 surfacePosition;
    vec3 geometryNormal;
    SpecularMaxEnt slowSignal;
    float secondMoment;
    SpecularMaxEnt responsiveSignal;
    float hitDistance;
    float roughness;
    float historyLength;
    uint materialID;
    float reprojectionConfidence;
};

uint packRelaxHistoryNormalMaterial(vec3 n, uint materialID) {
    n = normalize(n);
    vec2 p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(p, vec2(0.0)));
    uint oct8 = packUnorm4x8(vec4(p * 0.5 + 0.5, 0.0, 0.0)) & 0xffffu;
    return oct8 | ((materialID & 0xffffu) << 16u);
}

void unpackRelaxHistoryNormalMaterial(uint word, out vec3 n,
        out uint materialID) {
    vec2 f = unpackUnorm4x8(word & 0xffffu).xy * 2.0 - 1.0;
    n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(n.xy, vec2(0.0)));
    n = normalize(n);
    materialID = word >> 16u;
}

SpecularMaxEnt emptySpecularMaxEnt() {
    SpecularMaxEnt s;
    s.maxEntY = vec4(0.0);
    s.CoCg = vec2(0.0);
    return s;
}

SpecularMaxEnt sanitizeSpecularMaxEnt(SpecularMaxEnt s) {
    if (any(isnan(s.maxEntY)) || any(isinf(s.maxEntY)) ||
            any(isnan(s.CoCg)) || any(isinf(s.CoCg)) ||
            s.maxEntY.w <= 0.0)
        return emptySpecularMaxEnt();
    s.maxEntY.w = clamp(s.maxEntY.w, 0.0, 65504.0);
    float momentLength = length(s.maxEntY.xyz);
    if (momentLength > s.maxEntY.w)
        s.maxEntY.xyz *= s.maxEntY.w / max(momentLength, 1e-20);
    s.CoCg = clamp(s.CoCg, vec2(-65504.0), vec2(65504.0));
    return s;
}

SpecularMaxEnt specularMaxEntFromRgbDirection(vec3 color, vec3 direction) {
    color = max(color, vec3(0.0));
    SpecularMaxEnt s;
    float Y = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float l2 = dot(direction, direction);
    vec3 d = l2 > 1e-20 ? direction * inversesqrt(l2) : vec3(0.0);
    s.maxEntY = vec4(d * Y, Y);
    s.CoCg = vec2(0.5 * color.r - 0.5 * color.b,
        -0.25 * color.r + 0.5 * color.g - 0.25 * color.b);
    return sanitizeSpecularMaxEnt(s);
}

// Exact inverse of the Rec.709 Y + CoCg transform used by
// specularMaxEntFromRgbDirection(). This is not the conventional YCoCg
// inverse, whose Y row is (0.25, 0.5, 0.25).
vec3 specularRec709YCoCgToRgb(float Y, vec2 CoCg) {
    float Co = CoCg.x;
    float Cg = CoCg.y;
    return vec3(
        Y + 0.8596 * Co - 1.4304 * Cg,
        Y - 0.1404 * Co + 0.5696 * Cg,
        Y - 1.1404 * Co - 1.4304 * Cg);
}

vec3 specularMaxEntTotalRgb(SpecularMaxEnt s) {
    s = sanitizeSpecularMaxEnt(s);
    return max(specularRec709YCoCgToRgb(s.maxEntY.w, s.CoCg),
        vec3(0.0));
}

uvec3 packSpecularMaxEnt(SpecularMaxEnt s) {
    s = sanitizeSpecularMaxEnt(s);
    return uvec3(packHalf2x16(s.maxEntY.xy),
        packHalf2x16(s.maxEntY.zw), packHalf2x16(s.CoCg));
}

SpecularMaxEnt unpackSpecularMaxEnt(uvec3 p) {
    SpecularMaxEnt s;
    vec2 xy = unpackHalf2x16(p.x);
    vec2 zw = unpackHalf2x16(p.y);
    s.maxEntY = vec4(xy, zw);
    s.CoCg = unpackHalf2x16(p.z);
    return sanitizeSpecularMaxEnt(s);
}

void writeReflGeo(uvec2 xy, vec3 pos, vec3 R) {
    reflectBuffer.data[addr(SPEC_N_GEO, xy)] =
        uvec4(floatBitsToUint(pos), encodeNormalU(R));
}
void readReflGeo(uvec2 xy, out vec3 pos, out vec3 R) {
    uvec4 v = reflectBuffer.data[addr(SPEC_N_GEO, xy)];
    pos = uintBitsToFloat(v.xyz);
    R = decodeNormalU(v.w);
}
void writeRefrGeo(uvec2 xy, vec3 pos, vec3 T) {
    refractBuffer.data[addr(SPEC_N_GEO, xy)] =
        uvec4(floatBitsToUint(pos), encodeNormalU(T));
}
void readRefrGeo(uvec2 xy, out vec3 pos, out vec3 T) {
    uvec4 v = refractBuffer.data[addr(SPEC_N_GEO, xy)];
    pos = uintBitsToFloat(v.xyz);
    T = decodeNormalU(v.w);
}

// N=1 is exactly eight FP16 values: MaxEnt-4 Y, CoCg, hit distance, weight.
void writeReflMaxEnt(uvec2 xy, SpecularMaxEnt signal,
        float hitDistance, float debugWeight) {
    uvec3 p = packSpecularMaxEnt(signal);
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(p,
        packHalf2x16(clamp(vec2(hitDistance, debugWeight),
            vec2(-65504.0), vec2(65504.0))));
}
void readReflMaxEnt(uvec2 xy, out SpecularMaxEnt signal,
        out float hitDistance, out float debugWeight) {
    uvec4 p = reflectBuffer.data[addr(SPEC_N_LIGHT, xy)];
    signal = unpackSpecularMaxEnt(p.xyz);
    vec2 hd = unpackHalf2x16(p.w);
    hitDistance = max(hd.x, 0.0);
    debugWeight = hd.y;
}

// Compatibility helpers are retained for diagnostics and old utility code.
void writeReflLight(uvec2 xy, vec3 color, float vprojDist, float accumWeight) {
    writeReflMaxEnt(xy, specularMaxEntFromRgbDirection(color, vec3(0.0)),
        vprojDist, accumWeight);
}
void readReflLight(uvec2 xy, out vec3 color, out float vprojDist,
        out float accumWeight) {
    SpecularMaxEnt s;
    readReflMaxEnt(xy, s, vprojDist, accumWeight);
    color = specularMaxEntTotalRgb(s);
}

void writeRefrLight(uvec2 xy, vec3 color, float vprojDist, float accumWeight) {
    refractBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(
        pack2HalfClampedU(color.r, color.g),
        pack2HalfClampedU(color.b, vprojDist),
        floatBitsToUint(accumWeight), 0u);
}
void readRefrLight(uvec2 xy, out vec3 color, out float vprojDist,
        out float accumWeight) {
    uvec4 v = refractBuffer.data[addr(SPEC_N_LIGHT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    accumWeight = uintBitsToFloat(v.z);
}

void writeReflHistGeo(uvec2 xy, vec3 pos, vec3 R) {
    reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)] =
        uvec4(floatBitsToUint(pos), encodeNormalU(R));
}
void readReflHistGeo(uvec2 xy, out vec3 pos, out vec3 R) {
    uvec4 v = reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    pos = uintBitsToFloat(v.xyz);
    R = decodeNormalU(v.w);
}
void writeRefrHistGeo(uvec2 xy, vec3 pos, vec3 T) {
    refractBuffer.data[addr(SPEC_N_HISTGEO, xy)] =
        uvec4(floatBitsToUint(pos), encodeNormalU(T));
}
void readRefrHistGeo(uvec2 xy, out vec3 pos, out vec3 T) {
    uvec4 v = refractBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    pos = uintBitsToFloat(v.xyz);
    T = decodeNormalU(v.w);
}

void writeReflHistLight(uvec2 xy, vec3 color, float vprojDist, float weight) {
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = uvec4(
        pack2HalfClampedU(color.r, color.g),
        pack2HalfClampedU(color.b, vprojDist),
        floatBitsToUint(weight), 0u);
}
void readReflHistLight(uvec2 xy, out vec3 color, out float vprojDist,
        out float weight) {
    uvec4 v = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    weight = uintBitsToFloat(v.z);
}
void writeRefrHistLight(uvec2 xy, vec3 color, float vprojDist, float weight) {
    refractBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = uvec4(
        pack2HalfClampedU(color.r, color.g),
        pack2HalfClampedU(color.b, vprojDist),
        floatBitsToUint(weight), 0u);
}
void readRefrHistLight(uvec2 xy, out vec3 color, out float vprojDist,
        out float weight) {
    uvec4 v = refractBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    weight = uintBitsToFloat(v.z);
}

// N3: slow MaxEnt6 + sqrt(E[Y^2]) + hitDistance.
// N4: responsive MaxEnt6 + roughness + history length. Material is packed
// with the octahedral history normal in N2.w.
void writeRelaxSpecularHistory(uvec2 xy, RelaxSpecularHistory h) {
    h.slowSignal = sanitizeSpecularMaxEnt(h.slowSignal);
    h.responsiveSignal = sanitizeSpecularMaxEnt(h.responsiveSignal);
    reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)] = uvec4(
        floatBitsToUint(h.surfacePosition), packRelaxHistoryNormalMaterial(
            h.geometryNormal, h.materialID));
    uvec3 slow = packSpecularMaxEnt(h.slowSignal);
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = uvec4(slow,
        pack2HalfClampedU(encodeSqrtMomentFP16(h.secondMoment),
            h.hitDistance));
    uvec3 responsive = packSpecularMaxEnt(h.responsiveSignal);
    reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)] = uvec4(responsive,
        pack2HalfClampedU(h.roughness, h.historyLength));
}

RelaxSpecularHistory readRelaxSpecularHistory(uvec2 xy) {
    RelaxSpecularHistory h;
    uvec4 g = reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    uvec4 s = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    uvec4 m = reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)];
    vec2 m2Hit = unpackHalf2x16(s.w);
    vec2 roughnessHistory = unpackHalf2x16(m.w);
    h.surfacePosition = uintBitsToFloat(g.xyz);
    unpackRelaxHistoryNormalMaterial(g.w, h.geometryNormal, h.materialID);
    h.slowSignal = unpackSpecularMaxEnt(s.xyz);
    h.secondMoment = decodeSqrtMomentFP16(m2Hit.x);
    h.hitDistance = max(m2Hit.y, 0.0);
    h.responsiveSignal = unpackSpecularMaxEnt(m.xyz);
    h.roughness = clamp(roughnessHistory.x, 0.0, 1.0);
    h.historyLength = max(roughnessHistory.y, 0.0);
    h.reprojectionConfidence = 1.0;

    bool valid = !any(isnan(h.surfacePosition)) &&
        !any(isinf(h.surfacePosition)) &&
        h.historyLength <= 255.0 && h.materialID != 0xffffffffu;
    if (!valid) {
        h.surfacePosition = vec3(0.0);
        h.geometryNormal = vec3(0.0, 1.0, 0.0);
        h.slowSignal = emptySpecularMaxEnt();
        h.secondMoment = 0.0;
        h.responsiveSignal = emptySpecularMaxEnt();
        h.hitDistance = 0.0;
        h.roughness = 1.0;
        h.historyLength = 0.0;
        h.materialID = 0u;
        h.reprojectionConfidence = 0.0;
    }
    return h;
}

#endif
