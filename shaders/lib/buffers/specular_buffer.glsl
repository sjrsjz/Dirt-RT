#ifndef BUFFERS_SPECULAR_BUFFER_GLSL
#define BUFFERS_SPECULAR_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/buffers/debug_buffer.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/common/oct_encode.glsl"

// Reflection has no current-frame position plane. Primary position is
// reconstructed from the compact G-buffer, leaving four physical layers.
#define SPEC_N_LIGHT     0u
#define SPEC_N_HISTGEO   1u
#define SPEC_N_HISTLIGHT 2u
#define SPEC_N_HISTMETA  3u

// Refraction owns four PSR planes with unrelated semantics.
#define REFR_N_ENDPOINT  0u
#define REFR_N_SURFACE   1u
#define REFR_N_META      2u
#define REFR_N_TRANSPORT 3u

// MaxEnt-4 stores the energy weighted first directional moment in xyz and
// total luminance energy in w. CoCg is deliberately scalar/angularly shared.
struct SpecularMaxEnt {
    vec4 maxEntY;
    vec2 CoCg;
};

struct MaxEntSpecularHistory {
    vec3 surfacePosition;
    vec3 geometryNormal;
    SpecularMaxEnt signal;
    float rootMeanY2;
    float hitDistance;
    float roughness;
    float historyLength;
    uint materialID;
};

// ray3 publishes only PSR resolve metadata. The old refraction history planes
// are intentionally repurposed because refraction no longer has a standalone
// temporal/spatial denoiser.
struct PSRResolveData {
    vec3 endpointRelative;
    vec3 refractedDirection;
    vec3 geometryNormal;
    vec3 macroNormal;
    vec3 diffuseAlbedo;
    float roughness;
    float pathRoughness;
    vec3 transmittance;
    vec3 surfaceLight;
    bool endpointValid;
    bool environment;
    bool screenCandidate;
};

void writePSRResolve(uvec2 xy, PSRResolveData p) {
    uint flags = (p.endpointValid ? 1u : 0u)
        | (p.environment ? 2u : 0u)
        | (p.screenCandidate ? 4u : 0u);
    refractBuffer.data[addr(REFR_N_ENDPOINT, xy)] = uvec4(
        floatBitsToUint(p.endpointRelative),
        encodeNormalU(p.refractedDirection));
    refractBuffer.data[addr(REFR_N_SURFACE, xy)] = uvec4(
        encodeNormalU(p.geometryNormal), encodeNormalU(p.macroNormal),
        packHalf2x16(clamp(p.diffuseAlbedo.rg, vec2(0.0), vec2(65504.0))),
        packHalf2x16(clamp(vec2(p.diffuseAlbedo.b, p.roughness),
            vec2(0.0), vec2(65504.0))));
    refractBuffer.data[addr(REFR_N_META, xy)] = uvec4(flags,
        floatBitsToUint(p.pathRoughness), 0u, 0u);
    refractBuffer.data[addr(REFR_N_TRANSPORT, xy)] = uvec4(
        packHalf2x16(clamp(p.transmittance.rg, vec2(0.0), vec2(8.0))),
        packHalf2x16(clamp(vec2(p.transmittance.b, p.surfaceLight.r),
            vec2(0.0), vec2(8.0, 65504.0))),
        packHalf2x16(clamp(p.surfaceLight.gb, vec2(0.0), vec2(65504.0))),
        0u);
}

PSRResolveData readPSRResolve(uvec2 xy) {
    uvec4 endpoint = refractBuffer.data[addr(REFR_N_ENDPOINT, xy)];
    uvec4 surface = refractBuffer.data[addr(REFR_N_SURFACE, xy)];
    uvec4 metadata = refractBuffer.data[addr(REFR_N_META, xy)];
    uvec4 transport = refractBuffer.data[addr(REFR_N_TRANSPORT, xy)];
    vec2 albedoRG = unpackHalf2x16(surface.z);
    vec2 albedoBRoughness = unpackHalf2x16(surface.w);
    vec2 transRG = unpackHalf2x16(transport.x);
    vec2 transBLightR = unpackHalf2x16(transport.y);
    vec2 lightGB = unpackHalf2x16(transport.z);

    PSRResolveData p;
    p.endpointRelative = uintBitsToFloat(endpoint.xyz);
    p.refractedDirection = decodeNormalU(endpoint.w);
    p.geometryNormal = decodeNormalU(surface.x);
    p.macroNormal = decodeNormalU(surface.y);
    p.diffuseAlbedo = max(vec3(albedoRG, albedoBRoughness.x), vec3(0.0));
    p.roughness = clamp(albedoBRoughness.y, 0.0, 1.0);
    p.pathRoughness = uintBitsToFloat(metadata.y);
    p.transmittance = max(vec3(transRG, transBLightR.x), vec3(0.0));
    p.surfaceLight = max(vec3(transBLightR.y, lightGB), vec3(0.0));
    p.endpointValid = (metadata.x & 1u) != 0u;
    p.environment = (metadata.x & 2u) != 0u;
    p.screenCandidate = (metadata.x & 4u) != 0u;
    return p;
}

uint packMaxEntHistoryNormalMaterial(vec3 n, uint materialID) {
    n = normalize(n);
    vec2 p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(p, vec2(0.0)));
    uint oct8 = packUnorm4x8(vec4(p * 0.5 + 0.5, 0.0, 0.0)) & 0xffffu;
    return oct8 | ((materialID & 0xffffu) << 16u);
}

uint packMaxEntOct16(vec3 n) {
    n = normalize(n);
    vec2 p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(p, vec2(0.0)));
    return packUnorm4x8(vec4(p * 0.5 + 0.5, 0.0, 0.0)) & 0xffffu;
}

vec3 unpackMaxEntOct16(uint word) {
    vec2 f = unpackUnorm4x8(word & 0xffffu).xy * 2.0 - 1.0;
    vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

void unpackMaxEntHistoryNormalMaterial(uint word, out vec3 n,
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

void writeRefrGeo(uvec2 xy, vec3 pos, vec3 T) {
    refractBuffer.data[addr(REFR_N_ENDPOINT, xy)] =
        uvec4(floatBitsToUint(pos), encodeNormalU(T));
}
void readRefrGeo(uvec2 xy, out vec3 pos, out vec3 T) {
    uvec4 v = refractBuffer.data[addr(REFR_N_ENDPOINT, xy)];
    pos = uintBitsToFloat(v.xyz);
    T = decodeNormalU(v.w);
}

// N=0 is MaxEnt-4 Y, CoCg and radial metadata. Before temporal/spatial
// processing the lower half is ray hit distance; the final spatial resolve
// replaces it with filtered virtual distance. Diagnostics live in DebugBuffer
// and never change this production layout.
void writeReflMaxEnt(uvec2 xy, SpecularMaxEnt signal,
        float hitDistance, float debugWeight) {
    uvec3 p = packSpecularMaxEnt(signal);
    uint metadata = packHalf2x16(clamp(vec2(hitDistance, debugWeight),
        vec2(-65504.0), vec2(65504.0)));
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(p, metadata);
}

void writeReflMaxEntSample(uvec2 xy, SpecularMaxEnt signal,
        float hitDistance, vec3 sampledDirection) {
    uvec3 p = packSpecularMaxEnt(signal);
    uint metadata = packHalf2x16(clamp(vec2(hitDistance, 0.0),
        vec2(-65504.0), vec2(65504.0)));
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(p, metadata);
    debugWriteReflectionSampleDirection(xy, sampledDirection);
}

vec3 readReflSampleDirection(uvec2 xy) {
    return debugReadReflectionSampleDirection(xy);
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
    refractBuffer.data[addr(REFR_N_SURFACE, xy)] = uvec4(
        pack2HalfClampedU(color.r, color.g),
        pack2HalfClampedU(color.b, vprojDist),
        floatBitsToUint(accumWeight), 0u);
}
void readRefrLight(uvec2 xy, out vec3 color, out float vprojDist,
        out float accumWeight) {
    uvec4 v = refractBuffer.data[addr(REFR_N_SURFACE, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    accumWeight = uintBitsToFloat(v.z);
}

void writeRefrHistGeo(uvec2 xy, vec3 pos, vec3 T) {
    refractBuffer.data[addr(REFR_N_META, xy)] =
        uvec4(floatBitsToUint(pos), encodeNormalU(T));
}
void readRefrHistGeo(uvec2 xy, out vec3 pos, out vec3 T) {
    uvec4 v = refractBuffer.data[addr(REFR_N_META, xy)];
    pos = uintBitsToFloat(v.xyz);
    T = decodeNormalU(v.w);
}

void writeRefrHistLight(uvec2 xy, vec3 color, float vprojDist, float weight) {
    refractBuffer.data[addr(REFR_N_TRANSPORT, xy)] = uvec4(
        pack2HalfClampedU(color.r, color.g),
        pack2HalfClampedU(color.b, vprojDist),
        floatBitsToUint(weight), 0u);
}
void readRefrHistLight(uvec2 xy, out vec3 color, out float vprojDist,
        out float weight) {
    uvec4 v = refractBuffer.data[addr(REFR_N_TRANSPORT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    weight = uintBitsToFloat(v.z);
}

// N1: primary surface geometry plus FP16 hit distance and roughness.
// N2: the sole temporal MaxEnt6 history, rootMeanY2 and Kish N_eff.
// N3: previous final denoised MaxEnt6 plus filtered standard deviation and a
// negative layout stamp. It replaces the former secondary history in place.
void writeMaxEntSpecularTemporalHistory(uvec2 xy, MaxEntSpecularHistory h) {
    h.signal = sanitizeSpecularMaxEnt(h.signal);
    h.rootMeanY2 = sanitizeRootMeanSquareFP16(h.rootMeanY2);
    h.historyLength = isnan(h.historyLength) || isinf(h.historyLength)
        ? 1.0 : clamp(h.historyLength, 1.0, 65504.0);
    float surfaceDistance = length(h.surfacePosition);
    vec3 surfaceDirection = surfaceDistance > 1e-8
        ? h.surfacePosition / surfaceDistance : vec3(0.0, 0.0, -1.0);
    reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)] = uvec4(
        floatBitsToUint(surfaceDistance), encodeNormalU(surfaceDirection),
        packMaxEntHistoryNormalMaterial(h.geometryNormal, h.materialID),
        packHalf2x16(clamp(vec2(h.hitDistance, h.roughness), vec2(0.0), vec2(65504.0, 1.0))));
    uvec3 temporal = packSpecularMaxEnt(h.signal);
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = uvec4(temporal,
        pack2HalfClampedU(h.rootMeanY2,
            h.historyLength));
}

void writeMaxEntSpecularDenoisedHistory(uvec2 xy, SpecularMaxEnt signal, float stddev) {
    signal = sanitizeSpecularMaxEnt(signal);
    if (!(stddev >= 0.0) || isnan(stddev) || isinf(stddev)) {
        reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)] = uvec4(0u, 0u, 0u,
            packHalf2x16(vec2(-1.0, 0.0)));
        return;
    }
    uvec3 denoised = packSpecularMaxEnt(signal);
    reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)] = uvec4(denoised,
        packHalf2x16(vec2(min(stddev, 65504.0), -2.0)));
}

void writeMaxEntSpecularDenoisedHistoryInvalid(uvec2 xy) {
    reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)] = uvec4(0u, 0u, 0u,
        packHalf2x16(vec2(-1.0, 0.0)));
}

// Transient N0 layout consumed by the final spatial pass:
//   xy = reprojected previous denoised MaxEntY
//   z  = previous N_eff, valid mass
//   w  = filtered stddev, negative current-sample temporal weight
// CoCg is unnecessary because temporal difference detection uses MaxEntY only.
void writeMaxEntSpecularDenoisedReprojection(uvec2 xy, vec4 maxEntY,
        float standardDeviation, float historySamples, float validWeight,
        float temporalCurrentWeight) {
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(
        packHalf2x16(clamp(maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(min(historySamples, 65504.0), clamp(validWeight, 0.0, 1.0))),
        packHalf2x16(vec2(clamp(standardDeviation, 0.0, 65504.0),
            -clamp(temporalCurrentWeight, 0.0, 1.0))));
}

void writeMaxEntSpecularDenoisedReprojectionInvalid(uvec2 xy) {
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(0u, 0u, 0u,
        packHalf2x16(vec2(-1.0, 0.0)));
}

bool readMaxEntSpecularDenoisedReprojection(uvec2 xy, out vec4 maxEntY,
        out float stddev, out float historySamples, out float validWeight,
        out float temporalCurrentWeight) {
    uvec4 words = reflectBuffer.data[addr(SPEC_N_LIGHT, xy)];
    vec2 historyMeta = unpackHalf2x16(words.z);
    vec2 metadata = unpackHalf2x16(words.w);
    stddev = metadata.x;
    historySamples = historyMeta.x;
    validWeight = historyMeta.y;
    temporalCurrentWeight = -metadata.y;
    bool valid = stddev >= 0.0 && historySamples >= 1.0 && validWeight > 0.0
        && metadata.y < 0.0 && !any(isnan(historyMeta)) && !any(isinf(historyMeta))
        && !any(isnan(metadata)) && !any(isinf(metadata));
    maxEntY = valid ? vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y)) : vec4(0.0);
    return valid;
}

MaxEntSpecularHistory readMaxEntSpecularHistory(uvec2 xy) {
    MaxEntSpecularHistory h;
    uvec4 g = reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    uvec4 s = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    uvec4 m = reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)];
    vec2 m2History = unpackHalf2x16(s.w);
    vec2 hitRoughness = unpackHalf2x16(g.w);
    vec2 denoisedMetadata = unpackHalf2x16(m.w);
    float surfaceDistance = uintBitsToFloat(g.x);
    h.surfacePosition = decodeNormalU(g.y) * surfaceDistance;
    unpackMaxEntHistoryNormalMaterial(g.z, h.geometryNormal, h.materialID);
    h.signal = unpackSpecularMaxEnt(s.xyz);
    h.rootMeanY2 = sanitizeRootMeanSquareFP16(m2History.x);
    h.historyLength = max(m2History.y, 0.0);
    bool denoisedValid = denoisedMetadata.x >= 0.0 && denoisedMetadata.y == -2.0
        && !any(isnan(denoisedMetadata)) && !any(isinf(denoisedMetadata));
    h.hitDistance = max(hitRoughness.x, 0.0);
    h.roughness = clamp(hitRoughness.y, 0.0, 1.0);
    bool valid = surfaceDistance >= 0.0 && !isnan(surfaceDistance) &&
        !isinf(surfaceDistance) && !any(isnan(m2History))
        && !any(isinf(m2History)) && m2History.x >= 0.0 &&
        h.historyLength >= 1.0 && h.historyLength <= 65504.0 &&
        h.materialID != 0xffffffffu && denoisedValid;
    if (!valid) {
        h.surfacePosition = vec3(0.0);
        h.geometryNormal = vec3(0.0, 1.0, 0.0);
        h.signal = emptySpecularMaxEnt();
        h.rootMeanY2 = 0.0;
        h.hitDistance = 0.0;
        h.roughness = 1.0;
        h.historyLength = 0.0;
        h.materialID = 0u;
    }
    return h;
}

#endif
