#ifndef BUFFERS_SPECULAR_BUFFER_GLSL
#define BUFFERS_SPECULAR_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/common/oct_encode.glsl"

// ===========================================================================
// Binding 3/4 — SpecularBuffer (Reflect/Refract), native uvec4 storage
// ===========================================================================
// N=0: Current Geo+Dir — vec4(worldPos.xyz, oct(dir))
// N=1: Current Light   — xy=packed color/hit; zw=four-FP16 endpoint moments
//                        until resolve rewrites them as accumulation metadata.
// N=2: History Geo+Dir — vec4(hist_pos.xyz, oct(hist_dir))
// N=3: History Light   — vec4(packHalf(hcR,hcG), packHalf(hcB,h_vproj), hist_weight, pad)

#define SPEC_N_GEO       0u
#define SPEC_N_LIGHT     1u
#define SPEC_N_HISTGEO   2u
#define SPEC_N_HISTLIGHT 3u
#define SPEC_N_HISTMETA  4u

struct RelaxEndpointMoments {
    vec3 mean;
    float secondMoment;
};

struct RelaxSpecularHistory {
    vec3 surfacePosition;
    vec3 geometryNormal;
    vec3 slowRadiance;
    float secondMoment;
    vec3 responsiveRadiance;
    RelaxEndpointMoments endpoint;
    float roughness;
    float historyLength;
    uint materialID;
    float reprojectionConfidence;
};

RelaxEndpointMoments emptyRelaxEndpointMoments() {
    RelaxEndpointMoments m;
    m.mean = vec3(0.0);
    m.secondMoment = 0.0;
    return m;
}

bool relaxEndpointMomentsValid(RelaxEndpointMoments m) {
    return m.secondMoment > 0.0 &&
        !any(isnan(m.mean)) && !any(isinf(m.mean)) &&
        !isnan(m.secondMoment) && !isinf(m.secondMoment);
}

RelaxEndpointMoments sanitizeRelaxEndpointMoments(RelaxEndpointMoments m) {
    if (!relaxEndpointMomentsValid(m)) return emptyRelaxEndpointMoments();
    m.mean = clamp(m.mean, vec3(-1.0), vec3(1.0));
    float meanLengthSquared = dot(m.mean, m.mean);
    if (meanLengthSquared > 1.0) {
        m.mean *= inversesqrt(meanLengthSquared);
        meanLengthSquared = 1.0;
    }
    m.secondMoment = clamp(m.secondMoment, meanLengthSquared, 1.0);
    return m;
}

uvec2 relaxPackEndpointMoments(RelaxEndpointMoments moments) {
    moments = sanitizeRelaxEndpointMoments(moments);

    uint packedMeanXY = packHalf2x16(moments.mean.xy);
    uint meanZBits = packHalf2x16(vec2(moments.mean.z, 0.0)) & 0xffffu;

    vec2 storedMeanXY = unpackHalf2x16(packedMeanXY);
    float storedMeanZ = unpackHalf2x16(meanZBits).x;
    vec3 storedMean = vec3(storedMeanXY, storedMeanZ);

    float centralSecondMoment = max(moments.secondMoment - dot(moments.mean, moments.mean), 0.0);
    float storedMeanSquared = dot(storedMean, storedMean);
    float targetSecondMoment = clamp(storedMeanSquared + centralSecondMoment, storedMeanSquared, 1.0);

    float targetRoot = sqrt(targetSecondMoment);
    uint rootBits = packHalf2x16(vec2(targetRoot, 0.0)) & 0xffffu;

    return uvec2(packedMeanXY, meanZBits | (rootBits << 16u));
}

RelaxEndpointMoments relaxUnpackEndpointMoments(uvec2 packed_) {
    vec2 meanXY = unpackHalf2x16(packed_.x);
    vec2 meanZRms = unpackHalf2x16(packed_.y);
    RelaxEndpointMoments moments;
    moments.mean = vec3(meanXY, meanZRms.x);
    moments.secondMoment = meanZRms.y * meanZRms.y;
    return sanitizeRelaxEndpointMoments(moments);
}

// --- N=0: Current Geometry + Direction ---
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

// --- N=1: Current Light ---
void writeReflLight(uvec2 xy, vec3 color, float vprojDist, float accumWeight) {
    reflectBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(
            pack2HalfClampedU(color.r, color.g),
            pack2HalfClampedU(color.b, vprojDist),
            floatBitsToUint(accumWeight),
            0u
        );
}
void readReflLight(uvec2 xy, out vec3 color, out float vprojDist, out float accumWeight) {
    uvec4 v = reflectBuffer.data[addr(SPEC_N_LIGHT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    accumWeight = uintBitsToFloat(v.z);
}

// N=1.zw carry the ray's endpoint sample into the spatial endpoint pass.
// No pass reads these words while another invocation writes neighbouring
// pixels, so this introduces no SSBO read/write race.
void writeReflEndpointMoments(uvec2 xy, RelaxEndpointMoments moments) {
    uvec2 packed_ = relaxPackEndpointMoments(moments);
    uint address = addr(SPEC_N_LIGHT, xy);
    reflectBuffer.data[address].zw = packed_;
}

uvec2 readReflEndpointMomentsRaw(uvec2 xy) {
    return reflectBuffer.data[addr(SPEC_N_LIGHT, xy)].zw;
}

RelaxEndpointMoments readReflEndpointMoments(uvec2 xy) {
    return relaxUnpackEndpointMoments(readReflEndpointMomentsRaw(xy));
}

void writeRefrLight(uvec2 xy, vec3 color, float vprojDist, float accumWeight) {
    refractBuffer.data[addr(SPEC_N_LIGHT, xy)] = uvec4(
            pack2HalfClampedU(color.r, color.g),
            pack2HalfClampedU(color.b, vprojDist),
            floatBitsToUint(accumWeight),
            0u
        );
}
void readRefrLight(uvec2 xy, out vec3 color, out float vprojDist, out float accumWeight) {
    uvec4 v = refractBuffer.data[addr(SPEC_N_LIGHT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    accumWeight = uintBitsToFloat(v.z);
}

// --- N=2: History Geometry + Direction ---
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

// --- N=3: History Light ---
void writeReflHistLight(uvec2 xy, vec3 color, float vprojDist, float weight) {
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = uvec4(
            pack2HalfClampedU(color.r, color.g),
            pack2HalfClampedU(color.b, vprojDist),
            floatBitsToUint(weight),
            0u
        );
}
void readReflHistLight(uvec2 xy, out vec3 color, out float vprojDist, out float weight) {
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
            floatBitsToUint(weight),
            0u
        );
}
void readRefrHistLight(uvec2 xy, out vec3 color, out float vprojDist, out float weight) {
    uvec4 v = refractBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    vec2 rg = unpackHalf2x16(v.x);
    vec2 bv = unpackHalf2x16(v.y);
    color = vec3(rg.x, rg.y, bv.x);
    vprojDist = bv.y;
    weight = uintBitsToFloat(v.z);
}

// Reflection-only RELAX history. Five tiled virtual images share binding 3;
// refraction continues to use the four legacy images above.
void writeRelaxSpecularHistory(uvec2 xy, RelaxSpecularHistory h) {
    uint packedMeta = (h.materialID & 0xffffu) |
            ((packHalf2x16(vec2(clamp(h.reprojectionConfidence, 0.0, 1.0), 0.0))
                & 0xffffu) << 16u);
    reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)] = uvec4(
            floatBitsToUint(h.surfacePosition), encodeNormalU(h.geometryNormal));
    uvec2 packedEndpoint = relaxPackEndpointMoments(h.endpoint);
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)] = uvec4(
            pack2HalfClampedU(h.slowRadiance.r, h.slowRadiance.g),
            pack2HalfClampedU(h.slowRadiance.b,
            encodeSqrtMomentFP16(h.secondMoment)),
            pack2HalfClampedU(h.responsiveRadiance.r, h.responsiveRadiance.g),
            pack2HalfClampedU(h.responsiveRadiance.b, 0.0));
    reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)] = uvec4(
            pack2HalfClampedU(h.roughness, h.historyLength),
            packedMeta,
            packedEndpoint.x,
            packedEndpoint.y);
}

RelaxSpecularHistory readRelaxSpecularHistory(uvec2 xy, bool decodeEndpoint) {
    RelaxSpecularHistory h;
    uvec4 g = reflectBuffer.data[addr(SPEC_N_HISTGEO, xy)];
    uvec4 s = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, xy)];
    uvec4 m = reflectBuffer.data[addr(SPEC_N_HISTMETA, xy)];
    vec3 surfacePosition = uintBitsToFloat(g.xyz);
    vec2 slowRG = unpackHalf2x16(s.x);
    vec2 slowBM2 = unpackHalf2x16(s.y);
    vec2 fastRG = unpackHalf2x16(s.z);
    vec2 fastBUnused = unpackHalf2x16(s.w);
    vec2 roughHistory = unpackHalf2x16(m.x);
    uint packedMeta = m.y;
    float packedConfidence =
        unpackHalf2x16((packedMeta >> 16u) & 0xffffu).x;
    bool valid = !any(isnan(surfacePosition)) && !any(isinf(surfacePosition)) &&
            !any(isnan(slowRG)) && !any(isinf(slowRG)) &&
            !any(isnan(slowBM2)) && !any(isinf(slowBM2)) &&
            !any(isnan(fastRG)) && !any(isinf(fastRG)) &&
            !any(isnan(fastBUnused)) && !any(isinf(fastBUnused)) &&
            !any(isnan(roughHistory)) && !any(isinf(roughHistory)) &&
            !isnan(packedConfidence) && !isinf(packedConfidence) &&
            roughHistory.y >= 0.0 && roughHistory.y <= 255.0;
    h.surfacePosition = surfacePosition;
    h.geometryNormal = decodeNormalU(g.w);
    h.slowRadiance = vec3(slowRG, slowBM2.x);
    h.secondMoment = decodeSqrtMomentFP16(slowBM2.y);
    h.responsiveRadiance = vec3(fastRG, fastBUnused.x);
    h.endpoint = decodeEndpoint
        ? relaxUnpackEndpointMoments(m.zw)
        : emptyRelaxEndpointMoments();
    h.roughness = roughHistory.x;
    h.historyLength = roughHistory.y;
    h.materialID = valid ? (packedMeta & 0xffffu) : 0u;
    h.reprojectionConfidence = clamp(packedConfidence, 0.0, 1.0);
    if (!valid) {
        h.surfacePosition = vec3(0.0);
        h.geometryNormal = vec3(0.0, 1.0, 0.0);
        h.slowRadiance = vec3(0.0);
        h.secondMoment = 0.0;
        h.responsiveRadiance = vec3(0.0);
        h.endpoint = emptyRelaxEndpointMoments();
        h.roughness = 1.0;
        h.historyLength = 0.0;
        h.reprojectionConfidence = 0.0;
    }
    return h;
}

#endif // BUFFERS_SPECULAR_BUFFER_GLSL
