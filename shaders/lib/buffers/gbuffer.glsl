#ifndef BUFFERS_GBUFFER_GLSL
#define BUFFERS_GBUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack_half.glsl"

// ===========================================================================
// Binding 0 — GeometryMaterialBuffer pack/unpack
// ===========================================================================
// N=0: vec4(worldPos.xyz, distance)
// N=1: vec4(oct(geometryNormal), roughness, float(illumType), pathRoughness)
// N=2: vec4(packHalf(spR,spG), packHalf(spB,dfR), packHalf(dfG,dfB), pad)
// N=3: vec4(packHalf(trR,trG), packHalf(trB,emR), packHalf(emG,emB), oct(rd))
// N=4: vec4(packHalf(ltR,ltG), packHalf(ltB,abR), packHalf(abG,abB), pad)

// N=0..5 layer constants (semantic)
#define GEO_N_GEO          0u
#define GEO_N_NORMALS      1u
#define GEO_N_ALBEDOS      2u
#define GEO_N_MISC         3u
#define GEO_N_LIGHTABS     4u
#define GEO_N_MICRONORMAL  5u  // oct(macroNormal) — surface normal with detail map
#define GEO_N_MOTION       6u  // xyz=currentWorld-previousWorld, w=history validity

// --- N=0 ---
void writeGeo0(uint N, uvec2 xy, vec3 pos, float dist) {
    geomBuffer.data[addr(N, xy)] = vec4(pos, dist);
}
void readGeo0(uint N, uvec2 xy, out vec3 pos, out float dist) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    pos = v.xyz; dist = v.w;
}

void writeSurfaceMotion(uvec2 xy, vec3 motion, float valid) {
    geomBuffer.data[addr(GEO_N_MOTION, xy)] = vec4(motion, valid);
}

void readSurfaceMotion(uvec2 xy, out vec3 motion, out float valid) {
    vec4 v = geomBuffer.data[addr(GEO_N_MOTION, xy)];
    motion = v.xyz;
    valid = v.w;
}

// --- N=1 ---
void writeGeo1(uint N, uvec2 xy, vec3 geometryNormal, float rough, int illumType, float pathRoughness) {
    geomBuffer.data[addr(N, xy)] = vec4(encodeNormal(geometryNormal), rough, float(illumType), pathRoughness);
}
void readGeo1(uint N, uvec2 xy, out vec3 geometryNormal, out float rough, out int illumType, out float pathRoughness) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    geometryNormal = decodeNormal(v.x);
    rough = v.y;
    illumType = int(v.z);
    pathRoughness = v.w;
}

// Refraction pass only: update pathRoughness (RMW without half pack/unpack)
void writePathRoughness(uint N, uvec2 xy, float pathR) {
    uint idx = addr(N, xy);
    vec4 v = geomBuffer.data[idx];
    v.w = pathR;
    geomBuffer.data[idx] = v;
}
float readPathRoughness(uint N, uvec2 xy) {
    return geomBuffer.data[addr(N, xy)].w;
}

// --- N=2 ---
void writeAlbedosPath(uint N, uvec2 xy, vec3 spec, vec3 diff) {
    geomBuffer.data[addr(N, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(vec2(clamp(spec.r, -65504.0, 65504.0), clamp(spec.g, -65504.0, 65504.0)))),
        uintBitsToFloat(packHalf2x16(vec2(clamp(spec.b, -65504.0, 65504.0), clamp(diff.r, -65504.0, 65504.0)))),
        uintBitsToFloat(packHalf2x16(vec2(clamp(diff.g, -65504.0, 65504.0), clamp(diff.b, -65504.0, 65504.0)))),
        0.0
    );
}
void readAlbedosPath(uint N, uvec2 xy, out vec3 spec, out vec3 diff) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    vec2 sg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 sd = unpackHalf2x16(floatBitsToUint(v.y));
    vec2 db = unpackHalf2x16(floatBitsToUint(v.z));
    spec = vec3(sg.x, sg.y, sd.x);
    diff = vec3(sd.y, db.x, db.y);
}

// --- N=3 ---
void writeMisc(uint N, uvec2 xy, vec3 trans, vec3 emis, vec3 rd) {
    geomBuffer.data[addr(N, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(vec2(clamp(trans.r, -65504.0, 65504.0), clamp(trans.g, -65504.0, 65504.0)))),
        uintBitsToFloat(packHalf2x16(vec2(clamp(trans.b, -65504.0, 65504.0), clamp(emis.r, -65504.0, 65504.0)))),
        uintBitsToFloat(packHalf2x16(vec2(clamp(emis.g, -65504.0, 65504.0), clamp(emis.b, -65504.0, 65504.0)))),
        encodeNormal(rd)
    );
}
void readMisc(uint N, uvec2 xy, out vec3 trans, out vec3 emis, out vec3 rd) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    vec2 tg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 te = unpackHalf2x16(floatBitsToUint(v.y));
    vec2 eb = unpackHalf2x16(floatBitsToUint(v.z));
    trans = vec3(tg.x, tg.y, te.x);
    emis  = vec3(te.y, eb.x, eb.y);
    rd    = decodeNormal(v.w);
}

// --- N=4 ---
void writeLightAbs(uint N, uvec2 xy, vec3 light, vec3 absorption) {
    geomBuffer.data[addr(N, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(vec2(clamp(light.r, -65504.0, 65504.0), clamp(light.g, -65504.0, 65504.0)))),
        uintBitsToFloat(packHalf2x16(vec2(clamp(light.b, -65504.0, 65504.0), clamp(absorption.r, -65504.0, 65504.0)))),
        uintBitsToFloat(packHalf2x16(vec2(clamp(absorption.g, -65504.0, 65504.0), clamp(absorption.b, -65504.0, 65504.0)))),
        0.0
    );
}
void readLightAbs(uint N, uvec2 xy, out vec3 light, out vec3 absorption) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    vec2 lg = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 la = unpackHalf2x16(floatBitsToUint(v.y));
    vec2 ab = unpackHalf2x16(floatBitsToUint(v.z));
    light      = vec3(lg.x, lg.y, la.x);
    absorption = vec3(la.y, ab.x, ab.y);
}

// --- N=5 ---
void writeMicroNormal(uint N, uvec2 xy, vec3 microN) {
    geomBuffer.data[addr(N, xy)] = vec4(encodeNormal(microN), 0.0, 0.0, 0.0);
}
vec3 readMicroNormal(uint N, uvec2 xy) {
    return decodeNormal(geomBuffer.data[addr(N, xy)].x);
}

#endif // BUFFERS_GBUFFER_GLSL
