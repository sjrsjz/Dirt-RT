#ifndef BUFFERS_GBUFFER_GLSL
#define BUFFERS_GBUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack_half.glsl"

// ===========================================================================
// Binding 0 — GeometryMaterialBuffer pack/unpack (native uvec4 storage)
// ===========================================================================
// N=0: vec4(worldPos.xyz, distance)
// N=1: vec4(oct(geometryNormal), roughness, float(illumType), pathRoughness)
// N=2: vec4(packHalf(spR,spG), packHalf(spB,dfR), packHalf(dfG,dfB), oct(microN))
// N=3: vec4(packHalf(trR,trG), packHalf(trB,emR), packHalf(emG,emB), oct(rd))
// N=4: vec4(packHalf(ltR,ltG), packHalf(ltB,abR), packHalf(abG,abB), pad)

// N=0..6 layer constants (semantic)
#define GEO_N_GEO          0u
#define GEO_N_NORMALS      1u
#define GEO_N_ALBEDOS      2u
#define GEO_N_MISC         3u
#define GEO_N_LIGHTABS     4u
#define GEO_N_MICRONORMAL  GEO_N_ALBEDOS // Compatibility alias: stored in N=2.w
#define GEO_N_MOTION       5u  // xyz=currentWorld-previousWorld, w=history validity

#define GEO_N_PRIMARY_MAT  6u  // packed half(Cs.rgb, Cd.rgb, S.xy)

// --- N=0 ---
void writeGeo0(uint N, uvec2 xy, vec3 pos, float dist) {
    geomBuffer.data[addr(N, xy)] = floatBitsToUint(vec4(pos, dist));
}
void readGeo0(uint N, uvec2 xy, out vec3 pos, out float dist) {
    vec4 v = uintBitsToFloat(geomBuffer.data[addr(N, xy)]);
    pos = v.xyz; dist = v.w;
}

void writeSurfaceMotion(uvec2 xy, vec3 motion, float valid) {
    geomBuffer.data[addr(GEO_N_MOTION, xy)] =
        floatBitsToUint(vec4(motion, valid));
}

void readSurfaceMotion(uvec2 xy, out vec3 motion, out float valid) {
    vec4 v = uintBitsToFloat(geomBuffer.data[addr(GEO_N_MOTION, xy)]);
    motion = v.xyz;
    valid = v.w;
}

// --- N=1 ---
void writeGeo1(uint N, uvec2 xy, vec3 geometryNormal, float rough, int illumType, float pathRoughness) {
    geomBuffer.data[addr(N, xy)] = uvec4(
        encodeNormalU(geometryNormal),
        floatBitsToUint(rough),
        uint(illumType),
        floatBitsToUint(pathRoughness));
}
void readGeo1(uint N, uvec2 xy, out vec3 geometryNormal, out float rough, out int illumType, out float pathRoughness) {
    uvec4 v = geomBuffer.data[addr(N, xy)];
    geometryNormal = decodeNormalU(v.x);
    rough = uintBitsToFloat(v.y);
    illumType = int(v.z);
    pathRoughness = uintBitsToFloat(v.w);
}

// Refraction pass only: update pathRoughness (RMW without half pack/unpack)
void writePathRoughness(uint N, uvec2 xy, float pathR) {
    geomBuffer.data[addr(N, xy)].w = floatBitsToUint(pathR);
}
float readPathRoughness(uint N, uvec2 xy) {
    return uintBitsToFloat(geomBuffer.data[addr(N, xy)].w);
}

// --- N=2 ---
void writeAlbedosPath(uint N, uvec2 xy, vec3 spec, vec3 diff, vec3 microN) {
    geomBuffer.data[addr(N, xy)] = uvec4(
        packHalf2x16(vec2(clamp(spec.r, -65504.0, 65504.0), clamp(spec.g, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(spec.b, -65504.0, 65504.0), clamp(diff.r, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(diff.g, -65504.0, 65504.0), clamp(diff.b, -65504.0, 65504.0))),
        encodeNormalU(microN)
    );
}
void readAlbedosPath(uint N, uvec2 xy, out vec3 spec, out vec3 diff) {
    uvec4 v = geomBuffer.data[addr(N, xy)];
    vec2 sg = unpackHalf2x16(v.x);
    vec2 sd = unpackHalf2x16(v.y);
    vec2 db = unpackHalf2x16(v.z);
    spec = vec3(sg.x, sg.y, sd.x);
    diff = vec3(sd.y, db.x, db.y);
}

void readAlbedosPathMicroNormal(uint N, uvec2 xy, out vec3 spec,
        out vec3 diff, out vec3 microN) {
    uvec4 v = geomBuffer.data[addr(N, xy)];
    vec2 sg = unpackHalf2x16(v.x);
    vec2 sd = unpackHalf2x16(v.y);
    vec2 db = unpackHalf2x16(v.z);
    spec = vec3(sg.x, sg.y, sd.x);
    diff = vec3(sd.y, db.x, db.y);
    microN = decodeNormalU(v.w);
}

vec3 readPrimarySpecularAlbedo(uvec2 xy) {
    uvec4 v = geomBuffer.data[addr(GEO_N_ALBEDOS, xy)];
    vec2 sg = unpackHalf2x16(v.x);
    return vec3(sg, unpackHalf2x16(v.y).x);
}

vec3 readPrimarySpecularAlbedoMicroNormal(uvec2 xy, out vec3 microN) {
    uvec4 v = geomBuffer.data[addr(GEO_N_ALBEDOS, xy)];
    vec2 sg = unpackHalf2x16(v.x);
    microN = decodeNormalU(v.w);
    return vec3(sg, unpackHalf2x16(v.y).x);
}

// --- N=3 ---
void writeMisc(uint N, uvec2 xy, vec3 trans, vec3 emis, vec3 rd) {
    geomBuffer.data[addr(N, xy)] = uvec4(
        packHalf2x16(vec2(clamp(trans.r, -65504.0, 65504.0), clamp(trans.g, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(trans.b, -65504.0, 65504.0), clamp(emis.r, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(emis.g, -65504.0, 65504.0), clamp(emis.b, -65504.0, 65504.0))),
        encodeNormalU(rd)
    );
}
void readMisc(uint N, uvec2 xy, out vec3 trans, out vec3 emis, out vec3 rd) {
    uvec4 v = geomBuffer.data[addr(N, xy)];
    vec2 tg = unpackHalf2x16(v.x);
    vec2 te = unpackHalf2x16(v.y);
    vec2 eb = unpackHalf2x16(v.z);
    trans = vec3(tg.x, tg.y, te.x);
    emis  = vec3(te.y, eb.x, eb.y);
    rd    = decodeNormalU(v.w);
}

void readPrimaryTransmissionAndRay(uvec2 xy, out vec3 trans, out vec3 rd) {
    uvec4 v = geomBuffer.data[addr(GEO_N_MISC, xy)];
    vec2 tg = unpackHalf2x16(v.x);
    vec2 te = unpackHalf2x16(v.y);
    trans = vec3(tg, te.x);
    rd = decodeNormalU(v.w);
}

vec3 readPrimaryRayDirection(uvec2 xy) {
    return decodeNormalU(geomBuffer.data[addr(GEO_N_MISC, xy)].w);
}

// --- N=4 ---
void writeLightAbs(uint N, uvec2 xy, vec3 light, vec3 absorption) {
    geomBuffer.data[addr(N, xy)] = uvec4(
        packHalf2x16(vec2(clamp(light.r, -65504.0, 65504.0), clamp(light.g, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(light.b, -65504.0, 65504.0), clamp(absorption.r, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(absorption.g, -65504.0, 65504.0), clamp(absorption.b, -65504.0, 65504.0))),
        0u
    );
}
void readLightAbs(uint N, uvec2 xy, out vec3 light, out vec3 absorption) {
    uvec4 v = geomBuffer.data[addr(N, xy)];
    vec2 lg = unpackHalf2x16(v.x);
    vec2 la = unpackHalf2x16(v.y);
    vec2 ab = unpackHalf2x16(v.z);
    light      = vec3(lg.x, lg.y, la.x);
    absorption = vec3(la.y, ab.x, ab.y);
}

// --- N=5 ---
void writeMicroNormal(uint N, uvec2 xy, vec3 microN) {
    uint index = addr(GEO_N_ALBEDOS, xy);
    geomBuffer.data[index].w = encodeNormalU(microN);
}
vec3 readMicroNormal(uint N, uvec2 xy) {
    return decodeNormalU(geomBuffer.data[addr(GEO_N_ALBEDOS, xy)].w);
}

// --- N=7 ---
// The primary visibility pass evaluates the texture-backed material once.
// Dedicated lobe passes consume this compact transport representation instead
// of tracing and shading the primary surface again.
void writePrimaryMaterial(uvec2 xy, vec3 Cs, vec3 Cd, vec2 S) {
    geomBuffer.data[addr(GEO_N_PRIMARY_MAT, xy)] = uvec4(
        packHalf2x16(clamp(Cs.rg, -65504.0, 65504.0)),
        packHalf2x16(clamp(vec2(Cs.b, Cd.r), -65504.0, 65504.0)),
        packHalf2x16(clamp(Cd.gb, -65504.0, 65504.0)),
        packHalf2x16(clamp(S, -65504.0, 65504.0))
    );
}

void readPrimaryMaterial(uvec2 xy, out vec3 Cs, out vec3 Cd, out vec2 S) {
    uvec4 v = geomBuffer.data[addr(GEO_N_PRIMARY_MAT, xy)];
    vec2 csRG = unpackHalf2x16(v.x);
    vec2 csBcdR = unpackHalf2x16(v.y);
    vec2 cdGB = unpackHalf2x16(v.z);
    Cs = vec3(csRG, csBcdR.x);
    Cd = vec3(csBcdR.y, cdGB);
    S = unpackHalf2x16(v.w);
}

#endif // BUFFERS_GBUFFER_GLSL
