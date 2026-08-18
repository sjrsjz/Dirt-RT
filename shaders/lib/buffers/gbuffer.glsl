#ifndef BUFFERS_GBUFFER_GLSL
#define BUFFERS_GBUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack_half.glsl"

// ===========================================================================
// Binding 0 — GeometryMaterialBuffer pack/unpack (native uvec4 storage)
// ===========================================================================
// N=0: uvec4(oct(geometryNormal), half(roughness)|materialID16,
//             oct(textureNormal), floatBits(primaryDistance))
// N=1: vec4(packHalf(spR,spG), packHalf(spB,dfR), packHalf(dfG,dfB), pad)
// N=2: vec4(packHalf(trR,trG), packHalf(trB,emR), packHalf(emG,emB), pad)
// N=3: vec4(packHalf(ltR,ltG), packHalf(ltB,abR), packHalf(abG,abB), pad)
// The per-frame jittered primary ray is reconstructed from the native RT
// pixel grid and cached asymmetric projection; it is not stored per pixel.
// A negative distance is the sole sky/invalid-surface marker.

// N=0..5 layer constants (semantic)
#define GEO_N_GEOMETRY     0u
#define GEO_N_ALBEDOS      1u
#define GEO_N_MISC         2u
#define GEO_N_LIGHTABS     3u
#define GEO_N_MOTION       4u  // xyz=currentWorld-previousWorld, w=history validity

#define GEO_N_PRIMARY_MAT  5u  // packed half(Cs.rgb, Cd.rgb, S.xy)

// --- N=0: compact primary geometry ---
uvec4 readPrimaryGeometryWords(uvec2 xy) {
    return geomBuffer.data[addr(GEO_N_GEOMETRY, xy)];
}

float readPrimaryDistance(uvec2 xy) {
    return uintBitsToFloat(
        geomBuffer.data[addr(GEO_N_GEOMETRY, xy)].w);
}

vec3 readPrimaryGeometryNormal(uvec2 xy) {
    return decodeNormalU(readPrimaryGeometryWords(xy).x);
}

vec3 reconstructPrimaryRay(uvec2 xy, uvec2 rayResolution) {
    vec2 safeResolution = max(vec2(rayResolution), vec2(1.0));
    // Match raytrace_rgen exactly: the native RT grid uses pixel/resolution.
    // rtProjectionParams already contains this frame's TAA phase.
    vec2 ndc = vec2(xy) / safeResolution * 2.0 - 1.0;
    vec2 projectionScale = rtProjectionParams.xy;
    vec2 inverseScale = vec2(
        abs(projectionScale.x) > 1e-8 ? 1.0 / projectionScale.x : 0.0,
        abs(projectionScale.y) > 1e-8 ? 1.0 / projectionScale.y : 0.0);
    vec2 viewSlope = (ndc + rtProjectionParams.zw) * inverseScale;
    vec3 viewRay = normalize(vec3(viewSlope, -1.0));
    return transpose(mat3(rtModelView)) * viewRay;
}

vec3 reconstructPrimaryRay(uvec2 xy) {
    return reconstructPrimaryRay(xy, resolution_global);
}

vec3 readPrimaryTextureNormal(uvec2 xy) {
    return decodeNormalU(readPrimaryGeometryWords(xy).z);
}

vec3 reconstructPrimaryRelativePosition(uvec2 xy, float distance) {
    return distance >= 0.0
        ? reconstructPrimaryRay(xy) * distance
        : vec3(0.0);
}

void unpackPrimaryGeometry(uvec4 words, uvec2 xy,
        out vec3 position, out float distance, out vec3 geometryNormal,
        out float roughness, out int materialID, out float pathRoughness) {
    distance = uintBitsToFloat(words.w);
    position = distance >= 0.0
        ? reconstructPrimaryRay(xy) * distance : vec3(0.0);
    geometryNormal = decodeNormalU(words.x);
    roughness = unpackHalf2x16(words.y).x;
    // Active PSR stores its cascaded roughness in REFR_N_META. Retain the
    // legacy output as primary roughness for old, unscheduled consumers.
    pathRoughness = roughness;
    materialID = int(words.y >> 16u);
}

void readPrimaryPosition(uvec2 xy, out vec3 pos, out float dist) {
    uvec4 words = readPrimaryGeometryWords(xy);
    dist = uintBitsToFloat(words.w);
    pos = dist >= 0.0 ? reconstructPrimaryRay(xy) * dist : vec3(0.0);
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

void writePrimaryGeometry(uvec2 xy, vec3 geometryNormal, float rough,
        int illumType, vec3 textureNormal, float primaryDistance) {
    uint roughnessMaterial = packHalf2x16(clamp(vec2(rough, 0.0),
        vec2(-65504.0), vec2(65504.0)));
    roughnessMaterial = (roughnessMaterial & 0xffffu)
        | ((uint(illumType) & 0xffffu) << 16u);
    geomBuffer.data[addr(GEO_N_GEOMETRY, xy)] = uvec4(
        encodeNormalU(geometryNormal),
        roughnessMaterial,
        encodeNormalU(textureNormal),
        floatBitsToUint(primaryDistance));
}
// --- N=1 ---
void writeAlbedosPath(uint N, uvec2 xy, vec3 spec, vec3 diff) {
    geomBuffer.data[addr(N, xy)] = uvec4(
        packHalf2x16(vec2(clamp(spec.r, -65504.0, 65504.0), clamp(spec.g, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(spec.b, -65504.0, 65504.0), clamp(diff.r, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(diff.g, -65504.0, 65504.0), clamp(diff.b, -65504.0, 65504.0))),
        0u
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

vec3 readPrimarySpecularAlbedo(uvec2 xy) {
    uvec4 v = geomBuffer.data[addr(GEO_N_ALBEDOS, xy)];
    vec2 sg = unpackHalf2x16(v.x);
    return vec3(sg, unpackHalf2x16(v.y).x);
}

// --- N=2 ---
void writeMiscTransport(uint N, uvec2 xy, vec3 trans, vec3 emis) {
    geomBuffer.data[addr(N, xy)] = uvec4(
        packHalf2x16(vec2(clamp(trans.r, -65504.0, 65504.0), clamp(trans.g, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(trans.b, -65504.0, 65504.0), clamp(emis.r, -65504.0, 65504.0))),
        packHalf2x16(vec2(clamp(emis.g, -65504.0, 65504.0), clamp(emis.b, -65504.0, 65504.0))),
        0u
    );
}
void readMiscTransport(uint N, uvec2 xy, out vec3 trans, out vec3 emis) {
    uvec4 v = geomBuffer.data[addr(N, xy)];
    vec2 tg = unpackHalf2x16(v.x);
    vec2 te = unpackHalf2x16(v.y);
    vec2 eb = unpackHalf2x16(v.z);
    trans = vec3(tg.x, tg.y, te.x);
    emis = vec3(te.y, eb.x, eb.y);
}

vec3 readPrimaryTransmission(uvec2 xy) {
    uvec4 v = geomBuffer.data[addr(GEO_N_MISC, xy)];
    vec2 tg = unpackHalf2x16(v.x);
    vec2 te = unpackHalf2x16(v.y);
    return vec3(tg, te.x);
}

// --- N=3 ---
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
