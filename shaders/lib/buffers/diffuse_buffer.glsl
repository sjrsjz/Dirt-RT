#ifndef BUFFERS_DIFFUSE_BUFFER_GLSL
#define BUFFERS_DIFFUSE_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/lighting/alice_encode.glsl"

// ===========================================================================
// Binding 2 — DiffuseBuffer pack/unpack
// ===========================================================================
// N=0: Current Light  — vec4(aliceY_xy_f16, aliceY_zw_f16, CoCg_f16, surfaceMask)
// N=1: Current Geo    — vec4(worldPos.xyz, surfaceMask)
// N=2: History Light  — vec4(hist_aliceY_xy_f16, hist_aliceY_zw_f16, hist_CoCg_f16, hist_weight)
// N=3: History Geo    — vec4(hist_worldPos.xyz, surfaceMask)
// N=4: Swap Light     — vec4(swap_aliceY_xy_f16, swap_aliceY_zw_f16, swap_CoCg_f16, swap_weight)
// N=5: Path Guide     — vec4(packHalf(aliceY.xy), packHalf(aliceY.zw), M, 0)  same as colorimg6 layout
//
// surfaceMask: 1.0 = valid surface, 0.0 = sky/invalid.

#define DIF_N_LIGHT   0u
#define DIF_N_GEO     1u
#define DIF_N_HIST    2u
#define DIF_N_HISTGEO 3u
#define DIF_N_SWAP    4u
#define DIF_N_PATHGUIDE 5u

// --- N=0: Current RT Light ---
void writeDiffuseLightRT(uvec2 xy, AliceEncoding alice, float surfaceMask) {
    diffuseBuffer.data[addr(DIF_N_LIGHT, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(clamp(alice.aliceY.xy, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(alice.aliceY.zw, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(alice.CoCg, vec2(-65504.0), vec2(65504.0)))),
        surfaceMask
    );
}
void readDiffuseLightRT(uvec2 xy, out AliceEncoding alice, out float surfaceMask) {
    vec4 v = diffuseBuffer.data[addr(DIF_N_LIGHT, xy)];
    vec2 ay_xy = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 ay_zw = unpackHalf2x16(floatBitsToUint(v.y));
    vec2 cocg  = unpackHalf2x16(floatBitsToUint(v.z));
    alice.aliceY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    alice.CoCg = cocg;
    surfaceMask = v.w;
}

// Helper: write zero light (sky reset — surfaceMask=0.0 implicit)
void writeDiffuseLightRTSky(uvec2 xy) {
    diffuseBuffer.data[addr(DIF_N_LIGHT, xy)] = vec4(0.0);
}

// --- N=1: Current Geometry ---
void writeDiffuseGeo(uvec2 xy, vec3 pos, float surfaceMask) {
    diffuseBuffer.data[addr(DIF_N_GEO, xy)] = vec4(pos, surfaceMask);
}
void readDiffuseGeo(uvec2 xy, out vec3 pos, out float surfaceMask) {
    vec4 v = diffuseBuffer.data[addr(DIF_N_GEO, xy)];
    pos = v.xyz;
    surfaceMask = v.w;
}

// --- N=2: History Light ---
void writeDiffuseHist(uvec2 xy, AliceEncoding alice, float weight) {
    diffuseBuffer.data[addr(DIF_N_HIST, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(clamp(alice.aliceY.xy, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(alice.aliceY.zw, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(alice.CoCg, vec2(-65504.0), vec2(65504.0)))),
        clamp(weight, 0.0, 65504.0)
    );
}
void readDiffuseHist(uvec2 xy, out AliceEncoding alice, out float weight) {
    vec4 v = diffuseBuffer.data[addr(DIF_N_HIST, xy)];
    vec2 ay_xy = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 ay_zw = unpackHalf2x16(floatBitsToUint(v.y));
    vec2 cocg  = unpackHalf2x16(floatBitsToUint(v.z));
    alice.aliceY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    alice.CoCg = cocg;
    weight = v.w;
}

// --- N=3: History Geometry ---
void writeDiffuseHistGeo(uvec2 xy, vec3 pos, float surfaceMask) {
    diffuseBuffer.data[addr(DIF_N_HISTGEO, xy)] = vec4(pos, surfaceMask);
}
void readDiffuseHistGeo(uvec2 xy, out vec3 pos, out float surfaceMask) {
    vec4 v = diffuseBuffer.data[addr(DIF_N_HISTGEO, xy)];
    pos = v.xyz;
    surfaceMask = v.w;
}

// --- N=4: Swap Light ---
void writeDiffuseSwap(uvec2 xy, AliceEncoding alice, float weight) {
    diffuseBuffer.data[addr(DIF_N_SWAP, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(clamp(alice.aliceY.xy, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(alice.aliceY.zw, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(alice.CoCg, vec2(-65504.0), vec2(65504.0)))),
        clamp(weight, 0.0, 65504.0)
    );
}
void readDiffuseSwap(uvec2 xy, out AliceEncoding alice, out float weight) {
    vec4 v = diffuseBuffer.data[addr(DIF_N_SWAP, xy)];
    vec2 ay_xy = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 ay_zw = unpackHalf2x16(floatBitsToUint(v.y));
    vec2 cocg  = unpackHalf2x16(floatBitsToUint(v.z));
    alice.aliceY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    alice.CoCg = clamp(cocg, vec2(-65504.0), vec2(65504.0));
    weight = v.w;
}

// ===========================================================================
// N=5 — ReSTIR temporal reservoir (Path Guide Reservoir)
// ===========================================================================
// N=5 layout: vec4(
//   packHalf(aliceY.xy),  // sample direction × luminance (2×f16 → f32)
//   packHalf(aliceY.zw),  // total energy            (2×f16 → f32)
//   M,                     // effective sample count / reservoir weight sum
//   0.0)                   // pad

void writePathGuide(uvec2 xy, vec4 aliceY, float M) {
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)] = vec4(
        uintBitsToFloat(packHalf2x16(clamp(aliceY.xy, vec2(-65504.0), vec2(65504.0)))),
        uintBitsToFloat(packHalf2x16(clamp(aliceY.zw, vec2(-65504.0), vec2(65504.0)))),
        M,
        0.0);
}
void readPathGuide(uvec2 xy, out vec4 aliceY, out float M) {
    vec4 v = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)];
    vec2 ay_xy = unpackHalf2x16(floatBitsToUint(v.x));
    vec2 ay_zw = unpackHalf2x16(floatBitsToUint(v.y));
    aliceY = vec4(ay_xy, ay_zw);
    M = v.w;
}

// 2×2 bilinear path guide sampling with validity mask
vec4 samplePathGuide(vec2 prevCoord) {
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2  pf = prevCoord - vec2(p0);

    vec4 y00, y10, y01, y11;
    float M00, M10, M01, M11;
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 0), ivec2(0), ivec2(resolution_global) - 1)), y00, M00);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 0), ivec2(0), ivec2(resolution_global) - 1)), y10, M10);
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 1), ivec2(0), ivec2(resolution_global) - 1)), y01, M01);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 1), ivec2(0), ivec2(resolution_global) - 1)), y11, M11);

    // Validity mask: M > 0 means valid reservoir
    float w00 = (M00 > 0.0) ? (1.0 - pf.x) * (1.0 - pf.y) : 0.0;
    float w10 = (M10 > 0.0) ? pf.x * (1.0 - pf.y) : 0.0;
    float w01 = (M01 > 0.0) ? (1.0 - pf.x) * pf.y : 0.0;
    float w11 = (M11 > 0.0) ? pf.x * pf.y : 0.0;

    float sumW = w00 + w10 + w01 + w11;
    if (sumW < 1e-8) return vec4(0.0);

    return (y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11) / sumW;
}

#endif // BUFFERS_DIFFUSE_BUFFER_GLSL
