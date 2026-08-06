#ifndef BUFFERS_DIFFUSE_BUFFER_GLSL
#define BUFFERS_DIFFUSE_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/lighting/alice_encode.glsl"

// ===========================================================================
// Binding 2 — DiffuseBuffer pack/unpack (uvec4 raw-integer storage)
// ===========================================================================
// N=0: Current Light  — uvec4(pHalf2(aliceY.xy), pHalf2(aliceY.zw), pHalf2(CoCg), pHalf2(0, sqrt(meanY2)))
// N=1: Current Geo    — uvec4(fbits(pos.xyz), fbits(surfaceMask))
// N=2: History Light  — uvec4(pHalf2(hist_aliceY.xy), pHalf2(hist_aliceY.zw), pHalf2(hist_CoCg), pHalf2(weight, sqrt(meanY2)))
// N=3: History Geo    — uvec4(fbits(hist_pos.xyz), fbits(surfaceMask))
// N=4: Swap Light     — uvec4(pHalf2(swap_aliceY.xy), pHalf2(swap_aliceY.zw), pHalf2(swap_CoCg), pHalf2(weight, sqrt(meanY2)))
// N=5: Path Guide     — uvec4(pHalf2(aliceY.xy), pHalf2(aliceY.zw), fbits(M), 0u)
//
// .w lane uses packHalf2x16: weight:f16 + sqrt(meanY2):f16.
// sqrt compression keeps HDR second moments within f16 range (e.g. Y=1000 →
// sqrt(Y²)=1000 < 65504), at the cost of relative precision halved after squaring.
// surfaceMask lives only in N=1/N=3 (geo layers), not duplicated in light layers.

#define DIF_N_LIGHT    0u
#define DIF_N_GEO      1u
#define DIF_N_HIST     2u
#define DIF_N_HISTGEO  3u
#define DIF_N_SWAP     4u
#define DIF_N_PATHGUIDE 5u

// ===========================================================================
// N=0 — Current RT Light
// ===========================================================================
// .w = packHalf2x16(0.0, sqrt(meanY2))  —  single-sample second moment Y²

void writeDiffuseLightRT(uvec2 xy, AliceEncoding alice, float meanY2) {
    float sqrtM2 = sqrt(max(meanY2, 0.0));
    diffuseBuffer.data[addr(DIF_N_LIGHT, xy)] = uvec4(
        packHalf2x16(clamp(alice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(alice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(alice.CoCg,        vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(0.0, sqrtM2))
    );
}
void readDiffuseLightRT(uvec2 xy, out AliceEncoding alice, out float meanY2) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_LIGHT, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    vec2 cocg  = unpackHalf2x16(v.z);
    alice.aliceY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    alice.CoCg = cocg;
    vec2 wm = unpackHalf2x16(v.w);
    meanY2 = wm.y * wm.y;  // undo sqrt compression
}

// Helper: write zero light (sky reset)
void writeDiffuseLightRTSky(uvec2 xy) {
    diffuseBuffer.data[addr(DIF_N_LIGHT, xy)] = uvec4(0u);
}

// ===========================================================================
// N=1 — Current Geometry (surfaceMask lives ONLY here, not duplicated in N=0)
// ===========================================================================

void writeDiffuseGeo(uvec2 xy, vec3 pos, float surfaceMask) {
    diffuseBuffer.data[addr(DIF_N_GEO, xy)] = uvec4(
        floatBitsToUint(pos.x),
        floatBitsToUint(pos.y),
        floatBitsToUint(pos.z),
        floatBitsToUint(surfaceMask)
    );
}
void readDiffuseGeo(uvec2 xy, out vec3 pos, out float surfaceMask) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_GEO, xy)];
    pos = vec3(uintBitsToFloat(v.x), uintBitsToFloat(v.y), uintBitsToFloat(v.z));
    surfaceMask = uintBitsToFloat(v.w);
}

// Convenience: read only surfaceMask from N=1
float readDiffuseSurfaceMask(uvec2 xy) {
    return uintBitsToFloat(diffuseBuffer.data[addr(DIF_N_GEO, xy)].w);
}

// ===========================================================================
// N=2 — History Light
// ===========================================================================
// .w = packHalf2x16(weight, sqrt(meanY2))  →  weight:f16 + sqrt(meanY2):f16

void writeDiffuseHist(uvec2 xy, AliceEncoding alice, float weight, float meanY2) {
    float sqrtM2 = sqrt(max(meanY2, 0.0));
    diffuseBuffer.data[addr(DIF_N_HIST, xy)] = uvec4(
        packHalf2x16(clamp(alice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(alice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(alice.CoCg,        vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(weight, sqrtM2))
    );
}
void readDiffuseHist(uvec2 xy, out AliceEncoding alice, out float weight, out float meanY2) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_HIST, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    vec2 cocg  = unpackHalf2x16(v.z);
    alice.aliceY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    alice.CoCg = cocg;
    vec2 wm = unpackHalf2x16(v.w);
    weight = wm.x;
    meanY2 = wm.y * wm.y;
}

// ===========================================================================
// N=3 — History Geometry
// ===========================================================================

void writeDiffuseHistGeo(uvec2 xy, vec3 pos, float surfaceMask) {
    diffuseBuffer.data[addr(DIF_N_HISTGEO, xy)] = uvec4(
        floatBitsToUint(pos.x),
        floatBitsToUint(pos.y),
        floatBitsToUint(pos.z),
        floatBitsToUint(surfaceMask)
    );
}
void readDiffuseHistGeo(uvec2 xy, out vec3 pos, out float surfaceMask) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_HISTGEO, xy)];
    pos = vec3(uintBitsToFloat(v.x), uintBitsToFloat(v.y), uintBitsToFloat(v.z));
    surfaceMask = uintBitsToFloat(v.w);
}

// ===========================================================================
// N=4 — Swap Light
// ===========================================================================
// .w = packHalf2x16(weight, sqrt(meanY2))  →  weight:f16 + sqrt(meanY2):f16

void writeDiffuseSwap(uvec2 xy, AliceEncoding alice, float weight, float meanY2) {
    float sqrtM2 = sqrt(max(meanY2, 0.0));
    diffuseBuffer.data[addr(DIF_N_SWAP, xy)] = uvec4(
        packHalf2x16(clamp(alice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(alice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(alice.CoCg,        vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(weight, sqrtM2))
    );
}
void readDiffuseSwap(uvec2 xy, out AliceEncoding alice, out float weight, out float meanY2) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_SWAP, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    vec2 cocg  = unpackHalf2x16(v.z);
    alice.aliceY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    alice.CoCg = clamp(cocg, vec2(-65504.0), vec2(65504.0));
    vec2 wm = unpackHalf2x16(v.w);
    weight = wm.x;
    meanY2 = wm.y * wm.y;
}

// ===========================================================================
// N=5 — ReSTIR temporal reservoir (Path Guide Reservoir)
// ===========================================================================
// Layout: uvec4(
//   packHalf2x16(aliceY.xy),   // sample direction × luminance
//   packHalf2x16(aliceY.zw),   // total energy
//   floatBitsToUint(M),        // effective sample count
//   0u)                        // pad

void writePathGuide(uvec2 xy, vec4 aliceY, float M) {
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)] = uvec4(
        packHalf2x16(clamp(aliceY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(aliceY.zw, vec2(-65504.0), vec2(65504.0))),
        floatBitsToUint(M),
        0u);
}
void readPathGuide(uvec2 xy, out vec4 aliceY, out float M) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    aliceY = vec4(ay_xy, ay_zw);
    M = uintBitsToFloat(v.z);
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

    float w00 = (M00 > 0.0) ? (1.0 - pf.x) * (1.0 - pf.y) : 0.0;
    float w10 = (M10 > 0.0) ? pf.x * (1.0 - pf.y) : 0.0;
    float w01 = (M01 > 0.0) ? (1.0 - pf.x) * pf.y : 0.0;
    float w11 = (M11 > 0.0) ? pf.x * pf.y : 0.0;

    float sumW = w00 + w10 + w01 + w11;
    if (sumW < 1e-8) return vec4(0.0);

    return (y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11) / sumW;
}

#endif // BUFFERS_DIFFUSE_BUFFER_GLSL
