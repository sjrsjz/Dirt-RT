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
// N=3: History Geo    — uvec4(fbits(hist_pos.xyz), oct(hist_geometryNormal))
// N=4: Swap Light     — uvec4(pHalf2(swap_aliceY.xy), pHalf2(swap_aliceY.zw), pHalf2(swap_CoCg), pHalf2(weight, sqrt(meanY2)))
// N=5: Path Guide     — uvec4(pHalf2(aliceY.xy), pHalf2(aliceY.zw), fbits(W), fbits(M))
//
// .w lane uses packHalf2x16: weight:f16 + sqrt(meanY2):f16.
// sqrt compression keeps HDR second moments within f16 range (e.g. Y=1000 →
// sqrt(Y²)=1000 < 65504), at the cost of relative precision halved after squaring.
// surfaceMask lives in N=1. N=3 stores the reprojectable history normal instead.

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
uvec4 readDiffuseLightRTRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(DIF_N_LIGHT, xy)];
}
void readDiffuseLightRT(uvec2 xy, out AliceEncoding alice, out float meanY2) {
    uvec4 v = readDiffuseLightRTRaw(xy);
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
        floatBitsToUint(pos),
        floatBitsToUint(surfaceMask)
    );
}
void readDiffuseGeo(uvec2 xy, out vec3 pos, out float surfaceMask) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_GEO, xy)];
    pos = uintBitsToFloat(v.xyz);
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

uint encodeDiffuseHistoryNormalU(vec3 n) {
    n = normalize(n);
    vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    return packSnorm2x16(clamp(p, vec2(-1.0), vec2(1.0)));
}

vec3 decodeDiffuseHistoryNormalU(uint packed_) {
    vec2 p = unpackSnorm2x16(packed_);
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

void writeDiffuseHistGeo(uvec2 xy, vec3 pos, vec3 geometryNormal) {
    diffuseBuffer.data[addr(DIF_N_HISTGEO, xy)] = uvec4(
        floatBitsToUint(pos),
        encodeDiffuseHistoryNormalU(geometryNormal)
    );
}
void readDiffuseHistGeo(uvec2 xy, out vec3 pos, out vec3 geometryNormal) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_HISTGEO, xy)];
    pos = uintBitsToFloat(v.xyz);
    geometryNormal = decodeDiffuseHistoryNormalU(v.w);
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
uvec4 readDiffuseSwapRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(DIF_N_SWAP, xy)];
}
void readDiffuseSwap(uvec2 xy, out AliceEncoding alice, out float weight, out float meanY2) {
    uvec4 v = readDiffuseSwapRaw(xy);
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
//   floatBitsToUint(W),        // reservoir reciprocal-proposal normalization
//   floatBitsToUint(M))        // effective sample count

void writePathGuide(uvec2 xy, vec4 aliceY, float W, float M) {
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)] = uvec4(
        packHalf2x16(clamp(aliceY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(aliceY.zw, vec2(-65504.0), vec2(65504.0))),
        floatBitsToUint(W),
        floatBitsToUint(M));
}
void readPathGuide(uvec2 xy, out vec4 aliceY, out float W, out float M) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    aliceY = vec4(ay_xy, ay_zw);
    W = uintBitsToFloat(v.z);
    M = uintBitsToFloat(v.w);
}

bool pathGuideReservoirValid(vec4 aliceY, float W, float M) {
    return W > 0.0 && M > 0.0
        && !isnan(W) && !isinf(W) && !isnan(M) && !isinf(M)
        && !any(isnan(aliceY)) && !any(isinf(aliceY));
}

// 2×2 bilinear path guide sampling with validity mask
vec4 samplePathGuide(vec2 prevCoord) {
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2  pf = prevCoord - vec2(p0);

    vec4 y00, y10, y01, y11;
    float W00, W10, W01, W11;
    float M00, M10, M01, M11;
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 0), ivec2(0), ivec2(resolution_global) - 1)), y00, W00, M00);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 0), ivec2(0), ivec2(resolution_global) - 1)), y10, W10, M10);
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 1), ivec2(0), ivec2(resolution_global) - 1)), y01, W01, M01);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 1), ivec2(0), ivec2(resolution_global) - 1)), y11, W11, M11);

    bool v00 = pathGuideReservoirValid(y00, W00, M00);
    bool v10 = pathGuideReservoirValid(y10, W10, M10);
    bool v01 = pathGuideReservoirValid(y01, W01, M01);
    bool v11 = pathGuideReservoirValid(y11, W11, M11);

    float w00 = v00 ? (1.0 - pf.x) * (1.0 - pf.y) : 0.0;
    float w10 = v10 ? pf.x * (1.0 - pf.y) : 0.0;
    float w01 = v01 ? (1.0 - pf.x) * pf.y : 0.0;
    float w11 = v11 ? pf.x * pf.y : 0.0;

    float sumW = w00 + w10 + w01 + w11;
    if (sumW < 1e-8) return vec4(0.0);

    return (y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11) / sumW;
}

#endif // BUFFERS_DIFFUSE_BUFFER_GLSL
