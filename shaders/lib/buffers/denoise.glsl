
#include "/lib/buffers/frame_data.glsl"
#include "/lib/lighting/alice.glsl"
#include "/lib/common/tiled_addr.glsl"
#include "/lib/common/oct_encode.glsl"

// ===========================================================================
// Vec4-based SSBO addressing — 用 tiledAddr8x8 替代旧 getIndex
// ===========================================================================
// 所有 SSBO 均以 vec4 data[] 存储。每个"抽象 image"是一张分辨率×分辨率
// 的 vec4 格网，以 8×8 瓦片编码。N 选抽象 image 层级。

uint addr(uint N, uvec2 xy) {
    // 钳制到 resolution_global 范围内，匹配旧 getIndex 行为：防止越界坐标破坏 tile 计算
    uvec2 p = min(xy, uvec2(resolution_global) - 1u);
    return tiledAddr8x8(N, uint(resolution_global.x), uint(resolution_global.y), p.x, p.y);
}

uint addr(uint N, ivec2 xy) {
    return addr(N, uvec2(xy));
}

// ===========================================================================
// SSBO 声明 — 4 个 binding，全部 vec4 data[]
// ===========================================================================

layout(std430, set = 3, binding = 0) buffer GeometryMaterialBuffer {
    vec4 data[];
} geomBuffer;

layout(std430, set = 3, binding = 2) buffer DiffuseBuffer {
    vec4 data[];
} diffuseBuffer;

layout(std430, set = 3, binding = 3) buffer ReflectBuffer {
    vec4 data[];
} reflectBuffer;

layout(std430, set = 3, binding = 4) buffer RefractBuffer {
    vec4 data[];
} refractBuffer;

// ===========================================================================
// Binding 0 — GeometryMaterialBuffer pack/unpack
// ===========================================================================
// N=0: vec4(worldPos.xyz, distance)
// N=1: vec4(oct(macroNormal), roughness, float(illumType), pathRoughness)
// N=2: vec4(packHalf(spR,spG), packHalf(spB,dfR), packHalf(dfG,dfB), pad)
// N=3: vec4(packHalf(trR,trG), packHalf(trB,emR), packHalf(emG,emB), oct(rd))
// N=4: vec4(packHalf(ltR,ltG), packHalf(ltB,abR), packHalf(abG,abB), pad)

// --- N=0 ---
void writeGeo0(uint N, uvec2 xy, vec3 pos, float dist) {
    geomBuffer.data[addr(N, xy)] = vec4(pos, dist);
}
void readGeo0(uint N, uvec2 xy, out vec3 pos, out float dist) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    pos = v.xyz; dist = v.w;
}

// --- N=1 ---
void writeGeo1(uint N, uvec2 xy, vec3 macroN, float rough, int illumType, float pathRoughness) {
    geomBuffer.data[addr(N, xy)] = vec4(encodeNormal(macroN), rough, float(illumType), pathRoughness);
}
void readGeo1(uint N, uvec2 xy, out vec3 macroN, out float rough, out int illumType, out float pathRoughness) {
    vec4 v = geomBuffer.data[addr(N, xy)];
    macroN = decodeNormal(v.x);
    rough = v.y;
    illumType = int(v.z);
    pathRoughness = v.w;
}

// 折射 pass 专用：只更新 pathRoughness（RMW 但无需 half 打包/解包）
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

// N=0..5 常量（语义化）
#define GEO_N_GEO          0u
#define GEO_N_NORMALS      1u
#define GEO_N_ALBEDOS      2u
#define GEO_N_MISC         3u
#define GEO_N_LIGHTABS     4u
#define GEO_N_MICRONORMAL  5u  // oct(microNormal) — surface normal with detail map

// --- N=5 ---
void writeMicroNormal(uint N, uvec2 xy, vec3 microN) {
    geomBuffer.data[addr(N, xy)] = vec4(encodeNormal(microN), 0.0, 0.0, 0.0);
}
vec3 readMicroNormal(uint N, uvec2 xy) {
    return decodeNormal(geomBuffer.data[addr(N, xy)].x);
}

// ===========================================================================
// pack2Half / unpack2Half — 保留现有工具
// ===========================================================================

float pack2Half(float a, float b) {
    return uintBitsToFloat(packHalf2x16(vec2(a, b)));
}

void unpack2Half(float packed_, out float a, out float b) {
    vec2 v = unpackHalf2x16(floatBitsToUint(packed_));
    a = v.x;
    b = v.y;
}

const float VPROJDIST_SKY = 60000.0;

float pack2HalfClamped(float a, float b) {
    return uintBitsToFloat(packHalf2x16(
        vec2(clamp(a, -65504.0, 65504.0), clamp(b, -65504.0, 65504.0))));
}

// ===========================================================================
// ALICE 光照编码 — 保持现有逻辑不变
// ===========================================================================

struct AliceEncoding {
    vec4 aliceY;
    vec2 CoCg;
};

AliceEncoding radiance_to_alice(vec3 color, vec3 dir)
{
    AliceEncoding result;
    float Y = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float Co = 0.5 * color.r - 0.5 * color.b;
    float Cg = -0.25 * color.r + 0.5 * color.g - 0.25 * color.b;
    result.CoCg = vec2(Co, Cg);
    result.aliceY = vec4(dir * Y, Y);
    return result;
}

vec3 project_alice_irradiance(AliceEncoding encoded, vec3 N)
{
    float total_omega = encoded.aliceY.w;
    float irradiance = alice_irradiance(encoded.aliceY, N);
    float attenuation = (total_omega > 1e-10) ? (irradiance / total_omega) : 0.0;
    float Co = encoded.CoCg.x * attenuation;
    float Cg = encoded.CoCg.y * attenuation;
    float B = irradiance - 1.1404 * Co - 1.4304 * Cg;
    float R = B + 2.0 * Co;
    float G = irradiance - 0.1404 * Co + 0.5696 * Cg;
    return max(vec3(R, G, B), vec3(0.0));
}

// colortex5 双对偶向量打包/解包
vec4 packDualVector(vec3 dual_theta, float dual_beta) {
    return vec4(dual_theta, dual_beta);
}
vec4 packDualVectorFromEncoded(vec4 aliceEncoded) {
    vec4 tb = alice_theta_beta(aliceEncoded);
    return tb;
}
void unpackDualVector(vec4 packed_, out vec3 dual_theta, out float dual_beta) {
    dual_theta = packed_.xyz;
    dual_beta = packed_.w;
}

// ALICE 混合原语
AliceEncoding mix_alice(AliceEncoding a, AliceEncoding b, float s) {
    AliceEncoding result;
    result.aliceY = mix(a.aliceY, b.aliceY, s);
    result.CoCg = mix(a.CoCg, b.CoCg, s);
    return result;
}
AliceEncoding init_alice() {
    AliceEncoding result;
    result.aliceY = vec4(0.0);
    result.CoCg = vec2(0.0);
    return result;
}
AliceEncoding scale_alice(AliceEncoding A, float x) {
    AliceEncoding tmp;
    tmp.CoCg = A.CoCg * x;
    tmp.aliceY = A.aliceY * x;
    return tmp;
}
void accumulate_alice(inout AliceEncoding accum, AliceEncoding b, float scale) {
    accum.aliceY += b.aliceY * scale;
    accum.CoCg += b.CoCg * scale;
}

vec3 packAlice(AliceEncoding encoded) {
    float s0 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.x, encoded.aliceY.y)));
    float s1 = uintBitsToFloat(packHalf2x16(vec2(encoded.aliceY.z, encoded.aliceY.w)));
    float s2 = uintBitsToFloat(packHalf2x16(vec2(encoded.CoCg.x, encoded.CoCg.y)));
    return vec3(s0, s1, s2);
}
AliceEncoding unpackAlice(float s0, float s1, float s2) {
    AliceEncoding encoded;
    vec2 v0 = unpackHalf2x16(floatBitsToUint(s0));
    vec2 v1 = unpackHalf2x16(floatBitsToUint(s1));
    vec2 v2 = unpackHalf2x16(floatBitsToUint(s2));
    encoded.aliceY = vec4(v0.x, v0.y, v1.x, v1.y);
    encoded.CoCg = v2;
    return encoded;
}

// ===========================================================================
// Binding 2 — DiffuseBuffer pack/unpack
// ===========================================================================
// N=0: Current Light  — vec4(aliceY_xy_f16, aliceY_zw_f16, CoCg_f16, surfaceMask)
// N=1: Current Geo    — vec4(worldPos.xyz, surfaceMask)
// N=2: History Light  — vec4(hist_aliceY_xy_f16, hist_aliceY_zw_f16, hist_CoCg_f16, hist_weight)
// N=3: History Geo    — vec4(hist_worldPos.xyz, surfaceMask)
// N=4: Swap Light     — vec4(swap_aliceY_xy_f16, swap_aliceY_zw_f16, swap_CoCg_f16, swap_weight)
// N=5: Path Guide     — vec4(packHalf(aliceY.xy), packHalf(aliceY.zw), W, M)  同 colorimg6 布局
//
// surfaceMask: 1.0 = valid surface, 0.0 = sky/invalid.
// Replaces oct(normal) — ALICE encodes the demodulated incident light field;
// normals are only needed at final composite (macroNormal from Geo1 suffices).

#define DIF_N_LIGHT   0u
#define DIF_N_GEO     1u
#define DIF_N_HIST    2u
#define DIF_N_HISTGEO 3u
#define DIF_N_SWAP    4u

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
    alice.CoCg = cocg;
    weight = v.w;
}

// ===========================================================================
// Binding 2 N=5 — ReSTIR 时域蓄水池 (Path Guide Reservoir)
// ===========================================================================
// N=5 布局: vec4(
//   packHalf(aliceY.xy),  // 样本方向×亮度 (2×f16 → f32)
//   packHalf(aliceY.zw),  // 样本总能量       (2×f16 → f32)
//   M,                     // 有效样本数 / 蓄水池权重和
//   0.0)                   // pad
//
// composite58 ReSTIR 合并后写入 colorimg4，composite59 抄入 N=5。
// raygen 通过 samplePathGuide 做 2×2 双线性采样（含有效性 mask）。

#define DIF_N_PATHGUIDE 5u

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

// 2×2 双线性采样路径引导，含有效性 mask 修正
vec4 samplePathGuide(vec2 prevCoord) {
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2  pf = prevCoord - vec2(p0);

    vec4 y00, y10, y01, y11;
    float M00, M10, M01, M11;
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 0), ivec2(0), ivec2(resolution_global) - 1)), y00, M00);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 0), ivec2(0), ivec2(resolution_global) - 1)), y10, M10);
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 1), ivec2(0), ivec2(resolution_global) - 1)), y01, M01);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 1), ivec2(0), ivec2(resolution_global) - 1)), y11, M11);

    // 有效性 mask: M > 0 表示有效蓄水池
    float w00 = (M00 > 0.0) ? (1.0 - pf.x) * (1.0 - pf.y) : 0.0;
    float w10 = (M10 > 0.0) ? pf.x * (1.0 - pf.y) : 0.0;
    float w01 = (M01 > 0.0) ? (1.0 - pf.x) * pf.y : 0.0;
    float w11 = (M11 > 0.0) ? pf.x * pf.y : 0.0;

    float sumW = w00 + w10 + w01 + w11;
    if (sumW < 1e-8) return vec4(0.0);

    return (y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11) / sumW;
}

// ===========================================================================
// Binding 3/4 — SpecularBuffer (Reflect/Refract) pack/unpack
// ===========================================================================
// N=0: Current Geo+Dir — vec4(worldPos.xyz, oct(dir))
// N=1: Current Light   — vec4(packHalf(cR,cG), packHalf(cB,vprojDist), accum_weight, pad)
// N=2: History Geo+Dir — vec4(hist_pos.xyz, oct(hist_dir))
// N=3: History Light   — vec4(packHalf(hcR,hcG), packHalf(hcB,h_vproj), hist_weight, pad)

#define SPEC_N_GEO      0u
#define SPEC_N_LIGHT    1u
#define SPEC_N_HISTGEO  2u
#define SPEC_N_HISTLIGHT 3u

// --- N=0: Current Geometry + Direction ---
void writeSpecGeo(uint N, uvec2 xy, vec3 pos, vec3 dir) {
    vec4 v = vec4(pos, encodeNormal(dir));
    if (N == SPEC_N_GEO) {
        // use binding-specific buffer
    }
}
// 为 reflect/refract 分别提供（binding 不同）
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

// ===========================================================================
// 兼容旧接口的结构体（内存中解包表示，不影响存储）
// ===========================================================================

struct diffuseIlluminationData {
    AliceEncoding data;
    AliceEncoding data_swap;
    vec3 pos;
    float surfaceMask;   // was: lowp vec3 normal
    float histSurfaceMask; // was: lowp vec3 normal2
    float weight;
    float prev_weight;
};

struct DiffuseIlluminationWriteData {
    AliceEncoding data_swap;
    vec3 pos;
    float surfaceMask;   // was: lowp vec3 normal
    float weight;
};

struct vec3IlluminationData {
    vec3 data;
    vec3 data_swap;
    vec3 pos;
    vec3 normal;
    float weight;
    float prev_weight;
};

// ===========================================================================
// PackedLightSample — 保留给 colortex3/4 纹理 I/O（非 SSBO）
// ===========================================================================

struct PackedLightSample {
    vec4 data0;
    vec4 data1;
};

PackedLightSample packSpecularSample(vec3 pos, vec3 R, vec3 radiance,
    float roughness, float variance, float virtualProjDist, vec3 H) {
    PackedLightSample s;
    s.data0 = vec4(pos, encodeNormal(R));
    s.data1 = vec4(
        pack2HalfClamped(radiance.r, radiance.g),
        pack2HalfClamped(radiance.b, roughness),
        pack2HalfClamped(variance, virtualProjDist),
        encodeNormal(H)
    );
    return s;
}

void unpackSpecularSample(PackedLightSample s,
    out vec3 pos, out vec3 R, out vec3 radiance,
    out float roughness, out float variance, out float virtualProjDist, out vec3 H) {
    pos = s.data0.xyz;
    R = decodeNormal(s.data0.w);
    vec2 rg = unpackHalf2x16(floatBitsToUint(s.data1.x));
    vec2 br = unpackHalf2x16(floatBitsToUint(s.data1.y));
    vec2 vv = unpackHalf2x16(floatBitsToUint(s.data1.z));
    radiance  = vec3(rg.x, rg.y, br.x);
    roughness = br.y;
    variance  = vv.x;
    virtualProjDist = vv.y;
    H = decodeNormal(s.data1.w);
}

// ===========================================================================
// SpecularRT pack/unpack — 兼容 raytrace_rgen 和 temporal 接口
// ===========================================================================

struct SpecularRTWriteData {
    vec3 pos;
    vec3 R;
    float virtualProjDist;
    vec3 color;
};

SpecularRTWriteData packSpecularRT(vec3 pos, vec3 R, float virtualProjDist, vec3 color) {
    SpecularRTWriteData e;
    e.pos = pos;
    e.R = R;
    e.virtualProjDist = virtualProjDist;
    e.color = color;
    return e;
}

void unpackSpecularRT_Refl(uvec2 xy, out vec3 pos, out vec3 R, out vec3 color, out float virtualProjDist) {
    readReflGeo (xy, pos, R);
    float accumW;
    readReflLight(xy, color, virtualProjDist, accumW);
}

void unpackSpecularRT_Refr(uvec2 xy, out vec3 pos, out vec3 R, out vec3 color, out float virtualProjDist) {
    readRefrGeo (xy, pos, R);
    float accumW;
    readRefrLight(xy, color, virtualProjDist, accumW);
}

// ===========================================================================
// Diffuse load/fetch/write — 兼容 temporal_diffuse 和 composite
// ===========================================================================

// 从当前帧 RT 输出加载漫反射输入（temporal_diffuse 使用）
DiffuseIlluminationWriteData loadDiffuseInput(ivec2 p) {
    uvec2 xy = uvec2(p);
    DiffuseIlluminationWriteData t;
    AliceEncoding alice;
    float mask;
    readDiffuseLightRT(xy, alice, mask);
    t.data_swap = alice;
    readDiffuseGeo(xy, t.pos, mask);
    t.surfaceMask = mask;
    t.weight = 1.0;
    return t;
}

// 读取历史漫反射光照（temporal_diffuse 使用）
diffuseIlluminationData fetchDiffuse(ivec2 p) {
    uvec2 xy = uvec2(p);
    diffuseIlluminationData tmp;

    // swap = 当前帧累积结果（N=4）
    AliceEncoding alice;
    float weight;
    readDiffuseSwap(xy, alice, weight);
    tmp.data_swap = alice;
    tmp.weight = weight;

#ifndef DIFFUSE_BUFFER_MIN2
    // hist = 上一帧历史（N=2）
    readDiffuseHist(xy, alice, weight);
    tmp.data = alice;
    tmp.prev_weight = weight;

    // 历史几何（N=3）— surfaceMask replaces oct(normal)
    float mask;
    readDiffuseHistGeo(xy, tmp.pos, mask);
    tmp.histSurfaceMask = mask;
#endif
    return tmp;
}

diffuseIlluminationData blendDiffuse(diffuseIlluminationData A, diffuseIlluminationData B, float x) {
    diffuseIlluminationData t;
    t.data_swap = mix_alice(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
#ifndef DIFFUSE_BUFFER_MIN2
    t.data = mix_alice(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.prev_weight = (B.prev_weight - A.prev_weight) * x + A.prev_weight;
#endif
    return t;
}

diffuseIlluminationData sampleDiffuse(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    diffuseIlluminationData A = fetchDiffuse(p1);
    diffuseIlluminationData B = fetchDiffuse(p1 + ivec2(1, 0));
    diffuseIlluminationData C = fetchDiffuse(p1 + ivec2(0, 1));
    diffuseIlluminationData D = fetchDiffuse(p1 + ivec2(1, 1));
    return blendDiffuse(blendDiffuse(A, B, p2.x), blendDiffuse(C, D, p2.x), p2.y);
}

vec3 sampleDiffusePos(vec2 p) {
    uvec2 xy = uvec2(ivec2(floor(p) + round(fract(p))));
    vec3 pos; float mask;
    readDiffuseHistGeo(xy, pos, mask);
    return pos;
}

void WriteDiffuse(diffuseIlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);

    // Always write swap (N=4)
    writeDiffuseSwap(xy, data.data_swap, data.weight);

#if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
    // Full write: also update hist (N=2) + hist geometry (N=3)
    writeDiffuseHist(xy, data.data, data.prev_weight);
    writeDiffuseHistGeo(xy, data.pos, data.histSurfaceMask);
#endif
}

// ===========================================================================
// Diffuse prev-frame (ray0.rgen guiding)
// ===========================================================================

#if defined(PREV_DIFFUSE_BUFFER)

DiffuseIlluminationWriteData fetchPrevDiffuse(ivec2 p) {
    uvec2 xy = uvec2(p);
    DiffuseIlluminationWriteData t;

    // swap = 上一帧最终降噪结果（ray0.rgen 用其引导采样）
    AliceEncoding alice;
    float weight;
    readDiffuseSwap(xy, alice, weight);
    t.data_swap = alice;
    t.weight = weight;

    // 从当前几何读取位置+mask（它们不会被 ray0.rgen 改变，因为 ray0 只写 N=0,1）
    float mask;
    readDiffuseGeo(xy, t.pos, mask);
    t.surfaceMask = mask;

    return t;
}

DiffuseIlluminationWriteData blendPrevDiffuse(DiffuseIlluminationWriteData A, DiffuseIlluminationWriteData B, float x) {
    DiffuseIlluminationWriteData t;
    t.data_swap = mix_alice(A.data_swap, B.data_swap, x);
    t.pos = mix(A.pos, B.pos, x);
    t.weight = mix(A.weight, B.weight, x);
    return t;
}

DiffuseIlluminationWriteData samplePrevDiffuse(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    DiffuseIlluminationWriteData A = fetchPrevDiffuse(p1);
    DiffuseIlluminationWriteData B = fetchPrevDiffuse(p1 + ivec2(1, 0));
    DiffuseIlluminationWriteData C = fetchPrevDiffuse(p1 + ivec2(0, 1));
    DiffuseIlluminationWriteData D = fetchPrevDiffuse(p1 + ivec2(1, 1));
    return blendPrevDiffuse(blendPrevDiffuse(A, B, p2.x), blendPrevDiffuse(C, D, p2.x), p2.y);
}

void WritePrevDiffuse(DiffuseIlluminationWriteData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    writeDiffuseSwap(xy, data.data_swap, data.weight);
}

#endif

// ===========================================================================
// Reflect fetch/write — 兼容 temporal_reflect + composite
// ===========================================================================

#if defined(REFLECT_BUFFER) || defined(REFLECT_BUFFER_MIN) || defined(REFLECT_BUFFER_MIN2)

vec3IlluminationData fetchReflect(ivec2 p) {
    uvec2 xy = uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1));
    vec3IlluminationData tmp;

    // 当前帧累积（N=1）
    float vproj;
    readReflLight(xy, tmp.data_swap, vproj, tmp.weight);

#ifndef REFLECT_BUFFER_MIN2
    // 历史（N=3 + N=2）
    readReflHistLight(xy, tmp.data, vproj, tmp.prev_weight);
    vec3 R;
    readReflHistGeo(xy, tmp.pos, R);
    tmp.normal = R * vproj;
#endif
    return tmp;
}

vec3IlluminationData blendReflect(vec3IlluminationData A, vec3IlluminationData B, float x) {
    vec3IlluminationData t;
    t.data_swap = mix(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
#ifndef REFLECT_BUFFER_MIN2
    t.prev_weight = (B.prev_weight - A.prev_weight) * x + A.prev_weight;
    t.data = mix(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = mix(A.normal, B.normal, x);
#endif
    return t;
}

bool fetchReflectHistoryGeometry(ivec2 p, out vec3 pos, out vec3 normal) {
    uvec2 xy = uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1));
    float dist;
    readGeo0(GEO_N_GEO, xy, pos, dist);
    if (dist < -0.5) return false;
    vec3 R; float vproj;
    readReflHistGeo(xy, pos, R);
    readReflHistLight(xy, normal, vproj, dist); // dist reused as discard
    normal = R * vproj;
    return true;
}

vec3IlluminationData sampleReflect(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    vec3IlluminationData A = fetchReflect(p1);
    vec3IlluminationData B = fetchReflect(p1 + ivec2(1, 0));
    vec3IlluminationData C = fetchReflect(p1 + ivec2(0, 1));
    vec3IlluminationData D = fetchReflect(p1 + ivec2(1, 1));
    return blendReflect(blendReflect(A, B, p2.x), blendReflect(C, D, p2.x), p2.y);
}

void WriteReflect(vec3IlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    // 读取 vprojDist 保留，只更新 color + weight
    vec3 color; float vproj, accumW;
    readReflLight(xy, color, vproj, accumW);
    writeReflLight(xy, data.data_swap, vproj, data.weight);
}

void WriteReflectHistory(vec3 preDenoiseColor, float prevWeight, vec3 pos, vec3 R, float virtualProjDist, ivec2 p) {
    uvec2 xy = uvec2(p);
    writeReflHistGeo(xy, pos, R);
    writeReflHistLight(xy, preDenoiseColor, virtualProjDist, prevWeight);
}
#endif

// ===========================================================================
// Refract fetch/write — 兼容 temporal_refract + composite
// ===========================================================================

#if defined(REFRACT_BUFFER) || defined(REFRACT_BUFFER_MIN) || defined(REFRACT_BUFFER_MIN2)

vec3IlluminationData fetchRefract(ivec2 p) {
    uvec2 xy = uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1));
    vec3IlluminationData tmp;

    float vproj;
    readRefrLight(xy, tmp.data_swap, vproj, tmp.weight);

#ifndef REFRACT_BUFFER_MIN2
    readRefrHistLight(xy, tmp.data, vproj, tmp.prev_weight);
    vec3 T;
    readRefrHistGeo(xy, tmp.pos, T);
    tmp.normal = T * vproj;
#endif
    return tmp;
}

vec3IlluminationData blendRefract(vec3IlluminationData A, vec3IlluminationData B, float x) {
    vec3IlluminationData t;
    t.data_swap = mix(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
#ifndef REFRACT_BUFFER_MIN2
    t.data = mix(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = mix(A.normal, B.normal, x);
#endif
    return t;
}

vec3IlluminationData sampleRefract(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    vec3IlluminationData A = fetchRefract(p1);
    vec3IlluminationData B = fetchRefract(p1 + ivec2(1, 0));
    vec3IlluminationData C = fetchRefract(p1 + ivec2(0, 1));
    vec3IlluminationData D = fetchRefract(p1 + ivec2(1, 1));
    return blendRefract(blendRefract(A, B, p2.x), blendRefract(C, D, p2.x), p2.y);
}

void WriteRefract(vec3IlluminationData data, ivec2 p) {
    uvec2 xy = uvec2(p);
    vec3 color; float vproj, accumW;
    readRefrLight(xy, color, vproj, accumW);
    writeRefrLight(xy, data.data_swap, vproj, data.weight);
}

void WriteRefractHistory(vec3 preDenoiseColor, float prevWeight, vec3 pos, vec3 R, float virtualProjDist, ivec2 p) {
    uvec2 xy = uvec2(p);
    writeRefrHistGeo(xy, pos, R);
    writeRefrHistLight(xy, preDenoiseColor, virtualProjDist, prevWeight);
}
#endif
