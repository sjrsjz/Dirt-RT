
#include "/lib/buffers/frame_data.glsl"
#include "/lib/lighting/alice.glsl"
uint getIdx(uvec2 xy) {
    //return xy.y * 1024u + clamp(xy.x, 0, 1023u);
    return xy.y * resolution_global.x + clamp(xy.x, 0, resolution_global.x - 1);
}
struct bufferData {
    vec3 macroNormal;
    vec3 light;
    vec3 albedo;
    vec3 albedo2;
    float distance;
    vec3 absorption;
    vec3 emission;
    vec3 rd;
    int illumiantionType;
    float reflectWeight;
    float refractWeight;
    float last_rd_dot_n;
    float roughness;
};

layout(std430, set = 3, binding = 0) buffer DenoiseBuffer {
    bufferData data[];
} denoiseBuffer;

// 辅助函数：安全的非零符号函数
float sign_not_zero(float v) {
    return (v >= 0.0 ? 1.0 : -1.0);
}

vec2 sign_not_zero(vec2 v) {
    return vec2(sign_not_zero(v.x), sign_not_zero(v.y));
}

// 打包两个 half 为单个 float
float pack2Half(float a, float b) {
    return uintBitsToFloat(packHalf2x16(vec2(a, b)));
}

// 解包
void unpack2Half(float packed_, out float a, out float b) {
    vec2 v = unpackHalf2x16(floatBitsToUint(packed_));
    a = v.x;
    b = v.y;
}

// 编码：vec3 -> float
float encodeNormal(vec3 n) {
    // 确保输入是单位向量（非零）
    n = normalize(n);
    // 八面体映射  (Cigolle, 2014 等标准形式)
    vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * sign_not_zero(p);
    }
    // 映射到 [0, 1] 范围以备打包
    p = p * 0.5 + 0.5;
    // 打包为 32 位 uint，再按位解释为 float
    // packed 是保留关键字，因此使用 packed_ 作为变量名
    uint packed_ = packUnorm2x16(p);
    return uintBitsToFloat(packed_);
}

// 解码：float -> vec3
vec3 decodeNormal(float f) {
    uint packed_ = floatBitsToUint(f);
    vec2 p = unpackUnorm2x16(packed_);
    // 从 [0,1] 映射回 [-1,1]
    p = p * 2.0 - 1.0;
    // 逆八面体映射
    vec3 n = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * sign_not_zero(n.xy);
    }
    return normalize(n);
}

// ===========================================================================
// ALICE 光照编码与辐照度重建 (NaCg-Safe, O(1) 闭型逼近)
// ===========================================================================
// SH.shY = vec4(v, ω) — 即 ALICE 线性嵌入表示，与 alice_encode 输出兼容
// SH.CoCg = vec2(Co, Cg) — 色度 (YCoCg 空间)
//
// 辐照度解码使用 ALICE 最大熵半球余弦投影解析逼近，全域误差 < 0.4%

struct SH {
    mediump vec4 shY; // ALICE 嵌入: xyz = 方向向量 v, w = 总能量 ω = |v| + I
    mediump vec2 CoCg; // (Co, Cg)
};

// ---------------------------------------------------------------------------
// 编解码与投影核心接口
// ---------------------------------------------------------------------------

SH irradiance_to_SH(vec3 color, vec3 dir)
{
    SH result;

    float Y = dot(color, vec3(0.2126, 0.7152, 0.0722));

    float Co = 0.5 * color.r - 0.5 * color.b;
    float Cg = -0.25 * color.r + 0.5 * color.g - 0.25 * color.b;

    result.CoCg = vec2(Co, Cg);
    // ALICE 编码: v = dir*Y, ω = |v| + 0 = Y (单样本 I=0)
    result.shY = vec4(dir * Y, Y);

    return result;
}

// ALICE 辐照度投影 (替代原 SG 模型)
// sh.shY 即为 ALICE 编码 vec4(v, ω)
// 返回余弦加权漫反射辐照度 RGB
vec3 project_SH_irradiance(SH sh, vec3 N)
{
    float total_omega = sh.shY.w;
    
    float irradiance = alice_irradiance(sh.shY, N);

    float attenuation = (total_omega > 1e-10) ? (irradiance / total_omega) : 0.0;

    float Co = sh.CoCg.x * attenuation;
    float Cg = sh.CoCg.y * attenuation;

    float B = irradiance - 1.1404 * Co - 1.4304 * Cg;
    float R = B + 2.0 * Co;
    float G = irradiance - 0.1404 * Co + 0.5696 * Cg;
    return max(vec3(R, G, B), vec3(0.0));
}

// ---------------------------------------------------------------------------
// colortex5 双对偶向量 (θ, β) 打包/解包 — 用于 Jeffreys 散度计算
// ---------------------------------------------------------------------------
// colortex5 格式: RGBA32F — 直接存储 vec4(theta.xyz, beta)
//   θ = 自然参数空间方向分量 (3D 向量)
//   β = 自然参数空间能量分量 (标量)

vec4 packDualVector(vec3 dual_theta, float dual_beta) {
    return vec4(dual_theta, dual_beta);
}

vec4 packDualVectorFromEncoded(vec4 aliceEncoded) {
    vec4 tb = alice_theta_beta(aliceEncoded);
    return tb; // vec4(theta.xyz, beta)
}

void unpackDualVector(vec4 packed_, out vec3 dual_theta, out float dual_beta) {
    dual_theta = packed_.xyz;
    dual_beta = packed_.w;
}

// ---------------------------------------------------------------------------
// 基础混合原语
// ---------------------------------------------------------------------------
SH mix_SH(SH a, SH b, float s)
{
    SH result;
    result.shY = mix(a.shY, b.shY, s);
    result.CoCg = mix(a.CoCg, b.CoCg, s);
    return result;
}

SH init_SH()
{
    SH result;
    result.shY = vec4(0.0);
    result.CoCg = vec2(0.0);
    return result;
}

SH scaleSH(SH A, float x) {
    SH tmp;
    tmp.CoCg = A.CoCg * x;
    tmp.shY = A.shY * x;
    return tmp;
}

void accumulate_SH(inout SH accum, SH b, float scale)
{
    accum.shY += b.shY * scale;
    accum.CoCg += b.CoCg * scale;
}

// 将 SH 压缩为 3 个 float
vec3 packSH(SH sh) {
    // 注意：shY 和 CoCg 可能超出 half 范围（但通常不会），必要时 clamp
    float s0 = uintBitsToFloat(packHalf2x16(vec2(sh.shY.x, sh.shY.y)));
    float s1 = uintBitsToFloat(packHalf2x16(vec2(sh.shY.z, sh.shY.w)));
    float s2 = uintBitsToFloat(packHalf2x16(vec2(sh.CoCg.x, sh.CoCg.y)));
    return vec3(s0, s1, s2);
}

SH unpackSH(float s0, float s1, float s2) {
    SH sh;
    vec2 v0 = unpackHalf2x16(floatBitsToUint(s0));
    vec2 v1 = unpackHalf2x16(floatBitsToUint(s1));
    vec2 v2 = unpackHalf2x16(floatBitsToUint(s2));
    sh.shY = vec4(v0.x, v0.y, v1.x, v1.y);
    sh.CoCg = v2;
    return sh;
}

struct PackedLightSample {
    vec4 data0; // (pos.xyz, encoded_normal)
    vec4 data1; // (encoded_shY.xy, encoded_shY.zw, encoded_CoCg.xy)
};


PackedLightSample packSpecularSample(vec3 pos, vec3 normal, vec3 radiance, float weight, float roughness) {
    PackedLightSample sample_data;
    sample_data.data0 = vec4(pos, encodeNormal(normal));
    sample_data.data1 = vec4(radiance, pack2Half(weight, roughness));
    return sample_data;
}

void unpackSpecularSample(PackedLightSample sample_data, out vec3 pos, out vec3 normal, out vec3 radiance, out float weight, out float roughness) {
    pos = sample_data.data0.xyz;
    normal = decodeNormal(sample_data.data0.w);
    radiance = sample_data.data1.xyz;
    unpack2Half(sample_data.data1.w, weight, roughness);
}

struct diffuseIllumiantionData {
    SH data;
    SH data_swap;
    vec3 pos;
    lowp vec3 normal;
    lowp vec3 normal2;
    mediump float weight;
    mediump float variance;
    mediump float prev_weight;
    mediump float prev_variance;
};

struct diffuseIllumiantionBufferData {
    SH data_swap;
    vec3 pos;
    lowp vec3 normal;
    lowp vec3 normal2;
};
struct diffuseIllumiantionBufferDataW {
    SH data_swap;
    vec3 pos;
    lowp vec3 normal;
    lowp vec3 normal2;
    mediump float weight;
};
layout(std430, set = 3, binding = 2) buffer DiffuseIllumiantionDataBuffer {
    diffuseIllumiantionBufferData data[];
} diffuseIllumiantionBuffer;

// Half-packed temporal history — replaces 4 custom images + old binding 6.
// Layout matches the old RGBA32F texture packing bit-for-bit.
// 14 floats = 56 bytes, all float alignment (no padding).
struct DiffuseHistoryElement {
    // Previous frame (matches old shY texture):
    //   packHalf2x16(data.shY.xy), packHalf2x16(data.shY.zw),
    //   packHalf2x16(data.CoCg), packHalf2x16(prev_weight, prev_variance)
    float prev_shY_xy, prev_shY_zw, prev_CoCg, prev_w_v;
    // Current frame (matches old shY_swap texture):
    float swap_shY_xy, swap_shY_zw, swap_CoCg, swap_w_v; // w_v: weight, variance
    // Position (matches old lpos texture, without the .w=0 padding):
    float px, py, pz;
    // Normal (matches old lnormal texture, without the .w=0 padding):
    float nx, ny, nz;
};

layout(std430, set = 3, binding = 6) buffer DiffuseHistoryBuffer {
    DiffuseHistoryElement data[];
} diffuseHistory;

struct vec3IllumiantionData {
    mediump vec3 data;
    mediump vec3 data_swap;
    vec3 pos;
    mediump vec3 normal;
    mediump float weight;
    mediump float prev_weight;
    mediump float mixWeight;
};

layout(std430, set = 3, binding = 3) buffer ReflectIllumiantionDataBuffer {
    vec3IllumiantionData data[];
} reflectIllumiantionBuffer;

layout(std430, set = 3, binding = 4) buffer RefractIllumiantionDataBuffer {
    vec3IllumiantionData data[];
} refractIllumiantionBuffer;

#if defined(PREV_DIFFUSE_BUFFER)

// Read previous frame's accumulated SH for ray guiding (ray0.rgen).
// Only data_swap.shY fields are used by the caller; the rest are filled with
// best-effort values from the history SSBO.
diffuseIllumiantionBufferDataW fetchPrevDiffuse(ivec2 p) {
    DiffuseHistoryElement e = diffuseHistory.data[getIdx(p)];
    diffuseIllumiantionBufferDataW t;
    mediump vec2 shY_xy = unpackHalf2x16(floatBitsToUint(e.swap_shY_xy));
    mediump vec2 shY_zw = unpackHalf2x16(floatBitsToUint(e.swap_shY_zw));
    t.data_swap.shY = clamp(vec4(shY_xy, shY_zw), vec4(-10000), vec4(10000));
    t.data_swap.CoCg = unpackHalf2x16(floatBitsToUint(e.swap_CoCg));
    t.pos = vec3(e.px, e.py, e.pz);
    t.normal = vec3(e.nx, e.ny, e.nz);
    t.normal2 = vec3(0.0);
    mediump vec2 w_v = unpackHalf2x16(floatBitsToUint(e.swap_w_v));
    t.weight = w_v.x;
    return t;
}

diffuseIllumiantionBufferDataW blendPrevDiffuse(diffuseIllumiantionBufferDataW A, diffuseIllumiantionBufferDataW B, float x) {
    diffuseIllumiantionBufferDataW t;
    t.data_swap = mix_SH(A.data_swap, B.data_swap, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = normalize(mix(A.normal, B.normal, x));
    t.weight = mix(A.weight, B.weight, x);
    return t;
}

diffuseIllumiantionBufferDataW samplePrevDiffuse(vec2 p) {
    ivec2 p1 = ivec2(p);
    vec2 p2 = fract(p);
    diffuseIllumiantionBufferDataW A = fetchPrevDiffuse(p1);
    diffuseIllumiantionBufferDataW B = fetchPrevDiffuse(p1 + ivec2(1, 0));
    diffuseIllumiantionBufferDataW C = fetchPrevDiffuse(p1 + ivec2(0, 1));
    diffuseIllumiantionBufferDataW D = fetchPrevDiffuse(p1 + ivec2(1, 1));
    return blendPrevDiffuse(blendPrevDiffuse(A, B, p2.x), blendPrevDiffuse(C, D, p2.x), p2.y);
}

void WritePrevDiffuse(diffuseIllumiantionBufferDataW data, ivec2 p) {
    uint idx = getIdx(p);
    // Only update the swap half (data_swap goes to swap fields for ray guiding)
    diffuseHistory.data[idx].swap_shY_xy = uintBitsToFloat(packHalf2x16(data.data_swap.shY.xy));
    diffuseHistory.data[idx].swap_shY_zw = uintBitsToFloat(packHalf2x16(data.data_swap.shY.zw));
    diffuseHistory.data[idx].swap_CoCg   = uintBitsToFloat(packHalf2x16(data.data_swap.CoCg));
    // Preserve variance in w_v; only update weight
    mediump vec2 old_wv = unpackHalf2x16(floatBitsToUint(diffuseHistory.data[idx].swap_w_v));
    diffuseHistory.data[idx].swap_w_v = uintBitsToFloat(packHalf2x16(vec2(data.weight, old_wv.y)));
}

#endif

#if defined(DIFFUSE_BUFFER) || defined(DIFFUSE_BUFFER_MIN) || defined(DIFFUSE_BUFFER_MIN2)

// All diffuse temporal history now lives in diffuseHistory SSBO (binding 6).
// No more custom images — texelFetch/imageStore replaced with SSBO reads/writes.

diffuseIllumiantionData fetchDiffuse(ivec2 p) {
    diffuseIllumiantionData tmp;
    DiffuseHistoryElement e = diffuseHistory.data[getIdx(p)];

    // Unpack current frame (swap)
    mediump vec2 shY_xy = unpackHalf2x16(floatBitsToUint(e.swap_shY_xy));
    mediump vec2 shY_zw = unpackHalf2x16(floatBitsToUint(e.swap_shY_zw));
    tmp.data_swap.shY = clamp(vec4(shY_xy, shY_zw), vec4(-10000), vec4(10000));
    tmp.data_swap.CoCg = unpackHalf2x16(floatBitsToUint(e.swap_CoCg));
    mediump vec2 w_v = unpackHalf2x16(floatBitsToUint(e.swap_w_v));
    tmp.weight = w_v.x;
    tmp.variance = w_v.y;

    #ifndef DIFFUSE_BUFFER_MIN2
    // Unpack previous frame (prev)
    shY_xy = unpackHalf2x16(floatBitsToUint(e.prev_shY_xy));
    shY_zw = unpackHalf2x16(floatBitsToUint(e.prev_shY_zw));
    tmp.data.shY = clamp(vec4(shY_xy, shY_zw), vec4(-10000), vec4(10000));
    tmp.data.CoCg = unpackHalf2x16(floatBitsToUint(e.prev_CoCg));
    w_v = unpackHalf2x16(floatBitsToUint(e.prev_w_v));
    tmp.prev_weight = w_v.x;
    tmp.prev_variance = w_v.y;

    tmp.pos = vec3(e.px, e.py, e.pz);
    tmp.normal = vec3(e.nx, e.ny, e.nz);
    #endif
    return tmp;
}

diffuseIllumiantionData blendDiffuse(diffuseIllumiantionData A, diffuseIllumiantionData B, float x) {
    diffuseIllumiantionData t;
    t.data_swap = mix_SH(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
    t.variance = (B.variance - A.variance) * x + A.variance;
    #ifndef DIFFUSE_BUFFER_MIN2
    t.data = mix_SH(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = mix(A.normal, B.normal, x);
    t.prev_weight = (B.prev_weight - A.prev_weight) * x + A.prev_weight;
    t.prev_variance = (B.prev_variance - A.prev_variance) * x + A.prev_variance;
    #endif
    return t;
}

diffuseIllumiantionData sampleDiffuse(vec2 p) {

    //p*=textureSize(diffuseIllumiantionData_CoCg_swap_Sampler,0);
    ivec2 p1 = ivec2(p);

    vec2 p2 = fract(p);
    diffuseIllumiantionData A = fetchDiffuse(p1);
    diffuseIllumiantionData B = fetchDiffuse(p1 + ivec2(1, 0));
    diffuseIllumiantionData C = fetchDiffuse(p1 + ivec2(0, 1));
    diffuseIllumiantionData D = fetchDiffuse(p1 + ivec2(1, 1));
    diffuseIllumiantionData data = blendDiffuse(blendDiffuse(A, B, p2.x), blendDiffuse(C, D, p2.x), p2.y);
    #ifndef DIFFUSE_BUFFER_MIN2
    data.normal = normalize(data.normal);
    #endif
    return data;
}
vec3 sampleDiffusePos(vec2 p) {
    DiffuseHistoryElement e = diffuseHistory.data[getIdx(ivec2(floor(p) + round(fract(p))))];
    return vec3(e.px, e.py, e.pz);
}
void WriteDiffuse(diffuseIllumiantionData data, ivec2 p) {
    uint idx = getIdx(p);

    // Always write swap (current frame) — half-packing, bit-identical to old texture path
    data.weight = clamp(data.weight, 0.0, 65504);
    data.variance = clamp(data.variance, 0.0, 65504);

    diffuseHistory.data[idx].swap_shY_xy = uintBitsToFloat(packHalf2x16(data.data_swap.shY.xy));
    diffuseHistory.data[idx].swap_shY_zw = uintBitsToFloat(packHalf2x16(data.data_swap.shY.zw));
    diffuseHistory.data[idx].swap_CoCg   = uintBitsToFloat(packHalf2x16(data.data_swap.CoCg));
    diffuseHistory.data[idx].swap_w_v    = uintBitsToFloat(packHalf2x16(vec2(data.weight, data.variance)));

    #if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
    // Full write: also update prev (history) and geometry
    data.data.shY = clamp(data.data.shY, vec4(-65504), vec4(65504));
    data.data.CoCg = clamp(data.data.CoCg, vec2(-65504), vec2(65504));
    data.prev_weight = clamp(data.prev_weight, 0.0, 65504);
    data.prev_variance = clamp(data.prev_variance, 0.0, 65504);

    diffuseHistory.data[idx].prev_shY_xy = uintBitsToFloat(packHalf2x16(data.data.shY.xy));
    diffuseHistory.data[idx].prev_shY_zw = uintBitsToFloat(packHalf2x16(data.data.shY.zw));
    diffuseHistory.data[idx].prev_CoCg   = uintBitsToFloat(packHalf2x16(data.data.CoCg));
    diffuseHistory.data[idx].prev_w_v    = uintBitsToFloat(packHalf2x16(vec2(data.prev_weight, data.prev_variance)));

    diffuseHistory.data[idx].px = data.pos.x;
    diffuseHistory.data[idx].py = data.pos.y;
    diffuseHistory.data[idx].pz = data.pos.z;
    diffuseHistory.data[idx].nx = data.normal.x;
    diffuseHistory.data[idx].ny = data.normal.y;
    diffuseHistory.data[idx].nz = data.normal.z;
    #endif
}
#endif

#if defined(REFLECT_BUFFER) || defined(REFLECT_BUFFER_MIN) || defined(REFLECT_BUFFER_MIN2)

layout(rgba32f) uniform image2D reflectIllumiantionData_swap_color;
uniform sampler2D reflectIllumiantionData_color_Sampler;
uniform sampler2D reflectIllumiantionData_color_swap_Sampler;
uniform sampler2D reflectIllumiantionData_lnormal_Sampler;
uniform sampler2D reflectIllumiantionData_lpos_Sampler;
#if !defined(REFLECT_BUFFER_MIN) && !defined(REFLECT_BUFFER_MIN2)
layout(rgba32f) uniform image2D reflectIllumiantionData_color;
layout(rgba32f) uniform image2D reflectIllumiantionData_lnormal;
layout(rgba32f) uniform image2D reflectIllumiantionData_lpos;
#endif

// vec3IllumiantionData sampleReflect(vec2 p) {
//     vec3IllumiantionData tmp;
//     vec4 tmp4 = texture(reflectIllumiantionData_color_swap_Sampler, p);
//     tmp.data_swap = tmp4.xyz;
//     #ifndef REFLECT_BUFFER_MIN
//     tmp4 = texture(reflectIllumiantionData_color_Sampler, p);
//     tmp.data = tmp4.xyz;
//     tmp.weight = tmp4.w;
//     tmp.normal = texture(reflectIllumiantionData_lnormal_Sampler, p).xyz;
//     tmp.pos = texture(reflectIllumiantionData_lpos_Sampler, p).xyz;
//     #endif
//     return tmp;
// }

vec3IllumiantionData fetchReflect(ivec2 p) {
    vec3IllumiantionData tmp;
    vec4 tmp4 = texelFetch(reflectIllumiantionData_color_swap_Sampler, p, 0);
    tmp.data_swap = tmp4.xyz;
    vec2 w_mw = unpackHalf2x16(floatBitsToUint(tmp4.w)); // weight, mixWeight
    tmp.weight = w_mw.x;
    tmp.mixWeight = w_mw.y;
    #ifndef REFLECT_BUFFER_MIN2
    tmp4 = texelFetch(reflectIllumiantionData_color_Sampler, p, 0);
    tmp.data = tmp4.xyz;
    w_mw = unpackHalf2x16(floatBitsToUint(tmp4.w)); // prev_weight, 0
    tmp.prev_weight = w_mw.x;
    tmp.normal = texelFetch(reflectIllumiantionData_lnormal_Sampler, p, 0).xyz;
    tmp.pos = texelFetch(reflectIllumiantionData_lpos_Sampler, p, 0).xyz;
    #endif
    return tmp;
}

vec3IllumiantionData blendReflect(vec3IllumiantionData A, vec3IllumiantionData B, float x) {
    vec3IllumiantionData t;
    t.data_swap = mix(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;
    #ifndef REFLECT_BUFFER_MIN2
    t.prev_weight = (B.prev_weight - A.prev_weight) * x + A.prev_weight;
    t.data = mix(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = mix(A.normal, B.normal, x);
    t.mixWeight = (B.mixWeight - A.mixWeight) * x + A.mixWeight;
    #endif
    return t;
}

vec3IllumiantionData sampleReflect(vec2 p) {

    //p*=textureSize(diffuseIllumiantionData_CoCg_swap_Sampler,0);
    //p-=0.25;
    ivec2 p1 = ivec2(p);

    vec2 p2 = fract(p);
    vec3IllumiantionData A = fetchReflect(p1);
    vec3IllumiantionData B = fetchReflect(p1 + ivec2(1, 0));
    vec3IllumiantionData C = fetchReflect(p1 + ivec2(0, 1));
    vec3IllumiantionData D = fetchReflect(p1 + ivec2(1, 1));
    return blendReflect(blendReflect(A, B, p2.x), blendReflect(C, D, p2.x), p2.y);
}

void WriteReflect(vec3IllumiantionData data, ivec2 p) {
    float packed_w_mw = uintBitsToFloat(packHalf2x16(vec2(data.weight, data.mixWeight)));
    imageStore(reflectIllumiantionData_swap_color, p, vec4(data.data_swap, packed_w_mw));
    #if !defined(REFLECT_BUFFER_MIN) && !defined(REFLECT_BUFFER_MIN2)
    packed_w_mw = uintBitsToFloat(packHalf2x16(vec2(data.prev_weight, 0)));
    imageStore(reflectIllumiantionData_color, p, vec4(data.data, packed_w_mw));
    imageStore(reflectIllumiantionData_lpos, p, vec4(data.pos, 0));
    imageStore(reflectIllumiantionData_lnormal, p, vec4(data.normal, 0));
    #endif
}
#endif

#if defined(REFRACT_BUFFER) || defined(REFRACT_BUFFER_MIN) || defined(REFRACT_BUFFER_MIN2)

layout(rgba32f) uniform image2D refractIllumiantionData_swap_color;
uniform sampler2D refractIllumiantionData_color_Sampler;
uniform sampler2D refractIllumiantionData_color_swap_Sampler;
uniform sampler2D refractIllumiantionData_lnormal_Sampler;
uniform sampler2D refractIllumiantionData_lpos_Sampler;
#if !defined(REFRACT_BUFFER_MIN) && !defined(REFRACT_BUFFER_MIN2)
layout(rgba32f) uniform image2D refractIllumiantionData_color;
layout(rgba32f) uniform image2D refractIllumiantionData_lnormal;
layout(rgba32f) uniform image2D refractIllumiantionData_lpos;

#endif

// vec3IllumiantionData sampleRefract(vec2 p) {
//     vec3IllumiantionData tmp;
//     vec4 tmp4 = texture(refractIllumiantionData_color_swap_Sampler, p);
//     tmp.data_swap = tmp4.xyz;
//     #ifndef REFRACT_BUFFER_MIN
//     tmp4 = texture(refractIllumiantionData_color_Sampler, p);
//     tmp.data = tmp4.xyz;
//     tmp.weight = tmp4.w;

//     tmp.normal = texture(refractIllumiantionData_lnormal_Sampler, p).xyz;
//     tmp.pos = texture(refractIllumiantionData_lpos_Sampler, p).xyz;
//     #endif
//     return tmp;
// }

vec3IllumiantionData fetchRefract(ivec2 p) {
    vec3IllumiantionData tmp;
    vec4 tmp4 = texelFetch(refractIllumiantionData_color_swap_Sampler, p, 0);
    tmp.data_swap = tmp4.xyz;
    tmp.weight = tmp4.w;
    #ifndef REFRACT_BUFFER_MIN2
    tmp4 = texelFetch(refractIllumiantionData_color_Sampler, p, 0);
    tmp.data = tmp4.xyz;
    tmp.mixWeight = tmp4.w;
    tmp.normal = texelFetch(refractIllumiantionData_lnormal_Sampler, p, 0).xyz;
    tmp.pos = texelFetch(refractIllumiantionData_lpos_Sampler, p, 0).xyz;
    #endif
    return tmp;
}

vec3IllumiantionData blendRefract(vec3IllumiantionData A, vec3IllumiantionData B, float x) {
    vec3IllumiantionData t;
    t.data_swap = mix(A.data_swap, B.data_swap, x);
    t.weight = (B.weight - A.weight) * x + A.weight;

    #ifndef REFRACT_BUFFER_MIN2
    t.data = mix(A.data, B.data, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = mix(A.normal, B.normal, x);
    t.mixWeight = (B.mixWeight - A.mixWeight) * x + A.mixWeight;
    #endif
    return t;
}

vec3IllumiantionData sampleRefract(vec2 p) {

    //p*=textureSize(diffuseIllumiantionData_CoCg_swap_Sampler,0);
    //p-=0.375;
    ivec2 p1 = ivec2(p);

    vec2 p2 = fract(p);
    vec3IllumiantionData A = fetchRefract(p1);
    vec3IllumiantionData B = fetchRefract(p1 + ivec2(1, 0));
    vec3IllumiantionData C = fetchRefract(p1 + ivec2(0, 1));
    vec3IllumiantionData D = fetchRefract(p1 + ivec2(1, 1));
    return blendRefract(blendRefract(A, B, p2.x), blendRefract(C, D, p2.x), p2.y);
}

void WriteRefract(vec3IllumiantionData data, ivec2 p) {
    imageStore(refractIllumiantionData_swap_color, p, vec4(data.data_swap, data.weight));
    #if !defined(REFRACT_BUFFER_MIN) && !defined(REFRACT_BUFFER_MIN2)
    imageStore(refractIllumiantionData_color, p, vec4(data.data, data.mixWeight));
    imageStore(refractIllumiantionData_lpos, p, vec4(data.pos, 0));
    imageStore(refractIllumiantionData_lnormal, p, vec4(data.normal, 0));
    #endif
}
#endif
