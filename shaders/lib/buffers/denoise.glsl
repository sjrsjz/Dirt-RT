
#include "/lib/buffers/frame_data.glsl"
#include "/lib/lighting/alice.glsl"
uint getIndex(uvec2 xy) {
    //return xy.y * 1024u + clamp(xy.x, 0, 1023u);
    return xy.y * resolution_global.x + clamp(xy.x, 0, resolution_global.x - 1);
}
// 逐 lobe 材质乘数 + G-Buffer: 128B/元素, scalar 填入 vec3 尾部 4B padding.
// specularAlbedo / diffuseAlbedo / transmissionAlbedo 由 ray0.rgen 写入,
// 供 composite_lighting.glsl 合成时与各自 denoised 光照相乘 (每个 buffer 存材质无关光场信号)。
struct bufferData {
    vec3 macroNormal;        // offset   0 (12B)
    float distance;          // offset  12 (4B)
    vec3 light;              // offset  16 (12B)
    float roughness;         // offset  28 (4B)
    vec3 specularAlbedo;     // offset  32 (12B) — rC.rgb * S.x
    float reflectWeight;     // offset  44 (4B)
    vec3 diffuseAlbedo;      // offset  48 (12B) — nonSpecColor * diffuseSelector
    float refractWeight;     // offset  60 (4B)
    vec3 transmissionAlbedo; // offset  64 (12B) — nonSpecColor * transmissionSelector
    int illuminationType;    // offset  76 (4B)
    vec3 emission;           // offset  80 (12B)
    // [4B pad to 96]
    vec3 absorption;         // offset  96 (12B) — primary-segment atmospheric transmission
    // [4B pad to 112]
    vec3 rd;                 // offset 112 (12B) — primary ray direction (sky branch)
    // [4B pad to 128]
};                           // 128B total

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

// 镜面反射射线击中天空时的虚拟投射距离哨兵值 (f16 可表示, ~6e4)
const float VPROJDIST_SKY = 60000.0;

// 打包两个 half 并钳制到 f16 范围, 避免 HDR 颜色溢出为 Inf
float pack2HalfClamped(float a, float b) {
    return uintBitsToFloat(packHalf2x16(
        vec2(clamp(a, -65504.0, 65504.0), clamp(b, -65504.0, 65504.0))));
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
// AliceEncoding.aliceY = vec4(v, ω) — 即 ALICE 线性嵌入表示，与 alice_encode 输出兼容
// AliceEncoding.CoCg = vec2(Co, Cg) — 色度 (YCoCg 空间)
//
// 辐照度解码使用 ALICE 最大熵半球余弦投影解析逼近，全域误差 < 0.4%

struct AliceEncoding {
    mediump vec4 aliceY; // ALICE 嵌入: xyz = 方向向量 v, w = 总能量 ω = |v| + I
    mediump vec2 CoCg; // (Co, Cg)
};

// ---------------------------------------------------------------------------
// 编解码与投影核心接口
// ---------------------------------------------------------------------------

AliceEncoding irradiance_to_alice(vec3 color, vec3 dir)
{
    AliceEncoding result;

    float Y = dot(color, vec3(0.2126, 0.7152, 0.0722));

    float Co = 0.5 * color.r - 0.5 * color.b;
    float Cg = -0.25 * color.r + 0.5 * color.g - 0.25 * color.b;

    result.CoCg = vec2(Co, Cg);
    // ALICE 编码: v = dir*Y, ω = |v| + 0 = Y (单样本 I=0)
    result.aliceY = vec4(dir * Y, Y);

    return result;
}

// ALICE 辐照度投影 (替代原 SG 模型)
// encoded.aliceY 即为 ALICE 编码 vec4(v, ω)
// 返回余弦加权漫反射辐照度 RGB
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
AliceEncoding mix_alice(AliceEncoding a, AliceEncoding b, float s)
{
    AliceEncoding result;
    result.aliceY = mix(a.aliceY, b.aliceY, s);
    result.CoCg = mix(a.CoCg, b.CoCg, s);
    return result;
}

AliceEncoding init_alice()
{
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

void accumulate_alice(inout AliceEncoding accum, AliceEncoding b, float scale)
{
    accum.aliceY += b.aliceY * scale;
    accum.CoCg += b.CoCg * scale;
}

// 将 AliceEncoding 压缩为 3 个 float
vec3 packAlice(AliceEncoding encoded) {
    // 注意：aliceY 和 CoCg 可能超出 half 范围（但通常不会），必要时 clamp
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

// 镜面降噪纹理打包 (colortex3 + colortex4)
//   data0 (colortex3): pos.xyz + oct(R)            R = 主导反射/折射方向
//   data1 (colortex4): f16(R,G) | f16(B,roughness) | f16(variance, virtualProjDist) | oct(H)
//   variance < 0 复用为天空 mask; virtualProjDist = VPROJDIST_SKY 表示反射射线击中天空
//   H = GGX 主半向量 (虚拟平面法线)
//   weight 不再走纹理 — 由 101/102 直接写入 image, 降噪 pass 不参与
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

struct diffuseIlluminationData {
    AliceEncoding data;
    AliceEncoding data_swap;
    vec3 pos;
    lowp vec3 normal;
    lowp vec3 normal2;
    mediump float weight;
    mediump float prev_weight;
};

// ===========================================================================
// Unified diffuse buffer — replaces old binding 2 + binding 6 + 4 custom images.
// 20 floats = 80 bytes, normals oct-encoded into 1 float each and packed with positions.
//
// Layout rationale:
//   - rt_* fields: ray0.rgen writes current frame RT output; temporal_diffuse.glsl reads via loadDiffuseInput
//   - px/py/pz + oct_n: current geometry + oct-encoded normal packed together
//   - oct_n2: oct-encoded normal2 (no position to pair with; 1 float vs old 3)
//   - hist_px/hist_py/hist_pz + hist_oct_n: history geometry from swap3,
//     read by temporal_diffuse.glsl via fetchDiffuse for reprojection edge-stopping.
//     MUST be separate from px/py/pz because ray0.rgen overwrites those each frame.
// ===========================================================================
struct UnifiedDiffuseElement {
    // --- RT output (ray0.rgen writes, temporal_diffuse.glsl reads) — half-packed AliceEncoding: 12B ---
    float rt_aliceY_xy, rt_aliceY_zw, rt_CoCg;
    // --- Current geometry: pos.xyz + oct(normal) + oct(normal2) — 20B ---
    float px, py, pz, oct_n;    // position + oct-encoded current normal
    float oct_n2;                // oct-encoded normal2
    // --- History geometry: pos.xyz + oct(normal) — 16B ---
    float hist_px, hist_py, hist_pz, hist_oct_n;
    // --- Temporal history prev frame (swap3 writes, temporal_diffuse.glsl reads): 14B ---
    float hist_aliceY_xy, hist_aliceY_zw, hist_CoCg, hist_weight;
    // --- Temporal history swap frame (temporal_diffuse.glsl/swap3 write, swap2/fog read): 14B ---
    float swap_aliceY_xy, swap_aliceY_zw, swap_CoCg, swap_weight;
};  // 18 floats = 72 bytes (was 80 with variance)

layout(std430, set = 3, binding = 2) buffer DiffuseBuffer {
    UnifiedDiffuseElement data[];
} diffuseIlluminationBuffer;

// Keep old struct types for function interfaces (unpacked representation).
// DiffuseIlluminationWriteData is still returned by fetchPrevDiffuse/samplePrevDiffuse.
struct DiffuseIlluminationWriteData {
    AliceEncoding data_swap;
    vec3 pos;
    lowp vec3 normal;
    lowp vec3 normal2;
    mediump float weight;
};

// Helper: unpack RT output from unified SSBO into full-precision struct.
// Used by temporal_diffuse.glsl to read the current frame's ray-traced input.
DiffuseIlluminationWriteData loadDiffuseInput(uint idx) {
    UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[idx];
    DiffuseIlluminationWriteData t;
    mediump vec2 aliceY_xy = unpackHalf2x16(floatBitsToUint(e.rt_aliceY_xy));
    mediump vec2 aliceY_zw = unpackHalf2x16(floatBitsToUint(e.rt_aliceY_zw));
    t.data_swap.aliceY = clamp(vec4(aliceY_xy, aliceY_zw), vec4(-10000), vec4(10000));
    t.data_swap.CoCg = unpackHalf2x16(floatBitsToUint(e.rt_CoCg));
    t.pos = vec3(e.px, e.py, e.pz);
    t.normal = decodeNormal(e.oct_n);
    t.normal2 = decodeNormal(e.oct_n2);
    t.weight = 1.0;  // not stored in RT output
    return t;
}

struct vec3IlluminationData {
    mediump vec3 data;
    mediump vec3 data_swap;
    vec3 pos;
    mediump vec3 normal;
    mediump float weight;
    mediump float prev_weight;
    mediump float mixWeight;
};

// ---------------------------------------------------------------------------
// 镜面反射/折射 RT 输出 + 时域历史 (SSBO, 64B/元素) — ray0.rgen 写当前帧,
// 101/102 写累积颜色+权重, swap5/7 写历史, swap4/6 读累积颜色.
// 替代了原先 8 个独立的 rgba32f image (reflect+refract 各4个, 共~253MB VRAM).
// 方向以八面体压缩存储, 虚拟投射距离单独存放, 颜色 f16 压缩.
// normal = decodeNormal(oct_dir)*virtualProjDist 可按需重建 (dir*dist 语义).
// H (GGX 主半向量) 不存于此 — 由 swap4/6 从 R+V 计算后写入 colortex4.w.
// ---------------------------------------------------------------------------
struct SpecularRTElement {
    // === 当前帧 (32B) — ray0.rgen 写入 raw RT, 101/102 覆写为累积色 ===
    float px, py, pz;    // 主命中点世界坐标 (12B)
    float oct_dir;       // 八面体压缩主导方向 R (4B)
    float virtualProjDist;     // 虚拟投射距离 (4B)
    float color_rg;      // packHalf2x16: ray0→raw RT, 101/102→accumulated (4B)
    float color_b;       // packHalf2x16: ray0→raw RT, 101/102→accumulated (4B)
    float accum_weight;  // 时域累积权重 (4B, 101/102 写, swap4/5/6/7 读)
    // [4B 隐式填充到 32B 对齐]

    // === 时域历史 (32B) — swap5/7 写, 101/102 下帧读 ===
    float hist_px, hist_py, hist_pz; // 上帧世界坐标 (12B)
    float hist_oct_dir;  // 上帧八面体压缩方向 R (4B)
    float hist_vprojdist;// 上帧虚拟投射距离 (4B)
    float hist_color_rg; // packHalf2x16: 上帧 pre-denoise 累积色 (4B)
    float hist_color_b;  // packHalf2x16 (4B)
    float hist_weight;   // 上帧累积权重 (4B)
}; // 64B 总计, 16B 对齐

SpecularRTElement packSpecularRT(vec3 pos, vec3 R, float virtualProjDist, vec3 color) {
    SpecularRTElement e;
    e.px         = pos.x;
    e.py         = pos.y;
    e.pz         = pos.z;
    e.oct_dir    = encodeNormal(R);
    e.virtualProjDist  = virtualProjDist;
    e.color_rg   = pack2HalfClamped(color.r, color.g);
    e.color_b    = pack2HalfClamped(color.b, 0.0);
    e.accum_weight = 0.0;
    // hist_* fields left uninitialized (written later by swap5/7)
    return e;
}

// 重建当前帧: pos + normal(=R*virtualProjDist) + raw RT color
void unpackSpecularRT(SpecularRTElement e, out vec3 pos, out vec3 normal, out vec3 color) {
    pos    = vec3(e.px, e.py, e.pz);
    normal = decodeNormal(e.oct_dir) * e.virtualProjDist;
    vec2 rg = unpackHalf2x16(floatBitsToUint(e.color_rg));
    float b = unpackHalf2x16(floatBitsToUint(e.color_b)).x;
    color  = vec3(rg.x, rg.y, b);
}

// 读取时域历史 (swap5/7 写入, 101/102 下帧读)
void unpackSpecularHistory(SpecularRTElement e, out vec3 histPos, out vec3 histNormal, out vec3 histColor, out float histWeight) {
    histPos    = vec3(e.hist_px, e.hist_py, e.hist_pz);
    histNormal = decodeNormal(e.hist_oct_dir) * e.hist_vprojdist;
    vec2 rg = unpackHalf2x16(floatBitsToUint(e.hist_color_rg));
    float b = unpackHalf2x16(floatBitsToUint(e.hist_color_b)).x;
    histColor  = vec3(rg.x, rg.y, b);
    histWeight = e.hist_weight;
}

layout(std430, set = 3, binding = 3) buffer ReflectIlluminationDataBuffer {
    SpecularRTElement data[];
} reflectIlluminationBuffer;

layout(std430, set = 3, binding = 4) buffer RefractIlluminationDataBuffer {
    SpecularRTElement data[];
} refractIlluminationBuffer;

#if defined(PREV_DIFFUSE_BUFFER)

// Read previous frame's accumulated AliceEncoding for ray guiding (ray0.rgen).
// Only data_swap.aliceY fields are used by the caller; the rest are filled with
// best-effort values from the history SSBO.
DiffuseIlluminationWriteData fetchPrevDiffuse(ivec2 p) {
    UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[getIndex(p)];
    DiffuseIlluminationWriteData t;
    mediump vec2 aliceY_xy = unpackHalf2x16(floatBitsToUint(e.swap_aliceY_xy));
    mediump vec2 aliceY_zw = unpackHalf2x16(floatBitsToUint(e.swap_aliceY_zw));
    t.data_swap.aliceY = clamp(vec4(aliceY_xy, aliceY_zw), vec4(-10000), vec4(10000));
    t.data_swap.CoCg = unpackHalf2x16(floatBitsToUint(e.swap_CoCg));
    t.pos = vec3(e.px, e.py, e.pz);
    t.normal = decodeNormal(e.oct_n);
    t.normal2 = decodeNormal(e.oct_n2);
    t.weight = e.swap_weight;
    return t;
}

DiffuseIlluminationWriteData blendPrevDiffuse(DiffuseIlluminationWriteData A, DiffuseIlluminationWriteData B, float x) {
    DiffuseIlluminationWriteData t;
    t.data_swap = mix_alice(A.data_swap, B.data_swap, x);
    t.pos = mix(A.pos, B.pos, x);
    t.normal = normalize(mix(A.normal, B.normal, x));
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
    uint idx = getIndex(p);
    diffuseIlluminationBuffer.data[idx].swap_aliceY_xy = uintBitsToFloat(packHalf2x16(data.data_swap.aliceY.xy));
    diffuseIlluminationBuffer.data[idx].swap_aliceY_zw = uintBitsToFloat(packHalf2x16(data.data_swap.aliceY.zw));
    diffuseIlluminationBuffer.data[idx].swap_CoCg   = uintBitsToFloat(packHalf2x16(data.data_swap.CoCg));
    diffuseIlluminationBuffer.data[idx].swap_weight = data.weight;
}

#endif

#if defined(DIFFUSE_BUFFER) || defined(DIFFUSE_BUFFER_MIN) || defined(DIFFUSE_BUFFER_MIN2)

// All diffuse temporal history now lives in unified diffuseIlluminationBuffer (binding 2).

diffuseIlluminationData fetchDiffuse(ivec2 p) {
    diffuseIlluminationData tmp;
    UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[getIndex(p)];

    // Unpack current frame (swap)
    mediump vec2 aliceY_xy = unpackHalf2x16(floatBitsToUint(e.swap_aliceY_xy));
    mediump vec2 aliceY_zw = unpackHalf2x16(floatBitsToUint(e.swap_aliceY_zw));
    tmp.data_swap.aliceY = clamp(vec4(aliceY_xy, aliceY_zw), vec4(-10000), vec4(10000));
    tmp.data_swap.CoCg = unpackHalf2x16(floatBitsToUint(e.swap_CoCg));
    tmp.weight = e.swap_weight;

    #ifndef DIFFUSE_BUFFER_MIN2
    // Unpack previous frame (hist)
    aliceY_xy = unpackHalf2x16(floatBitsToUint(e.hist_aliceY_xy));
    aliceY_zw = unpackHalf2x16(floatBitsToUint(e.hist_aliceY_zw));
    tmp.data.aliceY = clamp(vec4(aliceY_xy, aliceY_zw), vec4(-10000), vec4(10000));
    tmp.data.CoCg = unpackHalf2x16(floatBitsToUint(e.hist_CoCg));
    tmp.prev_weight = e.hist_weight;

    tmp.pos = vec3(e.hist_px, e.hist_py, e.hist_pz);
    tmp.normal = decodeNormal(e.hist_oct_n);
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
    t.normal = mix(A.normal, B.normal, x);
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
    diffuseIlluminationData data = blendDiffuse(blendDiffuse(A, B, p2.x), blendDiffuse(C, D, p2.x), p2.y);
    #ifndef DIFFUSE_BUFFER_MIN2
    data.normal = normalize(data.normal);
    #endif
    return data;
}
vec3 sampleDiffusePos(vec2 p) {
    UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[getIndex(ivec2(floor(p) + round(fract(p))))];
    return vec3(e.hist_px, e.hist_py, e.hist_pz);
}
void WriteDiffuse(diffuseIlluminationData data, ivec2 p) {
    uint idx = getIndex(p);

    // Always write swap (current frame)
    data.weight = clamp(data.weight, 0.0, 65504);

    diffuseIlluminationBuffer.data[idx].swap_aliceY_xy = uintBitsToFloat(packHalf2x16(data.data_swap.aliceY.xy));
    diffuseIlluminationBuffer.data[idx].swap_aliceY_zw = uintBitsToFloat(packHalf2x16(data.data_swap.aliceY.zw));
    diffuseIlluminationBuffer.data[idx].swap_CoCg   = uintBitsToFloat(packHalf2x16(data.data_swap.CoCg));
    diffuseIlluminationBuffer.data[idx].swap_weight = data.weight;

    #if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
    // Full write: also update hist (history) and geometry
    data.data.aliceY = clamp(data.data.aliceY, vec4(-65504), vec4(65504));
    data.data.CoCg = clamp(data.data.CoCg, vec2(-65504), vec2(65504));
    data.prev_weight = clamp(data.prev_weight, 0.0, 65504);

    diffuseIlluminationBuffer.data[idx].hist_aliceY_xy = uintBitsToFloat(packHalf2x16(data.data.aliceY.xy));
    diffuseIlluminationBuffer.data[idx].hist_aliceY_zw = uintBitsToFloat(packHalf2x16(data.data.aliceY.zw));
    diffuseIlluminationBuffer.data[idx].hist_CoCg   = uintBitsToFloat(packHalf2x16(data.data.CoCg));
    diffuseIlluminationBuffer.data[idx].hist_weight = data.prev_weight;

    // Write history geometry for next frame's temporal reprojection.
    // These survive ray0.rgen's next-frame overwrite of px/py/pz/oct_n.
    diffuseIlluminationBuffer.data[idx].hist_px = data.pos.x;
    diffuseIlluminationBuffer.data[idx].hist_py = data.pos.y;
    diffuseIlluminationBuffer.data[idx].hist_pz = data.pos.z;
    diffuseIlluminationBuffer.data[idx].hist_oct_n = encodeNormal(data.normal);
    #endif
}
#endif

#if defined(REFLECT_BUFFER) || defined(REFLECT_BUFFER_MIN) || defined(REFLECT_BUFFER_MIN2)

// 时域历史全部存入 SSBO reflectIlluminationBuffer (SpecularRTElement.hist_*).
// 原先的 4 个 rgba32f image (swap_color/color/lpos/lnormal, ~32MB) 已删除.

vec3IlluminationData fetchReflect(ivec2 p) {
    vec3IlluminationData tmp;
    uint i = getIndex(uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1)));
    SpecularRTElement e = reflectIlluminationBuffer.data[i];

    // 当前帧累积颜色 + 权重 (101 写入, swap4/swap5 读取)
    vec2 rg = unpackHalf2x16(floatBitsToUint(e.color_rg));
    float b = unpackHalf2x16(floatBitsToUint(e.color_b)).x;
    tmp.data_swap = vec3(rg.x, rg.y, b);
    tmp.weight = e.accum_weight;
    tmp.mixWeight = 0.0;

    #ifndef REFLECT_BUFFER_MIN2
    // 时域历史 (swap5 写入, 101 下帧读取)
    rg = unpackHalf2x16(floatBitsToUint(e.hist_color_rg));
    b  = unpackHalf2x16(floatBitsToUint(e.hist_color_b)).x;
    tmp.data = vec3(rg.x, rg.y, b);
    tmp.prev_weight = e.hist_weight;
    tmp.normal = decodeNormal(e.hist_oct_dir) * e.hist_vprojdist;
    tmp.pos = vec3(e.hist_px, e.hist_py, e.hist_pz);
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
    t.mixWeight = (B.mixWeight - A.mixWeight) * x + A.mixWeight;
    #endif
    return t;
}

// 最近邻读取反射历史几何 (避免双线性插值破坏方向向量 R×vproj)
bool fetchReflectHistoryGeometry(ivec2 p, out vec3 pos, out vec3 normal) {
    ivec2 clamped_p = clamp(p, ivec2(0), ivec2(resolution_global) - 1);
    uint i = getIndex(uvec2(clamped_p));
    SpecularRTElement e = reflectIlluminationBuffer.data[i];
    if (denoiseBuffer.data[i].distance < -0.5) return false;
    pos    = vec3(e.hist_px, e.hist_py, e.hist_pz);
    normal = decodeNormal(e.hist_oct_dir) * e.hist_vprojdist;
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

// 101 调用: 写入累积颜色 + 权重到 SSBO 当前帧区段
void WriteReflect(vec3IlluminationData data, ivec2 p) {
    uint i = getIndex(uvec2(p));
    reflectIlluminationBuffer.data[i].color_rg = pack2HalfClamped(data.data_swap.r, data.data_swap.g);
    reflectIlluminationBuffer.data[i].color_b  = pack2HalfClamped(data.data_swap.b, 0.0);
    reflectIlluminationBuffer.data[i].accum_weight = data.weight;
}

// swap5 调用: 写入时域历史到 SSBO hist_* 区段 (供 101 下帧读取)
void WriteReflectHistory(vec3 preDenoiseColor, float prevWeight, vec3 pos, vec3 R, float virtualProjDist, ivec2 p) {
    uint i = getIndex(uvec2(p));
    reflectIlluminationBuffer.data[i].hist_px = pos.x;
    reflectIlluminationBuffer.data[i].hist_py = pos.y;
    reflectIlluminationBuffer.data[i].hist_pz = pos.z;
    reflectIlluminationBuffer.data[i].hist_oct_dir = encodeNormal(R);
    reflectIlluminationBuffer.data[i].hist_vprojdist = virtualProjDist;
    reflectIlluminationBuffer.data[i].hist_color_rg = pack2HalfClamped(preDenoiseColor.r, preDenoiseColor.g);
    reflectIlluminationBuffer.data[i].hist_color_b  = pack2HalfClamped(preDenoiseColor.b, 0.0);
    reflectIlluminationBuffer.data[i].hist_weight = prevWeight;
}
#endif

#if defined(REFRACT_BUFFER) || defined(REFRACT_BUFFER_MIN) || defined(REFRACT_BUFFER_MIN2)

// 时域历史全部存入 SSBO refractIlluminationBuffer (SpecularRTElement.hist_*).
// 原先的 4 个 rgba32f image 已删除.

vec3IlluminationData fetchRefract(ivec2 p) {
    vec3IlluminationData tmp;
    uint i = getIndex(uvec2(clamp(p, ivec2(0), ivec2(resolution_global) - 1)));
    SpecularRTElement e = refractIlluminationBuffer.data[i];

    vec2 rg = unpackHalf2x16(floatBitsToUint(e.color_rg));
    float b = unpackHalf2x16(floatBitsToUint(e.color_b)).x;
    tmp.data_swap = vec3(rg.x, rg.y, b);
    tmp.weight = e.accum_weight;

    #ifndef REFRACT_BUFFER_MIN2
    rg = unpackHalf2x16(floatBitsToUint(e.hist_color_rg));
    b  = unpackHalf2x16(floatBitsToUint(e.hist_color_b)).x;
    tmp.data = vec3(rg.x, rg.y, b);
    tmp.mixWeight = 0.0;
    tmp.normal = decodeNormal(e.hist_oct_dir) * e.hist_vprojdist;
    tmp.pos = vec3(e.hist_px, e.hist_py, e.hist_pz);
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
    t.mixWeight = (B.mixWeight - A.mixWeight) * x + A.mixWeight;
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
    uint i = getIndex(uvec2(p));
    refractIlluminationBuffer.data[i].color_rg = pack2HalfClamped(data.data_swap.r, data.data_swap.g);
    refractIlluminationBuffer.data[i].color_b  = pack2HalfClamped(data.data_swap.b, 0.0);
    refractIlluminationBuffer.data[i].accum_weight = data.weight;
}

void WriteRefractHistory(vec3 preDenoiseColor, float prevWeight, vec3 pos, vec3 R, float virtualProjDist, ivec2 p) {
    uint i = getIndex(uvec2(p));
    refractIlluminationBuffer.data[i].hist_px = pos.x;
    refractIlluminationBuffer.data[i].hist_py = pos.y;
    refractIlluminationBuffer.data[i].hist_pz = pos.z;
    refractIlluminationBuffer.data[i].hist_oct_dir = encodeNormal(R);
    refractIlluminationBuffer.data[i].hist_vprojdist = virtualProjDist;
    refractIlluminationBuffer.data[i].hist_color_rg = pack2HalfClamped(preDenoiseColor.r, preDenoiseColor.g);
    refractIlluminationBuffer.data[i].hist_color_b  = pack2HalfClamped(preDenoiseColor.b, 0.0);
    refractIlluminationBuffer.data[i].hist_weight = prevWeight;
}
#endif
