
#include "/lib/buffers/frame_data.glsl"
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

// ===========================================================================
// Unnormalized True-Physical YCoCg & SG Irradiance Integration (NaN-Safe)
// ===========================================================================
#define M_PI 3.14159265358979323846

struct SH {
    mediump vec4 shY;   // (dir * Y, Y)
    mediump vec2 CoCg;  // (Co, Cg)
};

struct SGLobe {
    vec3 axis;
    float sharpness;
    float logAmplitude;
};

// ---------------------------------------------------------------------------
// 辅助数学原语
// ---------------------------------------------------------------------------

float precise_erf(float x) {
    float sign_x = sign(x);
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-x * x);
    return sign_x * y;
}

float expm1_over_x(float x) {
    if (abs(x) < 1e-5) {
        return 1.0 + 0.5 * x;
    }
    return (exp(x) - 1.0) / x;
}

// 避开 0/0 无定义，并使用一阶泰勒展开平滑逼近低频极限 (确保绝对不溢出)
float safe_lambda_over_one_minus_exp_two_lambda(float lambda) {
    if (lambda < 1e-4) {
        // 当 lambda 趋于 0 时的极高精度泰勒级数展开
        return 0.5 * (1.0 + lambda + (1.0 / 3.0) * lambda * lambda);
    }
    return lambda / (1.0 - exp(-2.0 * lambda));
}

// 数值稳定的双 SG 乘积计算 (已修复浮点抖动导致的负数开方 NaN)
SGLobe SGProduct(vec3 axis1, float sharpness1, vec3 axis2, float sharpness2) {
    vec3 axis = axis1 * sharpness1 + axis2 * sharpness2;
    float sharpness = length(axis);
    float cosine = clamp(dot(axis1, axis2), -1.0, 1.0);
    
    float sharpnessMin = min(sharpness1, sharpness2);
    float sharpnessRatio = sharpnessMin / max(sharpness1, sharpness2);
    
    // 安全开方保护：使用 max(..., 0.0) 强行杜绝浮点抖动产生的极小负数开方
    float sqrt_term = sqrt(max(2.0 * sharpnessRatio * cosine + sharpnessRatio * sharpnessRatio + 1.0, 0.0));
    float logAmplitude = 2.0 * sharpnessMin * (cosine - 1.0) / (1.0 + sharpnessRatio + sqrt_term);

    SGLobe result;
    result.axis = axis / max(sharpness, 1e-6);
    result.sharpness = sharpness;
    result.logAmplitude = logAmplitude;
    return result;
}

float HSGIntegral(float cosine, float sharpness) {
    float steepness = sharpness * sqrt(
        (0.5 * sharpness + 0.65173288269070562) / 
        ((sharpness + 1.3418280033141288) * sharpness + 7.2216687798956709)
    );

    float s = 0.5 + 0.5 * (precise_erf(steepness * clamp(cosine, -1.0, 1.0)) / precise_erf(max(steepness, 1e-5)));

    return 2.0 * M_PI * mix(exp(-sharpness), 1.0, s) * expm1_over_x(-sharpness);
}

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
    result.shY = vec4(dir * Y, Y);

    return result;
}

vec3 project_SH_irradiance(SH sh, vec3 N)
{
    float Y_base = max(sh.shY.w, 0.0001);
    
    float Co = sh.CoCg.x;
    float Cg = sh.CoCg.y;

    float B = Y_base - 1.1404 * Co - 1.4304 * Cg;
    float R = B + 2.0 * Co;
    float G = Y_base - 0.1404 * Co + 0.5696 * Cg;
    
    vec3 base_color = max(vec3(R, G, B), vec3(0.0));

    // 从一阶方向矩估计 SG 物理参数
    float len_v = length(sh.shY.xyz);
    vec3 mu = sh.shY.xyz / max(len_v, 1e-5);
    
    float R_ratio = clamp(len_v / Y_base, 0.0, 0.999);
    float lambda = (R_ratio * (3.0 - R_ratio * R_ratio)) / max(1.0 - R_ratio * R_ratio, 1e-5);
    
    // 采用数学防崩溃函数计算振幅，确保 lambda -> 0 时数值依然绝对稳定
    float amplitude_factor = safe_lambda_over_one_minus_exp_two_lambda(lambda);
    float amplitude = Y_base * amplitude_factor / (2.0 * M_PI);

    // 构造入射光 SG
    SGLobe lightLobe;
    lightLobe.axis = mu;
    lightLobe.sharpness = lambda;
    lightLobe.logAmplitude = log(max(amplitude, 1e-6)); // 额外加一层 log 安全保护

    // 构造余弦波瓣 SG
    const float LAMBDA_C = 0.0008456087;
    const float ALPHA_C = LAMBDA_C / (2.0 * exp(LAMBDA_C) - 2.0 - 2.0 * LAMBDA_C);

    SGLobe cosineLobe;
    cosineLobe.axis = N;
    cosineLobe.sharpness = LAMBDA_C;
    cosineLobe.logAmplitude = 0.0;

    // 求解双 SG 乘积积分
    SGLobe prodLobe = SGProduct(lightLobe.axis, lightLobe.sharpness, cosineLobe.axis, cosineLobe.sharpness);
    
    float p = HSGIntegral(dot(prodLobe.axis, N), prodLobe.sharpness) * exp(LAMBDA_C + prodLobe.logAmplitude);
    float q = HSGIntegral(dot(lightLobe.axis, N), lightLobe.sharpness);
    
    float attenuation = exp(lightLobe.logAmplitude) * max(ALPHA_C * p - ALPHA_C * q, 0.0);
    attenuation = clamp(attenuation / max(Y_base, 1e-5), 0.0, 1.0);

    return max(base_color * attenuation, vec3(0)); 
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

// ===========================================================================
// ReSTIR GI 路径样本与储层结构体
// ===========================================================================
struct GI_Sample {
    vec3 pos;       // 盲追撞击点的坐标
    vec3 normal;    // 盲追撞击点的法线
    vec3 radiance;  // 撞击点发出的辐射度 (自发光 + 直射光)
    vec3 wi; 
    float is_sky; 
};

struct Reservoir {
    GI_Sample samplePoint;
    float w_sum;
    float M;
    float W;
};

// 绑定 10：当前帧写入，绑定 11：上一帧读取
layout(std430, set = 3, binding = 7) buffer CurReservoirBuffer {
    Reservoir data[];
} curReservoirs;

layout(std430, set = 3, binding = 8) buffer PrevReservoirBuffer {
    Reservoir data[];
} prevReservoirs;

// 储层更新原语
bool updateReservoir(inout Reservoir r, GI_Sample candidate, float p_hat, float weight, float randomValue) {
    r.w_sum += weight;
    r.M += 1.0;
    if (randomValue * r.w_sum < weight) {
        r.samplePoint = candidate;
        return true; 
    }
    return false;
}

// 随机双线性重投影采样函数
Reservoir samplePrevReservoirStochastic(vec2 uv, vec2 res, float random_val) {
    // 将 0~1 的连续 UV 映射到像素浮点网格（像素中心在 integer + 0.5）
    vec2 continuous_px = uv * res + 0.5;
    ivec2 base = ivec2(floor(continuous_px));
    vec2 f = fract(continuous_px);

    // 计算标准双线性插值的 4 个角权重
    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;

    // 按双线性权重作为概率，随机抽取 4 个角中的 1 个
    ivec2 offset;
    if (random_val < w00) {
        offset = ivec2(0, 0);
    } else if (random_val < w00 + w10) {
        offset = ivec2(1, 0);
    } else if (random_val < w00 + w10 + w01) {
        offset = ivec2(0, 1);
    } else {
        offset = ivec2(1, 1);
    }

    ivec2 fetch_coord = base + offset;

    // 边界安全钳制
    ivec2 max_coord = ivec2(res) - ivec2(1);
    fetch_coord = clamp(fetch_coord, ivec2(0), max_coord);

    uint nIdx = uint(fetch_coord.y) * uint(res.x) + uint(fetch_coord.x);
    return prevReservoirs.data[nIdx];
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

layout(std430, set = 3, binding = 6) buffer PrevDiffuseIllumiantionDataBuffer {
    diffuseIllumiantionBufferDataW data[];
} prevDiffuseIllumiantionBuffer; // for temporal reprojection

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

diffuseIllumiantionBufferDataW fetchPrevDiffuse(ivec2 p) {
    return prevDiffuseIllumiantionBuffer.data[getIdx(p)];
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
    prevDiffuseIllumiantionBuffer.data[getIdx(p)] = data;
}

#endif

#if defined(DIFFUSE_BUFFER) || defined(DIFFUSE_BUFFER_MIN) || defined(DIFFUSE_BUFFER_MIN2)

layout(rgba32f) uniform image2D diffuseIllumiantionData_shY_swap;
layout(rgba32f) uniform image2D diffuseIllumiantionData_CoCg_swap;
uniform sampler2D diffuseIllumiantionData_shY_Sampler;
uniform sampler2D diffuseIllumiantionData_CoCg_Sampler;
uniform sampler2D diffuseIllumiantionData_shY_swap_Sampler;
uniform sampler2D diffuseIllumiantionData_CoCg_swap_Sampler;
uniform sampler2D diffuseIllumiantionData_lnormal_Sampler;
uniform sampler2D diffuseIllumiantionData_lpos_Sampler;
#if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
layout(rgba32f) uniform image2D diffuseIllumiantionData_shY;
layout(rg32f) uniform image2D diffuseIllumiantionData_CoCg;
layout(rgba32f) uniform image2D diffuseIllumiantionData_lnormal;
layout(rgba32f) uniform image2D diffuseIllumiantionData_lpos;
#endif

/*diffuseIllumiantionData sampleDiffuse(vec2 p) {
    diffuseIllumiantionData tmp;

    vec4 tmp4 = texture(diffuseIllumiantionData_CoCg_swap_Sampler, p);
    tmp.data_swap.CoCg = tmp4.xy;
    tmp.data_swap.shY = texture(diffuseIllumiantionData_shY_swap_Sampler, p);
    #ifndef DIFFUSE_BUFFER_MIN
    tmp4 = texture(diffuseIllumiantionData_CoCg_Sampler, p);
    tmp.data.CoCg = tmp4.xy;
    tmp.data.shY = texture(diffuseIllumiantionData_shY_Sampler, p);
    tmp.weight = tmp4.z;
    tmp.normal = texture(diffuseIllumiantionData_lnormal_Sampler, p).xyz;
    tmp.pos = texture(diffuseIllumiantionData_lpos_Sampler, p).xyz;
    #endif
    return tmp;
}*/

diffuseIllumiantionData fetchDiffuse(ivec2 p) {
    diffuseIllumiantionData tmp;

    //vec4 tmp4 = texelFetch(diffuseIllumiantionData_CoCg_swap_Sampler, p, 0);
    vec4 tmp4 = texelFetch(diffuseIllumiantionData_shY_swap_Sampler, p, 0);

    //tmp.data_swap.CoCg = tmp4.xy;
    tmp.data_swap.CoCg = unpackHalf2x16(floatBitsToUint(tmp4.z));
    mediump vec2 w_v = unpackHalf2x16(floatBitsToUint(tmp4.w));

    mediump vec2 shY_xy = unpackHalf2x16(floatBitsToUint(tmp4.x));
    mediump vec2 shY_zw = unpackHalf2x16(floatBitsToUint(tmp4.y));

    tmp.data_swap.shY = clamp(vec4(shY_xy, shY_zw), vec4(-10000), vec4(10000));

    //tmp.data_swap.shY = texelFetch(diffuseIllumiantionData_shY_swap_Sampler, p, 0);
    //tmp.weight = tmp4.z;
    //tmp.variance = tmp4.w;
    tmp.weight = w_v.x;
    tmp.variance = w_v.y;

    #ifndef DIFFUSE_BUFFER_MIN2

    //tmp4 = texelFetch(diffuseIllumiantionData_CoCg_Sampler, p, 0);
    //tmp.data.CoCg = tmp4.xy;
    //tmp.data.shY = texelFetch(diffuseIllumiantionData_shY_Sampler, p, 0);
    tmp4 = texelFetch(diffuseIllumiantionData_shY_Sampler, p, 0);
    tmp.data.CoCg = unpackHalf2x16(floatBitsToUint(tmp4.z));
    w_v = unpackHalf2x16(floatBitsToUint(tmp4.w));
    tmp.prev_weight = w_v.x;
    tmp.prev_variance = w_v.y;
    shY_xy = unpackHalf2x16(floatBitsToUint(tmp4.x));
    shY_zw = unpackHalf2x16(floatBitsToUint(tmp4.y));
    tmp.data.shY = clamp(vec4(shY_xy, shY_zw), vec4(-10000), vec4(10000));

    tmp.normal = texelFetch(diffuseIllumiantionData_lnormal_Sampler, p, 0).xyz;
    tmp.pos = texelFetch(diffuseIllumiantionData_lpos_Sampler, p, 0).xyz;
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
    // ivec2 p1 = ivec2(p);
    // vec2 p2 = fract(p);
    // vec3 posA = texelFetch(diffuseIllumiantionData_lpos_Sampler, p1, 0).xyz;
    // vec3 posB = texelFetch(diffuseIllumiantionData_lpos_Sampler, p1 + ivec2(1,0), 0).xyz;
    // vec3 posC = texelFetch(diffuseIllumiantionData_lpos_Sampler, p1 + ivec2(0,1), 0).xyz;
    // vec3 posD = texelFetch(diffuseIllumiantionData_lpos_Sampler, p1 + ivec2(1,1), 0).xyz;
    // return mix(
    //     mix(posA, posB, round(p2.x)),
    //     mix(posC, posD, round(p2.x)),
    //     round(p2.y)
    // );
    return texelFetch(diffuseIllumiantionData_lpos_Sampler, ivec2(floor(p) + round(fract(p))), 0).xyz;
}
void WriteDiffuse(diffuseIllumiantionData data, ivec2 p) {
    data.weight = clamp(data.weight, 0.0, 65504);
    data.variance = clamp(data.variance, 0.0, 65504);

    float shY_xy = uintBitsToFloat(packHalf2x16(data.data_swap.shY.xy));
    float shY_zw = uintBitsToFloat(packHalf2x16(data.data_swap.shY.zw));
    float CoCg = uintBitsToFloat(packHalf2x16(data.data_swap.CoCg));
    float w_v = uintBitsToFloat(packHalf2x16(vec2(data.weight, data.variance)));
    imageStore(diffuseIllumiantionData_shY_swap, p, vec4(shY_xy, shY_zw, CoCg, w_v));
    //imageStore(diffuseIllumiantionData_shY_swap, p, data.data_swap.shY);
    //imageStore(diffuseIllumiantionData_CoCg_swap, p, vec4(CoCg, data.weight, data.variance));
    //imageStore(diffuseIllumiantionData_CoCg_swap, p, vec4(CoCg, w_v, 0, 0));

    #if !defined(DIFFUSE_BUFFER_MIN) && !defined(DIFFUSE_BUFFER_MIN2)
    //imageStore(diffuseIllumiantionData_shY, p, data.data.shY);
    //imageStore(diffuseIllumiantionData_CoCg, p, vec4(data.data.CoCg, 0, 0));
    data.data.shY = clamp(data.data.shY, vec4(-65504), vec4(65504));
    data.data.CoCg = clamp(data.data.CoCg, vec2(-65504), vec2(65504));
    data.prev_weight = clamp(data.prev_weight, 0.0, 65504);
    data.prev_variance = clamp(data.prev_variance, 0.0, 65504);

    shY_xy = uintBitsToFloat(packHalf2x16(data.data.shY.xy));
    shY_zw = uintBitsToFloat(packHalf2x16(data.data.shY.zw));
    CoCg = uintBitsToFloat(packHalf2x16(data.data.CoCg));
    w_v = uintBitsToFloat(packHalf2x16(vec2(data.prev_weight, data.prev_variance)));
    imageStore(diffuseIllumiantionData_shY, p, vec4(shY_xy, shY_zw, CoCg, w_v));

    imageStore(diffuseIllumiantionData_lpos, p, vec4(data.pos, 0));
    imageStore(diffuseIllumiantionData_lnormal, p, vec4(data.normal, 0));

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
