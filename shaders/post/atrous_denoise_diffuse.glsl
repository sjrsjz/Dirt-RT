#version 430 core
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"

// ==========================================================================
// Pass 300: 空间滤波器 — 漫反射 (MaxEnt) 降噪
// 修改版: Bures 距离 + 能量感知权重
//
// 管线 (6 级迭代):
//   à‑trous:  STEP=1 R0=1   STEP=2 R0=2   STEP=3 R0=4   (compute)
//   Poisson:  STEP=4 R0=8   STEP=5 R0=16  STEP=6 R0=32  (fragment)
// ==========================================================================

uniform sampler2D colortex3; // 几何 (pos + normal)
uniform usampler2D colortex4; // 光照 (MaxEnt + variance)

/* RENDERTARGETS: 4,5 */
layout(location = 0) out uvec4 out_light_sample;
layout(location = 1) out uvec4 out_light_sample_blurred;

// ---------------------------------------------------------------------------
// Poisson 圆盘采样表 (NRD)
// .z = length(.xy), 用于高斯核权重
// ---------------------------------------------------------------------------

// samples = 8, min distance = 0.5
// .xy = 归一化采样偏移, .z = length(.xy), .w = 高斯核权重 exp(-z^2/2)
const vec4 POISSON_8[8] = {
    vec4(-0.4706069, -0.4427112, +0.6461146, +0.81170),
    vec4(-0.9057375, +0.3003471, +0.9542373, +0.63422),
    vec4(-0.3487388, +0.4037880, +0.5335386, +0.86734),
    vec4(+0.1023042, +0.6439373, +0.6520134, +0.80847),
    vec4(+0.5699277, +0.3513750, +0.6695386, +0.79925),
    vec4(+0.2939128, -0.1131226, +0.3149309, +0.95161),
    vec4(+0.7836658, -0.4208784, +0.8895339, +0.67328),
    vec4(+0.1564120, -0.8198990, +0.8346850, +0.70589)
    };

#define POISSON_N 8

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

bool unpackLightSample(ivec2 coord, out vec3 pos, out float oct_normal,
    out MaxEntEncoding encoded, out float variance) {
    uvec4 light = texelFetch(colortex4, coord, 0);
    variance = uintBitsToFloat(light.w);
    if (variance < 0.0) {
        pos = vec3(0.0);
        oct_normal = 0.0;
        encoded.maxEntY = vec4(0.0);
        encoded.CoCg = vec2(0.0);
        return false;
    }
    vec4 sample_data0 = texelFetch(colortex3, coord, 0);
    pos = sample_data0.xyz;
    oct_normal = sample_data0.w;
    encoded.maxEntY = vec4(unpackHalf2x16(light.x), unpackHalf2x16(light.y));
    encoded.CoCg = unpackHalf2x16(light.z);
    return true;
}

// xy = eigen stddev, z = variance anisotropy, w = |v|^2.
// The trace is reconstructed as 3 * sigma_perp^2 + anisotropy.
vec4 makeBuresData(vec4 encoded) {
    float len_v_sq = dot(encoded.xyz, encoded.xyz);
    float kappa = maxent_kappa(sqrt(len_v_sq), encoded.w);
    vec2 stddev = maxent_eigen_std(encoded.w, kappa);
    vec2 stddev_sq = stddev * stddev;
    return vec4(stddev, stddev_sq.y - stddev_sq.x, len_v_sq);
}

float buresDistanceSq(vec3 center_v, vec4 center_data, float center_trace,
    vec3 sample_v, vec4 sample_data) {
    float dot_v = dot(center_v, sample_v);
    float c_sq = 0.0;
    if (center_data.w > 0.0 && sample_data.w > 1e-16) {
        c_sq = min(1.0, dot_v * dot_v * center_data.w / sample_data.w);
    }

    float sample_std_perp_sq = sample_data.x * sample_data.x;
    float sample_trace = 3.0 * sample_std_perp_sq + sample_data.z;
    float cross_ab = center_data.x * sample_data.y
            + center_data.y * sample_data.x;
    float cross_2d = sqrt(max(0.0, cross_ab * cross_ab
                    + c_sq * center_data.z * sample_data.z));

    vec3 delta_v = center_v - sample_v;
    float mean_distance_sq = dot(delta_v, delta_v);
    float cross_trace = center_data.x * sample_data.x + cross_2d;
    return max(0.0, mean_distance_sq + center_trace + sample_trace
            - 2.0 * cross_trace);
}

// à-trous 分数阶方差传播指数
// 注意，fs 采样使用的是泊松核，结论可能不完全适用，但仍然可以作为一个近似值
#if STEP == 1
#define ATROUS_POWER_COEFFICIENT 0.3339015144
#elif STEP == 2
#define ATROUS_POWER_COEFFICIENT 0.4375201036
#elif STEP == 3
#define ATROUS_POWER_COEFFICIENT 0.4592464660
#elif STEP == 4
#define ATROUS_POWER_COEFFICIENT 0.4644479501
#elif STEP == 5
#define ATROUS_POWER_COEFFICIENT 0.4657344365
#elif STEP == 6
#define ATROUS_POWER_COEFFICIENT 0.4660551979
#endif


// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

void main() {
    ivec2 texSize = textureSize(colortex3, 0);
    ivec2 pix = ivec2(gl_FragCoord.xy);

    // ---- 中心像素基础数据 -------------------------------------------------
    vec3 center_pos;
    float center_oct_n;
    MaxEntEncoding center_maxent;
    float center_var_est;
    if (!unpackLightSample(pix, center_pos, center_oct_n, center_maxent,
            center_var_est)) {
        // Fragment passes ping-pong colortex4. Explicitly propagate the
        // sentinel instead of leaving the destination attachment undefined.
        out_light_sample = uvec4(0u, 0u, 0u,
            floatBitsToUint(-1.0));
#ifdef FINAL_DENOISE_PASS
        out_light_sample_blurred = uvec4(0u);
#endif
        return;
    }

    float center_pos_sq = dot(center_pos, center_pos);

    const float power = max(1.0, 2.0
                - ATROUS_GAMMA * ATROUS_POWER_COEFFICIENT);
    const float variance_mix = 2.0 - exp2(2.0 - power);

    center_var_est = max(center_var_est, 1e-16);

    // ---- 从 colortex3.w 解码中心法线（方差滤波写入，零额外读取）-------
    vec3 center_normal = decodeNormal(center_oct_n);

    // ---- 预计算中心像素的统计特征 -----------------------------------------
    // 中心 MaxEnt 编码: maxEntY = vec4(v, ω)
    vec4 c_enc = center_maxent.maxEntY;
    vec4 c_bures = makeBuresData(c_enc);
    float c_len_v_sq = c_bures.w;
    c_bures.w = c_len_v_sq > 1e-16 ? 1.0 / c_len_v_sq : 0.0;
    float c_trace = 3.0 * c_bures.x * c_bures.x + c_bures.z;

    float resolution_y = float(resolution_global.y);
    float dist_to_cam = max(sqrt(center_pos_sq), 0.001);
    float inv_pixel_footprint = resolution_y / (ATROUS_POSITION_PARAM
                * max(dist_to_cam, resolution_y * 0.00001));
    float center_plane_distance = dot(center_pos, center_normal);

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0;
    vec2 sumVarEnergy = vec2(center_var_est);
    MaxEntEncoding accumMaxEnt = center_maxent;

    // ---- Poisson 圆盘采样 (大核 R0=8,16,32) -----------------------------------
    // 旋转器 + 高斯核权重, 均匀圆盘覆盖替代 3×3 网格
    float theta = 2.0 * PI * fract(rand(vec2(pix)) + R0 * 0.6180339887498949);
    float cos_theta = cos(theta) * R0 * 1.75;
    float sin_theta = sin(theta) * R0 * 1.75;
    mat2 rotM = mat2(cos_theta, -sin_theta, sin_theta, cos_theta);

    for (int k = 0; k < POISSON_N; k++) {
        vec4 ps = POISSON_8[k];
        vec2 offset = rotM * ps.xy;
        ivec2 sample_coord = pix + ivec2(round(offset));

        if (any(greaterThanEqual(uvec2(sample_coord), uvec2(texSize)))) continue;

        vec3 sample_world_pos;
        float sample_oct_n;
        MaxEntEncoding sample_maxent;
        float sample_var_est;
        if (!unpackLightSample(sample_coord, sample_world_pos, sample_oct_n,
                sample_maxent, sample_var_est)) continue;
        float w_geometry = abs(dot(sample_world_pos, center_normal)
                    - center_plane_distance) * inv_pixel_footprint;

        vec4 s_enc = sample_maxent.maxEntY;
        vec4 s_bures = makeBuresData(s_enc);
        float d_bures_sq = buresDistanceSq(c_enc.xyz, c_bures, c_trace,
                s_enc.xyz, s_bures);
        float w_luma = ATROUS_PHI_L * d_bures_sq / (center_var_est + sample_var_est);

        const float w_kernel = ps.w;
        float w0 = w_kernel * exp(-w_geometry - w_luma);

        // ---- 累积 ----------------------------------------------------
        accumulate_maxent(accumMaxEnt, sample_maxent, w0);
        sumWeight += w0;
        float weighted_var = w0 * sample_var_est;
        sumVarEnergy += vec2(weighted_var, w0 * weighted_var);
    }

    float inv_sumWeight = 1.0 / sumWeight;

    // ---- 归一化并输出 ----------------------------------------------------
    accumMaxEnt = scale_maxent(accumMaxEnt, inv_sumWeight);

    float varEnergyOut = mix(sumVarEnergy.x, sumVarEnergy.y, variance_mix)
            * pow(inv_sumWeight, power);
    uvec3 packedMaxEnt = uvec3(
            packHalf2x16(clamp(accumMaxEnt.maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(accumMaxEnt.maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(accumMaxEnt.CoCg, vec2(-65504.0), vec2(65504.0))));
    out_light_sample = uvec4(packedMaxEnt, floatBitsToUint(varEnergyOut));

    #ifdef FINAL_DENOISE_PASS
    out_light_sample_blurred = uvec4(packedMaxEnt, 0u);
    #endif
}
