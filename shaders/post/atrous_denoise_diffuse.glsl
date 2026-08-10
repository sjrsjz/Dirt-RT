#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/alice.glsl"

// ==========================================================================
// Pass 300: 空间滤波器 — 漫反射 (ALICE) 降噪
// 修改版: Bures 距离 + 能量感知权重
//
// 管线 (6 级迭代):
//   à‑trous:  STEP=1 R0=1   STEP=2 R0=2   STEP=3 R0=4   (compute)
//   Poisson:  STEP=4 R0=8   STEP=5 R0=16  STEP=6 R0=32  (fragment)
// ==========================================================================

uniform sampler2D colortex3; // 几何 (pos + normal)
uniform usampler2D colortex4; // 光照 (ALICE + variance)

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

void unpackLightSample(ivec2 coord, out vec3 pos, out float oct_normal,
    out AliceEncoding encoded, out float variance) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0);
    uvec4 light = texelFetch(colortex4, coord, 0);
    pos = sample_data0.xyz;
    oct_normal = sample_data0.w;
    encoded.aliceY = vec4(unpackHalf2x16(light.x), unpackHalf2x16(light.y));
    encoded.CoCg = unpackHalf2x16(light.z);
    variance = uintBitsToFloat(light.w);
}

// xy = eigen stddev, z = variance anisotropy, w = |v|^2.
// The trace is reconstructed as 3 * sigma_perp^2 + anisotropy.
vec4 makeBuresData(vec4 encoded) {
    float len_v_sq = dot(encoded.xyz, encoded.xyz);
    float kappa = alice_kappa(sqrt(len_v_sq), encoded.w);
    vec2 stddev = alice_eigen_std(encoded.w, kappa);
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
#if R0 == 8
#define ATROUS_POWER_COEFFICIENT 0.4644479501
#elif R0 == 16
#define ATROUS_POWER_COEFFICIENT 0.4657344365
#elif R0 == 32
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
    AliceEncoding center_alice;
    float center_var_est;
    unpackLightSample(pix, center_pos, center_oct_n, center_alice, center_var_est);

    // 天空像素跳过 (colortex3.xyz = 0 for sky)
    float center_pos_sq = dot(center_pos, center_pos);
    if (center_pos_sq < 1e-6) return;

    const float power = max(1.0, 2.0
                - ATROUS_GAMMA * ATROUS_POWER_COEFFICIENT);
    const float variance_mix = 2.0 - exp2(2.0 - power);

    center_var_est = max(center_var_est, 1e-12);

    // ---- 从 colortex3.w 解码中心法线（方差滤波写入，零额外读取）-------
    vec3 center_normal = decodeNormal(center_oct_n);

    // ---- 预计算中心像素的统计特征 -----------------------------------------
    // 中心 ALICE 编码: aliceY = vec4(v, ω)
    vec4 c_enc = center_alice.aliceY;
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
    AliceEncoding accumAlice = center_alice;

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
        AliceEncoding sample_alice;
        float sample_var_est;
        unpackLightSample(sample_coord, sample_world_pos, sample_oct_n,
            sample_alice, sample_var_est);

        if (dot(sample_world_pos, sample_world_pos) < 1e-6) continue; // 天空
        float w_geometry = abs(dot(sample_world_pos, center_normal)
                    - center_plane_distance) * inv_pixel_footprint;

        vec4 s_enc = sample_alice.aliceY;
        vec4 s_bures = makeBuresData(s_enc);
        float d_bures_sq = buresDistanceSq(c_enc.xyz, c_bures, c_trace,
                s_enc.xyz, s_bures);
        float w_luma = ATROUS_PHI_L * d_bures_sq / max(center_var_est + sample_var_est, 1e-12);

        const float w_kernel = ps.w;
        float w0 = w_kernel * exp(-w_geometry - w_luma);

        // ---- 累积 ----------------------------------------------------
        accumulate_alice(accumAlice, sample_alice, w0);
        sumWeight += w0;
        float weighted_var = w0 * sample_var_est;
        sumVarEnergy += vec2(weighted_var, w0 * weighted_var);
    }

    float inv_sumWeight = 1.0 / sumWeight;

    // ---- 归一化并输出 ----------------------------------------------------
    accumAlice = scale_alice(accumAlice, inv_sumWeight);

    float varEnergyOut = mix(sumVarEnergy.x, sumVarEnergy.y, variance_mix)
            * pow(inv_sumWeight, power);
    uvec3 packedAlice = uvec3(
            packHalf2x16(clamp(accumAlice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(accumAlice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(accumAlice.CoCg, vec2(-65504.0), vec2(65504.0))));
    out_light_sample = uvec4(packedAlice, floatBitsToUint(varEnergyOut));

    #if FINAL_DENOISE_PASS
    out_light_sample_blurred = uvec4(packedAlice, 0u);
    #endif
}
