#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/lighting/alice.glsl"

// ==========================================================================
// Pass 300: 空间滤波器 — 漫反射 (ALICE) 降噪
// 修改版: Bures 距离 + 能量感知权重
//
// 管线 (6 级 à-trous 迭代):
//   STEP=1 R0=1   STEP=2 R0=2   STEP=3 R0=4   (compute)
//   STEP=4 R0=8   STEP=5 R0=16  STEP=6 R0=32  (fragment)
// ==========================================================================

uniform sampler2D colortex3; // 几何 (pos + normal)
uniform sampler2D colortex4; // 光照 (ALICE + variance)

/* RENDERTARGETS: 4,5 */
layout(location = 0) out mediump vec4 out_light_sample;
layout(location = 1) out mediump vec4 out_light_sample_blurred;

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

// 能量项灵敏度 (相对于 v-space z-score 的权重)
// 1.0 = 能量差与方向差同等重要; <1.0 = 能量差更宽容
#ifndef PHI_ENERGY
#define PHI_ENERGY 1.0
#endif

// ---------------------------------------------------------------------------
// Poisson 圆盘采样表 (NRD)
// .z = length(.xy), 用于高斯核权重
// ---------------------------------------------------------------------------

// samples = 8, min distance = 0.5
// .xy = 归一化采样偏移, .z = length(.xy), .w = 高斯核权重 exp(-z^2/2)
const vec4 POISSON_8[8] = {
    vec4( -0.4706069, -0.4427112, +0.6461146, +0.81170 ),
    vec4( -0.9057375, +0.3003471, +0.9542373, +0.63422 ),
    vec4( -0.3487388, +0.4037880, +0.5335386, +0.86734 ),
    vec4( +0.1023042, +0.6439373, +0.6520134, +0.80847 ),
    vec4( +0.5699277, +0.3513750, +0.6695386, +0.79925 ),
    vec4( +0.2939128, -0.1131226, +0.3149309, +0.95161 ),
    vec4( +0.7836658, -0.4208784, +0.8895339, +0.67328 ),
    vec4( +0.1564120, -0.8198990, +0.8346850, +0.70589 )
};

#define POISSON_N 8

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

void unpackLightSample(ivec2 coord, out vec3 pos, out vec3 normal,
                       out AliceEncoding encoded, out float variance) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0);
    vec4 sample_data1 = texelFetch(colortex4, coord, 0);
    pos = sample_data0.xyz;
    normal = decodeNormal(sample_data0.w);
    encoded = unpackAlice(sample_data1.x, sample_data1.y, sample_data1.z);
    variance = sample_data1.w;
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

void main() {
    ivec2 texSize = textureSize(colortex3, 0) - 1;
    ivec2 pix = ivec2(gl_FragCoord.xy);

    // ---- 中心像素基础数据 -------------------------------------------------
    vec3 center_pos, center_normal;
    AliceEncoding center_alice;
    float center_var_est;
    unpackLightSample(pix, center_pos, center_normal, center_alice, center_var_est);

    // 天空像素跳过 (方差被 swap2 复用作天空 mask)
    if (center_var_est < 0.0) return;

    center_var_est = max(center_var_est, 4e-9);

    // ---- 预计算中心像素的统计特征 -----------------------------------------
    // 中心 ALICE 编码: aliceY = vec4(v, ω)
    vec4 c_enc = center_alice.aliceY;
    float c_len_v = length(c_enc.xyz);
    float c_omega = c_enc.w;
    float c_kappa = alice_kappa(c_len_v, c_omega);

    // 方差分解:
    //   c_inv_sqrt_var       → v-space z-score 归一化 (标量估计量标准差逆)
    //   c_inv_sqrt_var_omega → ω-space z-score 归一化 (径向估计量标准差逆)
    float c_var_omega_est = alice_radial_est_var_from_scalar(center_var_est, c_kappa);
    float c_inv_sqrt_var       = inversesqrt(center_var_est);
    float c_inv_sqrt_var_omega = inversesqrt(max(c_var_omega_est, 1e-12));

    // 高斯曲率标记
    #if ENABLE_GAUSSIAN_FILTER == 1
    float geomValid = float(c_omega >= 0.0);
    center_alice.aliceY.w = abs(c_omega);
    c_omega = center_alice.aliceY.w;
    #else
    const float geomValid = 1.0;
    #endif

    float dist_to_cam = max(length(center_pos), 0.001);
    float inv_pixel_footprint = 1.0 / (SVGF_POSITION_PARAM
        * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0;
    float sumVarEnergy = center_var_est;
    AliceEncoding accumAlice = center_alice;

    // ---- Poisson 圆盘采样 (NRD, STEP>=4) -----------------------------------
    // 旋转器 + 高斯核权重, 替代 3×3 网格 → 更均匀的圆盘覆盖
    float theta = 2.0 * PI * fract(rand(vec2(pix)) + R0 * 0.6180339887498949);
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0 * 1.75;

    for (int k = 0; k < POISSON_N; k++) {
        // 选取 8 或 16 采样表
        vec4 ps = POISSON_8[k];
        vec2 offset = rotM * ps.xy;
        ivec2 sample_coord = pix + ivec2(round(offset));

        if (sample_coord != clamp(sample_coord, ivec2(0), texSize)) continue;

        float w_kernel = ps.w; // 预计算: exp(-z^2/2)

        // ---- 加载邻居样本 ------------------------------------------------
        vec3 sample_world_pos, sample_normal;
        AliceEncoding sample_alice;
        float sample_var_est;
        unpackLightSample(sample_coord, sample_world_pos, sample_normal,
                          sample_alice, sample_var_est);

        if (sample_var_est < 0.0) continue; // 天空
        sample_alice.aliceY.w = abs(sample_alice.aliceY.w);

        // ---- 几何权重 (不变) --------------------------------------------
        vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
        float depthTerm = abs(dot(delta, center_normal));
        float w_geometry = SVGF_NORMAL_POWER * (1.0 - dot(center_normal, sample_normal))
                         + depthTerm * geomValid;

        // ---- Bures 距离 + 能量感知 -------------------------------------
        vec4 s_enc = sample_alice.aliceY;
        float s_len_v = length(s_enc.xyz);
        float s_kappa = alice_kappa(s_len_v, s_enc.w);

        float d_bures_sq = alice_bures_distance_sq(c_enc, c_kappa, s_enc, s_kappa);
        float z_bures = sqrt(d_bures_sq) * c_inv_sqrt_var;

        float delta_omega = c_omega - s_enc.w;
        float z_energy = abs(delta_omega) * c_inv_sqrt_var_omega * PHI_ENERGY;

        float w_luma = SVGF_PHI_L * sqrt(z_bures * z_bures + z_energy * z_energy);

        // ---- 组合权重 -----------------------------------------------
        float w0 = w_kernel * (1.0 + w_luma) * exp2(-(w_geometry + w_luma) * LOG2_E);

        // ---- 累积 ----------------------------------------------------
        accumulate_alice(accumAlice, sample_alice, w0);
        sumWeight += w0;
        sumVarEnergy += w0 * w0 * sample_var_est;
    }

    float inv_sumWeight = 1.0 / sumWeight;

    // ---- 归一化并输出 ----------------------------------------------------
    accumAlice = scale_alice(accumAlice, inv_sumWeight);

    #if ENABLE_GAUSSIAN_FILTER == 1
    #ifndef FINAL_DENOISE_PASS
    accumAlice.aliceY.w *= (2.0 * geomValid - 1.0);
    #endif
    #endif

    float varEnergyOut = sumVarEnergy * inv_sumWeight * inv_sumWeight;
    out_light_sample = vec4(packAlice(accumAlice), varEnergyOut);

    #if FINAL_DENOISE_PASS
    out_light_sample_blurred = vec4(packAlice(accumAlice), 0.0);
    #endif
}
