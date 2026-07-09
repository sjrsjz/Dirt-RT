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

    center_var_est = max(center_var_est, 1e-10);

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
    float geomValid = 1.0;
    #endif

    float dist_to_cam = max(length(center_pos), 0.001);
    float inv_pixel_footprint = 1.0 / (SVGF_POSITION_PARAM
        * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0;
    float sumVarEnergy = center_var_est;
    AliceEncoding accumAlice = center_alice;

    #if STEP >= 4
    float theta = 2.0 * PI * rand(vec2(pix + R0));
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    float hw[2] = float[](1.0, 0.66667);

    // ---- 主采样循环 --------------------------------------------------------
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            #if STEP >= 4
            ivec2 sample_coord = pix + ivec2(round(rotM * vec2(i, j)));
            #else
            ivec2 sample_coord = pix + R0 * ivec2(i, j);
            #endif

            if (sample_coord != clamp(sample_coord, ivec2(0), texSize)) continue;

            float w_kernel = hw[abs(i)] * hw[abs(j)];

            // ---- 加载邻居样本 ----------------------------------------------
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
            //
            // 原方案: w_luma = |Δv| / σ_est
            //   问题 1: 只看均值差, 丢弃分布形状差 (κ, ω 差异)
            //   问题 2: 不看能量差, 阴影边界处无信号级停止
            //
            // 新方案: 联合 z-score = √(z_bures² + z_energy²)
            //   z_bures:  Bures 距离 (均值差 + 协方差形状差) / 估计量标准差
            //   z_energy: |Δω| / 径向估计量标准差
            //
            // Bures vs Jeffreys (对称 KL):
            //   Jeffreys: β₁ω₂ + β₂ω₁  →  β ∝ 1/(1-κ²) → κ→1 指数爆炸
            //   Bures:    (σ_⊥,c - σ_⊥,s)²  →  σ_⊥ ∝ √(1-κ²) → κ→1 多项式收敛
            //   两个同向尖锐分布: Jeffreys 爆炸 (错误), Bures 趋零 (正确)

            vec4 s_enc = sample_alice.aliceY;
            float s_len_v = length(s_enc.xyz);
            float s_kappa = alice_kappa(s_len_v, s_enc.w);

            // Bures 距离 (同轴近似, n=3)
            float d_bures_sq = alice_bures_distance_sq(c_enc, c_kappa, s_enc, s_kappa);

            // v-space z-score: 分布差异是否超过估计噪声?
            float z_bures = sqrt(d_bures_sq) * c_inv_sqrt_var;

            // ω-space z-score: 能量差异是否超过估计噪声?
            float delta_omega = c_omega - s_enc.w;
            float z_energy = abs(delta_omega) * c_inv_sqrt_var_omega * PHI_ENERGY;

            // 联合 z-score (欧氏范数, 无额外参数)
            float w_luma = SVGF_PHI_L * sqrt(z_bures * z_bures + z_energy * z_energy);

            // ---- 组合权重 (形式不变) ---------------------------------------
            float w0 = w_kernel * (1.0 + w_luma) * exp2(-(w_geometry + w_luma) * LOG2_E);

            // ---- 累积 ------------------------------------------------------
            accumulate_alice(accumAlice, sample_alice, w0);
            sumWeight += w0;
            sumVarEnergy += w0 * w0 * sample_var_est;
        }
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
