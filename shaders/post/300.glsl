#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"

// ==========================================================================
// Pass 300: SVGF 空间滤波器 — 漫反射 (ALICE) 降噪
// ==========================================================================

// ===========================================================================
// SVGF 空间滤波器 — 漫反射 (ALICE) 降噪
// ===========================================================================
// 参考：Schied et al., “Spatiotemporal Variance-Guided Filtering”, HPG 2017
//
// 管线（6 级 à‑trous 迭代）：
//   composite50  STEP=1  R0=1   → 3×3 核，步长 1
//   composite52  STEP=2  R0=2   → 3×3 核，步长 2
//   composite53  STEP=3  R0=4   → 3×3 核，步长 4
//   composite54  STEP=4  R0=8   → 3×3 核，步长 8
//   composite55  STEP=5  R0=16  → 3×3 核，步长 16
//   composite56  STEP=6  R0=32  → 3×3 核，步长 32
//
// ALICE 光照编码 (Asymmetric Laplace Isomorphic Conic Encoding)：
//   colortex4 存储 ALICE 嵌入表示 vec4(v, ω) + CoCg + weight + variance
//
// ===========================================================================

// ---------------------------------------------------------------------------
// 输入纹理
// ---------------------------------------------------------------------------

// colortex3: 压缩几何信息缓冲（世界空间法线 + 世界空间位置）
uniform sampler2D colortex3;
// colortex4: 压缩光照信息缓冲（ALICE + 方差 + 权重）
uniform sampler2D colortex4;

// ---------------------------------------------------------------------------
// 输出声明
// ---------------------------------------------------------------------------

/* RENDERTARGETS: 4,5 */
layout(location = 0) out mediump vec4 out_light_sample; // 压缩光照样本输出（ALICE + 方差 + 权重）
layout(location = 1) out mediump vec4 out_light_sample_blurred; // 压缩光照样本输出（ALICE + 方差 + 权重）

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

// SVGF 亮度停止的主灵敏度参数 (phi_l)
// (SVGF_PHI_L 直接引用 settings.glsl 定义)

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

void unpackLightSample(ivec2 coord, out vec3 pos, out vec3 normal, out SH sh, out float variance) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0); // 几何信息
    vec4 sample_data1 = texelFetch(colortex4, coord, 0); // 光照样本信息
    pos = sample_data0.xyz;
    normal = decodeNormal(sample_data0.w);
    sh = unpackSH(sample_data1.x, sample_data1.y, sample_data1.z);
    variance = sample_data1.w;
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

void main() {
    // 获取纹理尺寸（用于边界裁剪）
    ivec2 texSize = textureSize(colortex3, 0) - 1;

    // ---- 中心像素基础数据 -------------------------------------------------
    ivec2 pix = ivec2(gl_FragCoord.xy);

    vec3 center_pos, center_normal;
    SH center_sh;
    float center_var_est;
    unpackLightSample(pix, center_pos, center_normal, center_sh, center_var_est);

    // 跳过天空像素 — 方差被 swap2 复用作天空 mask
    if (center_var_est < 0.0) return;

    // 高斯曲率标记: omega < 0 → 几何不可靠, geomValid=0 跳过几何权重
    float geomValid = float(center_sh.shY.w >= 0.0);
    center_sh.shY.w = abs(center_sh.shY.w);

    // 像素的世界空间 footprint，用于距离无关的深度边缘停止
    float dist_to_cam = max(length(center_pos), 0.001);
    float inv_pixel_footprint = 1.0 / (SVGF_POSITION_PARAM * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0; // 总权重（中心像素初始权重=1）
    float sumVarEnergy = center_var_est; // 中心权重 a0 = 1，所以 a0^2 * var = var
    SH accumulatedSH = center_sh; // 加权和，最后除以 sumWeight 得到平均

    #if STEP >= 4
    // ---- 抖动旋转 ---------------------------------------------------------
    float theta = 2.0 * PI * rand(vec2(pix + R0)); // 基于像素坐标和权重的随机旋转，避免固定模式
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    // B‑样条权重核（中心 1.0, 十字 0.66667）
    float hw[2] = float[](1.0, 0.66667);

    SH sample_sh;
    vec3 sample_world_pos, sample_normal;
    ivec2 sample_coord;

    // 方差预滤波使得下面的 sigma2 不再会导致降噪器彻底崩溃
    float sigma2 = max(center_var_est, 1e-8);
    float inv_sqrt_sigma2 = SVGF_PHI_L * inversesqrt(sigma2);

    // ---- 主采样循环 --------------------------------------------------------
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue; // 中心像素已在累加器中
            // à‑trous 采样位置
            #if STEP >= 4
            sample_coord = pix + ivec2(round(rotM * vec2(i, j))); // 注意：加了 round 防止截断误差
            #else
            sample_coord = pix + R0 * ivec2(i, j);
            #endif

            // ---- 有效性检查 ------------------------------------------------
            if (sample_coord != clamp(sample_coord, ivec2(0), texSize)) continue; // 越界

            // 贴图空间权重
            float w_kernel = hw[abs(i)] * hw[abs(j)];

            float sample_var_est;
            unpackLightSample(sample_coord, sample_world_pos, sample_normal, sample_sh, sample_var_est);

            // 天空检查 — 方差被 swap2 复用作天空 mask (负值 = 天空)
            if (sample_var_est < 0.0) continue;

            // 曲率标记: 仅中心点决定几何权重有效性
            sample_sh.shY.w = abs(sample_sh.shY.w);

            vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
            float depthTerm = abs(dot(delta, center_normal));

            float raw_geom = SVGF_NORMAL_POWER * (1.0 - dot(center_normal, sample_normal)) + depthTerm * geomValid;
            float w_geometry = raw_geom;

            // 在未使用方差预滤波的情况下，只有同时考虑到 sample_var_est 和 center_var_est 才能得到合理的亮度权重，使得降噪器不崩溃
            // 但是在使用了方差预滤波后，sample_var_est 的修正作用已经减弱，并且会带来极其严重的频闪副作用，因此这里直接使用 center_var_est 作为亮度权重的方差估计值
            // float sigma2 = max(center_var_est + sample_var_est, 1e-8);

            float delta_energy = length(center_sh.shY.xyz - sample_sh.shY.xyz);
            float w_luma = delta_energy * inv_sqrt_sigma2;

            // ---- 组合权重 -------------------------------------------------
            float w0 = w_kernel * (1 + w_luma) * exp2(-(w_geometry + w_luma) * LOG2_E);

            // ---- 累积加权样本 ---------------------------------------------
            accumulate_SH(accumulatedSH, sample_sh, w0);
            sumWeight += w0;
            // 方差传播
            sumVarEnergy += w0 * w0 * sample_var_est;
        }
    }

    float inv_sumWeight = 1.0 / sumWeight;

    // ---- 归一化并输出 ----------------------------------------------------
    accumulatedSH = scaleSH(accumulatedSH, inv_sumWeight);
    // 传递几何有效性 mask 到下一级 à-trous (最终 pass 不传递)
    #ifndef FINAL_DENOISE_PASS
    accumulatedSH.shY.w *= (2.0 * geomValid - 1.0);
    #endif
    float varEnergyOut = sumVarEnergy * inv_sumWeight * inv_sumWeight;
    out_light_sample = vec4(packSH(accumulatedSH), varEnergyOut);

    #if STEP == 6
    // --- 输出模糊后的结果（仅在最后一步） ------------------------------------
    out_light_sample_blurred = vec4(packSH(accumulatedSH), 0.0);
    #endif
}
