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

/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA16F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;

const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

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

// colortex0: 主颜色缓冲（仅用于分辨率查询等全局信息）
uniform sampler2D colortex0;

// colortex3: 压缩几何信息缓冲（世界空间法线 + 世界空间位置）
uniform sampler2D colortex3;
// colortex4: 压缩光照信息缓冲（ALICE + 方差 + 权重）
uniform sampler2D colortex4;

// ---------------------------------------------------------------------------
// 输出声明
// ---------------------------------------------------------------------------

/* RENDERTARGETS: 4 */
layout(location = 0) out mediump vec4 out_light_sample; // 压缩光照样本输出（ALICE + 方差 + 权重）

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

// SVGF 亮度停止的主灵敏度参数 (phi_l)
// (SVGF_PHI_L 直接引用 settings.glsl 定义)

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

// 各向异性滤波轴缩放因子 (Schied 2017 的 K 函数)
// 参数:
//   B, A : 屏幕空间的两个正交方向（如视线方向和屏幕 X 轴）
//   n    : 世界空间表面法线
// 返回:   沿该轴的拉伸因子（越小拉伸越大，即沿该方向越模糊）
float computeAnisotropicAxisScale(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    vec3 x = an * B - bn * A;
    float denom = max(dot(x, x), 0.01);
    return abs(bn) * sqrt(max(1.0 - an * an, 0.0)) / denom;
}

void unpackLightSample(ivec2 coord, out vec3 pos, out vec3 normal, out SH sh, out float weight) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0); // 几何信息
    vec4 sample_data1 = texelFetch(colortex4, coord, 0); // 光照样本信息
    pos = sample_data0.xyz;
    normal = decodeNormal(sample_data0.w);
    sh = unpackSH(sample_data1.x, sample_data1.y, sample_data1.z);
    weight = sample_data1.w;
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

void main() {
    // 获取纹理尺寸（用于边界裁剪）
    ivec2 texSize = textureSize(colortex3, 0) - 1;

    // 当前像素在去噪缓冲区中的索引
    uint idx = getIdx(uvec2(gl_FragCoord.xy));

    // 跳过天空像素（distance < 0 表示无几何体命中）
    bufferData info_ = denoiseBuffer.data[idx];
    if (info_.distance < -0.5) return;

    // ---- 中心像素基础数据 -------------------------------------------------
    ivec2 pix = ivec2(gl_FragCoord.xy);

    vec3 center_pos, center_normal;
    SH center_sh;
    float center_weight;
    unpackLightSample(pix, center_pos, center_normal, center_sh, center_weight);
    float sqrt_center_weight = min(sqrt(center_weight), 100.0);

    // 像素的世界空间 footprint，用于距离无关的深度边缘停止
    float dist_to_cam = max(length(center_pos - camPos), 0.001);
    float inv_pixel_footprint = 1 / (SVGF_POSITION_PARAM * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0; // 总权重（中心像素初始权重=1）
    float center_var_est = alice_radial_estimator_variance(center_sh.shY, center_weight);
    float sumVarEnergy = center_var_est; // 中心权重 a0 = 1，所以 a0^2 * var = var
    SH accumulatedSH = center_sh; // 加权和，最后除以 sumWeight 得到平均

    // ---- à‑trous 核半径定义 -----------------------------------------------
    #define KERNAL_R 1          // 3×3 核半径，步长由 R0 定义

    #if STEP <= 3
    // ---- 抖动旋转 ---------------------------------------------------------
    float theta = 2.0 * PI * rand(vec2(pix + R0 + center_weight)); // 基于像素坐标和权重的随机旋转，避免固定模式
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    const float axis_A = 1.0;
    const float axis_B = 1.0;
    #else
    // ---- 各向异性轴计算 ---------------------------------------------------
    vec3 viewDir = cross(camX_global, camY_global); // 视线方向
    float rawAxisA = computeAnisotropicAxisScale(viewDir, camX_global, center_normal);
    float rawAxisB = computeAnisotropicAxisScale(viewDir, camY_global, center_normal);
    // 钳制最小值并平方，使其与 i²/j² 项匹配
    float axis_A = 1.0 / max(rawAxisA, 0.75);
    float axis_B = 1.0 / max(rawAxisB, 0.75);
    axis_A *= axis_A;
    axis_B *= axis_B;
    #endif

    // B‑样条权重核（中心 1.0, 十字 0.66667, 对角 0.166667）
    float hw[3] = float[](1.0, 0.66667, 0.166667);

    SH sample_sh;
    vec3 sample_world_pos, sample_normal;
    ivec2 sample_coord;

    // ---- 主采样循环 --------------------------------------------------------
    for (int i = -KERNAL_R; i <= KERNAL_R; i++) {
        for (int j = -KERNAL_R; j <= KERNAL_R; j++) {
            if (i == 0 && j == 0) continue; // 中心像素已在累加器中
            // à‑trous 采样位置
            #if STEP <= 3
            sample_coord = pix + ivec2(round(rotM * vec2(i, j))); // 注意：加了 round 防止截断误差
            #else
            sample_coord = pix + R0 * ivec2(i, j);
            #endif

            // ---- 有效性检查 ------------------------------------------------
            float dist = denoiseBuffer.data[getIdx(uvec2(sample_coord))].distance;
            if (dist < -0.5) continue; // 天空
            if (sample_coord != clamp(sample_coord, ivec2(0), texSize)) continue; // 越界

            // 贴图空间权重
            float w_kernel = hw[abs(i)] * hw[abs(j)];
            float sample_weight;

            unpackLightSample(sample_coord, sample_world_pos, sample_normal, sample_sh, sample_weight);

            // ---- 深度权重 -------------------------------------------------
            // 采样点到中心平面的垂直距离，归一化到屏幕空间
            vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
            float k = abs(dot(delta, center_normal));
            float delta2 = dot(delta, delta);
            // 深度项 = k × 各向异性拉伸（i²/j² 加权）
            float depthTerm = R0 * R0 * k * (axis_A * float(i * i) + axis_B * float(j * j));

            // ---- 几何权重 -------------------------------------------------
            float w_geometry = SVGF_NORMAL_POWER * (1.0 - dot(center_normal, sample_normal)) + depthTerm;

            float sample_var_est = alice_radial_estimator_variance(sample_sh.shY, sample_weight);

            float sigma2 = max(center_var_est + sample_var_est, 1e-6);
            float delta_energy = abs(center_sh.shY.w - sample_sh.shY.w);
            float w_luma = SVGF_PHI_L * delta_energy * inversesqrt(sigma2);

            // ---- 组合权重 -------------------------------------------------
            float w0 = w_kernel * exp(-w_geometry - w_luma);

            // ---- 累积加权样本 ---------------------------------------------
            accumulate_SH(accumulatedSH, sample_sh, w0);
            sumWeight += w0;
            // 方差传播
            sumVarEnergy += w0 * w0 * sample_var_est;
        }
    }

    // ---- 时空方差混合 ------------------------------------------------
    float inv_sumWeight = 1.0 / sumWeight;

    // ---- 归一化并输出 ----------------------------------------------------
    accumulatedSH = scaleSH(accumulatedSH, inv_sumWeight);
    float varEnergyOut = sumVarEnergy / max(sumWeight * sumWeight, 1e-7);
    float intrinsicOut = alice_radial_variance(accumulatedSH.shY);
    float filtered_weight;
    if (intrinsicOut > 1e-10 && varEnergyOut > 1e-12) {
        filtered_weight = intrinsicOut / varEnergyOut;
    } else {
        // 黑场或极低能量退化情况，回退到原 ESS
        filtered_weight = center_weight;
    }
    out_light_sample = vec4(packSH(accumulatedSH), filtered_weight);
}
