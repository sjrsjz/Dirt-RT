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
const int colortex5Format = RGBA32F; // (θ, β) 双对偶向量需要高动态范围
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
//   colortex5 存储双对偶向量 (θ, β) ∈ R⁴ — 用于 Jeffreys 散度计算
//
//   边缘停止策略：
//     w_geometry = NORMAL_POWER · (1 - n1·n2) + depth_term    · 法线 + 深度
//     w_stat     = Jeffreys(L1, L2) / mcVariance               · 信息几何散度
//     weight     = w_kernel × exp(‑w_geometry ‑ w_luma)
//
//   其中 Jeffreys 散度替代了原 ad‑hoc 马氏距离，提供严格的信息几何意义下的
//   双样本分布差异度量。mcVariance 保持 SVGF 的方差引导自适应行为：
//   噪点多时散度被压低（宽松模糊），收敛后散度被放大（严苛保护细节）。
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
// colortex5: 双对偶向量 (θ, β) — 用于 Jeffreys 散度计算
uniform sampler2D colortex5;

// ---------------------------------------------------------------------------
// 输出声明
// ---------------------------------------------------------------------------

/* RENDERTARGETS: 4,5 */
layout(location = 0) out mediump vec4 out_light_sample; // 压缩光照样本输出（ALICE + 方差 + 权重）
layout(location = 1) out mediump vec4 out_dual_vector; // colortex5: 更新后的 (θ, β) 双对偶向量

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

// 法线边缘停止力度 (值越大，法线边界越锐利)
#define NORMAL_POWER SVGF_NORMAL_POWER

// 深度边缘停止灵敏度系数（已归一化到"像素等效"单位）
// 实际深度项会被 pixel_footprint 归一化，因此 k≈1 代表约 1px 的深度差
#define POSITION_PARAM SVGF_POSITION_PARAM

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

void unpackLightSample(ivec2 coord, out vec3 pos, out vec3 normal, out SH sh, out float weight, out vec4 dual) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0); // 几何信息
    vec4 sample_data1 = texelFetch(colortex4, coord, 0); // 光照样本信息
    pos = sample_data0.xyz;
    normal = decodeNormal(sample_data0.w);
    sh = unpackSH(sample_data1.x, sample_data1.y, sample_data1.z);
    weight = sample_data1.w;
    dual = texelFetch(colortex5, coord, 0); // 双对偶向量 (θ, β)
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
    vec4 center_dual; // (θ, β) 双对偶向量
    unpackLightSample(pix, center_pos, center_normal, center_sh, center_weight, center_dual);
    float sqrt_center_weight = min(sqrt(center_weight), 100.0);

    // 像素的世界空间 footprint，用于距离无关的深度边缘停止
    float dist_to_cam = max(length(center_pos - camPos), 0.001);
    float inv_pixel_footprint = 1 / (POSITION_PARAM * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0; // 总权重（中心像素初始权重=1）
    SH accumulatedSH = center_sh; // 加权和，最后除以 sumWeight 得到平均

    // ---- à‑trous 核半径定义 -----------------------------------------------
    #define KERNAL_R 1          // 3×3 核半径，步长由 R0 定义

    #if STEP <= 3
    // ---- 抖动旋转 ---------------------------------------------------------
    float theta = 2.0 * PI * rand(vec2(pix + 10 + R0 + center_weight)); // 基于像素坐标和权重的随机旋转，避免固定模式
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    const float axis_A = 1.0;
    const float axis_B = 1.0;
    #else
    // ---- 各向异性轴计算 ---------------------------------------------------
    vec3 viewDir = cross(camX_global, camY_global); // 视线方向
    float rawAxisA = computeAnisotropicAxisScale(viewDir, camX_global, center_normal);
    float rawAxisB = computeAnisotropicAxisScale(viewDir, camY_global, center_normal);
    // 钳制最小值并平方，使其与 i²/j² 项匹配
    float axis_A = 1.0 / max(rawAxisA, 0.5);
    float axis_B = 1.0 / max(rawAxisB, 0.5);
    axis_A *= axis_A;
    axis_B *= axis_B;
    #endif

    // B‑样条权重核（中心 1.0, 十字 0.66667, 对角 0.166667）
    float hw[3] = float[](1.0, 0.66667, 0.166667);

    SH sample_sh;
    vec3 sample_world_pos, sample_normal;
    vec4 sample_dual; // 采样点的 (θ, β)
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

            unpackLightSample(sample_coord, sample_world_pos, sample_normal, sample_sh, sample_weight, sample_dual);

            // ---- 深度权重 -------------------------------------------------
            // 采样点到中心平面的垂直距离，归一化到屏幕空间
            vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
            float k = abs(dot(delta, center_normal));
            float delta2 = dot(delta, delta);
            // 深度项 = k × 各向异性拉伸（i²/j² 加权）
            float depthTerm = R0 * R0 * k * (axis_A * float(i * i) + axis_B * float(j * j));

            // ---- 法线权重 -------------------------------------------------
            float w_geometry = NORMAL_POWER * (1.0 - dot(center_normal, sample_normal)) + depthTerm;

            // ---- ALICE Jeffreys 散度统计权重 (替换原 ad-hoc 马氏距离) ---------
            // 调和累积加权: W_eff = (N1·N2)/(N1+N2)，低SPP软包容，高SPP严拒绝
            float divergence = alice_weighted_jeffreys_with_N(
                    center_sh.shY, sample_sh.shY,
                    center_dual, sample_dual,
                    center_weight, sample_weight);
            float w_luma = sqrt(divergence * center_sh.shY.w * sample_sh.shY.w); // 将散度从相对尺度放大到绝对尺度，避免低能量噪点被误判为高散度

            // ---- 组合权重 -------------------------------------------------
            float w0 = w_kernel * exp(-w_geometry) / (1 + 0.02 * sqrt_center_weight * R0 * w_luma); // w_luma 越大，权重越小

            // ---- 累积加权样本 ---------------------------------------------
            accumulate_SH(accumulatedSH, sample_sh, w0);
            sumWeight += w0;
        }
    }

    // ---- 时空方差混合 ------------------------------------------------
    float inv_sumWeight = 1.0 / sumWeight;

    // ---- 归一化并输出 ----------------------------------------------------
    accumulatedSH = scaleSH(accumulatedSH, inv_sumWeight);
    out_light_sample = vec4(packSH(accumulatedSH), center_weight);

    // ---- 更新双对偶向量 (θ, β) → colortex5，供下一 à‑trous 级使用 ---------
    if (accumulatedSH.shY.w < 1e-10) {
        out_dual_vector = vec4(0.0);
    } else {
        out_dual_vector = packDualVectorFromEncoded(accumulatedSH.shY);
    }
}
