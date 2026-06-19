#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"

// ==========================================================================
// Pass 300: SVGF 空间滤波器 — 漫反射（SH）降噪
// ==========================================================================

/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA16F;
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
// SVGF 空间滤波器 — 漫反射（SH）降噪
// ===========================================================================
// 参考：Schied et al., "Spatiotemporal Variance-Guided Filtering", HPG 2017
//
// 管线（6 级 à‑trous 迭代）：
//   composite50  STEP=1  R0=1   → 3×3 核，步长 1
//   composite52  STEP=2  R0=2   → 3×3 核，步长 2
//   composite53  STEP=3  R0=4   → 3×3 核，步长 4
//   composite54  STEP=4  R0=8   → 3×3 核，步长 8
//   composite55  STEP=5  R0=16  → 3×3 核，步长 16
//   composite56  STEP=6  R0=32  → 3×3 核，步长 32
//
// 自定义光照数据格式 — SH（非球谐，而是“方向 + 环境”矢量能量模型）：
//   SH.shY  = vec4(dir * Y, Y)   · 方向因子 × 总亮度   + 总亮度标量
//   SH.CoCg = vec2(Co, Cg)       · 色度 (YCoCg 空间)
//
//   解码为 RGB：
//     T = Y - Cg/2
//     R = B + Co
//     G = T + Cg
//     B = T - Co/2
//
//   边缘停止策略（每个邻域样本，1 次 pow + 1 次 exp）：
//     weight = w_kernel × pow(dot(n1,n2), NORMAL_POWER)     · 法线相似度
//             × exp(‑depth_term)                           · 深度差异
//             × exp(‑mahalanobis_distance)                 · 亮度‑方向联合统计距离
//
//   深度项：depth_term = k · (axis_A·i² + axis_B·j²)
//     k = 采样点到中心平面的带符号距离 → 除以像素足迹实现屏幕空间归一化
//     各向异性拉伸（axis_A, axis_B）使得在掠射角时沿表面方向扩展核，
//     而垂直于表面的方向保持紧致，防止跨深度边缘模糊。
//
//   亮度‑方向项：
//     基于 SH 模型的固有散度方差 (light_sigma = Y² - |dir·Y|²)
//     与时域蒙特卡洛方差 (tex.z) 共同构建马哈拉诺比斯距离，
//     实现方差引导的自适应停止：噪点多时宽松模糊，收敛后紧致保护细节。
// ===========================================================================

// ---------------------------------------------------------------------------
// 输入纹理
// ---------------------------------------------------------------------------

// colortex0: 主颜色缓冲（仅用于分辨率查询等全局信息）
uniform sampler2D colortex0;

// colortex3: RGBA32F
//   .xyz = 世界空间法线 (归一化)
//   .w   = 二次射线距离（镜面反射相关，本 pass 未直接使用）
uniform sampler2D colortex3;

// colortex4: RGBA32F
//   .xyz = 世界空间表面位置
//   .w   = 深度/距离度量（备用）
uniform sampler2D colortex4;

// colortex5: RGBA16F — 存储 SH.shY（方向光 + 环境光）
//   使用半精度打包，但此处直接读取为 vec4
//   .xyzw = shY: (dir·Y, Y)
uniform mediump sampler2D colortex5;

// colortex6: RGBA16F — 存储 SH.CoCg + 方差 + 历史权重
//   .xy = CoCg 色度
//   .z  = 时域方差估计（经预平滑处理后的稳定方差）
//   .w  = 历史累积权重（用于时域重投影混合）
uniform mediump sampler2D colortex6;

// ---------------------------------------------------------------------------
// 输出声明
// ---------------------------------------------------------------------------

/* RENDERTARGETS: 5,6 */
layout(location = 0) out mediump vec4 out_shY; // → colortex5: 滤波后的 SH.shY
layout(location = 1) out mediump vec4 out_CoCg; // → colortex6: .xy = 滤波后 CoCg, .zw = (方差, 权重)

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

// Welford 在线方差更新 — 单遍扫描计算邻域 SH 的加权方差
// 参数:
//   M_n     : 当前加权和 (avg_SH)
//   D_n     : 当前方差
//   X       : 新样本
//   h_w     : 历史权重总和
//   w       : 新样本的权重
// 返回:      更新后的方差 D_{n+1}
float updateVariance(SH M_n, float D_n, SH X, float h_w, float w) {
    vec4 diff = X.shY - M_n.shY / h_w; // 新样本与当前均值的差
    float t = 1.0 / (h_w + w);
    return (D_n * h_w + dot(diff, diff) * w * t) * t;
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

    lowp vec3 centerNormal = texelFetch(colortex3, pix, 0).xyz; // 世界空间法线
    vec3 centerPos = texelFetch(colortex4, pix, 0).xyz; // 世界空间位置

    // 像素的世界空间足迹，用于距离无关的深度边缘停止
    // footprint ≈ distance / resolution.y  远处像素足迹大，归一化后深度敏感度保持一致
    float dist_to_cam = max(length(centerPos - camPos), 0.01);
    float inv_pixel_footprint = 1 / (POSITION_PARAM * max(dist_to_cam / float(resolution_global.y), 0.0001));

    // 中心 SH 数据
    SH centerSH;
    centerSH.shY = texelFetch(colortex5, pix, 0);
    vec4 tex = texelFetch(colortex6, pix, 0);
    centerSH.CoCg = tex.xy; // 色度
    // tex.z = 时域方差 (已预平滑)
    // tex.w = 历史权重

    // ---- 各向异性轴计算 ---------------------------------------------------
    vec3 viewDir = cross(camX_global, camY_global); // 视线方向
    float rawAxisA = computeAnisotropicAxisScale(viewDir, camX_global, centerNormal);
    float rawAxisB = computeAnisotropicAxisScale(viewDir, camY_global, centerNormal);
    // 钳制最小值并平方，使其与 i²/j² 项匹配
    float axis_A = 1.0 / max(rawAxisA, 0.5);
    float axis_B = 1.0 / max(rawAxisB, 0.5);
    axis_A *= axis_A;
    axis_B *= axis_B;

    // ---- 中心点光照散度方差（light_sigma）----------------------------------
    // light_sigma = Y² - |v|²  是模型内在的角分布度量
    float sigma_sq_center = centerSH.shY.w * centerSH.shY.w
            - dot(centerSH.shY.xyz, centerSH.shY.xyz);
    sigma_sq_center = max(sigma_sq_center, 0.0);

    // 时域蒙特卡洛方差（已预平滑）
    float var_MC = max(tex.z, 0.0);
    float mcVariance = SVGF_PHI_L * var_MC + 1e-2; // 加了小偏移避免除零
    float inv_mcVariance = 1.0 / mcVariance;
    
    float base_dirTolerance = sigma_sq_center + mcVariance + 0.125;

    // ---- 初始化累积器 ----------------------------------------------------
    float spatialVar = 0.0; // 空间方差 (Welford 算法)
    float sumWeight = 1.0; // 总权重（中心像素初始权重=1）
    SH accumulatedSH = centerSH; // 加权和，最后除以 sumWeight 得到平均

    // ---- à‑trous 核半径定义 -----------------------------------------------
    #define KERNAL_R 1          // 3×3 核半径，步长由 R0 定义

    #if STEP > 1 && STEP <= 2
    // ---- 抖动旋转 ---------------------------------------------------------
    float theta = 2.0 * PI * rand(vec2(pix + 10 + R0 + tex.z));
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    // B‑样条权重核（中心 1.0, 十字 0.66667, 对角 0.166667）
    float hw[3] = float[](1.0, 0.66667, 0.166667);

    SH sampleSH;
    ivec2 samplePos;

    // ---- 主采样循环 --------------------------------------------------------
    for (int i = -KERNAL_R; i <= KERNAL_R; i++) {
        for (int j = -KERNAL_R; j <= KERNAL_R; j++) {
            if (i == 0 && j == 0) continue; // 中心像素已在累加器中
            // à‑trous 采样位置
            #if STEP > 1 && STEP <= 2
            samplePos = pix + ivec2(round(rotM * vec2(i, j))); // 注意：加了 round 防止截断误差
            #else
            samplePos = pix + R0 * ivec2(i, j);
            #endif

            // ---- 有效性检查 ------------------------------------------------
            float dist = denoiseBuffer.data[getIdx(uvec2(samplePos))].distance;
            if (dist < -0.5) continue; // 天空
            if (samplePos != clamp(samplePos, ivec2(0), texSize)) continue; // 越界

            // 贴图空间权重
            float w_kernel = hw[abs(i)] * hw[abs(j)];

            // 读取邻域 SH 数据
            sampleSH.shY = texelFetch(colortex5, samplePos, 0);
            sampleSH.CoCg = texelFetch(colortex6, samplePos, 0).xy;

            // 邻域法线
            vec3 sampleNormal = texelFetch(colortex3, samplePos, 0).xyz;

            // ---- 法线权重 -------------------------------------------------
            float w_n = pow(max(dot(centerNormal, sampleNormal), 0.0), NORMAL_POWER);

            // ---- 深度权重 -------------------------------------------------
            // 采样点到中心平面的垂直距离，归一化到屏幕空间
            vec3 delta = (texelFetch(colortex4, samplePos, 0).xyz - centerPos) * inv_pixel_footprint;
            float k = abs(dot(delta, centerNormal)) ;
            float delta2 = dot(delta, delta);
            // 深度项 = k × 各向异性拉伸（i²/j² 加权）
            float depthTerm = R0 * R0 * k * (axis_A * float(i * i) + axis_B * float(j * j));
            float w_depth = exp(-depthTerm);

            // ---- 亮度‑方向统计权重（马哈拉诺比斯距离）-----------------------
            // 1. 采样点的光照散度方差
            float sigma_sq_sample = sampleSH.shY.w * sampleSH.shY.w
                    - dot(sampleSH.shY.xyz, sampleSH.shY.xyz);
            sigma_sq_sample = max(sigma_sq_sample, 0.0);

            // 2. 总强度（标量亮度）的统计距离
            float lumaDiff = centerSH.shY.w - sampleSH.shY.w;
            float lumaDistSq = (lumaDiff * lumaDiff) * inv_mcVariance;

            // 3. 方向矢量的统计距离
            vec3 dirDiff = centerSH.shY.xyz - sampleSH.shY.xyz;
            float dirDiffSq = dot(dirDiff, dirDiff);
            float dirTolerance = base_dirTolerance + sigma_sq_sample;
            float dirDistSq = dirDiffSq / dirTolerance;

            // 4. 联合马哈拉诺比斯距离
            float inv_w_luma = 1 + delta2 * (lumaDistSq + dirDistSq);

            // ---- 组合权重 -------------------------------------------------
            float w0 = w_kernel * w_n * w_depth / inv_w_luma;

            // ---- 累积加权样本 ---------------------------------------------
            spatialVar = updateVariance(accumulatedSH, spatialVar, sampleSH, sumWeight, w0);
            accumulate_SH(accumulatedSH, sampleSH, w0);
            sumWeight += w0;
        }
    }

    // // ---- 时空方差混合 ------------------------------------------------
    tex.z = tex.z * 0.875 + spatialVar * 0.125;

    // ---- 归一化并输出 ----------------------------------------------------
    accumulatedSH = scaleSH(accumulatedSH, 1.0 / sumWeight); // 加权和 → 加权平均

    out_shY = accumulatedSH.shY;
    out_CoCg = vec4(accumulatedSH.CoCg, tex.zw); // 色度 + 方差 + 历史权重
    if (any(isnan(out_shY))) out_shY = vec4(0.0);
    if (any(isnan(out_CoCg))) out_CoCg = vec4(0.0);
}
