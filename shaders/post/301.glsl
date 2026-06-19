#version 430 compatibility

// ===========================================================================
// Pass 301: 屏幕空间模糊滤波器 (Screen-Space Blur Filter)
// ===========================================================================
// 管线位置: 在 à-trous SH 滤波管线中的一环，专注于粗糙度感知的屏幕空间
//           各向异性模糊，通常用于漫反射的补充平滑。
//
// 通过 composite5x + R0/STEP 宏进行多级迭代 (与 300.glsl 共享机制)
//
// 滤波器特性:
//   - 各向异性核: 沿反射平面的屏幕空间投影方向拉伸
//   - 粗糙度门控: 使用 GetRoughnessWeight 对不同 roughness 的像素降权
//   - 深度感知: 近处表面锐利，远处/掠射角表面更模糊
//   - 法线感知: 动态法线一致性阈值
//
// 与 300.glsl (SH SVGF) 的对比:
//   - 301 直接在 vec3 颜色空间操作 (而非 SH 空间)
//   - 301 使用 roughness 辅助权重
//   - 301 的模糊因子随表面距离和法线变化自动调节
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

uniform sampler2D colortex3;  // RGBA32F: 世界空间法线 (.xyz) + 第二射线距离 (.w)
uniform sampler2D colortex4;  // RGBA32F: 世界空间位置 (.xyz) + 深度/距离 (.w)
uniform sampler2D colortex5;  // RGBA16F: 待滤波颜色 (.xyz) + roughness (.w)

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;

uniform float near;
uniform float far;
uniform vec2 resolution;
uniform int worldTime;

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

// 法线一致性基础指数 — 实际指数随距离动态缩放
const float NORMAL_PARAM = 8.0;

// 位置/平面距离灵敏度 (越小越容易跨平面模糊)
const float POSITION_PARAM = 1.0;

const float LUMINANCE_PARAM = 4.0;

// ---------------------------------------------------------------------------
// 边缘停止 / 权重函数
// ---------------------------------------------------------------------------

// 法线权重: 动态指数 = S (由外部传入的 normal_factor)
float svgfNormalWeight(vec3 centerNormal, vec3 normal, float S) {
    return pow(max(dot(centerNormal, normal), 0.0), S);
}

// 位置权重: 基于采样点与中心平面的有符号距离
float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

// ---------------------------------------------------------------------------
// 各向异性轴缩放 (Schied 2017 的 K 函数)
// ---------------------------------------------------------------------------
// 计算沿屏幕空间方向 (A) 相对于视线方向 (B) 的各向异性拉伸因子
float computeAnisotropicAxisScale(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    vec3 x = an * B - bn * A;
    return abs(bn) * sqrt(max(1.0 - an * an, 0.0)) / max(0.01, dot(x, x));
}

// ---------------------------------------------------------------------------
// 粗糙度权重: 只有 roughness 相近的像素才充分参与模糊
// ---------------------------------------------------------------------------
float GetRoughnessWeight(float roughness0, float roughness) {
    float norm = roughness0 * roughness0 * 0.99 + 0.01;
    float w = abs(roughness0 - roughness) * (1.0 / norm);
    return clamp(1.0 - w, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------

vec3 prevScreenPos;
bufferData info_;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
}

/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 color;

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    idx = getIdx(uvec2(gl_FragCoord.xy));

    // ---- 跳过无效像素 (天空/未命中) --------------------------------------
    info_ = denoiseBuffer.data[idx];
    if (info_.distance < -0.5) {
        return;
    }

    // ---- 中心像素数据 ----------------------------------------------------
    ivec2 pix = ivec2(gl_FragCoord.xy);

    // 几何法线 (来自 denoise buffer，用于构建反射平面)
    vec3 geoNormal = diffuseIllumiantionBuffer.data[idx].normal2;

    // 表面法线及其长度 (长度用于检测细节/粗糙度变化)
    vec4 centerNormal_ = texelFetch(colortex3, pix, 0);
    vec3 centerNormal = normalize(centerNormal_.xyz);
    float second_ray_distance = length(centerNormal_.xyz);

    // 中心位置与颜色
    vec4 centerPos = texelFetch(colortex4, pix, 0);
    vec4 centerColor = texelFetch(colortex5, pix, 0);

    // ---- 各向异性轴计算 (基于反射平面的屏幕投影) ---------------------------
    // planeN: 反射平面法线 (视线方向关于表面法线的反射)
    vec3 planeN = -reflect(centerNormal, geoNormal);

    // 计算沿屏幕 X 和 Y 轴的拉伸因子
    vec3 viewDir = cross(camX_global, camY_global);
    float axis_A = 0.75 + max(computeAnisotropicAxisScale(viewDir, camX_global, planeN), 0.0);
    float axis_B = 0.75 + max(computeAnisotropicAxisScale(viewDir, camY_global, planeN), 0.0);
    axis_A *= axis_A;
    axis_B *= axis_B;

    // ---- 动态模糊因子 ----------------------------------------------------
    // blur_factor: 随表面距离增大而增大 (远处更模糊)
    //             = 1 - exp(-0.25 * depth) / (3 + 0.5 * secondary_ray_dist)
    float blur_factor = (1.0 - exp(-0.25 * centerPos.w))
                      / (3.0 + 0.5 * second_ray_distance);

    // normal_factor: 法线约束力度随深度减小 (近处严格, 远处宽松)
    float normal_factor = (1.0 - exp(-0.1 * centerPos.w)) * NORMAL_PARAM;

    // ---- à-trous 采样 ----------------------------------------------------
    ivec2 samplePos;
    ivec2 texSize = textureSize(colortex3, 0);

    // 随机旋转 (减少走样)
    float theta = 2.0 * PI * rand(vec2(pix + 11 + R0));

#if STEP != 1
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
#endif

    vec3 A = vec3(0.0);     // 累积加权颜色
    float w = 0.0;           // 累积总权重

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;  // 中心像素最后单独添加
            }

            // à-trous 采样位置 (带旋转抖动)
#if STEP == 1
            samplePos = pix + ivec2(i, j);
#else
            samplePos = pix + ivec2(rotM * vec2(i, j));
#endif

            // ---- 读取邻域数据 --------------------------------------------
            vec4 c = texelFetch(colortex5, samplePos, 0);

            // 粗糙度权重: 防止光滑反射与粗糙反射相互污染
            float rW = GetRoughnessWeight(centerColor.w, c.w);

            vec4 B = texelFetch(colortex3, samplePos, 0);

            // ---- 计算各分量权重 ------------------------------------------
            // 深度权重 (各向异性拉伸)
            float w1 = exp(-blur_factor * (axis_A * i * i + axis_B * j * j));

            // 位置/平面距离权重
            float w_pos = exp(-POSITION_PARAM * abs(dot(
                centerPos.xyz - texelFetch(colortex4, samplePos, 0).xyz,
                centerNormal)));

            // 法线一致性权重
            float w_norm = svgfNormalWeight(centerNormal, normalize(B.xyz),
                                            normal_factor);

            // 组合权重 + 边界裁剪
            float w0 = rW * w_pos * w_norm * w1
                     * float(samplePos == clamp(samplePos, vec2(0), texSize));

            A += c.xyz * w0;
            w += w0;
        }
    }

    // ---- 中心像素 (权重 = 1, 不受各滤波器影响) ----------------------------
    float w0 = 1.0;
    A += centerColor.xyz * w0;
    w += w0;

    // ---- NaN 保护与归一化输出 --------------------------------------------
    if (any(isnan(A))) A = vec3(0.0);
    color = vec4(A / max(w, 0.01), centerColor.w);
}
