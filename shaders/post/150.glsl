#version 430 compatibility

// ===========================================================================
// Pass 150: SH 空间方差引导滤波器 (Legacy Variant)
// ===========================================================================
// 这是一个较旧版本的空间去噪 pass，直接在 SH 域操作，采用:
//   - 5×5 双边核 (法线 + 位置 + SH 统计距离)
//   - 基于方差的异常值抑制 (diff > sigma → 替换为邻域均值)
//
// 与 300.glsl (当前主 SVGF 管线) 的对比:
//   - 150 使用固定 5×5 核，没有 à-trous 多尺度迭代
//   - 150 在 SH 空间协方差上做阈值检测，而非像素级权重衰减
//   - 150 直接读写 colortex5/6，不经过 diffuseIllumiantionData 结构
//
// 注意: 此 pass 当前可能未被管线使用或作为备选路径保留。
//       主要空间滤波现在由 300.glsl (SVGF à-trous) 处理。
// ===========================================================================

#define DIFFUSE_BUFFER_MIN

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

in vec2 texCoord;

uniform sampler2D colortex0;
uniform sampler2D colortex3;  // 世界空间法线
uniform sampler2D colortex4;  // 世界空间位置
uniform sampler2D colortex5;  // SH.shY
uniform sampler2D colortex6;  // SH.CoCg + 方差 + 权重

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

const float NORMAL_PARAM = 4.0;
const float POSITION_PARAM = 4.0;
const float LUMINANCE_PARAM = 4.0;

// ---------------------------------------------------------------------------
// 边缘停止权重
// ---------------------------------------------------------------------------

float svgfNormalWeight(vec3 centerNormal, vec3 normal) {
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    return exp2(-POSITION_PARAM * LOG2_E * abs(dot(pixelPos - centerPos, normal)));
}

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------

vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
}

/* RENDERTARGETS: 5,6 */
layout(location = 0) out vec4 shY;
layout(location = 1) out vec4 CoCg;

// ---------------------------------------------------------------------------
// 方差估计 (旧版 SH 协方差方法)
// ---------------------------------------------------------------------------
float updateVariance(SH M_n, float D_n, SH X_nplus1, float w) {
    vec2 diff_CoCg = X_nplus1.CoCg - M_n.CoCg;
    vec4 diff_shY = X_nplus1.shY - M_n.shY;
    return w * (D_n + (dot(diff_CoCg, diff_CoCg)
                + dot(diff_shY, diff_shY)) / (1.0 + w)) / (1.0 + w);
}

// ---------------------------------------------------------------------------
// 从 colortex 直接读取 SH (绕过 diffuseIllumiantionData 结构)
// ---------------------------------------------------------------------------

SH fetchSH(ivec2 coord) {
    SH sh;
    sh.shY = texelFetch(colortex5, coord, 0);
    sh.CoCg = texelFetch(colortex6, coord, 0).xy;
    return sh;
}

void writeSH(SH sh, ivec2 coord) {
    shY = sh.shY;
    CoCg = vec4(sh.CoCg, texelFetch(colortex6, coord, 0).zw);
}

// ===========================================================================
// 空间滤波核心
// ===========================================================================
void MixDiffuse() {
    SH centerSH = fetchSH(ivec2(gl_FragCoord.xy));
    ivec2 texSize = ivec2(textureSize(colortex0, 0));

    // 中心像素几何数据
    vec3 centerNormal = texelFetch(colortex3, ivec2(gl_FragCoord.xy), 0).xyz;
    vec3 centerPos = texelFetch(colortex4, ivec2(gl_FragCoord.xy), 0).xyz;

    // ---- 5×5 双边采样 ------------------------------------------------
    const int S = 2;  // 核半径 → 5×5 窗口
    SH avgSH = init_SH();
    float D = 0.0;     // 累积方差
    float w = 0.0;     // 总权重

    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            if (i == 0 && j == 0) continue;  // 跳过中心
            ivec2 pix = ivec2(gl_FragCoord.xy) + ivec2(i, j);
            uint idx2 = getIdx(uvec2(pix));

            SH sample1 = fetchSH(pix);
            vec3 sampleNormal = texelFetch(colortex3, pix, 0).xyz;
            vec3 samplePos = texelFetch(colortex4, pix, 0).xyz;

            // 有效性 + 双边权重
            float w0 = float(denoiseBuffer.data[idx2].distance > -0.5)
                     * svgfNormalWeight(sampleNormal, centerNormal)
                     * svgfPositionWeight(samplePos, centerPos, centerNormal);
            w0 *= float(!notInRange(pix));

            // 累积方差与加权和
            D = updateVariance(avgSH, D, sample1, 1.0 / max(1e-1, w0));
            accumulate_SH(avgSH, sample1, w0);
            w += w0;
        }
    }

    // ---- 归一化加权平均 ------------------------------------------------
    avgSH = scaleSH(avgSH, 1.0 / (w + 0.01));

    // ---- 方差引导异常值抑制 -------------------------------------------
    // 如果中心 SH 与邻域均值的差异大于方差的 10 倍 → 替换为邻域均值
    float diff = dot(centerSH.shY - avgSH.shY, centerSH.shY - avgSH.shY)
               + dot(centerSH.CoCg - avgSH.CoCg, centerSH.CoCg - avgSH.CoCg);
    float sigma = 10.0 * D;
    if (diff > sigma) {
        centerSH = avgSH;
    }

    writeSH(centerSH, ivec2(gl_FragCoord.xy));
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    texSize = vec2(textureSize(colortex0, 0));
    MixDiffuse();
}
