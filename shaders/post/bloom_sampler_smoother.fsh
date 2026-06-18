#version 430

// ===========================================================================
// Pass bloom_sampler_smoother: 绽放平滑 / 色调映射 (Bloom Smooth + Tonemap)
// ===========================================================================
// 双模式 pass (由 FINAL 宏控制):
//
// 模式 1 — FINAL (最终输出):
//   1. 5×5 邻域平均 (轻度模糊)
//   2. 与原始图像 0.875:0.125 混合
//   3. 应用神经网络色调映射器 (tonemap_NeuralNetwork)
//   4. Gamma 校正 (pow 1/2.2)
//   5. 输出到 colortex0 (最终显示) + colortex1 (bloom 缓冲区)
//
// 模式 2 — 方向模糊 (中间 pass):
//   在 X (STEP=1) 或 Y (STEP≠1) 方向进行一次 33 像素的高斯模糊
//   输出到 colortex1
// ===========================================================================

#include "/lib/common.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;
uniform sampler2D colortex1;

const bool colortex0MipmapEnabled = true;

/* RENDERTARGETS: 0,1 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 bloomColor;

void main() {
    fragColor = texture(colortex0, texCoord);

#ifdef FINAL
    // =========================================================================
    // 最终合成模式
    // =========================================================================

    // 5×5 邻域平均 (轻度去噪)
    vec3 avg = vec3(0.0);
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            avg += texelFetch(colortex1, ivec2(gl_FragCoord.xy + vec2(i, j)), 0).rgb;
    avg /= 25.0;

    // 混合: 87.5% 平均 + 12.5% 原始
    fragColor.xyz = mix(avg, texture(colortex0, texCoord).rgb, 0.875);

    // 应用神经网络色调映射器
    // 将颜色归一化到 [0,1] 范围 (x/(1+x)) 后再映射，减去黑色偏移量
    fragColor.xyz = pow(max(
        tonemap_NeuralNetwork(fragColor.zyx / (1.0 + fragColor.zyx))
        - tonemap_NeuralNetwork(vec3(0.0)),
        vec3(0.0)),
        vec3(1.0 / 2.2));  // Gamma 校正

    // 写入泛光缓冲区
    bloomColor = texture(colortex1, texCoord);

    if (any(isnan(fragColor.xyz))) fragColor.xyz = vec3(0.0);
    return;

#else
    // =========================================================================
    // 方向高斯模糊模式 (单轴, 33 像素宽)
    // =========================================================================

    const int sampleN = 16;  // 单侧 = 16, 总长 = 33
    vec3 sumX = vec3(0.0);
    float w0 = 0.0;
    vec2 texSize = textureSize(colortex0, 0);

    for (int i = -sampleN; i <= sampleN; i++) {
#if STEP == 1
        // ---- 水平方向 (X 轴) -----------------------------------------------
        float w = exp(-float(i * i) * 0.05);
        w *= float(clamp(gl_FragCoord.xy + vec2(i * 5, 0),
                         vec2(0.0), texSize)
                   == gl_FragCoord.xy + vec2(i * 5, 0));
        sumX += texelFetch(colortex1,
                           ivec2(gl_FragCoord.xy + vec2(i * 5, 0)), 0).xyz * w;
#else
        // ---- 垂直方向 (Y 轴) -----------------------------------------------
        float w = exp(-float(i * i) * 0.05);
        w *= float(clamp(gl_FragCoord.xy + vec2(0, i * 5),
                         vec2(0.0), texSize)
                   == gl_FragCoord.xy + vec2(0, i * 5));
        sumX += texelFetch(colortex1,
                           ivec2(gl_FragCoord.xy + vec2(0, i * 5)), 0).xyz * w;
#endif
        w0 += w;
    }

    sumX /= w0 + 1e-3;
    if (any(isnan(sumX))) sumX = vec3(0.0);
    bloomColor.xyz = sumX;
#endif
}
