#version 430

// ===========================================================================
// Pass bloom_sampler_downscale: 高斯金字塔下采样 (Bloom Pyramid Downscale)
// ===========================================================================
// 双 pass 机制 (通过 PASS0 宏分派):
//
// PASS0 (第一级):
//   5×5 高斯模糊 → 输出到 colortex6
//   用于将全分辨率泛光图降至下一级
//
// !PASS0 (后续级):
//   5×5 高斯模糊 (从 colortex6 采样)
//   + 50% 混合原始 colortex0 纹理
//   输出到 colortex5
//   用于构建多级泛光金字塔
// ===========================================================================

#include "/lib/common.glsl"

in vec2 texCoord;

#ifdef PASS0
uniform sampler2D colortex0;
/* RENDERTARGETS: 6 */
layout(location = 0) out vec4 fragColor;
#else
uniform sampler2D colortex0;
uniform sampler2D colortex6;
/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 fragColor;
#endif

void main() {
    // 5×5 高斯模糊核 (σ² ≈ 2, 归一化在权重累积中完成)
    #define R 2

    vec3 sumX = vec3(0.0);
    float w0 = 0.0;

    for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
            float w = exp(-float(i * i + j * j) * 0.5);  // 高斯权重

#ifdef PASS0
            // 第一级: 从全分辨率源采样
            sumX += w * texelFetch(colortex0,
                                   ivec2(gl_FragCoord.xy + ivec2(i, j)), 0).rgb;
#else
            // 后续级: 从上一级金字塔采样
            sumX += w * texelFetch(colortex6,
                                   ivec2(gl_FragCoord.xy + ivec2(i, j)), 0).rgb;
#endif
            w0 += w;
        }
    }

#ifdef PASS0
    // 第一级: 纯高斯模糊输出 (供下一级金字塔使用)
    fragColor.rgb = sumX / w0;
#else
    // 后续级: 50% 模糊 + 50% 原始纹理 (保留原始高亮细节)
    fragColor.rgb = mix(sumX / w0, texture(colortex0, texCoord).rgb, 0.5);
#endif

    if (any(isnan(fragColor.rgb))) fragColor.rgb = vec3(0.0);
}
