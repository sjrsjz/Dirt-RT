#version 430

// ===========================================================================
// Pass godRays: 体积光 / 神光方向模糊 (Directional Blur)
// ===========================================================================
// 功能: 对 colortex2 进行轴向高斯模糊 (X 或 Y 方向)，模拟体积光散射效果。
//
// 多 pass 机制 (通过 STEP 宏分派):
//   - STEP=1 / STEP=3: 水平方向 (X 轴) 模糊
//   - STEP=2 / STEP=4: 垂直方向 (Y 轴) 模糊
//   - STEP=4:          额外将结果存入 denoiseBuffer.emission 供后续使用
//
// 关键实现:
//   - 使用 sign(distance) 检查几何连续性，防止天空与表面之间交叉模糊
//   - 高斯权重 exp(-i² × 0.05) 实现平滑衰减
//   - 8 像素半径采样 (sampleN=8, 即 ±8 = 17 个采样点)
// ===========================================================================

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex8;
uniform sampler2D colortex9;

/* RENDERTARGETS: 2 */
layout(location = 0) out vec4 fragColor;

void main() {
    // 检查当前像素的几何有效性 (天空 < 0, 表面 ≥ 0)
    float d = sign(denoiseBuffer.data[getIdx(uvec2(gl_FragCoord.xy))].distance);

    const int sampleN = 8;           // 单侧采样数 (总长 = 2*sampleN+1 = 17)
    vec3 sumX = vec3(0.0);           // 加权颜色累积
    float w0 = 0.0;                  // 总权重
    vec2 texSize = textureSize(colortex2, 0);

    // 缩放因子: STEP≤2 时用 ×3 加速收敛 (前几级需要更宽的模糊)
    const float scale = (STEP <= 2) ? 3.0 : 1.0;

    for (int k = -sampleN; k <= sampleN; k++) {
        int i = k;

#if STEP == 1 || STEP == 3
        // ---- 水平方向模糊 --------------------------------------------------
        float w = exp(-i * i * 0.05)
                * float(d == sign(denoiseBuffer.data[getIdx(uvec2(
                    gl_FragCoord.xy + vec2(i * scale, 0)))].distance))
                * float(clamp(gl_FragCoord.xy + vec2(i * scale, 0),
                              vec2(0), texSize)
                        == gl_FragCoord.xy + vec2(i * scale, 0));

        sumX += texelFetch(colortex2,
                           ivec2(gl_FragCoord.xy + vec2(i * scale, 0)), 0).xyz * w;
#else
        // ---- 垂直方向模糊 --------------------------------------------------
        float w = exp(-i * i * 0.05)
                * float(d == sign(denoiseBuffer.data[getIdx(uvec2(
                    gl_FragCoord.xy + vec2(0, i * scale)))].distance))
                * float(clamp(gl_FragCoord.xy + vec2(0, i * scale),
                              vec2(0), texSize)
                        == gl_FragCoord.xy + vec2(0, i * scale));

        sumX += texelFetch(colortex2,
                           ivec2(gl_FragCoord.xy + vec2(0, i * scale)), 0).xyz * w;
#endif
        w0 += w;
    }

    // ---- 归一化并输出 ----------------------------------------------------
    sumX /= w0 + 1e-3;
    if (any(isnan(sumX))) sumX = vec3(0.0);
    fragColor.xyz = sumX;

    // ---- STEP=4: 最后一轮 — 将结果存入 emission 供 fog.fsh 使用 ----------
#if STEP == 4
    denoiseBuffer.data[getIdx(uvec2(gl_FragCoord.xy))].emission = sumX;
#endif
}
