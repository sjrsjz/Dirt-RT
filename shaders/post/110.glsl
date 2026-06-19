#version 430 compatibility

// ===========================================================================
// Pass 110: 漫反射状态回写 (Diffuse State Writeback)
// ===========================================================================
// 管线位置: 在 100.glsl (时域累积) 之后，300.glsl (空间滤波) 之前
//
// 功能:
//   1. 读取 100.glsl 写入的 extInfoBuffer (时域方差 + 混合权重)
//   2. 将 (weight, variance) 写回到 diffuseIllumiantionData 中
//      以供后续 SVGF 空间滤波器 (300.glsl) 使用
//   3. 同时读取 denoiseBuffer 中的 emission 数据并输出
//
// 这是一个精简的"桥接"pass — 职责单一，仅做元数据回写。
// 之前的版本包含一个完整的空间滤波器，现已拆分到 300.glsl。
// ===========================================================================

#define DIFFUSE_BUFFER_MIN

#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

/* RENDERTARGETS: 2 */
layout(location = 0) out vec4 Emission;

// 外部信息缓冲区 (由 100.glsl 写入)
//   .x = 时域累积权重 (output_weight)
//   .y = 时域方差     (output_variance)
uniform sampler2D extInfoBuffer_Sampler;

void main() {
    // ---- 读取 emission 并直接输出 ----------------------------------------
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    Emission = vec4(denoiseBuffer.data[idx].emission, 0.0);

    // ---- 回写时域方差与权重到 diffuse 数据结构 ----------------------------
    // 后续 SVGF 空间滤波器 (300.glsl) 通过 colortex6 读取这些值
    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_FragCoord.xy));
    vec4 prev_data = texelFetch(extInfoBuffer_Sampler, ivec2(gl_FragCoord.xy), 0);
    tmp.weight = prev_data.x;
    tmp.variance = prev_data.y;
    WriteDiffuse(tmp, ivec2(gl_FragCoord.xy));
}
