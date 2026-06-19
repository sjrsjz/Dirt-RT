#version 430 compatibility

// ===========================================================================
// Pass swap: (已禁用) 双缓冲数据交换
// ===========================================================================
// 原本意图: 将 data_swap 复制到 data (实现双缓冲 flip)
//   为漫反射、反射、折射三种光照类型的数据结构执行 swap 操作
//
// 当前状态: 所有操作均已注释。此 pass 不产出任何有效输出。
//           数据交换现在通过各 swap2-7.glsl 的显式读取/写入完成。
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    // 已禁用 — swap 操作已分散到 swap2-7.glsl

    // 原始实现 (保留供参考):
    // uint idx = getIdx(uvec2(gl_FragCoord.xy));
    // diffuseIllumiantionBuffer.data[idx].data_swap = diffuseIllumiantionBuffer.data[idx].data;
    // reflectIllumiantionBuffer.data[idx].data_swap = reflectIllumiantionBuffer.data[idx].data;
    // refractIllumiantionBuffer.data[idx].data_swap = refractIllumiantionBuffer.data[idx].data;
}
