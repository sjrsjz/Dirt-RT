#version 430 compatibility

// ===========================================================================
// Pass 111: (已禁用) Compute Shader 版漫反射状态回写
// ===========================================================================
// 原本意图: 使用 compute shader (local_size_x=16) 并行回写时域方差/权重，
//           与 110.glsl 功能相同但以 CS 方式执行。
//
// 当前状态: main() 以 `return;` 截断，整个 pass 不执行任何操作。
//           功能已由 fragment shader 版本 110.glsl 完全替代。
// ===========================================================================

#define DIFFUSE_BUFFER_MIN

layout(local_size_x = 16, local_size_y = 16) in;
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

void main() {
    return;  // 已禁用 — 功能由 110.glsl 替代

    // 以下是原始实现 (保留供参考):
    // uint idx = getIdx(uvec2(gl_GlobalInvocationID.xy));
    // diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_GlobalInvocationID.xy));
    // vec2 prev_data = extInfoBuffer.data[idx];
    // tmp.weight = prev_data.x;
    // tmp.variance = prev_data.y;
    // WriteDiffuse(tmp, ivec2(gl_GlobalInvocationID.xy));
}
