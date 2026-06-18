#version 430 compatibility

// ===========================================================================
// Pass swap5: 反射缓冲交换 (Reflect Buffer Swap — Compute)
// ===========================================================================
// 管线位置: 在反射滤波处理之后，将结果交换回主缓冲区
//
// 功能:
//   1. 从 colortex5 读取滤波后的反射颜色 → 写入 data_swap
//   2. 将 data_swap 复制到 data (双缓冲 flip)
//   3. 保存 prev_weight (为下一帧时域累积做准备)
//   4. 更新法线、位置、mixWeight (refractWeight 来自 denoiseBuffer)
//
// Compute Shader 版本 (local_size_x=16)
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define REFLECT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

uniform sampler2D colortex3;  // 世界空间法线
uniform sampler2D colortex4;  // 世界空间位置
uniform sampler2D colortex5;  // 滤波后反射颜色

/* RENDERTARGETS: 0 */

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uint idx = getIdx(uvec2(gl_GlobalInvocationID.xy));
    vec3IllumiantionData tmp = fetchReflect(pix);

    // ---- NaN 保护 --------------------------------------------------------
    if (any(isnan(tmp.data_swap))) tmp.data_swap = vec3(0.0);

    // ---- 双缓冲 flip: 当前帧 → 历史帧 -----------------------------------
    tmp.data = tmp.data_swap;

    // ---- 从滤波后的 colortex 读取新数据 ----------------------------------
    tmp.data_swap = texelFetch(colortex5, pix, 0).xyz;

    // ---- 更新几何 + 权重数据 --------------------------------------------
    tmp.normal    = texelFetch(colortex3, pix, 0).xyz;
    tmp.pos       = texelFetch(colortex4, pix, 0).xyz;
    tmp.mixWeight = denoiseBuffer.data[idx].reflectWeight;  // 反射权重

    // ---- 保存历史权重 (供下一帧时域累积) ---------------------------------
    tmp.prev_weight = tmp.weight;

    WriteReflect(tmp, pix);
}
