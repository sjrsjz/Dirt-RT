#version 430 compatibility

// ===========================================================================
// Pass swap7: 折射缓冲交换 (Refract Buffer Swap — Fragment)
// ===========================================================================
// 管线位置: 在折射滤波处理之后，将结果交换回主缓冲区
//
// 功能:
//   1. 从 colortex5 读取滤波后的折射颜色 → 写入 data_swap
//   2. 将 data_swap 复制到 data (双缓冲 flip)
//   3. 更新法线、位置、mixWeight (refractWeight 来自 denoiseBuffer)
//
// 与 swap5.glsl (反射 Compute Shader 版本) 不同，这是 Fragment Shader 版本。
// ===========================================================================

#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

uniform sampler2D colortex3;  // 世界空间法线
uniform sampler2D colortex4;  // 世界空间位置
uniform sampler2D colortex5;  // 滤波后折射颜色

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp = fetchRefract(pix);

    // ---- NaN 保护 --------------------------------------------------------
    if (any(isnan(tmp.data_swap))) tmp.data_swap = vec3(0.0);

    // ---- 双缓冲 flip: 当前帧 → 历史帧 -----------------------------------
    tmp.data = tmp.data_swap;

    // ---- 从滤波后的 colortex 读取新数据 ----------------------------------
    tmp.data_swap = texelFetch(colortex5, pix, 0).xyz;

    // ---- 更新几何 + 权重数据 --------------------------------------------
    tmp.normal    = texelFetch(colortex3, pix, 0).xyz;
    tmp.pos       = texelFetch(colortex4, pix, 0).xyz;
    tmp.mixWeight = denoiseBuffer.data[idx].refractWeight;  // 折射权重

    WriteRefract(tmp, pix);
}
