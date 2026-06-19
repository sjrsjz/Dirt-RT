#version 430 compatibility

// ===========================================================================
// Pass swap6: 折射缓冲写入 (Refract Buffer Write)
// ===========================================================================
// 管线位置: 在折射时域累积 (102.glsl) 后，将数据写入 colortex 供后续使用
//
// 输出布局:
//   - RENDERTARGET 3 (refractNormal): .xyz = 法线, .w = 几何有效性遮罩
//   - RENDERTARGET 4 (refractPos):    .xyz = 世界位置, .w = 1.0
//   - RENDERTARGET 5 (color):         .xyz = 折射颜色, .w = roughness
//
// 注: 与 swap4.glsl (反射) 相比，refractPos.w 固定为 1.0 而非时域权重。
// ===========================================================================

#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5 */
layout(location = 0) out vec4 refractNormal;
layout(location = 1) out vec4 refractPos;
layout(location = 2) out vec4 color;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp = fetchRefract(ivec2(gl_FragCoord.xy));

    // 法线 + 几何有效性遮罩
    refractNormal.xyz = refractIllumiantionBuffer.data[idx].normal;
    refractNormal.w   = step(-0.5, denoiseBuffer.data[idx].distance);

    // 世界位置 (w 固定为 1.0，与反射的时域权重不同)
    refractPos.xyz = refractIllumiantionBuffer.data[idx].pos;
    refractPos.w   = 1.0;

    // 折射颜色 + 表面粗糙度
    color.xyz = tmp.data_swap;
    color.w   = denoiseBuffer.data[idx].roughness;
}
