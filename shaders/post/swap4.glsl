#version 430 compatibility

// ===========================================================================
// Pass swap4: 反射缓冲写入 (Reflect Buffer Write)
// ===========================================================================
// 管线位置: 在反射时域累积 (101.glsl) 后，将数据写入 colortex 供后续使用
//
// 输出布局:
//   - RENDERTARGET 3 (reflectNormal): .xyz = 法线, .w = 几何有效性遮罩
//   - RENDERTARGET 4 (reflectPos):    .xyz = 世界位置, .w = 时域权重
//   - RENDERTARGET 5 (color):         .xyz = 反射颜色, .w = roughness
// ===========================================================================

#define REFLECT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5 */
layout(location = 0) out vec4 reflectNormal;
layout(location = 1) out vec4 reflectPos;
layout(location = 2) out vec4 color;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp = fetchReflect(ivec2(gl_FragCoord.xy));

    // 法线 + 几何有效性遮罩 (1.0 = 有效, 0.0 = 天空)
    reflectNormal.xyz = reflectIllumiantionBuffer.data[idx].normal;
    reflectNormal.w   = step(-0.5, denoiseBuffer.data[idx].distance);

    // 世界位置 + 时域累积权重
    reflectPos.xyz = reflectIllumiantionBuffer.data[idx].pos;
    reflectPos.w   = tmp.weight;

    // 反射颜色 + 表面粗糙度
    color.xyz = tmp.data_swap;
    color.w   = denoiseBuffer.data[idx].roughness;
}
