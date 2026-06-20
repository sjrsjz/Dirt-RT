#version 430 compatibility

// ===========================================================================
// Pass swap5: 反射缓冲交换 (Reflect Buffer Swap — Compute) 压缩兼容版
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define REFLECT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform sampler2D colortex3;   // geometry: pos + encoded normal
uniform sampler2D colortex4;   // light_sample: radiance + packed(weight, roughness)

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uint idx = getIdx(uvec2(pix));

    // 读取压缩格式
    vec4 geom  = texelFetch(colortex3, pix, 0);
    vec4 light = texelFetch(colortex4, pix, 0);

    vec3 pos, normal, radiance;
    float weight, roughness;
    unpackSpecularSample(PackedLightSample(geom, light),
                         pos, normal, radiance, weight, roughness);

    // 获取当前帧的反射缓冲条目
    vec3IllumiantionData tmp = fetchReflect(pix);

    // 更新数据
    tmp.normal     = normal;
    tmp.pos        = pos;
    tmp.data       = tmp.data_swap;                   // 双缓冲 flip
    tmp.data_swap  = radiance;                        // 滤波后的反射颜色
    tmp.weight     = weight;                          // 更新时域权重
    tmp.prev_weight = weight;                         // 保存为下一帧的历史权重
    tmp.mixWeight  = denoiseBuffer.data[idx].reflectWeight; // 反射混合系数


    // NaN 保护
    if (any(isnan(tmp.data_swap))) tmp.data_swap = vec3(0.0);

    WriteReflect(tmp, pix);
}