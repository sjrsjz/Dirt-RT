#version 430 compatibility

// ===========================================================================
// Pass 200: (已禁用) Compute Shader 版 NaN 清理 / 数据交换
// ===========================================================================
// 原本意图: 使用 compute shader (local_size_x=8) 遍历所有像素:
//   1. 检查并清零 NaN 值 (diffuse SH, reflect color, refract color)
//   2. 将 data_swap 复制到 data (双缓冲交换)
//
// 当前状态: main() 以 `return;` 截断，整个 pass 不执行任何操作。
//           NaN 保护现在内联在各 pass 的输出阶段完成。
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

/* RENDERTARGETS: 0 */

void main() {
    return;  // 已禁用 — NaN 保护已内联到各输出 pass

    // 以下是原始实现 (保留供参考):
    // uint idx = getIdx(uvec2(gl_GlobalInvocationID.xy));
    // if (denoiseBuffer.data[idx].distance < -0.5) return;
    // SH tmp = diffuseIllumiantionBuffer.data[idx].data_swap;
    // if (any(isnan(tmp.shY)))   tmp.shY   = vec4(0);
    // if (any(isnan(tmp.CoCg)))  tmp.CoCg  = vec2(0);
    // diffuseIllumiantionBuffer.data[idx].data = tmp;
    // vec3 tmp2 = reflectIllumiantionBuffer.data[idx].data_swap;
    // if (any(isnan(tmp2))) tmp2 = vec3(0);
    // reflectIllumiantionBuffer.data[idx].data = tmp2;
    // tmp2 = refractIllumiantionBuffer.data[idx].data_swap;
    // if (any(isnan(tmp2))) tmp2 = vec3(0);
    // refractIllumiantionBuffer.data[idx].data = tmp2;
}
