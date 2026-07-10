#version 430 compatibility

// ===========================================================================
// Pass swap3: 漫反射缓冲交换 (Diffuse Buffer Swap — Compute)
// ===========================================================================
// 管线位置: 在空间滤波 (atrous_denoise_diffuse.glsl) 之后，将滤波结果交换回主缓冲区
//
// 功能:
//   1. 从 colortex5/6 读取滤波后的 AliceEncoding 数据 → 写入 data_swap
//   2. 将 data_swap 复制到 data (完成双缓冲 flip)
//   3. 保存 prev_weight / prev_variance (为下一帧时域累积做准备)
//   4. 更新法线与位置 (从 colortex3/4)
//
// 这是 Compute Shader 版本 (local_size_x=16)，与 variance_prefilter_diffuse.glsl 的 fragment
// shader 版本输出到同一缓冲区集合。
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

uniform sampler2D colortex3;  // 几何信息
uniform sampler2D colortex4;  // 光照信息
uniform sampler2D colortex5;  // 滤波后的 AliceEncoding 数据 (Y)

void unpackLightSample(ivec2 coord, out vec3 pos, out vec3 normal, out AliceEncoding encoded, out AliceEncoding blurred_alice) {
    vec4 sample_data0 = texelFetch(colortex3, coord, 0); // 几何信息
    vec4 sample_data1 = texelFetch(colortex4, coord, 0); // 光照样本信息
    vec4 sample_data2 = texelFetch(colortex5, coord, 0); // 滤波后的 AliceEncoding 数据 (Y)
    pos = sample_data0.xyz;
    normal = decodeNormal(sample_data0.w);
    encoded = unpackAlice(sample_data1.x, sample_data1.y, sample_data1.z);
    blurred_alice = unpackAlice(sample_data2.x, sample_data2.y, sample_data2.z);
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    diffuseIlluminationData tmp = fetchDiffuse(pix);

    // ---- NaN 保护 --------------------------------------------------------
    if (any(isnan(tmp.data_swap.aliceY)))  tmp.data_swap.aliceY  = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);

    // ---- 保存历史统计信息 (供下一帧时域累积使用) -------------------------
    tmp.prev_weight   = tmp.weight;

    // // ---- 双缓冲 flip: 当前帧 → 历史帧 -----------------------------------
    tmp.data = tmp.data_swap;
    AliceEncoding blurred_alice;
    // ---- 从滤波后的 colortex 读取新数据 ----------------------------------
    unpackLightSample(pix, tmp.pos, tmp.normal, tmp.data_swap, blurred_alice);

    // 在低权重的时候传播滤波结果
    // 这受到 NRD 的启发，实际上移除后对降噪质量不产生明显影响
    tmp.data = mix_alice(tmp.data, blurred_alice, clamp(NRD_BLEND_STRENGTH / max(tmp.weight, 1.0), 0.0, 1.0));
    WriteDiffuse(tmp, pix);
}
