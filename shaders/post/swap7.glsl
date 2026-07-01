#version 430 compatibility

// ===========================================================================
// Pass swap7: 折射缓冲写回 (Compute) — swap_color + color(flip+mixWeight) + lpos + lnormal
// ===========================================================================
// 102 (REFRACT_BUFFER_MIN) 已写入 swap_color = (accumulated, weight).
// 本 pass: 降噪颜色写回 swap_color.xyz, 保留 .w(weight); color(flip + mixWeight);
// lpos/lnormal 重建后写出.
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // 降噪后: f16(R,G)|f16(B,roughness)|f16(variance,vprojdist)|oct(H)

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 texSize = textureSize(colortex4, 0);
    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    vec4 geom = texelFetch(colortex3, pix, 0);
    vec4 light = texelFetch(colortex4, pix, 0);

    vec2 rg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 br = unpackHalf2x16(floatBitsToUint(light.y));
    vec2 vv = unpackHalf2x16(floatBitsToUint(light.z));
    vec3 denoised = vec3(rg.x, rg.y, br.x);
    float vprojdist = vv.y;
    if (vv.x < 0.0) denoised = vec3(0.0);
    if (any(isnan(denoised))) denoised = vec3(0.0);

    // 读 swap_color (102 写入: (accumulated, weight_raw))
    vec4 sc = texelFetch(refractIllumiantionData_color_swap_Sampler, pix, 0);

    // 写 swap_color: 降噪后颜色 + 保留 weight
    imageStore(refractIllumiantionData_swap_color, pix, vec4(denoised, sc.w));

    // flip: color (history) = 累积颜色 (pre-denoise = sc.xyz), mixWeight = refractWeight
    uint idx = getIdx(uvec2(pix));
    imageStore(refractIllumiantionData_color, pix,
        vec4(sc.xyz, denoiseBuffer.data[idx].refractWeight));

    // lpos = pos, lnormal = R * vprojdist
    vec3 R = decodeNormal(geom.w);
    imageStore(refractIllumiantionData_lpos, pix, vec4(geom.xyz, 0.0));
    imageStore(refractIllumiantionData_lnormal, pix, vec4(R * vprojdist, 0.0));
}
