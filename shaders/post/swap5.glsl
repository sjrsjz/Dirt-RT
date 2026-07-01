#version 430 compatibility

// ===========================================================================
// Pass swap5: 反射缓冲写回 (Compute) — swap_color + color(flip) + lpos + lnormal
// ===========================================================================
// 101 (REFLECT_BUFFER_MIN) 已写入 swap_color = (accumulated, pack(weight, mixWeight)).
// 本 pass: 将 301 降噪后颜色写回 swap_color.xyz, 保留 .w; 同时写 color(flip) / lpos / lnormal.
// swap_color 的 sampler 读+image 写 UB 与原始 swap5 一致 (已验证可行).
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define REFLECT_BUFFER

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

    // 解包降噪后颜色 + vprojdist
    vec2 rg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 br = unpackHalf2x16(floatBitsToUint(light.y));
    vec2 vv = unpackHalf2x16(floatBitsToUint(light.z));
    vec3 denoised = vec3(rg.x, rg.y, br.x);
    float vprojdist = vv.y;
    if (vv.x < 0.0) denoised = vec3(0.0);  // 天空
    if (any(isnan(denoised))) denoised = vec3(0.0);

    // 读 swap_color (101 写入: (accumulated, pack(weight, mixWeight)))
    vec4 sc = texelFetch(reflectIllumiantionData_color_swap_Sampler, pix, 0);
    vec2 w_mw = unpackHalf2x16(floatBitsToUint(sc.w));
    float weight = w_mw.x;

    // 写 swap_color: 降噪后颜色 + 保留 weight/mixWeight
    imageStore(reflectIllumiantionData_swap_color, pix, vec4(denoised, sc.w));

    // flip: color (history) = 累积颜色 (pre-denoise = sc.xyz), prev_weight = weight
    imageStore(reflectIllumiantionData_color, pix,
        vec4(sc.xyz, pack2HalfClamped(weight, 0.0)));

    // lpos = pos, lnormal = R * vprojdist (供下一帧 101 hitWeight)
    vec3 R = decodeNormal(geom.w);
    imageStore(reflectIllumiantionData_lpos, pix, vec4(geom.xyz, 0.0));
    imageStore(reflectIllumiantionData_lnormal, pix, vec4(R * vprojdist, 0.0));
}
