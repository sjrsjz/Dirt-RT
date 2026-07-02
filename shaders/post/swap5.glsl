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

    vec2 rg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 br = unpackHalf2x16(floatBitsToUint(light.y));
    vec2 vv = unpackHalf2x16(floatBitsToUint(light.z));
    float vprojdist = vv.y;
    if (vv.x < 0.0) return; // 天空 → 跳过, 保留 SSBO 原有值

    // 从 SSBO 读 101 写入的累积颜色 + 权重 (pre-denoise history 源)
    uint idx = getIdx(uvec2(pix));
    SpecularRTElement e = reflectIllumiantionBuffer.data[idx];
    vec2 erg = unpackHalf2x16(floatBitsToUint(e.color_rg));
    float eb = unpackHalf2x16(floatBitsToUint(e.color_b)).x;
    vec3 preDenoise = vec3(erg.x, erg.y, eb);
    float weight = e.accum_weight;
    if (any(isnan(preDenoise))) preDenoise = vec3(0.0);

    // 写回降噪颜色到 SSBO 当前帧区段 (fog.fsh 通过 fetchReflect 读取)
    vec2 drg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 dbr = unpackHalf2x16(floatBitsToUint(light.y));
    vec3 denoised = vec3(drg.x, drg.y, dbr.x);
    if (any(isnan(denoised))) denoised = vec3(0.0);
    reflectIllumiantionBuffer.data[idx].color_rg = pack2HalfClamped(denoised.r, denoised.g);
    reflectIllumiantionBuffer.data[idx].color_b  = pack2HalfClamped(denoised.b, 0.0);

    // 写时域历史到 SSBO hist_* 区段 (替代原先 4 个 rgba32f image write)
    vec3 R = decodeNormal(geom.w);
    WriteReflectHistory(preDenoise, weight, geom.xyz, R, vprojdist, pix);
}
