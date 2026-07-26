#version 430 compatibility

// ===========================================================================
// Pass swap7: 折射缓冲写回 (Compute) — swap_color + color(flip) + lpos + lnormal
// ===========================================================================
// 102 (REFRACT_BUFFER_MIN) 已写入 swap_color = (accumulated, weight).
// 本 pass: 降噪颜色写回 swap_color.xyz, 保留 .w(weight); color(flip);
// lpos/lnormal 重建后写出.
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // 降噪后: f16(R,G)|f16(B,roughness)|f16(variance,virtualProjDist)|oct(H)

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 texSize = textureSize(colortex4, 0);
    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    vec4 geom = texelFetch(colortex3, pix, 0);
    vec4 light = texelFetch(colortex4, pix, 0);

    vec2 rg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 br = unpackHalf2x16(floatBitsToUint(light.y));
    vec2 vv = unpackHalf2x16(floatBitsToUint(light.z));
    float virtualProjDist = vv.y;
    if (vv.x < 0.0) return; // 天空

    // 从 SSBO 读 102 写入的累积颜色 + 权重 (pre-denoise history 源)
    uvec2 xy = uvec2(pix);
    vec3 preDenoise; float vproj, weight;
    readRefrLight(xy, preDenoise, vproj, weight);
    if (any(isnan(preDenoise))) preDenoise = vec3(0.0);

    // 写回降噪颜色到 SSBO 当前帧区段
    vec2 drg = unpackHalf2x16(floatBitsToUint(light.x));
    vec2 dbr = unpackHalf2x16(floatBitsToUint(light.y));
    vec3 denoised = vec3(drg.x, drg.y, dbr.x);
    if (any(isnan(denoised))) denoised = vec3(0.0);
    writeRefrLight(xy, denoised, vproj, weight);

    vec3 R = decodeNormal(geom.w);
    writeRefractHistory(preDenoise, weight, geom.xyz, R, virtualProjDist, pix);
}
