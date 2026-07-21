#version 430 compatibility

// ===========================================================================
// Pass: swap3 + 蓄水池 SSBO 回写 (composite59)
// ===========================================================================
layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex6;

void unpackLightSample(ivec2 coord, out vec3 pos, out float surfaceMask, out AliceEncoding encoded, out AliceEncoding blurred_alice) {
    vec4 d0 = texelFetch(colortex3, coord, 0);
    vec4 d1 = texelFetch(colortex4, coord, 0);
    vec4 d2 = texelFetch(colortex5, coord, 0);
    pos = d0.xyz;
    surfaceMask = d0.w; // was oct(normal), now surfaceMask
    encoded = unpackAlice(d1.x, d1.y, d1.z);
    blurred_alice = unpackAlice(d2.x, d2.y, d2.z);
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(gid, uvec2(resolution_global)))) return;
    ivec2 pix = ivec2(gid);
    uvec2 gxy = uvec2(pix);

    // Phase 1: swap3
    diffuseIlluminationData tmp = fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.aliceY))) tmp.data_swap.aliceY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);
    tmp.prev_weight = tmp.weight;
    tmp.data = tmp.data_swap;

    AliceEncoding blurred_alice, encoded;
    vec3 pos;
    float mask;
    unpackLightSample(pix, pos, mask, encoded, blurred_alice);
    tmp.pos = pos;
    tmp.surfaceMask = mask;
    tmp.histSurfaceMask = mask;
    tmp.data_swap = encoded;
    tmp.data = mix_alice(tmp.data, blurred_alice,
            clamp(NRD_BLEND_STRENGTH * exp(-NRD_BLEND_STRENGTH * clamp(tmp.weight, 0.0, 100.0)), 0.0, 1.0));
    WriteDiffuse(tmp, pix);

    // Phase 2: colortex6 → N=5 原样拷贝
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex6, pix, 0);
}
