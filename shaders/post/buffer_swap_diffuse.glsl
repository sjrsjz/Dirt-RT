#version 430 core
// ===========================================================================
// Pass: swap3 + 蓄水池 SSBO 回写 (composite59)
// ===========================================================================
layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

uniform sampler2D colortex3;
uniform usampler2D colortex4;
uniform usampler2D colortex5;
uniform usampler2D colortex6;

void unpackLightSample(ivec2 coord, vec4 d0, uvec4 d1, out vec3 pos,
        out MaxEntEncoding encoded,
        out MaxEntEncoding blurred_maxent) {
    uvec4 d2 = texelFetch(colortex5, coord, 0);
    pos = d0.xyz;
    encoded.maxEntY       = vec4(unpackHalf2x16(d1.x), unpackHalf2x16(d1.y));
    encoded.CoCg         =       unpackHalf2x16(d1.z);
    blurred_maxent.maxEntY = vec4(unpackHalf2x16(d2.x), unpackHalf2x16(d2.y));
    blurred_maxent.CoCg   =       unpackHalf2x16(d2.z);
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(gid, uvec2(resolution_global)))) return;
    ivec2 pix = ivec2(gid);
    uvec2 gxy = uvec2(pix);

    uvec4 packedLight = texelFetch(colortex4, pix, 0);
    if (uintBitsToFloat(packedLight.w) < 0.0) {
        // No surface: invalidate every temporal consumer with four raw stores
        // instead of decoding two light textures and previous histories.
        diffuseBuffer.data[addr(DIF_N_HIST, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_HISTGEO, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_SWAP, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = uvec4(0u);
        return;
    }
    vec4 packedGeometry = texelFetch(colortex3, pix, 0);

    // Phase 1: swap3
    diffuseIlluminationData tmp = fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.maxEntY))) tmp.data_swap.maxEntY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);
    tmp.prev_weight = tmp.weight;
    tmp.prev_meanY2 = tmp.meanY2;
    tmp.data = tmp.data_swap;

    MaxEntEncoding blurred_maxent, encoded;
    vec3 pos;
    unpackLightSample(pix, packedGeometry, packedLight, pos, encoded,
        blurred_maxent);
    tmp.pos = pos;
    float roughness_unused, pathRoughness_unused;
    int illumType_unused;
    readGeo1(GEO_N_NORMALS, gxy, tmp.histNormal, roughness_unused, illumType_unused, pathRoughness_unused);
    tmp.data_swap = encoded;
    tmp.data = mix_maxent(tmp.data, blurred_maxent,
            clamp(NRD_BLEND_STRENGTH * exp(-NRD_BLEND_STRENGTH * clamp(tmp.weight, 0.0, 100.0)), 0.0, 1.0));
    writeDiffuse(tmp, pix);

    // Phase 2: colortex6 → N=5 (both RGBA32UI, raw copy)
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex6, pix, 0);
}
