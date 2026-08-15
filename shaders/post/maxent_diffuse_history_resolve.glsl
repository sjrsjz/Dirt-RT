#version 430 core
// ===========================================================================
// MaxEnt diffuse history resolve and path-guide SSBO publication.
// ===========================================================================
layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex6;

void unpackLightSample(uvec4 d1, out MaxEntEncoding encoded) {
    encoded.maxEntY       = vec4(unpackHalf2x16(d1.x), unpackHalf2x16(d1.y));
    encoded.CoCg         =       unpackHalf2x16(d1.z);
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
    // Phase 1: swap3
    diffuseIlluminationData tmp = fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.maxEntY))) tmp.data_swap.maxEntY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);
    tmp.prev_weight = tmp.weight;
    tmp.prev_meanY2 = tmp.meanY2;
    tmp.data = tmp.data_swap;

    MaxEntEncoding encoded;
    unpackLightSample(packedLight, encoded);
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, tmp.pos, primaryDistance);
    tmp.histNormal = readPrimaryGeometryNormal(gxy);
    tmp.data_swap = encoded;
    tmp.data = mix_maxent(tmp.data, encoded,
            clamp(MAXENT_SPATIAL_DIFFUSE_LOW_CONFIDENCE_BLEND * exp(-MAXENT_SPATIAL_DIFFUSE_LOW_CONFIDENCE_BLEND * clamp(tmp.weight, 0.0, 100.0)), 0.0, 1.0));
    writeDiffuse(tmp, pix);

    // Phase 2: publish the path-guide reservoir from temporal scratch.
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex6, pix, 0);
}
