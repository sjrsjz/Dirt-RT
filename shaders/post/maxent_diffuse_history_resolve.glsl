#version 430 core
// ===========================================================================
// MaxEnt diffuse history resolve and path-guide SSBO publication.
// ===========================================================================
layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_bures.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex6;

MaxEntEncoding unpackLightSample(uvec4 d1) {
    MaxEntEncoding encoded;
    encoded.maxEntY = vec4(unpackHalf2x16(d1.x), unpackHalf2x16(d1.y));
    encoded.CoCg = unpackHalf2x16(d1.z);
    return encoded;
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(gid, uvec2(resolution_global)))) return;
    ivec2 pix = ivec2(gid);
    uvec2 gxy = uvec2(pix);

    // Temporal used this parity as its reprojected-denoised scratch. Consume it
    // before replacing it with the exact current spatial result.
    uvec4 reprojectedWords = readDiffuseDenoisedCurrentRaw(gxy);
    uvec4 packedLight = texelFetch(colortex4, pix, 0);
    writeDiffuseDenoisedCurrentRaw(gxy, packedLight);
    if (unpackHalf2x16(packedLight.w).x < 0.0) {
        // No surface: invalidate every temporal consumer with four raw stores
        // instead of decoding two light textures and previous histories.
        diffuseBuffer.data[addr(DIF_N_HIST, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_SWAP, gxy)] = uvec4(0u);
        diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = uvec4(0u);
        writeDiffuseHistGeoInvalid(gxy);
        writeDiffuseDenoisedDifference(gxy, -1.0);
        return;
    }
    // Phase 1: swap3
    diffuseIlluminationData tmp = fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.maxEntY))) tmp.data_swap.maxEntY = vec4(0.0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0.0);
    vec2 historyMeta = unpackHalf2x16(reprojectedWords.z);
    vec2 historyDeviationAlpha = unpackHalf2x16(reprojectedWords.w);
    vec4 currentMaxEntY = vec4(unpackHalf2x16(packedLight.x), unpackHalf2x16(packedLight.y));
    vec4 historyMaxEntY = vec4(unpackHalf2x16(reprojectedWords.x), unpackHalf2x16(reprojectedWords.y));
    float normalizedDistance;
    tmp.weight = maxentClampHistoryWeightByDenoisedDifference(tmp.weight,
            currentMaxEntY, historyMaxEntY, historyDeviationAlpha.x,
            historyMeta.x, historyMeta.y, historyDeviationAlpha.y,
            float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY),
            MAXENT_DIFFUSE_TEMPORAL_DIFFERENCE_TOLERANCE, normalizedDistance);
    writeDiffuseDenoisedDifference(gxy, normalizedDistance);
    tmp.prev_weight = tmp.weight;
    tmp.data = tmp.data_swap;
    tmp.prev_meanY2 = tmp.meanY2;

    MaxEntEncoding encoded = unpackLightSample(packedLight);
    float primaryDistance;
    readDiffusePrimaryGeometry(gxy, tmp.pos, primaryDistance);
    tmp.histNormal = readPrimaryGeometryNormal(gxy);
    tmp.data_swap = encoded;

    writeDiffuse(tmp, pix);

    // Phase 2: publish the path-guide reservoir from temporal scratch.
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, gxy)] = texelFetch(colortex6, pix, 0);
}
