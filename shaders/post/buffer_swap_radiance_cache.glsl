#version 430 compatibility

#ifndef BUFFER_SWAP_RADIANCE_CACHE_GLSL
#define BUFFER_SWAP_RADIANCE_CACHE_GLSL

#include "/lib/buffers/radiance_cache.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(rgba32ui) uniform readonly uimage3D radianceCacheTemporal;

// Iris parses workGroups arguments with Integer.parseInt; use literal values.
const ivec3 workGroups = ivec3(32, 32, 32);

void main() {
    uvec3 voxelCoord = gl_GlobalInvocationID.xyz;
    vec4 packedCache = uintBitsToFloat(imageLoad(radianceCacheTemporal, ivec3(voxelCoord)));
    storeRadianceCacheHist(voxelCoord, unpackRadianceCache(packedCache));
}

#endif // BUFFER_SWAP_RADIANCE_CACHE_GLSL
