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
    ivec3 word0Coord = ivec3(voxelCoord);
    ivec3 word1Coord = word0Coord + ivec3(0, 0, RADIANCE_CACHE_D);
    vec4 word0 = uintBitsToFloat(imageLoad(radianceCacheTemporal, word0Coord));
    vec4 word1 = uintBitsToFloat(imageLoad(radianceCacheTemporal, word1Coord));
    storeRadianceCacheHist(voxelCoord, unpackRadianceCache(word0, word1));
}

#endif // BUFFER_SWAP_RADIANCE_CACHE_GLSL
