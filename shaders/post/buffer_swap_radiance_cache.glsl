#version 430 core
#ifndef BUFFER_SWAP_RADIANCE_CACHE_GLSL
#define BUFFER_SWAP_RADIANCE_CACHE_GLSL
// Compatibility pass: sparse temporal accumulation writes history in place.
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);
void main() {}
#endif
