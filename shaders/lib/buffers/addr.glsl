#ifndef BUFFERS_ADDR_GLSL
#define BUFFERS_ADDR_GLSL

#include "/lib/buffers/frame_data.glsl"
#include "/lib/common/tiled_addr.glsl"

// ===========================================================================
// Uvec4-based SSBO addressing — tiledAddr8x8 for 8×8 tile-encoded SSBO access
// ===========================================================================
// Screen SSBOs use raw uvec4 words. Packed values stay integer end-to-end;
// genuine FP32 lanes cross the storage boundary through vector bitcasts.

uint addr(uint N, uvec2 xy) {
    // Clamp to resolution_global to prevent out-of-bounds coordinates from corrupting tile calculation
    uvec2 p = min(xy, uvec2(resolution_global) - 1u);
    return tiledAddr8x8(N, uint(resolution_global.x), uint(resolution_global.y), p.x, p.y);
}

uint addr(uint N, ivec2 xy) {
    return addr(N, uvec2(xy));
}

// ===========================================================================
// SSBO declarations — screen virtual layers use uvec4 data[]
// ===========================================================================

layout(std430, set = 3, binding = 0) buffer GeometryMaterialBuffer {
    uvec4 data[];
} geomBuffer;

layout(std430, set = 3, binding = 2) buffer DiffuseBuffer {
    uvec4 data[];  // raw uint storage — no implicit float interpretation
} diffuseBuffer;

layout(std430, set = 3, binding = 3) buffer ReflectBuffer {
    uvec4 data[];
} reflectBuffer;

layout(std430, set = 3, binding = 4) buffer RefractBuffer {
    uvec4 data[];
} refractBuffer;

layout(std430, set = 3, binding = 5) buffer RadianceCacheBuffer {
    uint data[];
} radianceCacheBuffer;

#endif // BUFFERS_ADDR_GLSL
