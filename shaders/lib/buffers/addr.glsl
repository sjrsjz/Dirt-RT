#ifndef BUFFERS_ADDR_GLSL
#define BUFFERS_ADDR_GLSL

#include "/lib/buffers/frame_data.glsl"
#include "/lib/common/tiled_addr.glsl"

// ===========================================================================
// Vec4-based SSBO addressing — tiledAddr8x8 for 8×8 tile-encoded SSBO access
// ===========================================================================
// All SSBOs are stored as vec4 data[]. Each abstract "image" is a resolution×resolution
// vec4 grid encoded in 8×8 tiles. N selects the abstract image layer.

uint addr(uint N, uvec2 xy) {
    // Clamp to resolution_global to prevent out-of-bounds coordinates from corrupting tile calculation
    uvec2 p = min(xy, uvec2(resolution_global) - 1u);
    return tiledAddr8x8(N, uint(resolution_global.x), uint(resolution_global.y), p.x, p.y);
}

uint addr(uint N, ivec2 xy) {
    return addr(N, uvec2(xy));
}

// ===========================================================================
// SSBO declarations — 4 bindings, all vec4 data[]
// ===========================================================================

layout(std430, set = 3, binding = 0) buffer GeometryMaterialBuffer {
    vec4 data[];
} geomBuffer;

layout(std430, set = 3, binding = 2) buffer DiffuseBuffer {
    uvec4 data[];  // raw uint storage — no implicit float interpretation
} diffuseBuffer;

layout(std430, set = 3, binding = 3) buffer ReflectBuffer {
    vec4 data[];
} reflectBuffer;

layout(std430, set = 3, binding = 4) buffer RefractBuffer {
    vec4 data[];
} refractBuffer;

layout(std430, set = 3, binding = 5) buffer RadianceCacheBuffer {
    vec4 data[];
} radianceCacheBuffer;

#endif // BUFFERS_ADDR_GLSL
