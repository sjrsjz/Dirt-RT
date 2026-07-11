#ifndef PACK8_GLSL
#define PACK8_GLSL

// ===========================================================================
// unorm8 pack/unpack utilities — thin wrappers around GLSL packUnorm4x8
// ===========================================================================

// Pack 4 × [0,1] floats into one uint (4 × unorm8). Input clamped.
uint pack4Unorm8(vec4 v) {
    return packUnorm4x8(clamp(v, 0.0, 1.0));
}

// Unpack one uint into 4 × [0,1] floats (4 × unorm8)
vec4 unpack4Unorm8(uint u) {
    return unpackUnorm4x8(u);
}

// Pack 2 × [0,1] floats into low 16 bits of a uint (2 × unorm8)
uint pack2Unorm8(vec2 v) {
    uvec2 b = uvec2(clamp(v, 0.0, 1.0) * 255.0 + 0.5);
    return b.x | (b.y << 8);
}

// Unpack low 16 bits into 2 × [0,1] floats
vec2 unpack2Unorm8(uint u) {
    return vec2(u & 0xFFu, (u >> 8) & 0xFFu) / 255.0;
}

#endif // PACK8_GLSL
