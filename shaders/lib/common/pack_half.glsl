#ifndef PACK_HALF_GLSL
#define PACK_HALF_GLSL

// ===========================================================================
// Half-float pack/unpack utilities
// ===========================================================================

float pack2Half(float a, float b) {
    return uintBitsToFloat(packHalf2x16(vec2(a, b)));
}

void unpack2Half(float packed_, out float a, out float b) {
    vec2 v = unpackHalf2x16(floatBitsToUint(packed_));
    a = v.x;
    b = v.y;
}

float pack2HalfClamped(float a, float b) {
    return uintBitsToFloat(packHalf2x16(
        vec2(clamp(a, -65504.0, 65504.0), clamp(b, -65504.0, 65504.0))));
}

#endif // PACK_HALF_GLSL
