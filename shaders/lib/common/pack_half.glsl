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

uint pack2HalfClampedU(float a, float b) {
    return packHalf2x16(
        vec2(clamp(a, -65504.0, 65504.0), clamp(b, -65504.0, 65504.0)));
}

float pack2HalfClamped(float a, float b) {
    return uintBitsToFloat(pack2HalfClampedU(a, b));
}

// Storage APIs traffic in root second moments directly.  They deliberately do
// not accept E[L^2] and hide a sqrt in the packer: the caller owns the one
// conversion from its arithmetic representation to the storage ABI.
float sanitizeRootMeanSquareFP16(float rootMeanSquare) {
    if (isnan(rootMeanSquare) || rootMeanSquare <= 0.0) return 0.0;
    if (isinf(rootMeanSquare)) return 65504.0;
    return min(rootMeanSquare, 65504.0);
}

#endif // PACK_HALF_GLSL
