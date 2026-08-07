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

// A luminance second moment grows quadratically and otherwise overflows FP16
// once luminance exceeds sqrt(65504) ~= 256. Store sqrt(E[L^2]) at FP16
// boundaries and square it immediately after loading. This also preserves
// moments that would underflow in dark regions. All filtering code continues
// to operate on the decoded E[L^2], never on the encoded value.
float encodeSqrtMomentFP16(float secondMoment) {
    if (isnan(secondMoment) || secondMoment <= 0.0) return 0.0;
    if (isinf(secondMoment)) return 65504.0;
    return min(sqrt(secondMoment), 65504.0);
}

float decodeSqrtMomentFP16(float encodedMoment) {
    if (isnan(encodedMoment) || isinf(encodedMoment) || encodedMoment <= 0.0)
        return 0.0;
    float rootMoment = min(encodedMoment, 65504.0);
    return rootMoment * rootMoment;
}

#endif // PACK_HALF_GLSL
