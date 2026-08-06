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

const float VPROJDIST_SKY = 60000.0;

float pack2HalfClamped(float a, float b) {
    return uintBitsToFloat(packHalf2x16(
        vec2(clamp(a, -65504.0, 65504.0), clamp(b, -65504.0, 65504.0))));
}

// ===========================================================================
// ufloat24 — 24-bit unsigned float (8-bit exponent + 16-bit mantissa)
// ===========================================================================
// Derived from IEEE float32 by dropping the sign bit and 7 low mantissa bits.
// Dynamic range matches f32; relative precision ≈ 2⁻¹⁷.
// Used for storing non-negative second-moment values in packed SSBO lanes.

const uint U24_MASK  = 0x00FFFFFFu;
const uint UF24_MAX  = 0x00FEFFFFu; // max finite, excludes Inf/NaN

uint packUFloat24(float x) {
    if (isnan(x) || x <= 0.0) return 0u;
    if (isinf(x)) return UF24_MAX;

    uint bits = floatBitsToUint(x);
    // Round-to-nearest-even: add 0x3f + retained_lsb
    uint rounded = bits + 0x3fu + ((bits >> 7u) & 1u);
    uint encoded = rounded >> 7u;
    return min(encoded, UF24_MAX);
}

float unpackUFloat24(uint encoded) {
    encoded &= U24_MASK;
    return uintBitsToFloat(encoded << 7u);
}

// ===========================================================================
// History meta pack — weight:u8 | meanY2:ufloat24
// ===========================================================================
// Packs an 8-bit integer history weight and a non-negative second moment
// into a single 32-bit unsigned integer. Use with uvec4 SSBO storage.

uint packHistoryMeta(uint historyWeight, float meanY2) {
    uint w = min(historyWeight, 255u);
    return (w << 24u) | packUFloat24(meanY2);
}

void unpackHistoryMeta(uint p, out uint historyWeight, out float meanY2) {
    historyWeight = p >> 24u;
    meanY2 = unpackUFloat24(p & U24_MASK);
}

#endif // PACK_HALF_GLSL
