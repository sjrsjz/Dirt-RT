#ifndef PAYLOAD_PACK_GLSL
#define PAYLOAD_PACK_GLSL

#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack8.glsl"

// Block ID macros — synchronized with shaders/block.properties and lib/constants.glsl.
#ifndef BLOCK_WATER
#define BLOCK_WATER  1000 // water
#define BLOCK_GLASS  1001 // ice, stained glass (all colors + panes), blue_ice, packed_ice
#define BLOCK_PORTAL 1002 // nether_portal (frosted / translucent emissive)
#endif

// ===========================================================================
// Payload pack/unpack — 9/16 slots used, 7 free.
//
// Slot map:
//  [0] f32 pos.x                                — hit position (floatBitsToUint)
//  [1] f32 pos.y
//  [2] f32 pos.z
//  [3] f32 hitT                                 — hit distance
//  [4] instanceCustomIndex (u32)                — geometryBuffers[] index
//  [5] geometryIndex (u16 low) | primitiveID (u16 high)
//  [6] f16(bary.x) | f16(bary.y)               — packHalf2x16
//  [7] packUnorm4x8(shadow.r, shadow.g, shadow.b, blockID_enc)
//  [8] f16(prevDist) | u16(flags)
//      flags: bit0=inside, bit1=handedness, bit2=isNEE
//  [9-15] FREE
// ===========================================================================

#define PAYLOAD_SLOTS 16

// ---------------------------------------------------------------------------
// Hit position + distance [0-3]
// ---------------------------------------------------------------------------
void payload_packHitPos(inout uint d[PAYLOAD_SLOTS], vec3 pos, float t) {
    d[0] = floatBitsToUint(pos.x);
    d[1] = floatBitsToUint(pos.y);
    d[2] = floatBitsToUint(pos.z);
    d[3] = floatBitsToUint(t);
}
vec3 payload_unpackHitPos(uint d[PAYLOAD_SLOTS], out float t) {
    t = uintBitsToFloat(d[3]);
    return vec3(uintBitsToFloat(d[0]), uintBitsToFloat(d[1]), uintBitsToFloat(d[2]));
}

// ---------------------------------------------------------------------------
// Quad identification [4-5]
// ---------------------------------------------------------------------------
void payload_packQuadIDs(inout uint d[PAYLOAD_SLOTS], uint instanceIdx, uint geomIdx, uint primID) {
    d[4] = instanceIdx;
    d[5] = (geomIdx & 0xFFFFu) | ((primID & 0xFFFFu) << 16);
}
void payload_unpackQuadIDs(uint d[PAYLOAD_SLOTS], out uint instanceIdx, out uint geomIdx, out uint primID) {
    instanceIdx = d[4];
    geomIdx = d[5] & 0xFFFFu;
    primID = (d[5] >> 16) & 0xFFFFu;
}

// ---------------------------------------------------------------------------
// Barycentric coordinates [6] — f16×2
// ---------------------------------------------------------------------------
void payload_packBarycentrics(inout uint d[PAYLOAD_SLOTS], vec2 bary) {
    d[6] = packHalf2x16(bary);
}
vec2 payload_unpackBarycentrics(uint d[PAYLOAD_SLOTS]) {
    return unpackHalf2x16(d[6]);
}

// ---------------------------------------------------------------------------
// Shadow + blockID [7] — 3×unorm8 + blockID_enc in 4th byte
// ---------------------------------------------------------------------------
void payload_packShadow(inout uint d[PAYLOAD_SLOTS], vec3 st, int blockID) {
    float bEnc = 0.0;
    if      (blockID == BLOCK_WATER)  bEnc = 1.0 / 255.0;
    else if (blockID == BLOCK_GLASS)  bEnc = 2.0 / 255.0;
    else if (blockID == BLOCK_PORTAL) bEnc = 3.0 / 255.0;
    d[7] = packUnorm4x8(vec4(st, bEnc));
}
vec3 payload_unpackShadow(uint d[PAYLOAD_SLOTS], out int blockID) {
    vec4 v = unpackUnorm4x8(d[7]);
    int bEnc = int(v.a * 255.0 + 0.5);
    blockID = bEnc == 1 ? BLOCK_WATER : (bEnc == 2 ? BLOCK_GLASS : (bEnc == 3 ? BLOCK_PORTAL : 0));
    return v.rgb;
}
vec3 payload_unpackShadow(uint d[PAYLOAD_SLOTS]) {
    int _;
    return payload_unpackShadow(d, _);
}

// ---------------------------------------------------------------------------
// Flags + prevDist [8]
//   bit0: inside
//   bit1: handedness
//   bit2: isNEE  (direct sunlight shadow ray)
// ---------------------------------------------------------------------------
void payload_packFlags(inout uint d[PAYLOAD_SLOTS], float prevDist,
    bool inside, bool handedness, bool isNEE) {
    uint f = (inside ? 1u : 0u)
            | (handedness ? 2u : 0u)
            | (isNEE ? 4u : 0u);
    d[8] = packHalf2x16(vec2(prevDist, float(f)));
}

float payload_unpackFlags(uint d[PAYLOAD_SLOTS],
    out bool inside, out bool handedness, out bool isNEE) {
    vec2 v = unpackHalf2x16(d[8]);
    uint f = uint(v.y + 0.5);
    inside     = (f & 1u) != 0u;
    handedness = (f & 2u) != 0u;
    isNEE      = (f & 4u) != 0u;
    return v.x;
}

// ---------------------------------------------------------------------------
// Quad-derived data for material evaluation in rgen [9-15]
// ---------------------------------------------------------------------------

void payload_packQuadUV(inout uint d[16], vec2 uv) {
    d[9]  = floatBitsToUint(uv.x);
    d[10] = floatBitsToUint(uv.y);
}
vec2 payload_unpackQuadUV(uint d[16]) {
    return vec2(uintBitsToFloat(d[9]), uintBitsToFloat(d[10]));
}

void payload_packAtlasBox(inout uint d[16], vec4 box) {
    d[11] = packHalf2x16(box.xy);
    d[12] = packHalf2x16(box.zw);
}
vec4 payload_unpackAtlasBox(uint d[16]) {
    return vec4(unpackHalf2x16(d[11]), unpackHalf2x16(d[12]));
}

void payload_packGeomNormal(inout uint d[16], vec3 n) {
    d[13] = encodeNormalU(n);
}
vec3 payload_unpackGeomNormal(uint d[16]) {
    return decodeNormalU(d[13]);
}

void payload_packTangent(inout uint d[16], vec3 t) {
    d[14] = encodeNormalU(t);
}
vec3 payload_unpackTangent(uint d[16]) {
    return decodeNormalU(d[14]);
}

void payload_packQuadExtras(inout uint d[16], vec3 tint, float sky) {
    d[15] = packUnorm4x8(vec4(tint, sky));
}
void payload_unpackQuadExtras(uint d[16], out vec3 tint, out float sky) {
    vec4 v = unpackUnorm4x8(d[15]);
    tint = v.rgb;
    sky  = v.a;
}

#endif // PAYLOAD_PACK_GLSL
