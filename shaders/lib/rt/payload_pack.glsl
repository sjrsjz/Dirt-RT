#ifndef PAYLOAD_PACK_GLSL
#define PAYLOAD_PACK_GLSL

#include "/lib/common/oct_encode.glsl"
#include "/lib/common/pack8.glsl"

// Block ID macros — synchronized with shaders/block.properties and lib/constants.glsl.
// Conditionally defined so they are always available regardless of include order.
#ifndef BLOCK_WATER
#define BLOCK_WATER  1000 // water
#define BLOCK_GLASS  1001 // ice, stained glass (all colors + panes), blue_ice, packed_ice
#define BLOCK_PORTAL 1002 // nether_portal (frosted / translucent emissive)
#endif

// ===========================================================================
// Ultra-minimal payload pack/unpack — 9/16 slots used, 7 free.
//
// rgen reconstructs everything from {instanceIndex, primitiveID, barycentrics}
// via geometryBuffers[]. The payload only carries geometric identification +
// accumulated volume data that must survive traceRayEXT traversal.
//
// Slot map:
//  [0] f32 pos.x                                — hit position (floatBitsToUint)
//  [1] f32 pos.y
//  [2] f32 pos.z
//  [3] f32 hitT                                 — hit distance
//  [4] instanceCustomIndex (u32)                — geometryBuffers[] index
//  [5] geometryIndex (u16 low) | primitiveID (u16 high)
//  [6] f16(bary.x) | f16(bary.y)               — packHalf2x16
//  [7] packUnorm4x8(shadow.r, shadow.g, shadow.b, 0) — volume transmission
//  [8] f16(prevDist) | u16(flags)              — packHalf2x16
//      flags: bit0=inside, bits1-2=bounce(0-3), bit3=handedness,
//             bits4-5=ignoreID_enc, bits6-15=free
//  [9-15] FREE
// ===========================================================================

#define PAYLOAD_SLOTS 16

// ---------------------------------------------------------------------------
// Hit position + distance [0-3] — f32, full precision for ray continuation
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
// instanceIdx = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT
// primID      = gl_PrimitiveID  (quad index = primID >> 1)
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
// Shadow [7] — 3×unorm8 + blockID_enc in 4th byte
// blockID_enc: 0=normal, 1/255=water(BLOCK_WATER), 2/255=glass(BLOCK_GLASS), 3/255=portal(BLOCK_PORTAL)
// ---------------------------------------------------------------------------
void payload_packShadow(inout uint d[PAYLOAD_SLOTS], vec3 st, int blockID) {
    float bEnc = 0.0;
    if (blockID == BLOCK_WATER) bEnc = 1.0 / 255.0;
    else if (blockID == BLOCK_GLASS) bEnc = 2.0 / 255.0;
    else if (blockID == BLOCK_PORTAL) bEnc = 3.0 / 255.0;
    d[7] = packUnorm4x8(vec4(st, bEnc));
}
vec3 payload_unpackShadow(uint d[PAYLOAD_SLOTS], out int blockID) {
    vec4 v = unpackUnorm4x8(d[7]);
    int bEnc = int(v.a * 255.0 + 0.5);
    blockID = bEnc == 1 ? BLOCK_WATER : (bEnc == 2 ? BLOCK_GLASS : (bEnc == 3 ? BLOCK_PORTAL : 0));
    return v.rgb;
}
// Backward-compat overload
vec3 payload_unpackShadow(uint d[PAYLOAD_SLOTS]) {
    int _;
    return payload_unpackShadow(d, _);
}

// ---------------------------------------------------------------------------
// Flags + prevDist [8]
//
// ignoreID encoding (2 bits):
//   0 = none, 1 = water (BLOCK_WATER), 2 = glass (BLOCK_GLASS), 3 = portal (BLOCK_PORTAL)
// ---------------------------------------------------------------------------
void payload_packFlags(inout uint d[PAYLOAD_SLOTS], float prevDist,
    bool inside, uint bounce, bool handedness, uint ignoreID_enc) {
    uint f = (inside ? 1u : 0u)
            | ((bounce & 0x3u) << 1)
            | (handedness ? 8u : 0u)
            | ((ignoreID_enc & 0x3u) << 4);
    d[8] = packHalf2x16(vec2(prevDist, float(f)));
}

float payload_unpackFlags(uint d[PAYLOAD_SLOTS],
    out bool inside, out uint bounce, out bool handedness,
    out uint ignoreID_enc) {
    vec2 v = unpackHalf2x16(d[8]);
    uint f = uint(v.y + 0.5);
    inside = (f & 1u) != 0u;
    bounce = (f >> 1) & 0x3u;
    handedness = (f & 8u) != 0u;
    ignoreID_enc = (f >> 4) & 0x3u;
    return v.x;
}

// ---------------------------------------------------------------------------
// ignoreID helpers — encode/decode block ID ↔ 2-bit compact form
// ---------------------------------------------------------------------------
int payload_decodeIgnoreID(uint enc) {
    return enc == 1u ? BLOCK_WATER : (enc == 2u ? BLOCK_GLASS : (enc == 3u ? BLOCK_PORTAL : 0));
}
uint payload_encodeIgnoreID(int blockID) {
    if (blockID == BLOCK_WATER) return 1u;
    if (blockID == BLOCK_GLASS) return 2u;
    if (blockID == BLOCK_PORTAL) return 3u;
    return 0u;
}

// ---------------------------------------------------------------------------
// Quad-derived data for material evaluation in rgen [9-15]
// Packed by rchit so rgen doesn't need geometryBuffers[] (avoids set=1 binding issues).
// ---------------------------------------------------------------------------

// Global UV [9-10] — f32 for sub-texel precision (f16 loses ~0.5 texel on 512×512 atlas)
void payload_packQuadUV(inout uint d[16], vec2 uv) {
    d[9] = floatBitsToUint(uv.x);
    d[10] = floatBitsToUint(uv.y);
}
vec2 payload_unpackQuadUV(uint d[16]) {
    return vec2(uintBitsToFloat(d[9]), uintBitsToFloat(d[10]));
}

// Atlas box [11-12] — f16 is fine for clamping
void payload_packAtlasBox(inout uint d[16], vec4 box) {
    d[11] = packHalf2x16(box.xy);
    d[12] = packHalf2x16(box.zw);
}
vec4 payload_unpackAtlasBox(uint d[16]) {
    return vec4(unpackHalf2x16(d[11]), unpackHalf2x16(d[12]));
}

// Geometry normal — oct-encoded [13]
void payload_packGeomNormal(inout uint d[16], vec3 n) {
    d[13] = floatBitsToUint(encodeNormal(n));
}
vec3 payload_unpackGeomNormal(uint d[16]) {
    return decodeNormal(uintBitsToFloat(d[13]));
}

// Tangent — oct-encoded [14]
void payload_packTangent(inout uint d[16], vec3 t) {
    d[14] = floatBitsToUint(encodeNormal(t));
}
vec3 payload_unpackTangent(uint d[16]) {
    return decodeNormal(uintBitsToFloat(d[14]));
}

// Vertex tint + skylight [15] — 4×unorm8
// blockID is in shadow[7].a, bitangent sign in flags[8].bit3 (handedness)
void payload_packQuadExtras(inout uint d[16], vec3 tint, float sky) {
    d[15] = packUnorm4x8(vec4(tint, sky));
}
void payload_unpackQuadExtras(uint d[16], out vec3 tint, out float sky) {
    vec4 v = unpackUnorm4x8(d[15]);
    tint = v.rgb;
    sky = v.a;
}

#endif // PAYLOAD_PACK_GLSL
