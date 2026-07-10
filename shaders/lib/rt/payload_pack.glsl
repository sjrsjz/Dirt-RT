#ifndef PAYLOAD_PACK_GLSL
#define PAYLOAD_PACK_GLSL

#include "/lib/common/oct_encode.glsl"

// ===========================================================================
// Payload pack/unpack accessors for uint data[16] packed layout.
//
// Slot map:
//  [0] f32(pos.x)                              — hit position x (full float)
//  [1] f32(pos.y)                              — hit position y (full float)
//  [2] f32(pos.z)                              — hit position z (full float)
//  [3] f32(hitT)                               — hit distance (full float)
//  [4] encodeNormal(geometryNormal)            — macro surface normal
//  [5] f16(shadow.r)   | f16(shadow.g)        — volume transmission rg
//  [6] f16(shadow.b)   | f16(f0Channel)       — volume b + F0 index
//  [7] f16(albedo.r)   | f16(albedo.g)        — albedo rg
//  [8] f16(albedo.b)   | f16(alpha)           — albedo b + translucency
//  [9] f16(roughness)  | f16(subsurface)      — GGX roughness + SSS
// [10] encodeNormal(materialNormal)            — detail normal
// [11] f16(emission.r) | f16(emission.g)      — emission rg
// [12] f16(emission.b) | f16(ao)              — emission b + ambient occlusion
// [13] f16(wetStr)     | f16(wetness)         — wetness params
// [14] f16(block.x)    | f16(ignore.x)        — block type ID + self-intersect
// [15] f16(prev_dist)  | f16(flags)           — prev travel + flags
//       flags: bit0=inside_block, bit1=metallic, bits[2-4]=bounce_depth
// ===========================================================================

#define PAYLOAD_SLOTS 16

// ---------------------------------------------------------------------------
// Hit data — f32 (full precision for ray origin & distance)
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
// Macro normal (geometry)
// ---------------------------------------------------------------------------
void payload_packGeomNormal(inout uint d[PAYLOAD_SLOTS], vec3 n) {
    d[4] = floatBitsToUint(encodeNormal(n));
}
vec3 payload_unpackGeomNormal(uint d[PAYLOAD_SLOTS]) {
    return decodeNormal(uintBitsToFloat(d[4]));
}

// ---------------------------------------------------------------------------
// Volume transmission
// ---------------------------------------------------------------------------
void payload_packShadow(inout uint d[PAYLOAD_SLOTS], vec3 st) {
    d[5] = packHalf2x16(st.xy);
    vec2 v6 = unpackHalf2x16(d[6]);
    d[6] = packHalf2x16(vec2(st.z, v6.y));
}
vec3 payload_unpackShadow(uint d[PAYLOAD_SLOTS]) {
    vec2 v5 = unpackHalf2x16(d[5]);
    vec2 v6 = unpackHalf2x16(d[6]);
    return vec3(v5, v6.x);
}

// ---------------------------------------------------------------------------
// F0 channel (packed in upper half of data[6])
// ---------------------------------------------------------------------------
void payload_setF0(inout uint d[PAYLOAD_SLOTS], int channel) {
    vec2 v6 = unpackHalf2x16(d[6]);
    d[6] = packHalf2x16(vec2(v6.x, float(channel)));
}
int payload_getF0(uint d[PAYLOAD_SLOTS]) {
    return int(unpackHalf2x16(d[6]).y + 0.5);
}

// ---------------------------------------------------------------------------
// Albedo (rgb) + translucency (alpha)
// ---------------------------------------------------------------------------
void payload_packAlbedo(inout uint d[PAYLOAD_SLOTS], vec3 rgb, float alpha) {
    d[7] = packHalf2x16(rgb.xy);
    d[8] = packHalf2x16(vec2(rgb.z, alpha));
}
vec4 payload_unpackAlbedo(uint d[PAYLOAD_SLOTS]) {
    vec2 v7 = unpackHalf2x16(d[7]);
    vec2 v8 = unpackHalf2x16(d[8]);
    return vec4(v7, v8.x, v8.y);
}

// ---------------------------------------------------------------------------
// BSDF: roughness + subsurface scattering
// ---------------------------------------------------------------------------
void payload_packBSDF(inout uint d[PAYLOAD_SLOTS], float rough, float subsurf) {
    d[9] = packHalf2x16(vec2(rough, subsurf));
}
vec2 payload_unpackBSDF(uint d[PAYLOAD_SLOTS]) {
    return unpackHalf2x16(d[9]);
}

// ---------------------------------------------------------------------------
// Detail (material) normal
// ---------------------------------------------------------------------------
void payload_packMatNormal(inout uint d[PAYLOAD_SLOTS], vec3 n) {
    d[10] = floatBitsToUint(encodeNormal(n));
}
vec3 payload_unpackMatNormal(uint d[PAYLOAD_SLOTS]) {
    return decodeNormal(uintBitsToFloat(d[10]));
}

// ---------------------------------------------------------------------------
// Emission + ambient occlusion
// ---------------------------------------------------------------------------
void payload_packEmissionAO(inout uint d[PAYLOAD_SLOTS], vec3 em, float ao) {
    d[11] = packHalf2x16(em.xy);
    d[12] = packHalf2x16(vec2(em.z, ao));
}
void payload_unpackEmissionAO(uint d[PAYLOAD_SLOTS], out vec3 em, out float ao) {
    vec2 v11 = unpackHalf2x16(d[11]);
    vec2 v12 = unpackHalf2x16(d[12]);
    em = vec3(v11, v12.x);
    ao = v12.y;
}

// ---------------------------------------------------------------------------
// Wetness
// ---------------------------------------------------------------------------
void payload_packWetness(inout uint d[PAYLOAD_SLOTS], float strength, float wet) {
    d[13] = packHalf2x16(vec2(strength, wet));
}
vec2 payload_unpackWetness(uint d[PAYLOAD_SLOTS]) {
    return unpackHalf2x16(d[13]);
}

// ---------------------------------------------------------------------------
// Block ID + ignore block ID
// ---------------------------------------------------------------------------
void payload_packBlockIDs(inout uint d[PAYLOAD_SLOTS], int blockID, int ignoreID) {
    d[14] = packHalf2x16(vec2(float(blockID), float(ignoreID)));
}
void payload_unpackBlockIDs(uint d[PAYLOAD_SLOTS], out int blockID, out int ignore) {
    vec2 v = unpackHalf2x16(d[14]);
    blockID = int(v.x + 0.5);
    ignore  = int(v.y + 0.5);
}

// ---------------------------------------------------------------------------
// Flags: inside_block(bit0) | metallic(bit1) | bounce_depth(bits 2-4)
// ---------------------------------------------------------------------------
void payload_packFlags(inout uint d[PAYLOAD_SLOTS], float prevDist,
                       bool inside, bool metal, uint bounce) {
    uint f = (inside ? 1u : 0u) | (metal ? 2u : 0u) | ((bounce & 0x7u) << 2u);
    d[15] = packHalf2x16(vec2(prevDist, float(f)));
}
float payload_unpackFlags(uint d[PAYLOAD_SLOTS],
                          out bool inside, out bool metal, out uint bounce) {
    vec2 v = unpackHalf2x16(d[15]);
    uint f = uint(v.y + 0.5);
    inside = (f & 1u) != 0u;
    metal  = (f & 2u) != 0u;
    bounce = (f >> 2u) & 0x7u;
    return v.x;
}

#endif // PAYLOAD_PACK_GLSL
