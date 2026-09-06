#ifndef BUFFERS_DIFFUSE_BUFFER_GLSL
#define BUFFERS_DIFFUSE_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/buffers/debug_buffer.glsl"
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/math/denoiser_uncertainty.glsl"
#include "/lib/lighting/maxent_encode.glsl"

// ===========================================================================
// Binding 2 — DiffuseBuffer pack/unpack (uvec4 raw-integer storage)
// ===========================================================================
// N=0: Current Light  — MaxEnt6 + current second moment.
// N=1: History Light  — MaxEnt6 + Kish N_eff/rootMeanY2.
// N=2: History Geo A  — F32 distance + oct ray + oct normal + N_eff/frame stamp.
// N=3: Swap Light     — MaxEnt6 + Kish N_eff/rootMeanY2.
// N=4: Denoised Path Guide -- packed directional/energy moments + validity tags.
// N=5: Macro normal, diffuse material, motion and denoised-history difference.
// N=6: Alternate history geometry for race-free frame ping-pong.
// N=7..8: Exact previous/current denoiser output ping-pong.
// N=9..10: Shared independent-current ping-pong, reused serially by domains.
//
// .w lane uses packHalf2x16: Kish N_eff:f16 + rootMeanY2:f16.
// Storage APIs accept sqrt(E[Y²]) directly. Arithmetic code squares it only
// where linear second-moment operations require E[Y²].
// Current position, geometry normal and validity come from compact primary
// geometry; a negative primary distance is the only sky/no-surface marker.

MaxEntEncoding sanitizeDiffuseMaxEntEncoding(MaxEntEncoding maxent) {
    if (any(isnan(maxent.maxEntY)) || any(isinf(maxent.maxEntY))
            || any(isnan(maxent.CoCg)) || any(isinf(maxent.CoCg)))
        return init_maxent();
    // This helper enforces only the FP16 storage domain. Moment-cone and RGB-feasibility projections belong to
    // the decoder; applying them to encoder or latent state would make linear moment filtering nonlinear.
    maxent.maxEntY = clamp(maxent.maxEntY, vec4(-65504.0), vec4(65504.0));
    maxent.CoCg = clamp(maxent.CoCg, vec2(-65504.0), vec2(65504.0));
    return maxent;
}

// Active eleven-plane layout. Current geometry is owned by geomBuffer and is
// reconstructed from pixel + RT projection and F32 distance; it is not
// duplicated here.
// DIF_N_HISTGEO/ALT = F32 distance + oct ray + oct normal + N_eff/frame stamp.
#define DIF_N_LIGHT    0u
#define DIF_N_HIST     1u
#define DIF_N_HISTGEO  2u
#define DIF_N_SWAP     3u
#define DIF_N_PATHGUIDE 4u
#define DIF_N_SURFACE          5u
#define DIF_N_HISTGEO_ALT      6u
#define DIF_N_DENOISED_A      7u
#define DIF_N_DENOISED_B      8u
#define DIF_N_CURRENT_A       9u
#define DIF_N_CURRENT_B       10u

uint diffuseHistoryGeometryWritePlane() {
    return (uint(frame_id) & 1u) == 0u ? DIF_N_HISTGEO : DIF_N_HISTGEO_ALT;
}

uint diffuseHistoryGeometryReadPlane() {
    return (uint(frame_id) & 1u) == 0u ? DIF_N_HISTGEO_ALT : DIF_N_HISTGEO;
}

uint diffuseDenoisedWritePlane() {
    return (uint(frame_id) & 1u) == 0u ? DIF_N_DENOISED_A : DIF_N_DENOISED_B;
}

uint diffuseDenoisedReadPlane() {
    return (uint(frame_id) & 1u) == 0u ? DIF_N_DENOISED_B : DIF_N_DENOISED_A;
}

uint packDiffuseHistoryWeightStamp(float historyWeight) {
    if (!(historyWeight >= 0.0) || isinf(historyWeight)) historyWeight = 0.0;
    uint packedWeight = packHalf2x16(vec2(historyWeight, 0.0)) & 0xffffu;
    uint stamp = (uint(frame_id) & 0xffffu) ^ 0x2500u;
#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 1
    stamp ^= 0x5a00u;
#endif
    return packedWeight | (stamp << 16u);
}

bool unpackDiffusePreviousHistoryWeight(uint packed_, out float historyWeight) {
    historyWeight = unpackHalf2x16(packed_).x;
    uint expectedStamp = ((uint(frame_id) - 1u) & 0xffffu) ^ 0x2500u;
#if MAXENT_TEMPORAL_CONFIDENCE_CLAMP == 1
    expectedStamp ^= 0x5a00u;
#endif
    return (packed_ >> 16u) == expectedStamp && historyWeight > 0.0 && !isnan(historyWeight) && !isinf(historyWeight);
}

void writeDiffuseDenoisedCurrentRaw(uvec2 xy, uvec4 words) {
    diffuseBuffer.data[addr(diffuseDenoisedWritePlane(), xy)] = words;
}

void writeDiffuseDenoisedCurrentRaw(uvec2 xy, uvec4 words, float effectiveSamples) {
    vec2 metadata = unpackHalf2x16(words.w);
    // Negative N_eff is the persistent diffuse-history layout tag; spatial signals use this lane for nonnegative
    // virtual distance, so stale pre-change records cannot be mistaken for valid estimator statistics.
    metadata.y = -clamp(effectiveSamples, 1.0, 65504.0);
    words.w = packHalf2x16(metadata);
    writeDiffuseDenoisedCurrentRaw(xy, words);
}

uvec4 readDiffuseDenoisedCurrentRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(diffuseDenoisedWritePlane(), xy)];
}

uvec4 readDiffuseDenoisedPreviousRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(diffuseDenoisedReadPlane(), xy)];
}

void writeDiffuseIndependentCurrentA(uvec2 xy, uvec4 words) {
    diffuseBuffer.data[addr(DIF_N_CURRENT_A, xy)] = words;
}

void writeDiffuseIndependentCurrentB(uvec2 xy, uvec4 words) {
    diffuseBuffer.data[addr(DIF_N_CURRENT_B, xy)] = words;
}

uvec4 readDiffuseIndependentCurrentA(uvec2 xy) {
    return diffuseBuffer.data[addr(DIF_N_CURRENT_A, xy)];
}

uvec4 readDiffuseIndependentCurrentB(uvec2 xy) {
    return diffuseBuffer.data[addr(DIF_N_CURRENT_B, xy)];
}

// Before history resolve, the current parity is scratch for the previous
// denoised signal reprojected by the real diffuse temporal pass. Resolve reads
// it once, then replaces it with the exact current colortex4 words. Scratch z
// stores the CoCg belonging to the same denoised estimator as maxEntY; w stores
// (estimator standardDeviation, valid reprojection mass). The legacy sample-count
// argument is reserved; filtered uncertainty is carried entirely by sigma.
void writeDiffuseDenoisedReprojection(uvec2 xy, vec4 maxEntY, vec2 CoCg,
        float monteCarloStandardDeviation, float effectiveSamples, float validWeight) {
    diffuseBuffer.data[addr(diffuseDenoisedWritePlane(), xy)] = uvec4(
        packHalf2x16(clamp(maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(CoCg, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(denoiserSigmaOrUnknown(monteCarloStandardDeviation),
            clamp(validWeight, 0.0, 1.0))));
}

void writeDiffuseDenoisedReprojectionInvalid(uvec2 xy) {
    diffuseBuffer.data[addr(diffuseDenoisedWritePlane(), xy)] = uvec4(
        0u, 0u, 0u, packHalf2x16(vec2(-1.0, 0.0)));
}

bool readDiffuseDenoisedReprojection(uvec2 xy, out vec4 maxEntY,
        out vec2 CoCg, out float monteCarloStandardDeviation, out float validWeight) {
    uvec4 words = readDiffuseDenoisedCurrentRaw(xy);
    CoCg = unpackHalf2x16(words.z);
    vec2 deviationWeight = unpackHalf2x16(words.w);
    monteCarloStandardDeviation = deviationWeight.x;
    validWeight = deviationWeight.y;
    bool valid = denoiserSigmaUsable(monteCarloStandardDeviation) && validWeight > 0.0
        && !any(isnan(CoCg)) && !any(isinf(CoCg))
        && !any(isnan(deviationWeight)) && !any(isinf(deviationWeight));
    maxEntY = valid
        ? vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y))
        : vec4(0.0);
    valid = valid && !any(isnan(maxEntY)) && !any(isinf(maxEntY));
    if (!valid) CoCg = vec2(0.0);
    return valid;
}

bool readDiffuseDenoisedReprojection(uvec2 xy, out vec4 maxEntY, out vec2 CoCg,
        out float monteCarloStandardDeviation, out float effectiveSamples, out float validWeight) {
    bool valid = readDiffuseDenoisedReprojection(xy, maxEntY, CoCg, monteCarloStandardDeviation, validWeight);
    effectiveSamples = 1.0; // Reserved legacy ABI, no metadata fetch.
    return valid && effectiveSamples >= 1.0 && !isnan(effectiveSamples) && !isinf(effectiveSamples);
}

uvec4 packDiffuseTemporalState(MaxEntEncoding maxent,
        float effectiveSamples, float rootMeanY2) {
    // Do not clear only the mean and keep a seemingly valid orphaned RMS.
    if (!denoiserTemporalMomentsFinite(maxent.maxEntY, maxent.CoCg, rootMeanY2)) {
        effectiveSamples = 0.0;
        rootMeanY2 = 0.0;
    }
    maxent = sanitizeDiffuseMaxEntEncoding(maxent);
    effectiveSamples = effectiveSamples >= 1.0
            && effectiveSamples <= 65504.0
            && !isnan(effectiveSamples) && !isinf(effectiveSamples)
        ? effectiveSamples : 0.0;
    rootMeanY2 = sanitizeRootMeanSquareFP16(rootMeanY2);
    return uvec4(
        packHalf2x16(clamp(maxent.maxEntY.xy,
            vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxent.maxEntY.zw,
            vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxent.CoCg,
            vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(effectiveSamples, rootMeanY2)));
}

void unpackDiffuseTemporalState(uvec4 words, out MaxEntEncoding maxent,
        out float effectiveSamples, out float rootMeanY2) {
    maxent.maxEntY = clamp(vec4(unpackHalf2x16(words.x),
        unpackHalf2x16(words.y)), vec4(-65504.0), vec4(65504.0));
    maxent.CoCg = clamp(unpackHalf2x16(words.z),
        vec2(-65504.0), vec2(65504.0));
    maxent = sanitizeDiffuseMaxEntEncoding(maxent);
    vec2 metadata = unpackHalf2x16(words.w);
    effectiveSamples = metadata.x;
    rootMeanY2 = sanitizeRootMeanSquareFP16(metadata.y);
}

// ===========================================================================
// N=0 — Current RT Light
// ===========================================================================
// .w = packHalf2x16(0.0, rootMeanY2)

void writeDiffuseLightRT(uvec2 xy, MaxEntEncoding maxent,
        float rootMeanY2) {
    maxent = sanitizeDiffuseMaxEntEncoding(maxent);
    // One RT packet: RMS=abs(R), paired with the same sanitized packet.
    rootMeanY2 = max(maxent.maxEntY.w, 0.0);
    diffuseBuffer.data[addr(DIF_N_LIGHT, xy)] = uvec4(
        packHalf2x16(clamp(maxent.maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxent.maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxent.CoCg,        vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(0.0, rootMeanY2))
    );
}
uvec4 readDiffuseLightRTRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(DIF_N_LIGHT, xy)];
}
void readDiffuseLightRT(uvec2 xy, out MaxEntEncoding maxent,
        out float rootMeanY2) {
    uvec4 v = readDiffuseLightRTRaw(xy);
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    vec2 cocg  = unpackHalf2x16(v.z);
    maxent.maxEntY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    maxent.CoCg = cocg;
    maxent = sanitizeDiffuseMaxEntEncoding(maxent);
    vec2 wm = unpackHalf2x16(v.w);
    rootMeanY2 = sanitizeRootMeanSquareFP16(wm.y);
}

// Helper: write zero light (sky reset)
void writeDiffuseLightRTSky(uvec2 xy) {
    diffuseBuffer.data[addr(DIF_N_LIGHT, xy)] = uvec4(0u);
}

// ===========================================================================
// Current Geometry (shared compact primary G-buffer; no diffuse plane)
// ===========================================================================

void readDiffusePrimaryGeometry(uvec2 xy, out vec3 position,
        out float distance) {
    readPrimaryPosition(xy, position, distance);
}

float readDiffuseSurfaceMask(uvec2 xy) {
    return readPrimaryDistance(xy) >= 0.0 ? 1.0 : 0.0;
}

// ===========================================================================
// N=5 -- current diffuse-domain material and motion state
// ===========================================================================

uint encodeDiffuseNormalOct8(vec3 n) {
    n = normalize(n);
    vec2 p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(p, vec2(0.0)));
    return packUnorm4x8(vec4(p * 0.5 + 0.5, 0.0, 0.0)) & 0xffffu;
}

vec3 decodeDiffuseNormalOct8(uint packedNormal) {
    vec2 p = unpackUnorm4x8(packedNormal & 0xffffu).xy * 2.0 - 1.0;
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0),
            greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

void writeDiffuseSurface(uvec2 xy, vec3 macroNormal,
        vec3 diffuseAlbedo, float roughness, vec3 motion,
        float motionValid) {
    diffuseBuffer.data[addr(DIF_N_SURFACE, xy)] = uvec4(
        encodeDiffuseNormalOct8(macroNormal),
        packUnorm4x8(clamp(vec4(diffuseAlbedo, roughness), 0.0, 1.0)),
        packSnorm4x8(vec4(clamp(motion / 4.0, -1.0, 1.0),
            motionValid >= 0.5 ? 1.0 : 0.0)), 0u);
}

void readDiffuseSurface(uvec2 xy, out vec3 geometryNormal,
        out vec3 macroNormal, out vec3 diffuseAlbedo, out float roughness) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_SURFACE, xy)];
    geometryNormal = readPrimaryGeometryNormal(xy);
    macroNormal = decodeDiffuseNormalOct8(v.x);
    vec4 materialState = unpackUnorm4x8(v.y);
    diffuseAlbedo = materialState.rgb;
    roughness = materialState.a;
}

void readDiffuseMotion(uvec2 xy, out vec3 motion, out float valid) {
    vec4 packedMotion = unpackSnorm4x8(
        diffuseBuffer.data[addr(DIF_N_SURFACE, xy)].z);
    motion = packedMotion.xyz * 4.0;
    valid = packedMotion.w > 0.5 ? 1.0 : 0.0;
}

// ===========================================================================
// N=1 — History Light
// ===========================================================================
// .w = packHalf2x16(N_eff, rootMeanY2)

void writeDiffuseHist(uvec2 xy, MaxEntEncoding maxent, float weight,
        float rootMeanY2) {
    diffuseBuffer.data[addr(DIF_N_HIST, xy)] =
        packDiffuseTemporalState(maxent, weight, rootMeanY2);
}
void readDiffuseHist(uvec2 xy, out MaxEntEncoding maxent, out float weight,
        out float rootMeanY2) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_HIST, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    vec2 cocg  = unpackHalf2x16(v.z);
    maxent.maxEntY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    maxent.CoCg = cocg;
    maxent = sanitizeDiffuseMaxEntEncoding(maxent);
    vec2 wm = unpackHalf2x16(v.w);
    weight = wm.x;
    rootMeanY2 = sanitizeRootMeanSquareFP16(wm.y);
}

// ===========================================================================
// N=2 — History Geometry
// ===========================================================================

uint encodeDiffuseHistoryNormalU(vec3 n) {
    n = normalize(n);
    vec2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    return packSnorm2x16(clamp(p, vec2(-1.0), vec2(1.0)));
}

vec3 decodeDiffuseHistoryNormalU(uint packed_) {
    vec2 p = unpackSnorm2x16(packed_);
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

void writeDiffuseHistGeo(uvec2 xy, vec3 pos, vec3 geometryNormal, float historyWeight) {
    float distance = length(pos);
    vec3 primaryRay = distance > 1e-8 ? pos / distance : vec3(0.0, 0.0, -1.0);
    diffuseBuffer.data[addr(diffuseHistoryGeometryWritePlane(), xy)] = uvec4(
        floatBitsToUint(distance),
        encodeDiffuseHistoryNormalU(primaryRay),
        encodeDiffuseHistoryNormalU(geometryNormal),
        packDiffuseHistoryWeightStamp(historyWeight)
    );
}

void writeDiffuseHistGeoInvalid(uvec2 xy) {
    diffuseBuffer.data[addr(diffuseHistoryGeometryWritePlane(), xy)] = uvec4(
        floatBitsToUint(-1.0), 0u, 0u, packDiffuseHistoryWeightStamp(0.0));
}

uvec4 readDiffuseHistGeoRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(diffuseHistoryGeometryReadPlane(), xy)];
}

void readDiffuseHistGeo(uvec2 xy, out vec3 pos, out vec3 geometryNormal) {
    uvec4 v = readDiffuseHistGeoRaw(xy);
    float historyWeight;
    if (!unpackDiffusePreviousHistoryWeight(v.w, historyWeight)) {
        pos = vec3(0.0);
        geometryNormal = vec3(0.0);
        return;
    }
    float distance = uintBitsToFloat(v.x);
    pos = decodeDiffuseHistoryNormalU(v.y) * distance;
    geometryNormal = decodeDiffuseHistoryNormalU(v.z);
}

// ===========================================================================
// N=3 — Swap Light
// ===========================================================================
// .w = packHalf2x16(N_eff, rootMeanY2)

void writeDiffuseSwap(uvec2 xy, MaxEntEncoding maxent, float weight,
        float rootMeanY2) {
    diffuseBuffer.data[addr(DIF_N_SWAP, xy)] =
        packDiffuseTemporalState(maxent, weight, rootMeanY2);
}
uvec4 readDiffuseSwapRaw(uvec2 xy) {
    return diffuseBuffer.data[addr(DIF_N_SWAP, xy)];
}
void readDiffuseSwap(uvec2 xy, out MaxEntEncoding maxent, out float weight,
        out float rootMeanY2) {
    uvec4 v = readDiffuseSwapRaw(xy);
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    vec2 cocg  = unpackHalf2x16(v.z);
    maxent.maxEntY = clamp(vec4(ay_xy, ay_zw), vec4(-65504.0), vec4(65504.0));
    maxent.CoCg = clamp(cocg, vec2(-65504.0), vec2(65504.0));
    maxent = sanitizeDiffuseMaxEntEncoding(maxent);
    vec2 wm = unpackHalf2x16(v.w);
    weight = wm.x;
    rootMeanY2 = sanitizeRootMeanSquareFP16(wm.y);
}

// ===========================================================================
// N=4 -- Final denoised path-guide moments
// ===========================================================================
// Layout: uvec4(
//   packHalf2x16(maxEntY.xy),   // sample direction × luminance
//   packHalf2x16(maxEntY.zw),   // total energy
//   floatBitsToUint(1.0),      // valid-state tag
//   floatBitsToUint(1.0))      // valid-state tag

bool readPathGuide(uvec2 xy, out vec4 maxEntY) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)];
    maxEntY = vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
    // Fixed tags preserve the tested direct-guide layout; no W/M state.
    bool valid = v.z == floatBitsToUint(1.0) && v.w == floatBitsToUint(1.0)
        && maxEntY.w > 1e-8 && !any(isnan(maxEntY)) && !any(isinf(maxEntY));
    if (!valid) maxEntY = vec4(0.0);
    return valid;
}

// 2×2 bilinear path guide sampling with validity mask
vec4 samplePathGuide(vec2 prevCoord) {
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2  pf = prevCoord - vec2(p0);

    vec4 y00, y10, y01, y11;
    bool v00 = readPathGuide(uvec2(clamp(p0 + ivec2(0, 0), ivec2(0), ivec2(resolution_global) - 1)), y00);
    bool v10 = readPathGuide(uvec2(clamp(p0 + ivec2(1, 0), ivec2(0), ivec2(resolution_global) - 1)), y10);
    bool v01 = readPathGuide(uvec2(clamp(p0 + ivec2(0, 1), ivec2(0), ivec2(resolution_global) - 1)), y01);
    bool v11 = readPathGuide(uvec2(clamp(p0 + ivec2(1, 1), ivec2(0), ivec2(resolution_global) - 1)), y11);


    float w00 = v00 ? (1.0 - pf.x) * (1.0 - pf.y) : 0.0;
    float w10 = v10 ? pf.x * (1.0 - pf.y) : 0.0;
    float w01 = v01 ? (1.0 - pf.x) * pf.y : 0.0;
    float w11 = v11 ? pf.x * pf.y : 0.0;

    float sumW = w00 + w10 + w01 + w11;
    if (sumW < 1e-8) return vec4(0.0);

    return (y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11) / sumW;
}

#endif // BUFFERS_DIFFUSE_BUFFER_GLSL
