#ifndef BUFFERS_DIFFUSE_BUFFER_GLSL
#define BUFFERS_DIFFUSE_BUFFER_GLSL

#include "/lib/buffers/addr.glsl"
#include "/lib/buffers/debug_buffer.glsl"
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/common/pack_half.glsl"
#include "/lib/lighting/maxent_encode.glsl"

// ===========================================================================
// Binding 2 — DiffuseBuffer pack/unpack (uvec4 raw-integer storage)
// ===========================================================================
// N=0: Current Light  — MaxEnt6 + current second moment.
// N=1: History Light  — MaxEnt6 + Kish N_eff/rootMeanY2.
// N=2: History Geo A  — F32 distance + oct ray + oct normal + N_eff/frame stamp.
// N=3: Swap Light     — MaxEnt6 + Kish N_eff/rootMeanY2.
// N=4: Path Guide     — MaxEnt4 + F32 reservoir W/M.
// N=5: ReSTIR GI first-hit direct-light MaxEnt atom.
// N=6: ReSTIR GI endpoint distance/direction + signed first-direction PDF.
// N=7: Current-frame biased ReSTIR GI path-guide prewarm MaxEnt atom.
// N=8: Macro normal, diffuse material, motion and denoised-history difference.
// N=9: Alternate history geometry for race-free frame ping-pong.
// N=10..11: Exact previous/current denoiser RGBA32UI output ping-pong.
// N=12..13: Shared independent-current spatial-filter ping-pong. Diffuse and
// reflection execute serially and reuse these transient planes.
// N=14: Shared scalar metadata: x/y are FP32 spatial N_eff ping-pong and z is
// the FP32 diffuse filtered-history N_eff reconstructed for the current pixel.
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

// Active fifteen-plane layout. Current geometry is owned by geomBuffer and is
// reconstructed from pixel + RT projection and F32 distance; it is not
// duplicated here.
// DIF_N_HISTGEO/ALT = F32 distance + oct ray + oct normal + N_eff/frame stamp.
// DIF_N_RESTIR_ENDPOINT = F32 distance + oct direction + F32 signed PDF.
#define DIF_N_LIGHT    0u
#define DIF_N_HIST     1u
#define DIF_N_HISTGEO  2u
#define DIF_N_SWAP     3u
#define DIF_N_PATHGUIDE 4u
#define DIF_N_RESTIR_DIRECT   5u
#define DIF_N_RESTIR_ENDPOINT 6u
#define DIF_N_RESTIR_PREWARM  7u
#define DIF_N_SURFACE          8u
#define DIF_N_HISTGEO_ALT      9u
#define DIF_N_DENOISED_A      10u
#define DIF_N_DENOISED_B      11u
#define DIF_N_CURRENT_A       12u
#define DIF_N_CURRENT_B       13u
#define DIF_N_DENOISER_META   14u

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
    return packedWeight | ((uint(frame_id) & 0xffffu) << 16u);
}

bool unpackDiffusePreviousHistoryWeight(uint packed_, out float historyWeight) {
    historyWeight = unpackHalf2x16(packed_).x;
    uint expectedStamp = (uint(frame_id) - 1u) & 0xffffu;
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

float readDenoiserIndependentCurrentEffectiveSamplesA(ivec2 pixel) {
    return uintBitsToFloat(diffuseBuffer.data[addr(DIF_N_DENOISER_META, pixel)].x);
}

float readDenoiserIndependentCurrentEffectiveSamplesB(ivec2 pixel) {
    return uintBitsToFloat(diffuseBuffer.data[addr(DIF_N_DENOISER_META, pixel)].y);
}

void writeDenoiserIndependentCurrentEffectiveSamplesA(ivec2 pixel, float effectiveSamples) {
    atomicExchange(diffuseBuffer.data[addr(DIF_N_DENOISER_META, pixel)].x, floatBitsToUint(effectiveSamples));
}

void writeDenoiserIndependentCurrentEffectiveSamplesB(ivec2 pixel, float effectiveSamples) {
    atomicExchange(diffuseBuffer.data[addr(DIF_N_DENOISER_META, pixel)].y, floatBitsToUint(effectiveSamples));
}

void writeDiffuseDenoisedReprojectedEffectiveSamples(uvec2 xy, float effectiveSamples) {
    atomicExchange(diffuseBuffer.data[addr(DIF_N_DENOISER_META, xy)].z, floatBitsToUint(effectiveSamples));
}

float readDiffuseDenoisedReprojectedEffectiveSamples(uvec2 xy) {
    return uintBitsToFloat(diffuseBuffer.data[addr(DIF_N_DENOISER_META, xy)].z);
}

// Before history resolve, the current parity is scratch for the previous
// denoised signal reprojected by the real diffuse temporal pass. Resolve reads
// it once, then replaces it with the exact current colortex4 words. Scratch z
// stores the CoCg belonging to the same denoised estimator as maxEntY; w stores
// (filtered MC standardDeviation, valid reprojection mass). The denoised estimator N_eff is reconstructed separately
// into DIF_N_DENOISER_META.z because Raw history owns an independent N_eff.
void writeDiffuseDenoisedReprojection(uvec2 xy, vec4 maxEntY, vec2 CoCg,
        float monteCarloStandardDeviation, float effectiveSamples, float validWeight) {
    diffuseBuffer.data[addr(diffuseDenoisedWritePlane(), xy)] = uvec4(
        packHalf2x16(clamp(maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(CoCg, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(clamp(monteCarloStandardDeviation, 0.0, 65504.0),
            clamp(validWeight, 0.0, 1.0))));
    writeDiffuseDenoisedReprojectedEffectiveSamples(xy, effectiveSamples);
}

void writeDiffuseDenoisedReprojectionInvalid(uvec2 xy) {
    diffuseBuffer.data[addr(diffuseDenoisedWritePlane(), xy)] = uvec4(
        0u, 0u, 0u, packHalf2x16(vec2(-1.0, 0.0)));
    writeDiffuseDenoisedReprojectedEffectiveSamples(xy, 0.0);
}

bool readDiffuseDenoisedReprojection(uvec2 xy, out vec4 maxEntY,
        out vec2 CoCg, out float monteCarloStandardDeviation, out float validWeight) {
    uvec4 words = readDiffuseDenoisedCurrentRaw(xy);
    CoCg = unpackHalf2x16(words.z);
    vec2 deviationWeight = unpackHalf2x16(words.w);
    monteCarloStandardDeviation = deviationWeight.x;
    validWeight = deviationWeight.y;
    bool valid = monteCarloStandardDeviation >= 0.0 && validWeight > 0.0
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
    effectiveSamples = readDiffuseDenoisedReprojectedEffectiveSamples(xy);
    return valid && effectiveSamples >= 1.0 && !isnan(effectiveSamples) && !isinf(effectiveSamples);
}

uvec4 packDiffuseTemporalState(MaxEntEncoding maxent,
        float effectiveSamples, float rootMeanY2) {
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
    rootMeanY2 = sanitizeRootMeanSquareFP16(rootMeanY2);
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
// N=8 -- current diffuse-domain material and motion state
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
// N=4 — ReSTIR temporal reservoir (Path Guide Reservoir)
// ===========================================================================
// Layout: uvec4(
//   packHalf2x16(maxEntY.xy),   // sample direction × luminance
//   packHalf2x16(maxEntY.zw),   // total energy
//   floatBitsToUint(W),        // reservoir reciprocal-proposal normalization
//   floatBitsToUint(M))        // effective sample count

void writePathGuide(uvec2 xy, vec4 maxEntY, float W, float M) {
    diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)] = uvec4(
        packHalf2x16(clamp(maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
        floatBitsToUint(W),
        floatBitsToUint(M));
}
void readPathGuide(uvec2 xy, out vec4 maxEntY, out float W, out float M) {
    uvec4 v = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, xy)];
    vec2 ay_xy = unpackHalf2x16(v.x);
    vec2 ay_zw = unpackHalf2x16(v.y);
    maxEntY = vec4(ay_xy, ay_zw);
    W = uintBitsToFloat(v.z);
    M = uintBitsToFloat(v.w);
}

bool pathGuideReservoirValid(vec4 maxEntY, float W, float M) {
    return W > 0.0 && M > 0.0
        && !isnan(W) && !isinf(W) && !isnan(M) && !isinf(M)
        && !any(isnan(maxEntY)) && !any(isinf(maxEntY));
}

// 2×2 bilinear path guide sampling with validity mask
vec4 samplePathGuide(vec2 prevCoord) {
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2  pf = prevCoord - vec2(p0);

    vec4 y00, y10, y01, y11;
    float W00, W10, W01, W11;
    float M00, M10, M01, M11;
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 0), ivec2(0), ivec2(resolution_global) - 1)), y00, W00, M00);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 0), ivec2(0), ivec2(resolution_global) - 1)), y10, W10, M10);
    readPathGuide(uvec2(clamp(p0 + ivec2(0, 1), ivec2(0), ivec2(resolution_global) - 1)), y01, W01, M01);
    readPathGuide(uvec2(clamp(p0 + ivec2(1, 1), ivec2(0), ivec2(resolution_global) - 1)), y11, W11, M11);

    bool v00 = pathGuideReservoirValid(y00, W00, M00);
    bool v10 = pathGuideReservoirValid(y10, W10, M10);
    bool v01 = pathGuideReservoirValid(y01, W01, M01);
    bool v11 = pathGuideReservoirValid(y11, W11, M11);

    float w00 = v00 ? (1.0 - pf.x) * (1.0 - pf.y) : 0.0;
    float w10 = v10 ? pf.x * (1.0 - pf.y) : 0.0;
    float w01 = v01 ? (1.0 - pf.x) * pf.y : 0.0;
    float w11 = v11 ? pf.x * pf.y : 0.0;

    float sumW = w00 + w10 + w01 + w11;
    if (sumW < 1e-8) return vec4(0.0);

    return (y00 * w00 + y10 * w10 + y01 * w01 + y11 * w11) / sumW;
}

// ===========================================================================
// N=5..7 -- low-history ReSTIR GI path-guiding prewarm scratch
// ===========================================================================

struct RestirGIFreshCandidate {
    vec3 endpointRelative;
    float firstPdf;
    bool environment;
};

void writeRestirGIMaxEntPlane(uint plane, uvec2 xy,
        MaxEntEncoding encoded, float rootMeanY2) {
    // Never allow a bad donor or color transform to poison the finalize and
    // temporal passes. The biased prewarm itself performs no density division.
    if (any(isnan(encoded.maxEntY)) || any(isinf(encoded.maxEntY))
            || any(isnan(encoded.CoCg)) || any(isinf(encoded.CoCg)))
        encoded = init_maxent();
    rootMeanY2 = sanitizeRootMeanSquareFP16(rootMeanY2);
    diffuseBuffer.data[addr(plane, xy)] = uvec4(
        packHalf2x16(clamp(encoded.maxEntY.xy,
            vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(encoded.maxEntY.zw,
            vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(encoded.CoCg,
            vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(vec2(0.0, rootMeanY2)));
}

void readRestirGIMaxEntPlane(uint plane, uvec2 xy,
        out MaxEntEncoding encoded, out float rootMeanY2) {
    uvec4 packedValue = diffuseBuffer.data[addr(plane, xy)];
    vec2 xyValue = unpackHalf2x16(packedValue.x);
    vec2 zwValue = unpackHalf2x16(packedValue.y);
    encoded.maxEntY = vec4(xyValue, zwValue);
    encoded.CoCg = unpackHalf2x16(packedValue.z);
    rootMeanY2 = unpackHalf2x16(packedValue.w).y;
    if (any(isnan(encoded.maxEntY)) || any(isinf(encoded.maxEntY))
            || any(isnan(encoded.CoCg)) || any(isinf(encoded.CoCg)))
        encoded = init_maxent();
    if (isnan(rootMeanY2) || isinf(rootMeanY2) || rootMeanY2 < 0.0)
        rootMeanY2 = 0.0;
}

void clearRestirGIScratch(uvec2 xy) {
    diffuseBuffer.data[addr(DIF_N_RESTIR_DIRECT, xy)] = uvec4(0u);
    diffuseBuffer.data[addr(DIF_N_RESTIR_ENDPOINT, xy)] = uvec4(0u);
    diffuseBuffer.data[addr(DIF_N_RESTIR_PREWARM, xy)] = uvec4(0u);
}

void writeRestirGIDirect(uvec2 xy, MaxEntEncoding direct) {
    writeRestirGIMaxEntPlane(DIF_N_RESTIR_DIRECT, xy, direct,
        max(direct.maxEntY.w, 0.0));
}

void readRestirGIDirect(uvec2 xy, out MaxEntEncoding direct) {
    float unusedRootMeanY2;
    readRestirGIMaxEntPlane(DIF_N_RESTIR_DIRECT, xy,
        direct, unusedRootMeanY2);
}

void writeRestirGIPrewarm(uvec2 xy, MaxEntEncoding indirect) {
    writeRestirGIMaxEntPlane(DIF_N_RESTIR_PREWARM, xy, indirect,
        max(indirect.maxEntY.w, 0.0));
}

void readRestirGIPrewarm(uvec2 xy, out MaxEntEncoding indirect) {
    float unusedRootMeanY2;
    readRestirGIMaxEntPlane(DIF_N_RESTIR_PREWARM, xy,
        indirect, unusedRootMeanY2);
}

void writeRestirGIFreshCandidate(uvec2 xy,
        RestirGIFreshCandidate candidate) {
    float signedPdf = candidate.environment
        ? -abs(candidate.firstPdf) : abs(candidate.firstPdf);
    float endpointDistance = length(candidate.endpointRelative);
    vec3 endpointDirection = endpointDistance > 1e-8
        ? candidate.endpointRelative / endpointDistance
        : vec3(0.0, 0.0, -1.0);
    diffuseBuffer.data[addr(DIF_N_RESTIR_ENDPOINT, xy)] = uvec4(
        floatBitsToUint(endpointDistance),
        encodeDiffuseHistoryNormalU(endpointDirection),
        floatBitsToUint(signedPdf),
        0u);
}

RestirGIFreshCandidate readRestirGIFreshCandidate(uvec2 xy) {
    uvec4 endpointData =
        diffuseBuffer.data[addr(DIF_N_RESTIR_ENDPOINT, xy)];
    float endpointDistance = uintBitsToFloat(endpointData.x);
    float signedPdf = uintBitsToFloat(endpointData.z);
    RestirGIFreshCandidate candidate;
    candidate.endpointRelative =
        decodeDiffuseHistoryNormalU(endpointData.y) * endpointDistance;
    candidate.firstPdf = abs(signedPdf);
    candidate.environment = signedPdf < 0.0;
    return candidate;
}

bool restirGIFreshCandidateValid(RestirGIFreshCandidate candidate) {
    return candidate.firstPdf > 1e-8
        && !isnan(candidate.firstPdf) && !isinf(candidate.firstPdf)
        && !any(isnan(candidate.endpointRelative))
        && !any(isinf(candidate.endpointRelative));
}

#endif // BUFFERS_DIFFUSE_BUFFER_GLSL
