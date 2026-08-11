#version 430 core

// ===========================================================================
// Pass: ReSTIR 空域+时域加权蓄水池 (composite58)
// ===========================================================================
// 8 点 Poisson 盘空间采样 (无共享内存) + 降噪先验 + 时域重投影
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/common.glsl"
#include "/lib/lighting/alice.glsl"

uniform usampler2D colortex4; // atrous 降噪 ALICE (packAlice 格式, RGBA32UI)
uniform usampler2D colortex6; // temporal_diffuse validKernelWeight

layout(rgba32ui) uniform writeonly uimage2D colorimg6;

const uint POISSON_N = 8u;
#ifndef PATHGUIDE_SPATIAL_RADIUS
#define PATHGUIDE_SPATIAL_RADIUS 16.0
#endif
#ifndef PATHGUIDE_MAX_TEMPORAL_M
#define PATHGUIDE_MAX_TEMPORAL_M 64.0
#endif

// NRD Poisson 盘: .xy=偏移 .z=length .w=高斯权重
const vec4 POISSON[8] = {
    vec4(-0.4706069, -0.4427112, +0.6461146, +0.81170),
    vec4(-0.9057375, +0.3003471, +0.9542373, +0.63422),
    vec4(-0.3487388, +0.4037880, +0.5335386, +0.86734),
    vec4(+0.1023042, +0.6439373, +0.6520134, +0.80847),
    vec4(+0.5699277, +0.3513750, +0.6695386, +0.79925),
    vec4(+0.2939128, -0.1131226, +0.3149309, +0.95161),
    vec4(+0.7836658, -0.4208784, +0.8895339, +0.67328),
    vec4(+0.1564120, -0.8198990, +0.8346850, +0.70589)
    };

// ---------------------------------------------------------------------------
// PCG RNG
// ---------------------------------------------------------------------------
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float nextFloat(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967296.0;
}

// ---------------------------------------------------------------------------
// 蓄水池
// ---------------------------------------------------------------------------
struct Reservoir {
    vec4 y;
    float weightSum;
    float M;
    float target;
};
void reservoirInit(out Reservoir r) {
    r.y = vec4(0.0);
    r.weightSum = 0.0;
    r.M = 0.0;
    r.target = 0.0;
}
// Weighted reservoir update. selectionWeight controls candidate selection,
// while candidateTarget is p_hat(candidate) evaluated at the current pixel.
// A reused reservoir uses selectionWeight = p_hat * W * M, but its selected
// target remains p_hat.
void reservoirUpdate(inout Reservoir r, vec4 candidate, float selectionWeight,
        float candidateTarget, float m, inout uint seed) {
    if (!(m > 0.0) || isnan(m) || isinf(m)
            || isnan(selectionWeight) || isinf(selectionWeight)
            || isnan(candidateTarget) || isinf(candidateTarget)
            || any(isnan(candidate)) || any(isinf(candidate))) {
        return;
    }

    r.M += m;
    // Zero-target candidates are still part of M, but can never be selected.
    if (!(selectionWeight > 0.0) || !(candidateTarget > 0.0)) {
        return;
    }

    float newWeightSum = r.weightSum + selectionWeight;
    if (nextFloat(seed) * newWeightSum < selectionWeight) {
        r.y = candidate;
        r.target = candidateTarget;
    }
    r.weightSum = newWeightSum;
}

// Limiting temporal memory must scale both M and sum(w_i). Scaling only M
// changes W = sum(w_i) / (M * p_hat(y)) and artificially amplifies history.
void reservoirClampM(inout Reservoir r, float maxM) {
    if (r.M > maxM) {
        float scale = maxM / r.M;
        r.weightSum *= scale;
        r.M = maxM;
    }
}
float reservoirW(Reservoir r) {
    float denominator = r.M * r.target;
    if (!(r.weightSum > 0.0) || !(denominator > 0.0)
            || isnan(r.weightSum) || isinf(r.weightSum)
            || isnan(denominator) || isinf(denominator)) {
        return 0.0;
    }
    float W = r.weightSum / denominator;
    return (!isnan(W) && !isinf(W)) ? W : 0.0;
}

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------
bool isSky(vec4 y) {
    return y.w <= 1e-8 || any(isnan(y)) || any(isinf(y));
}

uvec2 packAliceHalf(vec4 y) {
    return uvec2(packHalf2x16(y.xy), packHalf2x16(y.zw));
}
vec4 unpackAliceHalf(uvec2 p) {
    return vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
}

float guideTarget(vec4 y, vec3 N) {
    return alice_irradiance(y, N);
}

// ---------------------------------------------------------------------------
// Phase 2: 8 点 Poisson 盘空间蓄水池 (直接 SSBO 读)
// ---------------------------------------------------------------------------
Reservoir spatialReservoir(uvec2 gid, vec3 centerNormal, vec3 centerPos, inout uint seed) {
    Reservoir r;
    reservoirInit(r);
    ivec2 texSize = ivec2(resolution_global);

    float centerDistance = max(length(centerPos), 0.01);
    float pixelFootprint = max(centerDistance /
        max(float(resolution_global.y), 1.0), 1e-4);
    float invGeometryScale = 1.0 / max(
        float(ATROUS_POSITION_PARAM) * pixelFootprint, 1e-6);
    float centerPlaneDistance = dot(centerPos, centerNormal);

    float theta = 2.0 * PI * nextFloat(seed);
    mat2 rot = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * PATHGUIDE_SPATIAL_RADIUS;

    for (uint k = 0u; k < POISSON_N; k++) {
        vec4 ps = POISSON[k];
        ivec2 off = ivec2(round(rot * ps.xy));
        ivec2 sc = ivec2(gid) + off;
        if (any(lessThan(sc, ivec2(0))) || any(greaterThanEqual(sc, texSize))) continue;

        uvec2 xy = uvec2(sc);
        AliceEncoding alice;
        float meanY2_unused;
        readDiffuseLightRT(xy, alice, meanY2_unused);
        vec4 y = alice.aliceY;
        if (isSky(y) || readDiffuseSurfaceMask(xy) < 0.5) continue;

        // 深度不连续拒绝
        vec3 pos;
        float dist;
        readGeo0(GEO_N_GEO, xy, pos, dist);
        // Plane distance in units of the center pixel's world-space footprint.
        // ATROUS_POSITION_PARAM is a scale, so it belongs in the denominator;
        // multiplying by its small value would make almost every edge weight 1.
        float planeDistance = abs(dot(pos, centerNormal) -
            centerPlaneDistance);
        float geomW = exp2(-planeDistance * invGeometryScale);
        if (geomW <= 1e-4) continue;

        float target = max(guideTarget(y, centerNormal), 0.0);
        reservoirUpdate(r, y, target * geomW, target, 1.0, seed);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Phase 3: 降噪先验
// ---------------------------------------------------------------------------
void addDenoisedPrior(inout Reservoir r, ivec2 pix, vec3 centerNormal, inout uint seed) {
    uvec4 raw = texelFetch(colortex4, pix, 0);
    AliceEncoding a;
    a.aliceY = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
    a.CoCg   = unpackHalf2x16(raw.z);
    vec4 y = a.aliceY;
    if (isSky(y)) return;
    float target = max(guideTarget(y, centerNormal), 0.0);
    reservoirUpdate(r, y, target, target, 1.0, seed);
}

// ---------------------------------------------------------------------------
// Phase 4: 2×2 随机重投影历史蓄水池
// ---------------------------------------------------------------------------
struct StoredReservoir {
    vec4 y;
    float W;
    float M;
};

StoredReservoir loadStored(ivec2 pix) {
    uvec4 raw = diffuseBuffer.data[addr(DIF_N_PATHGUIDE, uvec2(pix))];
    StoredReservoir s;
    s.y = vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
    s.W = uintBitsToFloat(raw.z);
    s.M = uintBitsToFloat(raw.w);
    return s;
}

bool sampleHistory(uvec2 gxy, vec3 curPos, float curDist, inout uint seed,
        out StoredReservoir result) {
    result.y = vec4(0.0);
    result.W = 0.0;
    result.M = 0.0;

    if (curDist <= -0.5) return false;

    vec3 surfaceMotion;
    float motionValid;
    readSurfaceMotion(gxy, surfaceMotion, motionValid);
    if (motionValid < 0.5) return false;

    vec3 prevPos = curPos + camPos - prevRaytracingCamPos - surfaceMotion;
    vec4 clip = rtPrevViewProjection * vec4(prevPos, 1.0);
    if (clip.w <= 1e-6) return false;
    vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) return false;

    vec2 prevCoord = uv * vec2(resolution_global);
    ivec2 p0 = ivec2(floor(prevCoord));
    vec2 f = fract(prevCoord);

    ivec2 offset = ivec2(nextFloat(seed) < f.x ? 1 : 0, nextFloat(seed) < f.y ? 1 : 0);
    result = loadStored(clamp(p0 + offset, ivec2(0), ivec2(resolution_global) - 1));
    return result.W > 0.0 && result.M > 0.0
        && !isnan(result.W) && !isinf(result.W)
        && !isnan(result.M) && !isinf(result.M)
        && !any(isnan(result.y)) && !any(isinf(result.y));
}

// ---------------------------------------------------------------------------
// Phase 5: 历史 RIS 合并
// ---------------------------------------------------------------------------
void combineWithHistory(inout Reservoir r, StoredReservoir hist, float temporalConf, vec3 centerNormal, inout uint seed) {
    float targetNow = max(guideTarget(hist.y, centerNormal), 0.0);

    float reusedM = min(hist.M, PATHGUIDE_MAX_TEMPORAL_M) * clamp(temporalConf, 0.0, 1.0);
    if (reusedM <= 0.0) return;

    float candidateWeight = targetNow * hist.W * reusedM;
    reservoirUpdate(r, hist.y, candidateWeight, targetNow, reusedM, seed);
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    ivec2 texSize = ivec2(resolution_global);
    if (any(greaterThanEqual(gid, uvec2(texSize)))) return;

    uvec2 gxy = gid;
    ivec2 pix = ivec2(gid);
    uint seed = pcg_hash(gid.x ^ pcg_hash(gid.y ^ pcg_hash(frame_id)));

    // 中心像素数据
    vec3 centerNormal, centerPos;
    float centerDist;
    {
        readGeo0(GEO_N_GEO, gxy, centerPos, centerDist);
    }

    // Guiding is consumed only by surface rays. Skip eight scattered probes,
    // random rotation and history reprojection for sky pixels.
    if (centerDist <= -0.5) {
        imageStore(colorimg6, pix, uvec4(0u));
        return;
    }
    float roughnessUnused, pathRoughnessUnused;
    int materialUnused;
    readGeo1(GEO_N_NORMALS, gxy, centerNormal, roughnessUnused,
        materialUnused, pathRoughnessUnused);

    // Phase 2: Poisson 盘空间蓄水池
    Reservoir r = spatialReservoir(gid, centerNormal, centerPos, seed);

    // Phase 3: 降噪先验
    addDenoisedPrior(r, pix, centerNormal, seed);

    // Phase 4+5: 历史重投影 + RIS 合并
    StoredReservoir hist;
    if (sampleHistory(gxy, centerPos, centerDist, seed, hist)) {
        float temporalConf = clamp(uintBitsToFloat(texelFetch(colortex6, pix, 0).r), 0.0, 1.0);
        combineWithHistory(r, hist, temporalConf, centerNormal, seed);
    }

    reservoirClampM(r, PATHGUIDE_MAX_TEMPORAL_M);
    float W = reservoirW(r);

    uvec2 halfY = packAliceHalf(r.y);
    imageStore(colorimg6, pix, uvec4(halfY.x, halfY.y, floatBitsToUint(W), floatBitsToUint(r.M)));
}
