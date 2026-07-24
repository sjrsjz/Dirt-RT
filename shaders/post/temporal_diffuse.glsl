#version 430 compatibility

// ===========================================================================
// Pass 100 CS: 漫反射时域累积 (当前像素梯形 → 历史空间厚梯形版)
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f) uniform writeonly image2D colorimg6;

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/lighting/alice.glsl"

// ===========================================================================
// Uniform 输入 & 宏定义
// ===========================================================================

uniform vec2 resolution;

#ifndef TEMPORAL_MAX_HISTORY
#define TEMPORAL_MAX_HISTORY 32.0
#endif

#ifndef TEMPORAL_HISTORY_MIN_WEIGHT
#define TEMPORAL_HISTORY_MIN_WEIGHT 1e-4
#endif

#ifndef TEMPORAL_DEPTH_FOOTPRINT_SCALE
#define TEMPORAL_DEPTH_FOOTPRINT_SCALE 1.0
#endif

#ifndef TEMPORAL_CLIP_PIXEL_RADIUS
#define TEMPORAL_CLIP_PIXEL_RADIUS 1.5
#endif

#ifndef TEMPORAL_GEOMETRY_EPSILON
#define TEMPORAL_GEOMETRY_EPSILON 1e-5
#endif

#ifndef TEMPORAL_AABB_ENABLE
#define TEMPORAL_AABB_ENABLE 1
#endif

#ifndef TEMPORAL_AABB_NEIGHBOR_RADIUS
#define TEMPORAL_AABB_NEIGHBOR_RADIUS 2
#endif

#ifndef TEMPORAL_AABB_EXPAND
#define TEMPORAL_AABB_EXPAND 2.0
#endif

#ifndef TEMPORAL_AABB_SIGMA_SCALE
#define TEMPORAL_AABB_SIGMA_SCALE 3.0
#endif

#ifndef TEMPORAL_AABB_MIN_EXTENT
#define TEMPORAL_AABB_MIN_EXTENT 1.0
#endif

#ifndef TEMPORAL_AABB_BOX_SCALE
#define TEMPORAL_AABB_BOX_SCALE 1.0
#endif

#ifndef TEMPORAL_AABB_MIN_VALID_NEIGHBORS
#define TEMPORAL_AABB_MIN_VALID_NEIGHBORS 2
#endif

// ===========================================================================
// 共享内存 AABB tile
// ===========================================================================

#define TILE_SIZE 16u

#if TEMPORAL_AABB_ENABLE
#define AABB_HALO uint(TEMPORAL_AABB_NEIGHBOR_RADIUS)
#define AABB_SM_W (TILE_SIZE + 2u * AABB_HALO)
#define AABB_SM_H (TILE_SIZE + 2u * AABB_HALO)

struct AABBTileSample {
    bool valid;
    vec4 aliceY;
    vec2 CoCg;
};

shared AABBTileSample sm_aabbTile[AABB_SM_H][AABB_SM_W];
#endif

// ===========================================================================
// 局部数据结构 & 全局变量
// ===========================================================================

struct TemporalFootprint {
    vec3 origin;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    vec2 plane0, plane1, plane2, plane3;
    float depthHalfExtent;
    float planeEdgeEpsilon;
};

vec3 prevScreenPos;
vec3 cameraDelta;
float info_distance;
vec3 currentNormal; // from Geo1, for buildTemporalFootprint tangent frame

DiffuseIlluminationWriteData current_data;
diffuseIlluminationData out_data;
float output_weight = 0.0;

// ===========================================================================
// 基础数学与几何工具
// ===========================================================================

float cross2(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

vec3 intersectCorner(vec2 ndc, mat4 invVP, vec3 planeO, vec3 planeN, out bool valid) {
    vec4 nearH = invVP * vec4(ndc, -1.0, 1.0);
    vec4 farH = invVP * vec4(ndc, 1.0, 1.0);
    vec3 ro = nearH.xyz / nearH.w;
    vec3 rd = farH.xyz / farH.w - ro;
    float denom = dot(rd, planeN);
    valid = abs(denom) > 1e-6;
    return ro + rd * (dot(planeO - ro, planeN) / denom);
}

bool buildTemporalFootprint(uvec2 pix, vec3 currentPos, vec3 geometricNormal, vec3 camDelta, out TemporalFootprint fp) {
    float nLenSq = dot(geometricNormal, geometricNormal);
    if (nLenSq < 1e-8) return false;

    fp.normal = geometricNormal * inversesqrt(nLenSq);

    // Frisvad 标准正交基
    if (fp.normal.z < -0.999999) {
        fp.tangent = vec3(0.0, -1.0, 0.0);
        fp.bitangent = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + fp.normal.z);
        float c = -fp.normal.x * fp.normal.y * a;
        fp.tangent = vec3(1.0 - fp.normal.x * fp.normal.x * a, c, -fp.normal.x);
        fp.bitangent = vec3(c, 1.0 - fp.normal.y * fp.normal.y * a, -fp.normal.y);
    }

    mat4 invVP = inverse(rtProjection * rtModelView);
    vec2 curRes = vec2(resolution);
    vec2 uvMin = (vec2(pix) - TEMPORAL_CLIP_PIXEL_RADIUS) / curRes * 2.0 - 1.0;
    vec2 uvMax = (vec2(pix) + TEMPORAL_CLIP_PIXEL_RADIUS) / curRes * 2.0 - 1.0;

    bool v0, v1, v2, v3;
    fp.origin = currentPos + camDelta;

    vec3 w0 = intersectCorner(vec2(uvMin.x, uvMin.y), invVP, currentPos, fp.normal, v0) + camDelta;
    vec3 w1 = intersectCorner(vec2(uvMax.x, uvMin.y), invVP, currentPos, fp.normal, v1) + camDelta;
    vec3 w2 = intersectCorner(vec2(uvMax.x, uvMax.y), invVP, currentPos, fp.normal, v2) + camDelta;
    vec3 w3 = intersectCorner(vec2(uvMin.x, uvMax.y), invVP, currentPos, fp.normal, v3) + camDelta;

    if (!(v0 && v1 && v2 && v3)) return false;

    fp.plane0 = vec2(dot(w0 - fp.origin, fp.tangent), dot(w0 - fp.origin, fp.bitangent));
    fp.plane1 = vec2(dot(w1 - fp.origin, fp.tangent), dot(w1 - fp.origin, fp.bitangent));
    fp.plane2 = vec2(dot(w2 - fp.origin, fp.tangent), dot(w2 - fp.origin, fp.bitangent));
    fp.plane3 = vec2(dot(w3 - fp.origin, fp.tangent), dot(w3 - fp.origin, fp.bitangent));

    float footprintDiameter = max(length(fp.plane2 - fp.plane0), length(fp.plane3 - fp.plane1));
    fp.depthHalfExtent = footprintDiameter * TEMPORAL_DEPTH_FOOTPRINT_SCALE;
    fp.planeEdgeEpsilon = TEMPORAL_GEOMETRY_EPSILON * max(footprintDiameter, 1.0);

    return true;
}

bool strictHistoryGeometryTest(vec3 histPos, TemporalFootprint fp) {
    vec3 delta = histPos - fp.origin;
    if (abs(dot(delta, fp.normal)) > fp.depthHalfExtent) return false;

    vec2 p = vec2(dot(delta, fp.tangent), dot(delta, fp.bitangent));

    float e0 = cross2(fp.plane1 - fp.plane0, p - fp.plane0);
    float e1 = cross2(fp.plane2 - fp.plane1, p - fp.plane1);
    float e2 = cross2(fp.plane3 - fp.plane2, p - fp.plane2);
    float e3 = cross2(fp.plane0 - fp.plane3, p - fp.plane3);
    float eps = fp.planeEdgeEpsilon;

    if (!((e0 >= -eps && e1 >= -eps && e2 >= -eps && e3 >= -eps) ||
            (e0 <= eps && e1 <= eps && e2 <= eps && e3 <= eps))) {
        return false;
    }

    // ALICE 编码的方向信息已隐含法线一致性 — 移除显式半球检查
    return true;
}

// ===========================================================================
// AABB 邻域钳制
// ===========================================================================

#if TEMPORAL_AABB_ENABLE
void computeAABB_CS(out vec4 minAY, out vec4 maxAY, out vec2 minCC, out vec2 maxCC, out int validCnt) {
    int cx = int(gl_LocalInvocationID.x + AABB_HALO);
    int cy = int(gl_LocalInvocationID.y + AABB_HALO);

    vec4 cenAY = current_data.data_swap.aliceY;
    vec2 cenCC = current_data.data_swap.CoCg;

    minAY = maxAY = cenAY;
    minCC = maxCC = cenCC;
    validCnt = 1;

    vec4 sumAY = cenAY;
    vec4 sumSqAY = cenAY * cenAY;

    for (int dy = -TEMPORAL_AABB_NEIGHBOR_RADIUS; dy <= TEMPORAL_AABB_NEIGHBOR_RADIUS; dy++) {
        for (int dx = -TEMPORAL_AABB_NEIGHBOR_RADIUS; dx <= TEMPORAL_AABB_NEIGHBOR_RADIUS; dx++) {
            if (dx == 0 && dy == 0) continue;

            AABBTileSample s = sm_aabbTile[cy + dy][cx + dx];
            if (!s.valid) continue;

            minAY = min(minAY, s.aliceY);
            maxAY = max(maxAY, s.aliceY);
            minCC = min(minCC, s.CoCg);
            maxCC = max(maxCC, s.CoCg);

            sumAY += s.aliceY;
            sumSqAY += s.aliceY * s.aliceY;
            validCnt++;
        }
    }

    vec4 extAY = maxAY - minAY;
    vec2 extCC = maxCC - minCC;

    float sigmaA = sqrt(max(0.0, alice_variance(cenAY)));
    float invN = 1.0 / float(validCnt);
    vec4 meanAY = sumAY * invN;
    vec4 nbVar = max(vec4(0.0), sumSqAY * invN - meanAY * meanAY);
    float sigmaN = sqrt(max(0.0, max(max(nbVar.x, nbVar.y), max(nbVar.z, nbVar.w))));

    float sigCombined = max(sigmaA, sigmaN);
    float sigExp = sigCombined * TEMPORAL_AABB_SIGMA_SCALE;

    vec4 expAY = (extAY * TEMPORAL_AABB_EXPAND + sigExp + TEMPORAL_AABB_MIN_EXTENT) * TEMPORAL_AABB_BOX_SCALE;
    vec2 expCC = (extCC * TEMPORAL_AABB_EXPAND + sigExp * 0.5 + TEMPORAL_AABB_MIN_EXTENT) * TEMPORAL_AABB_BOX_SCALE;

    minAY -= expAY;
    maxAY += expAY;
    minCC -= expCC;
    maxCC += expCC;
}
#endif

float updateAABBScale(float scale, float val, float lo, float hi) {
    if (val > hi) return (val > 0.0 && hi >= 0.0) ? min(scale, hi / val) : 0.0;
    if (val < lo) return (val < 0.0 && lo <= 0.0) ? min(scale, lo / val) : 0.0;
    return scale;
}

void clampHistoryToAABB(inout AliceEncoding hist, vec4 minAY, vec4 maxAY, vec2 minCC, vec2 maxCC) {
    float s = 1.0;
    s = updateAABBScale(s, hist.aliceY.x, minAY.x, maxAY.x);
    s = updateAABBScale(s, hist.aliceY.y, minAY.y, maxAY.y);
    s = updateAABBScale(s, hist.aliceY.z, minAY.z, maxAY.z);
    s = updateAABBScale(s, hist.aliceY.w, minAY.w, maxAY.w);
    s = updateAABBScale(s, hist.CoCg.x, minCC.x, maxCC.x);
    s = updateAABBScale(s, hist.CoCg.y, minCC.y, maxCC.y);
    s = clamp(s, 0.0, 1.0);

    hist.aliceY *= s;
    hist.CoCg *= s;
}

// ===========================================================================
// 时域累积核心
// ===========================================================================

void resetToCurrentSample() {
    output_weight = 1.0;
    out_data.data_swap = current_data.data_swap;
}

void MixDiffuse() {
    if (any(lessThan(prevScreenPos, vec3(0.0))) || any(greaterThan(prevScreenPos, vec3(1.0)))) {
        resetToCurrentSample();
        return;
    }

    TemporalFootprint fp;
    if (!buildTemporalFootprint(uvec2(gl_GlobalInvocationID.xy), current_data.pos, currentNormal, cameraDelta, fp)) {
        resetToCurrentSample();
        return;
    }

    vec2 prevCoord = prevScreenPos.xy * vec2(resolution_global);
    ivec2 prevBase = ivec2(floor(prevCoord));
    vec2 prevFrac = fract(prevCoord);

    AliceEncoding accumAlice = init_alice();
    float validKernelWeight = 0.0;
    float accumHistWeight = 0.0;

    float w[4] = {
        (1.0 - prevFrac.x) * (1.0 - prevFrac.y),
        prevFrac.x * (1.0 - prevFrac.y),
        (1.0 - prevFrac.x) * prevFrac.y,
        prevFrac.x * prevFrac.y
        };

    float currVoN = dot(normalize(current_data.pos), currentNormal);

    for (int i = 0; i < 4; i++) {
        ivec2 sampleTexel = prevBase + ivec2(i & 1, i >> 1);

        if (any(lessThan(sampleTexel, ivec2(0))) || any(greaterThanEqual(sampleTexel, ivec2(resolution_global)))) continue;

        diffuseIlluminationData tap = fetchDiffuse(sampleTexel);
        if (tap.prev_weight < TEMPORAL_HISTORY_MIN_WEIGHT) continue;
        // 几何一致性测试（纯位置，ALICE 方向编码隐式保证法线一致性）
        if (!strictHistoryGeometryTest(tap.pos, fp)) continue;

        float d1_sq = dot(tap.pos, tap.pos);
        vec3 histPosCur = tap.pos - cameraDelta;
        float d2_sq = dot(histPosCur, histPosCur);
        float histVoN = dot(normalize(histPosCur), currentNormal);

        float scale = clamp(d2_sq * abs(histVoN) / max(d1_sq * abs(currVoN), 1e-3), 0.0, 1.0);
        float correctedTapW = min(tap.prev_weight * scale, float(TEMPORAL_MAX_HISTORY));

        accumulate_alice(accumAlice, tap.data, w[i]);
        validKernelWeight += w[i];
        accumHistWeight += w[i] * correctedTapW;
    }

    if (validKernelWeight < 1e-5) {
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    AliceEncoding histAlice = scale_alice(accumAlice, 1.0 / validKernelWeight);
    float histWeight = accumHistWeight / validKernelWeight;

    histWeight = clamp(histWeight, 0.0, float(TEMPORAL_MAX_HISTORY));
    if (histWeight <= TEMPORAL_HISTORY_MIN_WEIGHT) {
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    #if TEMPORAL_AABB_ENABLE
    vec4 minAY, maxAY;
    vec2 minCC, maxCC;
    int validCnt;

    computeAABB_CS(minAY, maxAY, minCC, maxCC, validCnt);
    if (validCnt >= TEMPORAL_AABB_MIN_VALID_NEIGHBORS) {
        clampHistoryToAABB(histAlice, minAY, maxAY, minCC, maxCC);
    }
    #endif

    float W = histWeight + 1.0;
    float curAlpha = 1.0 / max(W, 1e-6);
    output_weight = min(W, float(TEMPORAL_MAX_HISTORY));

    out_data.data_swap = curAlpha >= 0.9999 ? current_data.data_swap : mix_alice(histAlice, current_data.data_swap, curAlpha);
    imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), vec4(validKernelWeight, 0.0, 0.0, 0.0));
}

// ===========================================================================
// 主入口
// ===========================================================================

void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;

    #if TEMPORAL_AABB_ENABLE
    {
        uint tid = gl_LocalInvocationID.y * TILE_SIZE + gl_LocalInvocationID.x;
        uint totalSamples = AABB_SM_W * AABB_SM_H;

        for (uint i = tid; i < totalSamples; i += TILE_SIZE * TILE_SIZE) {
            uint row = i / AABB_SM_W;
            uint col = i % AABB_SM_W;

            ivec2 gc = ivec2(gl_WorkGroupID.xy * TILE_SIZE) - ivec2(AABB_HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), ivec2(resolution) - 1);
            uvec2 loadXY = uvec2(clamped);

            AABBTileSample s;
            AliceEncoding alice;
            float mask;
            readDiffuseLightRT(loadXY, alice, mask);
            s.valid = mask > 0.5;
            if (s.valid) {
                s.aliceY = alice.aliceY;
                s.CoCg = alice.CoCg;
            } else {
                s.aliceY = vec4(0.0);
                s.CoCg = vec2(0.0);
            }
            sm_aabbTile[row][col] = s;
        }
    }
    memoryBarrierShared();
    barrier();
    #endif

    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    {
        readGeo0(GEO_N_GEO, pix, current_data.pos, info_distance);
    }
    {
        // 从 DiffuseBuffer N=0 读 ALICE + surfaceMask，pos 复用 Geo0
        uvec2 _xy = uvec2(pix);
        AliceEncoding alice;
        float mask;
        readDiffuseLightRT(_xy, alice, mask);
        current_data.data_swap = alice;
        current_data.surfaceMask = mask;
        current_data.weight = 1.0;
        // 从 Geo1 取 macroNormal（仅用于 buildTemporalFootprint 切空间）
        float _r;
        int _it;
        float _pr;
        readGeo1(GEO_N_NORMALS, pix, currentNormal, _r, _it, _pr);
    }

    out_data.data_swap = current_data.data_swap;
    out_data.data = init_alice();
    out_data.surfaceMask = current_data.surfaceMask;
    out_data.pos = current_data.pos;
    output_weight = 1.0;

    if (info_distance < -0.5) {
        out_data.weight = 0.0;
        WriteDiffuse(out_data, ivec2(pix));
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    cameraDelta = camPos - prevRaytracingCamPos;

    vec4 clipPos = rtPrevProjection * rtPrevModelView * vec4(current_data.pos + cameraDelta, 1.0);
    prevScreenPos = abs(clipPos.w) > 1e-6 ? (clipPos.xyz / clipPos.w) * 0.5 + 0.5 : vec3(-1.0);

    MixDiffuse();

    out_data.weight = output_weight;
    WriteDiffuse(out_data, ivec2(pix));
}
