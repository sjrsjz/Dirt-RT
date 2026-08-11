#version 430 core

// ===========================================================================
// Pass 100 CS: 漫反射时域累积 (当前像素梯形 → 历史空间厚梯形版)
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/alice.glsl"

uniform vec2 resolution;

#ifndef TEMPORAL_DEPTH_FOOTPRINT_SCALE
#define TEMPORAL_DEPTH_FOOTPRINT_SCALE 1.0
#endif

#ifndef TEMPORAL_CLIP_PIXEL_RADIUS
#define TEMPORAL_CLIP_PIXEL_RADIUS 1.0
#endif

#ifndef TEMPORAL_GEOMETRY_EPSILON
#define TEMPORAL_GEOMETRY_EPSILON 1e-5
#endif

// ===========================================================================
// 共享内存 AABB tile
// ===========================================================================

#define TILE_SIZE 16u

#if TEMPORAL_AABB_ENABLE
#define AABB_HALO uint(TEMPORAL_AABB_NEIGHBOR_RADIUS)
#define AABB_SM_W (TILE_SIZE + 2u * AABB_HALO)
#define AABB_SM_H (TILE_SIZE + 2u * AABB_HALO)

// Keep the source FP16 encoding in LDS and decode only when a tap is consumed.
// N=0.w's unused low half carries validity; its high half remains sqrt(meanY2).
// At the default 20x20 tile this uses 6.4 KiB instead of 12.8 KiB.
shared uvec4 sm_aabbPacked[AABB_SM_H * AABB_SM_W];

bool aabbPackedValid(uvec4 packedLight) {
    return (packedLight.w & 0xffffu) != 0u;
}

void unpackAABBLight(uvec4 packedLight, out vec4 aliceY, out vec2 CoCg) {
    aliceY = vec4(unpackHalf2x16(packedLight.x),
        unpackHalf2x16(packedLight.y));
    CoCg = unpackHalf2x16(packedLight.z);
}

void loadAABBTileSample(uint index, uvec2 xy) {
    uvec4 packedLight = readDiffuseLightRTRaw(xy);
    if (readDiffuseSurfaceMask(xy) > 0.5) {
        // Low half was written as zero by writeDiffuseLightRT.
        packedLight.w = (packedLight.w & 0xffff0000u) | 1u;
        sm_aabbPacked[index] = packedLight;
    } else {
        sm_aabbPacked[index] = uvec4(0u);
    }
}
#endif

// ===========================================================================
// 局部数据结构 & 全局变量
// ===========================================================================

struct TemporalFootprint {
    vec3 origin;
    vec3 geometryNormal;
    vec3 tangent;
    vec3 bitangent;
    vec2 plane0, plane1, plane2, plane3;
    float depthHalfExtent;
    float planeEdgeEpsilon;
};

struct TemporalFootprintFast {
    vec3 origin;
    vec3 geometryNormal;
    float depthHalfExtent;
};

vec3 prevScreenPos;
vec3 cameraDelta;
float info_distance;
vec3 geometryNormal; // from Geo1, for buildTemporalFootprint tangent frame

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

bool buildTemporalFootprint(uvec2 pix, vec3 currentPos, vec3 geometryNormal, vec3 camDelta, out TemporalFootprint fp) {
    float nLenSq = dot(geometryNormal, geometryNormal);
    if (nLenSq < 1e-8) return false;

    fp.geometryNormal = geometryNormal * inversesqrt(nLenSq);

    // Frisvad 标准正交基
    if (fp.geometryNormal.z < -0.999999) {
        fp.tangent = vec3(0.0, -1.0, 0.0);
        fp.bitangent = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + fp.geometryNormal.z);
        float c = -fp.geometryNormal.x * fp.geometryNormal.y * a;
        fp.tangent = vec3(1.0 - fp.geometryNormal.x * fp.geometryNormal.x * a, c, -fp.geometryNormal.x);
        fp.bitangent = vec3(c, 1.0 - fp.geometryNormal.y * fp.geometryNormal.y * a, -fp.geometryNormal.y);
    }

    mat4 invVP = rtInverseViewProjection;
    vec2 curRes = vec2(resolution);
    vec2 uvMin = (vec2(pix) - TEMPORAL_CLIP_PIXEL_RADIUS) / curRes * 2.0 - 1.0;
    vec2 uvMax = (vec2(pix) + TEMPORAL_CLIP_PIXEL_RADIUS) / curRes * 2.0 - 1.0;

    bool v0, v1, v2, v3;
    fp.origin = currentPos + camDelta;

    vec3 w0 = intersectCorner(vec2(uvMin.x, uvMin.y), invVP, currentPos, fp.geometryNormal, v0) + camDelta;
    vec3 w1 = intersectCorner(vec2(uvMax.x, uvMin.y), invVP, currentPos, fp.geometryNormal, v1) + camDelta;
    vec3 w2 = intersectCorner(vec2(uvMax.x, uvMax.y), invVP, currentPos, fp.geometryNormal, v2) + camDelta;
    vec3 w3 = intersectCorner(vec2(uvMin.x, uvMax.y), invVP, currentPos, fp.geometryNormal, v3) + camDelta;

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
    if (abs(dot(delta, fp.geometryNormal)) > fp.depthHalfExtent) return false;

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

bool buildTemporalFootprintFast(
    vec3 currentPos,
    vec3 surfaceNormal,
    vec3 camDelta,
    out TemporalFootprintFast fp
) {
    float normalLengthSquared = dot(surfaceNormal, surfaceNormal);
    if (normalLengthSquared < 1e-8) return false;
    fp.geometryNormal = surfaceNormal * inversesqrt(normalLengthSquared);
    fp.origin = currentPos + camDelta;

    float positionLengthSquared = dot(currentPos, currentPos);
    float positionLength = sqrt(max(positionLengthSquared, 1e-8));
    float noV = abs(dot(currentPos, fp.geometryNormal)) / positionLength;
    float pixelWorldSize = max(positionLength /
        max(float(resolution_global.y), 1.0), 1e-4);

    // A two-pixel diagonal at unit aspect is approximately 4*d/resY in
    // world space. Division by NoV reproduces the ray/plane expansion at
    // grazing angles without reconstructing four near/far ray pairs.
    fp.depthHalfExtent = max(4.0 * TEMPORAL_CLIP_PIXEL_RADIUS *
        pixelWorldSize * TEMPORAL_DEPTH_FOOTPRINT_SCALE / max(noV, 0.05),
        1e-5);
    return true;
}

bool strictHistoryGeometryTestFast(
    vec3 historyPosition,
    TemporalFootprintFast fp,
    uvec2 currentPixel,
    vec3 camDelta
) {
    vec3 historyDelta = historyPosition - fp.origin;
    if (abs(dot(historyDelta, fp.geometryNormal)) > fp.depthHalfExtent)
        return false;

    vec3 historyCurrentSpace = historyPosition - camDelta;
    vec4 clip = rtViewProjection * vec4(historyCurrentSpace, 1.0);
    if (clip.w <= 1e-7 || any(isnan(clip)) || any(isinf(clip))) return false;
    vec2 projectedPixel = (clip.xy / clip.w * 0.5 + 0.5) *
        vec2(resolution_global);
    vec2 extent = vec2(TEMPORAL_CLIP_PIXEL_RADIUS +
        TEMPORAL_GEOMETRY_EPSILON);
    return all(lessThanEqual(abs(projectedPixel - vec2(currentPixel)), extent));
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

            uint sampleIndex = uint(cy + dy) * AABB_SM_W + uint(cx + dx);
            uvec4 samplePacked = sm_aabbPacked[sampleIndex];
            if (!aabbPackedValid(samplePacked)) continue;
            vec4 sampleAliceY;
            vec2 sampleCoCg;
            unpackAABBLight(samplePacked, sampleAliceY, sampleCoCg);

            minAY = min(minAY, sampleAliceY);
            maxAY = max(maxAY, sampleAliceY);
            minCC = min(minCC, sampleCoCg);
            maxCC = max(maxCC, sampleCoCg);

            sumAY += sampleAliceY;
            sumSqAY += sampleAliceY * sampleAliceY;
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
    out_data.meanY2 = current_data.meanY2;
}

void MixDiffuse() {
    if (any(lessThan(prevScreenPos, vec3(0.0))) || any(greaterThan(prevScreenPos, vec3(1.0)))) {
        resetToCurrentSample();
        return;
    }

    TemporalFootprintFast fp;
    if (!buildTemporalFootprintFast(current_data.pos, geometryNormal,
            cameraDelta, fp)) {
        resetToCurrentSample();
        return;
    }

    vec2 prevCoord = prevScreenPos.xy * vec2(resolution_global);
    ivec2 prevBase = ivec2(floor(prevCoord));
    vec2 prevFrac = fract(prevCoord);

    AliceEncoding accumAlice = init_alice();
    float accumMeanY2 = 0.0;
    float validKernelWeight = 0.0;
    float accumHistWeight = 0.0;

    float w[4] = {
        (1.0 - prevFrac.x) * (1.0 - prevFrac.y),
        prevFrac.x * (1.0 - prevFrac.y),
        (1.0 - prevFrac.x) * prevFrac.y,
        prevFrac.x * prevFrac.y
        };

    float currVoN = dot(normalize(current_data.pos), geometryNormal);

    for (int i = 0; i < 4; i++) {
        ivec2 sampleTexel = prevBase + ivec2(i & 1, i >> 1);

        if (any(lessThan(sampleTexel, ivec2(0))) || any(greaterThanEqual(sampleTexel, ivec2(resolution_global)))) continue;

        diffuseIlluminationData tap = fetchDiffuse(sampleTexel);
        if (tap.prev_weight < TEMPORAL_HISTORY_MIN_WEIGHT) continue;
        // 几何一致性测试（纯位置，ALICE 方向编码隐式保证法线一致性）
        if (!strictHistoryGeometryTestFast(tap.pos, fp,
                uvec2(gl_GlobalInvocationID.xy), cameraDelta)) continue;

        float normalWeight = max(dot(fp.geometryNormal, tap.histNormal), 0.0);
        if (normalWeight <= 0.0) continue;

        float d1_sq = dot(tap.pos, tap.pos);
        vec3 histPosCur = tap.pos - cameraDelta;
        float d2_sq = dot(histPosCur, histPosCur);
        float histVoN = dot(normalize(histPosCur), geometryNormal);

        float scale = clamp(d2_sq * abs(histVoN) / max(d1_sq * abs(currVoN), 1e-3), 0.0, 1.0);
        float correctedTapW = min(tap.prev_weight * scale, float(TEMPORAL_MAX_HISTORY));

        float tapWeight = w[i] * normalWeight;
        accumulate_alice(accumAlice, tap.data, tapWeight);
        accumMeanY2 += tapWeight * tap.prev_meanY2;
        validKernelWeight += tapWeight;
        accumHistWeight += tapWeight * correctedTapW;
    }

    if (validKernelWeight < 1e-5) {
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    AliceEncoding histAlice = scale_alice(accumAlice, 1.0 / validKernelWeight);
    float histWeight = accumHistWeight / validKernelWeight;

    float histMeanY2 = accumMeanY2 / validKernelWeight;

    histWeight = clamp(histWeight, 0.0, float(TEMPORAL_MAX_HISTORY));
    if (histWeight <= TEMPORAL_HISTORY_MIN_WEIGHT) {
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
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

    // Blend second moment: M₂,n = (1-α)·M₂,h + α·Y²_c
    float newMeanY2 = mix(histMeanY2, current_data.meanY2, curAlpha);

    out_data.data_swap = curAlpha >= 0.9999 ? current_data.data_swap : mix_alice(histAlice, current_data.data_swap, curAlpha);
    out_data.meanY2 = newMeanY2;
    imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(floatBitsToUint(validKernelWeight), 0u, 0u, 0u));
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

        uint centerCol = gl_LocalInvocationID.x + AABB_HALO;
        uint centerRow = gl_LocalInvocationID.y + AABB_HALO;
        uint centerIndex = centerRow * AABB_SM_W + centerCol;
        ivec2 centerCoord = clamp(ivec2(pix), ivec2(0),
            ivec2(resolution) - 1);
        loadAABBTileSample(centerIndex, uvec2(centerCoord));

        for (uint i = tid; i < totalSamples; i += TILE_SIZE * TILE_SIZE) {
            uint row = i / AABB_SM_W;
            uint col = i % AABB_SM_W;

            bool interior = col >= AABB_HALO &&
                col < AABB_HALO + TILE_SIZE && row >= AABB_HALO &&
                row < AABB_HALO + TILE_SIZE;
            if (interior) continue;

            ivec2 gc = ivec2(gl_WorkGroupID.xy * TILE_SIZE) - ivec2(AABB_HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), ivec2(resolution) - 1);
            loadAABBTileSample(i, uvec2(clamped));
        }
    }
    barrier();
    #endif

    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    {
        readGeo0(GEO_N_GEO, pix, current_data.pos, info_distance);
    }
    {
        // 从 DiffuseBuffer N=0 读 ALICE + meanY2，surfaceMask 从 N=1 读
        #if TEMPORAL_AABB_ENABLE
        uint outputCenterCol = gl_LocalInvocationID.x + AABB_HALO;
        uint outputCenterRow = gl_LocalInvocationID.y + AABB_HALO;
        uint outputCenterIndex = outputCenterRow * AABB_SM_W + outputCenterCol;
        uvec4 centerPacked = sm_aabbPacked[outputCenterIndex];
        unpackAABBLight(centerPacked, current_data.data_swap.aliceY,
            current_data.data_swap.CoCg);
        float centerSqrtM2 = unpackHalf2x16(centerPacked.w).y;
        current_data.meanY2 = centerSqrtM2 * centerSqrtM2;
        current_data.surfaceMask = aabbPackedValid(centerPacked) ? 1.0 : 0.0;
        #else
        AliceEncoding centerAlice;
        readDiffuseLightRT(pix, centerAlice, current_data.meanY2);
        current_data.data_swap = centerAlice;
        current_data.surfaceMask = readDiffuseSurfaceMask(pix);
        #endif
        current_data.weight = 1.0;
        // 从 Geo1 取 geometryNormal（仅用于 buildTemporalFootprint 切空间）
        float _r;
        int _it;
        float _pr;
        readGeo1(GEO_N_NORMALS, pix, geometryNormal, _r, _it, _pr);
    }

    out_data.data_swap = current_data.data_swap;
    out_data.meanY2 = current_data.meanY2;
    out_data.data = init_alice();
    out_data.surfaceMask = current_data.surfaceMask;
    out_data.pos = current_data.pos;
    output_weight = 1.0;

    if (info_distance < -0.5) {
        out_data.weight = 0.0;
        writeDiffuse(out_data, ivec2(pix));
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    cameraDelta = camPos - prevRaytracingCamPos;

    vec3 surfaceMotion;
    float motionValid;
    readSurfaceMotion(pix, surfaceMotion, motionValid);
    if (motionValid < 0.5) {
        resetToCurrentSample();
        out_data.weight = output_weight;
        writeDiffuse(out_data, ivec2(pix));
        imageStore(colorimg6, ivec2(pix), uvec4(0u));
        return;
    }
    cameraDelta -= surfaceMotion;

    vec4 clipPos = rtPrevViewProjection * vec4(current_data.pos + cameraDelta, 1.0);
    prevScreenPos = abs(clipPos.w) > 1e-6 ? (clipPos.xyz / clipPos.w) * 0.5 + 0.5 : vec3(-1.0);

    MixDiffuse();

    out_data.weight = output_weight;
    writeDiffuse(out_data, ivec2(pix));
}
