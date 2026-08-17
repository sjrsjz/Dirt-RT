#version 430 core

// ===========================================================================
// MaxEnt diffuse temporal accumulation.
// ===========================================================================

// A 128-thread group admits another resident block when registers are the
// limiting launch resource, while preserving 16-wide horizontal coherence.
layout(local_size_x = 16, local_size_y = 8) in;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_temporal_statistics.glsl"

uniform vec2 resolution;

#ifndef MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE
#define MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE 1.0
#endif

#ifndef MAXENT_TEMPORAL_REPROJECTION_RADIUS
#define MAXENT_TEMPORAL_REPROJECTION_RADIUS 1.0
#endif

#ifndef MAXENT_TEMPORAL_GEOMETRY_EPSILON
#define MAXENT_TEMPORAL_GEOMETRY_EPSILON 1e-5
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
vec3 geometryNormal; // from Geo1, for buildTemporalFootprint tangent frame
vec3 currentPosition;

MaxEntEncoding outputMaxEnt;
float outputMeanY2;
float output_weight = 0.0;

uvec4 currentLightPacked() {
    return readDiffuseLightRTRaw(uvec2(gl_GlobalInvocationID.xy));
}

void unpackCurrentLight(out MaxEntEncoding maxEnt, out float meanY2) {
    uvec4 packedLight = currentLightPacked();
    maxEnt.maxEntY = vec4(unpackHalf2x16(packedLight.x),
        unpackHalf2x16(packedLight.y));
    maxEnt.CoCg = unpackHalf2x16(packedLight.z);
    float sqrtMeanY2 = unpackHalf2x16(packedLight.w).y;
    meanY2 = sqrtMeanY2 * sqrtMeanY2;
}

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
    vec2 uvMin = (vec2(pix) - MAXENT_TEMPORAL_REPROJECTION_RADIUS) / curRes * 2.0 - 1.0;
    vec2 uvMax = (vec2(pix) + MAXENT_TEMPORAL_REPROJECTION_RADIUS) / curRes * 2.0 - 1.0;

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
    fp.depthHalfExtent = footprintDiameter * MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE;
    fp.planeEdgeEpsilon = MAXENT_TEMPORAL_GEOMETRY_EPSILON * max(footprintDiameter, 1.0);

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

    // MaxEnt 编码的方向信息已隐含法线一致性 — 移除显式半球检查
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
    fp.depthHalfExtent = max(4.0 * MAXENT_TEMPORAL_REPROJECTION_RADIUS *
        pixelWorldSize * MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE / max(noV, 0.05),
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
    // Only xy/w participates in the footprint test. An unordered comparison
    // rejects NaN w; the final ordered bounds test rejects non-finite xy/w.
    // Do not keep the unused clip.z row and four-component class tests live.
    if (!(clip.w > 1e-7) || isinf(clip.w)) return false;
    vec2 projectedPixel = (clip.xy / clip.w * 0.5 + 0.5) *
        vec2(resolution_global);
    vec2 extent = vec2(MAXENT_TEMPORAL_REPROJECTION_RADIUS +
        MAXENT_TEMPORAL_GEOMETRY_EPSILON);
    return all(lessThanEqual(abs(projectedPixel - vec2(currentPixel)), extent));
}

// ===========================================================================
// 时域累积核心
// ===========================================================================

void resetToCurrentSample() {
    output_weight = 1.0;
    unpackCurrentLight(outputMaxEnt, outputMeanY2);
}

void MixDiffuse() {
    if (any(lessThan(prevScreenPos, vec3(0.0))) || any(greaterThan(prevScreenPos, vec3(1.0)))) {
        resetToCurrentSample();
        return;
    }

    TemporalFootprintFast fp;
    if (!buildTemporalFootprintFast(currentPosition, geometryNormal,
            cameraDelta, fp)) {
        resetToCurrentSample();
        return;
    }

    vec2 prevCoord = prevScreenPos.xy * vec2(resolution_global);
    ivec2 prevBase = ivec2(floor(prevCoord));
    vec2 prevFrac = fract(prevCoord);

    MaxEntEncoding accumMaxEnt = init_maxent();
    float accumMeanY2 = 0.0;
    float validKernelWeight = 0.0;
    float sumWeightOverSamples = 0.0;
    float accumulatedHistoryEvidence = 0.0;

    float currVoN = dot(normalize(currentPosition), geometryNormal);

    for (int i = 0; i < 4; i++) {
        ivec2 sampleTexel = prevBase + ivec2(i & 1, i >> 1);

        if (any(lessThan(sampleTexel, ivec2(0))) || any(greaterThanEqual(sampleTexel, ivec2(resolution_global)))) continue;

        uvec2 historyTexel = uvec2(sampleTexel);
        vec2 historyMeta = unpackHalf2x16(
            diffuseBuffer.data[addr(DIF_N_HIST, historyTexel)].w);
        if (historyMeta.x < MAXENT_DIFFUSE_TEMPORAL_MIN_HISTORY_WEIGHT) continue;

        // Decode geometry before the six-component light signal. Rejected
        // taps therefore never make history light live across reprojection.
        uvec4 packedGeometry = diffuseBuffer.data[
            addr(DIF_N_HISTGEO, historyTexel)];
        float historyDistance = uintBitsToFloat(packedGeometry.x);
        vec3 historyPosition = decodeDiffuseHistoryNormalU(
            packedGeometry.y) * historyDistance;
        // 几何一致性测试（纯位置，MaxEnt 方向编码隐式保证法线一致性）
        if (!strictHistoryGeometryTestFast(historyPosition, fp,
                uvec2(gl_GlobalInvocationID.xy), cameraDelta)) continue;

        vec3 historyNormal = decodeDiffuseHistoryNormalU(packedGeometry.z);
        float normalWeight = max(dot(fp.geometryNormal, historyNormal), 0.0);
        if (normalWeight <= 0.0) continue;

        float d1_sq = dot(historyPosition, historyPosition);
        vec3 histPosCur = historyPosition - cameraDelta;
        float d2_sq = dot(histPosCur, histPosCur);
        float histVoN = dot(normalize(histPosCur), geometryNormal);

        float scale = clamp(d2_sq * abs(histVoN) / max(d1_sq * abs(currVoN), 1e-3), 0.0, 1.0);
        float tapSamples = clamp(historyMeta.x, 1.0,
            float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY));

        float bilinearX = (i & 1) == 0 ? 1.0 - prevFrac.x : prevFrac.x;
        float bilinearY = (i & 2) == 0 ? 1.0 - prevFrac.y : prevFrac.y;
        float tapWeight = bilinearX * bilinearY * normalWeight;

        uvec4 packedHistory = diffuseBuffer.data[
            addr(DIF_N_HIST, historyTexel)];
        MaxEntEncoding tapMaxEnt;
        tapMaxEnt.maxEntY = clamp(vec4(unpackHalf2x16(packedHistory.x),
            unpackHalf2x16(packedHistory.y)), vec4(-65504.0),
            vec4(65504.0));
        tapMaxEnt.CoCg = unpackHalf2x16(packedHistory.z);
        accumulate_maxent(accumMaxEnt, tapMaxEnt, tapWeight);
        accumMeanY2 += tapWeight * historyMeta.y * historyMeta.y;
        validKernelWeight += tapWeight;
        sumWeightOverSamples += tapWeight / tapSamples;
        accumulatedHistoryEvidence += tapWeight * tapSamples * scale;
    }

    if (validKernelWeight < 1e-5) {
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    MaxEntEncoding histMaxEnt = scale_maxent(accumMaxEnt, 1.0 / validKernelWeight);
    float historySamples = maxentTemporalReprojectedEffectiveSamples(
        validKernelWeight, sumWeightOverSamples,
        float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY));
    float historyEvidence = clamp(accumulatedHistoryEvidence
        / validKernelWeight, 0.0, float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY));

    float histMeanY2 = accumMeanY2 / validKernelWeight;

    if (historySamples < 1.0
            || historyEvidence <= MAXENT_DIFFUSE_TEMPORAL_MIN_HISTORY_WEIGHT) {
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    // Geometry confidence controls how much history is reused, while N_eff
    // describes the noise of the history estimator itself. Keeping these
    // quantities separate avoids inventing sub-unity Kish sample counts.
    float curAlpha = 1.0 / (historyEvidence + 1.0);
    output_weight = maxentTemporalUpdatedEffectiveSamples(historySamples,
        curAlpha, float(MAXENT_DIFFUSE_TEMPORAL_MAX_HISTORY));

    // Blend second moment: M₂,n = (1-α)·M₂,h + α·Y²_c
    MaxEntEncoding currentMaxEnt;
    float currentMeanY2;
    unpackCurrentLight(currentMaxEnt, currentMeanY2);
    outputMeanY2 = mix(histMeanY2, currentMeanY2, curAlpha);
    outputMaxEnt = curAlpha >= 0.9999
        ? currentMaxEnt : mix_maxent(histMaxEnt, currentMaxEnt, curAlpha);
    imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(floatBitsToUint(validKernelWeight), 0u, 0u, 0u));
}

// ===========================================================================
// 主入口
// ===========================================================================

void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;

    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    float infoDistance;
    readDiffusePrimaryGeometry(pix, currentPosition, infoDistance);

    if (infoDistance < -0.5) {
        // Clear the two outputs consumed downstream and skip light/normal
        // decoding for sky pixels.
        diffuseBuffer.data[addr(DIF_N_SWAP, pix)] = uvec4(0u);
        imageStore(colorimg6, ivec2(pix), uvec4(0u));
        return;
    }

    geometryNormal = readPrimaryGeometryNormal(pix);

    cameraDelta = camPos - prevRaytracingCamPos;

    vec3 surfaceMotion;
    float motionValid;
    readDiffuseMotion(pix, surfaceMotion, motionValid);
    if (motionValid < 0.5) {
        resetToCurrentSample();
        writeDiffuseSwap(pix, outputMaxEnt, output_weight, outputMeanY2);
        imageStore(colorimg6, ivec2(pix), uvec4(0u));
        return;
    }
    cameraDelta -= surfaceMotion;

    vec4 clipPos = rtPrevViewProjection * vec4(currentPosition + cameraDelta, 1.0);
    prevScreenPos = abs(clipPos.w) > 1e-6 ? (clipPos.xyz / clipPos.w) * 0.5 + 0.5 : vec3(-1.0);

    MixDiffuse();

    writeDiffuseSwap(pix, outputMaxEnt, output_weight, outputMeanY2);
}
