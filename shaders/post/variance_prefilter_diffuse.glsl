#version 430 compatibility

// ===========================================================================
// Pass swap2_c: Diffuse Variance Filter → Colortex Push (Compute)
// ===========================================================================
// 16x16 workgroups + 2px halo = 20x20 shared memory tile
// Precomputed raw ALICE variance loaded into shared memory
// 5x5 geometry-aware bilateral variance filter
// imageStore output to colorimg3 (geometry) and colorimg4 (AliceEncoding+variance)

layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/lighting/alice.glsl"

uniform vec2 resolution;

// --- Output images (colorimgN = writable colortex in compute shaders) ---
layout(rgba32f) uniform writeonly image2D colorimg3;
layout(rgba32f) uniform writeonly image2D colorimg4;

// --- Kernel constants ---
const float hw[3] = float[](1.0, 0.66667, 0.44444);

#ifndef VAR_FILTER_NORMAL_POWER
#define VAR_FILTER_NORMAL_POWER SVGF_NORMAL_POWER
#endif

#ifndef VAR_FILTER_POSITION_PARAM
#define VAR_FILTER_POSITION_PARAM SVGF_POSITION_PARAM
#endif

#define VAR_FILTER_CONSERVATIVE 0

// --- Shared memory tile: 20x20 (16+4 halo for 5x5 kernel) ---
const uint SM_W = 20u;
const uint SM_H = 20u;
const uint HALO = 2u;

struct TileSample {
    float dist;
    float px, py, pz;
    float oct_n;
    float rawVar;
    float omega; // aliceY.w = total ALICE energy, for 3-sigma clamping
};
shared TileSample sm_tile[SM_H][SM_W];

// --- Helpers ---

AliceEncoding sanitizeAlice(AliceEncoding encoded) {
    if (any(isnan(encoded.aliceY)) || any(isinf(encoded.aliceY))) encoded.aliceY = vec4(0.0);
    if (any(isnan(encoded.CoCg)) || any(isinf(encoded.CoCg))) encoded.CoCg = vec2(0.0);
    return encoded;
}

float sanitizeVariance(float v) {
    if (isnan(v) || isinf(v)) return 0.0;
    return max(v, 0.0);
}

// Unpack swap AliceEncoding + weight from unified SSBO, return raw ALICE variance and omega.
float computeRawVariance(uvec2 xy, out float outOmega) {
    AliceEncoding alice;
    float weight;
    readDiffuseSwap(xy, alice, weight);

    vec4 aliceY = alice.aliceY;

    if (any(isnan(aliceY)) || any(isinf(aliceY))) aliceY = vec4(0.0);
    if (isnan(weight) || isinf(weight)) weight = 0.0;

    outOmega = aliceY.w;
    return sanitizeVariance(alice_estimator_variance(aliceY, max(weight, 1.0)));
}

float varianceGeometryWeight(
    vec3 centerPos, vec3 centerNormal,
    vec3 samplePos, vec3 sampleNormal
) {
    float nd = clamp(dot(centerNormal, sampleNormal), 0.0, 1.0);
    float wNormal = pow(nd, VAR_FILTER_NORMAL_POWER);

    float distToCam = max(length(centerPos), 0.01);
    float pixelFootprint = max(distToCam / max(resolution.y, 1.0), 1e-4);

    float planeDist = abs(dot(samplePos - centerPos, centerNormal));
    float depthTerm = planeDist / max(VAR_FILTER_POSITION_PARAM * pixelFootprint, 1e-6);

    float wDepth = exp2(-depthTerm * LOG2_E);
    return wNormal * wDepth;
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution);

    // =========================================================================
    // Phase 1: Cooperative load into shared memory
    // =========================================================================
    uint threadIdx = lid.y * 16u + lid.x; // 线程在 Workgroup 内的 1D 索引 (0~255)

    // Phase 1a: Binding 0 N=0 — 距离（天空判定）
    for (uint i = threadIdx; i < 400u; i += 256u) {
        uint row = i / 20u;
        uint col = i % 20u;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(col, row);
        ivec2 clamped = clamp(gc, ivec2(0), texSize - ivec2(1));
        uvec2 loadXY = uvec2(clamped);

        vec3 dpos; float d;
        readGeo0(GEO_N_GEO, loadXY, dpos, d);
        sm_tile[row][col].dist = d;
        sm_tile[row][col].px = dpos.x; sm_tile[row][col].py = dpos.y; sm_tile[row][col].pz = dpos.z;
        sm_tile[row][col].oct_n = 0.0;
        sm_tile[row][col].rawVar = 0.0;
        sm_tile[row][col].omega = 0.0;
    }
    barrier();
    memoryBarrierShared();

    // Phase 1b: Binding 2 N=1 — 漫反射几何（位置 + 法线）
    for (uint i = threadIdx; i < 400u; i += 256u) {
        uint row = i / 20u;
        uint col = i % 20u;
        if (sm_tile[row][col].dist > -0.5) {
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), texSize - ivec2(1));
            uvec2 loadXY = uvec2(clamped);

            vec3 dgeoPos, dgeoN;
            readDiffuseGeo(loadXY, dgeoPos, dgeoN);
            sm_tile[row][col].px = dgeoPos.x; sm_tile[row][col].py = dgeoPos.y; sm_tile[row][col].pz = dgeoPos.z;
            sm_tile[row][col].oct_n = encodeNormal(dgeoN);
        }
    }
    barrier();
    memoryBarrierShared();

    // Phase 1c: Binding 2 N=4 — 漫反射 swap 光照（方差计算）
    for (uint i = threadIdx; i < 400u; i += 256u) {
        uint row = i / 20u;
        uint col = i % 20u;
        if (sm_tile[row][col].dist > -0.5) {
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), texSize - ivec2(1));
            uvec2 loadXY = uvec2(clamped);

            float om = 0.0;
            sm_tile[row][col].rawVar = computeRawVariance(loadXY, om);
            sm_tile[row][col].omega = om;
        }
    }
    barrier();
    memoryBarrierShared();

    // =========================================================================
    // Phase 2: Skip out-of-bounds and sky pixels
    // =========================================================================
    if (any(greaterThanEqual(gid, uvec2(resolution)))) return;

    uint cx = lid.x + HALO;
    uint cy = lid.y + HALO;

    TileSample centerTile = sm_tile[cy][cx];

    if (centerTile.dist < -0.5) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        // 天空像素: 写入负值方差作为天空 mask, 供 300/300_cs 使用
        // (方差合法值为非负数, 负值可安全复用为天空标记)
        imageStore(colorimg4, ivec2(gid), vec4(0.0, 0.0, 0.0, -1.0));
        return;
    }

    // =========================================================================
    // Phase 3: Unpack center AliceEncoding, apply 3-sigma energy clamp on outAlice
    // =========================================================================
    uvec2 xy = uvec2(clamp(ivec2(gid), ivec2(0), texSize - ivec2(1)));
    float cWeight;
    AliceEncoding outAlice;
    readDiffuseSwap(xy, outAlice, cWeight);
    outAlice = sanitizeAlice(outAlice);

    // --- 3-sigma energy clamp on output AliceEncoding ---
    // Compute neighborhood mean & sigma of ω, clamp center outAlice if outlier.
    vec3 centerPos = vec3(centerTile.px, centerTile.py, centerTile.pz);
    vec3 centerNormal = decodeNormal(centerTile.oct_n);

    float sumOmega = 0.0;
    float sumOmega2 = 0.0;
    float sumStatW = 0.0;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int sx = int(cx) + kx;
            int sy = int(cy) + ky;
            TileSample s = sm_tile[sy][sx];
            if (s.dist < -0.5) continue;

            vec3 sPos = vec3(s.px, s.py, s.pz);
            vec3 sNrm = decodeNormal(s.oct_n);

            float wK = hw[abs(kx)] * hw[abs(ky)];
            float wG = varianceGeometryWeight(centerPos, centerNormal, sPos, sNrm);
            float w = wK * wG;

            sumOmega += w * s.omega;
            sumOmega2 += w * s.omega * s.omega;
            sumStatW += w;
        }
    }

    float meanOmega = (sumStatW > 1e-8) ? (sumOmega / sumStatW) : centerTile.omega;
    float varOmega = (sumStatW > 1e-8) ? max(sumOmega2 / sumStatW - meanOmega * meanOmega, 0.0) : 0.0;
    float sigmaOmega = sqrt(varOmega);

    // Clamp center AliceEncoding energy to [μ-3σ, μ+3σ]; scale full aliceY + CoCg by r.
    // Preserves ρ=|v|/ω and cone constraint ω≥|v|.
    float centerOmega = outAlice.aliceY.w;
    float omegaClamped = clamp(centerOmega, meanOmega - 3.0 * sigmaOmega, meanOmega + 3.0 * sigmaOmega);
    float aliceY_scale = omegaClamped / max(centerOmega, 1e-8);
    outAlice.aliceY *= aliceY_scale;
    outAlice.CoCg *= aliceY_scale;

    float centerVariance = sanitizeVariance(
            alice_estimator_variance(outAlice.aliceY, max(cWeight, 1.0))
        );

    // =========================================================================
    // Phase 4: 5x5 geometry-aware bilateral variance filter
    // =========================================================================
    float sumVar = 0.0;
    float sumW = 0.0;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int sx = int(cx) + kx;
            int sy = int(cy) + ky;
            TileSample s = sm_tile[sy][sx];
            if (s.dist < -0.5) continue;

            vec3 sPos = vec3(s.px, s.py, s.pz);
            vec3 sNrm = decodeNormal(s.oct_n);

            float wKernel = hw[abs(kx)] * hw[abs(ky)];
            float wGeom = varianceGeometryWeight(centerPos, centerNormal, sPos, sNrm);
            float w = wKernel * wGeom;

            sumVar += w * s.rawVar;
            sumW += w;
        }
    }

    float filteredVariance = centerVariance;
    if (sumW > 1e-8) {
        filteredVariance = sumVar / sumW;
    }
    #if VAR_FILTER_CONSERVATIVE
    filteredVariance = max(filteredVariance, centerVariance);
    #endif
    filteredVariance = sanitizeVariance(filteredVariance);

    // =========================================================================
    // Phase 5: 高斯曲率 → omega 符号标记 (3×3 有限差分, 复用 LDS)
    // =========================================================================
    // |K| > threshold → 几何边缘不可靠(棱/角) → 标记 omega 为负
    // 300 降噪 pass 检测到负 omega 时跳过几何权重
    #if ENABLE_GAUSSIAN_FILTER == 1
    {
        // 中心 + 邻域位置 (复用 Phase 1 加载的 sm_tile 2D 数组)
        int scx = int(cx), scy = int(cy);
        #define P(dx,dy) vec3(sm_tile[scy+(dy)][scx+(dx)].px, \
                                      sm_tile[scy+(dy)][scx+(dx)].py, \
        #define P(dx,dy) vec3(sm_tile[scy+(dy)][scx+(dx)].px, \
                              sm_tile[scy+(dy)][scx+(dx)].py, \
                              sm_tile[scy+(dy)][scx+(dx)].pz)
        vec3 Pc = P(0, 0);

        vec3 Dx = (P(1, 0) - P(-1, 0)) * 0.5;
        vec3 Dy = (P(0, 1) - P(0, -1)) * 0.5;
        vec3 Dxx = P(1, 0) - 2.0 * Pc + P(-1, 0);
        vec3 Dyy = P(0, 1) - 2.0 * Pc + P(0, -1);
        vec3 Dxy = (P(1, 1) - P(1, -1) - P(-1, 1) + P(-1, -1)) * 0.25;
        #undef P

        float E = dot(Dx, Dx);
        float F = dot(Dx, Dy);
        float G = dot(Dy, Dy);
        float L = dot(Dxx, centerNormal);
        float M = dot(Dxy, centerNormal);
        float N2 = dot(Dyy, centerNormal);

        float denom = max(E * G - F * F, 1e-8);
        float K = (L * N2 - M * M) / denom;

        // 分支无关: 曲率超阈值 → omega 取负 (标记跳过几何权重)
        float kMask = float(abs(K) > CURVATURE_THRESHOLD);
        outAlice.aliceY.w = abs(outAlice.aliceY.w) * (1.0 - 2.0 * kMask);
    }
    #endif // ENABLE_GAUSSIAN_FILTER

    // =========================================================================
    // Phase 6: Write outputs
    // =========================================================================
    // colortex3: pos.xyz + oct-encoded normal (matches old swap2 geometry layout)
    imageStore(colorimg3, ivec2(gid), vec4(centerPos, centerTile.oct_n));
    // colortex4: packed AliceEncoding + filtered variance (matches old swap2 light_sample layout)
    imageStore(colorimg4, ivec2(gid), vec4(packAlice(outAlice), filteredVariance));
}
