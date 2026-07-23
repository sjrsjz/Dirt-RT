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


#ifndef VAR_FILTER_NORMAL_POWER
#define VAR_FILTER_NORMAL_POWER SVGF_NORMAL_POWER
#endif

#ifndef VAR_FILTER_POSITION_PARAM
#define VAR_FILTER_POSITION_PARAM SVGF_POSITION_PARAM
#endif

// --- Shared memory tile: 20x20 (16+4 halo for 5x5 kernel) ---
const uint SM_W = 20u;
const uint SM_H = 20u;
const uint HALO = 2u;

struct TileSample {
    bool valid;
    vec3 p;
    float weight;
    vec4 aliceY;
};
shared TileSample sm_tile[SM_H][SM_W];

// --- Helpers ---

AliceEncoding sanitizeAlice(AliceEncoding encoded) {
    if (any(isnan(encoded.aliceY)) || any(isinf(encoded.aliceY))) encoded.aliceY = vec4(0.0);
    if (any(isnan(encoded.CoCg)) || any(isinf(encoded.CoCg))) encoded.CoCg = vec2(0.0);
    return encoded;
}

vec4 sanitizeAliceY(vec4 aliceY) {
    if (any(isnan(aliceY)) || any(isinf(aliceY))) aliceY = vec4(0.0);
    return aliceY;
}

float sanitizeVariance(float v) {
    if (isnan(v) || isinf(v)) return 0.0;
    return max(v, 0.0);
}

// Unpack swap AliceEncoding + weight from unified SSBO, return raw ALICE variance.
void loadAliceY(uvec2 xy, out vec4 aliceY, out float weight) {
    AliceEncoding alice;
    readDiffuseSwap(xy, alice, weight);
    aliceY = sanitizeAliceY(alice.aliceY);
}

// Depth-only geometry weight — ALICE Bures distance replaces normal-based edge-stopping
float varianceGeometryWeight(
    vec3 centerPos, vec3 centerNormal,
    vec3 samplePos
) {
    float distToCam = max(length(centerPos), 0.01);
    float pixelFootprint = max(distToCam / max(resolution.y, 1.0), 1e-4);

    float planeDist = abs(dot(samplePos - centerPos, centerNormal));
    float depthTerm = planeDist / max(VAR_FILTER_POSITION_PARAM * pixelFootprint, 1e-6);

    return exp2(-depthTerm);
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texSize = ivec2(resolution) - 1;

    // =========================================================================
    // Phase 1: Cooperative load into shared memory (2 phases, was 3)
    // =========================================================================
    uint threadIdx = lid.y * 16u + lid.x; // 线程在 Workgroup 内的 1D 索引 (0~255)

    // Phase 1a: Binding 2 N=1 — 漫反射几何（位置 + surfaceMask 替代 dist+normal）
    for (uint i = threadIdx; i < 400u; i += 256u) {
        uint row = i / 20u;
        uint col = i % 20u;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(col, row);
        ivec2 clamped = clamp(gc, ivec2(0), texSize);
        uvec2 loadXY = uvec2(clamped);

        float mask;
        readDiffuseGeo(loadXY, sm_tile[row][col].p, mask);
        sm_tile[row][col].valid = mask > 0.5 && gc == clamped; // 仅当像素有效且未被裁剪时才标记为有效
    }
    barrier();
    memoryBarrierShared();

    // Phase 1b: Binding 2 N=4 — 漫反射 swap 光照（方差计算）
    for (uint i = threadIdx; i < 400u; i += 256u) {
        uint row = i / 20u;
        uint col = i % 20u;
        if (sm_tile[row][col].valid) {
            ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), texSize);
            uvec2 loadXY = uvec2(clamped);
            loadAliceY(loadXY, sm_tile[row][col].aliceY, sm_tile[row][col].weight);
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

    if (!centerTile.valid) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid), vec4(0.0, 0.0, 0.0, -1.0));
        return;
    }

    // 从 Geo1 读取中心法线（仅中心像素，非整 tile）
    vec3 centerNormal;
    {
        float _r;
        int _it;
        float _pr;
        readGeo1(GEO_N_NORMALS, uvec2(clamp(ivec2(gid), ivec2(0), texSize)), centerNormal, _r, _it, _pr);
    }

    // =========================================================================
    // Phase 3: Unpack center AliceEncoding
    // =========================================================================
    uvec2 xy = uvec2(clamp(ivec2(gid), ivec2(0), texSize));
    float cWeight;
    AliceEncoding outAlice;
    readDiffuseSwap(xy, outAlice, cWeight);
    outAlice = sanitizeAlice(outAlice);

    // =========================================================================
    // Phase 4: 5x5 geometry-aware bilateral variance filter
    // =========================================================================
    vec4 sumAliceY = vec4(0.0);
    float sumHistW = 0.0;
    float sumW = 0.0;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int sx = int(cx) + kx;
            int sy = int(cy) + ky;
            TileSample s = sm_tile[sy][sx];
            if (!s.valid) continue;
            float wKernel = exp(-0.75 * float(kx * kx + ky * ky)); // 高斯核权重
            float wGeom = varianceGeometryWeight(centerTile.p, centerNormal, s.p);

            float w = wKernel * wGeom;
            sumAliceY += w * s.aliceY;
            sumHistW += w * s.weight;
            sumW += w;
        }
    }

    vec4 filteredAliceY = sumAliceY / max(sumW, 1e-20);
    float filteredHistW = sumHistW / max(sumW, 1e-20);
    float filteredVariance = alice_estimator_variance(filteredAliceY, filteredHistW);

    // =========================================================================
    // Phase 5: Write outputs
    // =========================================================================
    // colortex3: pos.xyz + oct(centerNormal) — normal for plane-projected depth in atrous
    imageStore(colorimg3, ivec2(gid), vec4(centerTile.p, encodeNormal(centerNormal)));
    // colortex4: packed AliceEncoding + filtered variance (matches old swap2 light_sample layout)
    imageStore(colorimg4, ivec2(gid), vec4(packAlice(outAlice), filteredVariance));
}
