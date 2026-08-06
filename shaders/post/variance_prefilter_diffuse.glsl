#version 430 compatibility

// ===========================================================================
// Pass swap2_c: Diffuse Variance Filter → Colortex Push (Compute)
// ===========================================================================
// 16×16 workgroups + 3px halo → 22×22 shared tile (7×7 kernel).
// Output: colorimg3 = vec4(pos, oct(normal)), colorimg4 = vec4(packAlice, variance)

layout(local_size_x = 16, local_size_y = 16) in;

#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/alice.glsl"

uniform vec2 resolution;

layout(rgba32f) uniform writeonly image2D colorimg3;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

#ifndef VAR_FILTER_POSITION_PARAM
#define VAR_FILTER_POSITION_PARAM ATROUS_POSITION_PARAM
#endif

#ifndef VAR_FILTER_KERNEL_SIGMA
#define VAR_FILTER_KERNEL_SIGMA 1.25
#endif

#ifndef VAR_FILTER_HISTORY_BEGIN
#define VAR_FILTER_HISTORY_BEGIN 2.0
#endif

#ifndef VAR_FILTER_HISTORY_END
#define VAR_FILTER_HISTORY_END 12.0
#endif

// ---------------------------------------------------------------------------
// Shared tile — 22×22 with 3px halo for 7×7 kernel
// ---------------------------------------------------------------------------

const uint SM_W = 22u;
const uint SM_H = 22u;
const uint HALO = 3u;

struct TileSample {
    vec3 p;
    vec4 aliceY; // mean state: xyz = E[Y·u], w = E[Y]
    float meanY2; // raw second moment E[Y²]
    float weight; // temporal sample count
    float estVar; // post-blend estimator variance (Phase 5 → 3×3 blur input)
    bool valid;
};

shared TileSample sm_tile[SM_H][SM_W];

// ---------------------------------------------------------------------------
// Sanitization & canonicalisation
// ---------------------------------------------------------------------------

AliceEncoding sanitizeAlice(AliceEncoding a) {
    if (any(isnan(a.aliceY)) || any(isinf(a.aliceY))) a.aliceY = vec4(0.0);
    if (any(isnan(a.CoCg)) || any(isinf(a.CoCg))) a.CoCg = vec2(0.0);
    return a;
}

float sanitizeNonnegative(float x) {
    if (isnan(x) || isinf(x)) return 0.0;
    return max(x, 0.0);
}

// Project stored f16 state onto the realizable cone |v| ≤ ω.
vec4 canonicalAliceY(vec4 s) {
    if (any(isnan(s)) || any(isinf(s))) return vec4(0.0);
    vec3 v = s.xyz;
    float omega = abs(s.w);
    float v2 = dot(v, v);
    if (v2 > omega * omega && v2 > 0.0) v *= omega * inversesqrt(v2);
    return vec4(v, omega);
}

// Jensen bounds: E[Y²] ≥ max(E[Y]², |E[Y·u]|²). f16 packing may weakly violate.
float canonicalMeanY2(vec4 meanState, float m2) {
    m2 = sanitizeNonnegative(m2);
    return max(m2, max(meanState.w * meanState.w, dot(meanState.xyz, meanState.xyz)));
}

// ---------------------------------------------------------------------------
// Light-field variance (population level)
// ---------------------------------------------------------------------------
// Feature z = (Y·u, Y) with |u|=1. E[|z|²] = 2·E[Y²].
//   tr Cov(z) = 2·E[Y²] - |E[Y·u]|² - E[Y]² = 2·M₂ - |mean|²

float lightFieldPopVar(vec4 meanState, float m2) {
    return max(2.0 * canonicalMeanY2(meanState, m2) - dot(meanState, meanState), 0.0);
}

// Unbiased sample variance → estimator variance of the temporal mean.
// Var(mean) = V_pop_biased / (N-1).  Returns 0 for N ≤ 1.5 (cold start).
float temporalEstVar(vec4 meanState, float m2, float N) {
    N = clamp(sanitizeNonnegative(N), 1.0, float(TEMPORAL_MAX_HISTORY));
    if (N <= 1.5) return 0.0;
    return lightFieldPopVar(meanState, m2) / (N - 1.0);
}

// ---------------------------------------------------------------------------
// Kernel & geometry weights
// ---------------------------------------------------------------------------

float geometryWeight(vec3 centerPos, vec3 centerN, vec3 samplePos) {
    float d = max(length(centerPos), 0.01);
    float footprint = max(d / max(resolution.y, 1.0), 1e-4);
    float planeDist = abs(dot(samplePos - centerPos, centerN));
    return exp2(-planeDist / max(VAR_FILTER_POSITION_PARAM * footprint, 1e-6));
}

float kernelWeight(int kx, int ky) {
    float s2 = VAR_FILTER_KERNEL_SIGMA * VAR_FILTER_KERNEL_SIGMA;
    return exp(-0.5 * float(kx * kx + ky * ky) / max(s2, 1e-6));
}

// ---------------------------------------------------------------------------
// Swap-buffer load
// ---------------------------------------------------------------------------

void loadSwapSample(uvec2 xy, out vec4 aliceY, out float m2, out float w) {
    AliceEncoding alice;
    readDiffuseSwap(xy, alice, w, m2);
    alice = sanitizeAlice(alice);
    aliceY = canonicalAliceY(alice.aliceY);
    w = sanitizeNonnegative(w);
    m2 = canonicalMeanY2(aliceY, m2);
}

// ===========================================================================
// Main
// ===========================================================================

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texMax = ivec2(resolution) - 1;
    uint tid = lid.y * 16u + lid.x;

    // ---- Phase 1a: load geometry tile ----
    for (uint i = tid; i < SM_W * SM_H; i += 256u) {
        uint r = i / SM_W, c = i % SM_W;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(int(c), int(r));
        ivec2 clamped = clamp(gc, ivec2(0), texMax);
        vec3 pos;
        float mask;
        readDiffuseGeo(uvec2(clamped), pos, mask);
        sm_tile[r][c].p = pos;
        sm_tile[r][c].valid = mask > 0.5 && all(equal(gc, clamped));
        sm_tile[r][c].aliceY = vec4(0.0);
        sm_tile[r][c].meanY2 = 0.0;
        sm_tile[r][c].weight = 0.0;
    }
    memoryBarrierShared();
    barrier();

    // ---- Phase 1b: load swap light tile ----
    for (uint i = tid; i < SM_W * SM_H; i += 256u) {
        uint r = i / SM_W, c = i % SM_W;
        if (!sm_tile[r][c].valid) continue;
        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(int(c), int(r));
        loadSwapSample(uvec2(clamp(gc, ivec2(0), texMax)),
            sm_tile[r][c].aliceY, sm_tile[r][c].meanY2, sm_tile[r][c].weight);
    }
    memoryBarrierShared();
    barrier();

    // ---- Phase 2: bounds & sky ----
    if (any(greaterThanEqual(gid, uvec2(resolution)))) return;

    uint cx = lid.x + HALO, cy = lid.y + HALO;
    TileSample ctr = sm_tile[cy][cx];
    if (!ctr.valid) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid), uvec4(0u));
        return;
    }

    // ---- Phase 3: center normal & temporal variance ----
    vec3 centerN;
    {
        float _r, _pr;
        int _it;
        readGeo1(GEO_N_NORMALS, gid, centerN, _r, _it, _pr);
    }

    float cW, cM2;
    AliceEncoding outAlice;
    readDiffuseSwap(gid, outAlice, cW, cM2);
    outAlice = sanitizeAlice(outAlice);
    vec4 cState = canonicalAliceY(outAlice.aliceY);
    cW = sanitizeNonnegative(cW);
    cM2 = canonicalMeanY2(cState, cM2);
    float cN = clamp(cW, 1.0, float(TEMPORAL_MAX_HISTORY));
    float temporalVar = temporalEstVar(cState, cM2, cN);

    // ---- Phase 4: spatially-pooled variance ----
    // Pool neighbor moments with mass_i = wSpatial_i · N_i.
    // V_pool = 2·M2_pool - |mean_pool|² captures both within-history
    // variance and between-history variation. No hard-coded sample counts.
    float sumMass = 0.0, sumSqW = 0.0;
    vec4 sumState = vec4(0.0);
    float sumM2 = 0.0;

    for (int ky = -3; ky <= 3; ++ky) {
        for (int kx = -3; kx <= 3; ++kx) {
            TileSample s = sm_tile[cy + ky][cx + kx];
            if (!s.valid) continue;

            float wS = kernelWeight(kx, ky) * geometryWeight(ctr.p, centerN, s.p);
            float sN = clamp(s.weight, 1.0, float(TEMPORAL_MAX_HISTORY));
            float mass = wS * sN;

            sumMass += mass;
            sumSqW += sN * wS * wS; // Σ N_i·w_i² (not (N_i·w_i)²)
            sumState += mass * s.aliceY;
            sumM2 += mass * s.meanY2;
        }
    }

    float spatialVar = temporalVar; // fallback

    if (sumMass > 1e-8) {
        vec4 poolState = sumState / sumMass;
        float poolM2 = canonicalMeanY2(poolState, sumM2 / sumMass);
        float poolPopVar = lightFieldPopVar(poolState, poolM2);

        // N_eff = (Σ w)² / Σ w²  for weighted independent samples
        float N_eff = (sumMass * sumMass) / max(sumSqW, 1e-12);
        if (N_eff > 1.01) poolPopVar *= N_eff / (N_eff - 1.0);

        // Spatial neighbors estimate per-sample variance, not extra center samples
        spatialVar = poolPopVar / max(cN, 1.0);
    }

    // ---- Phase 5: spatial→temporal blend ----
    // Blend sigma (not variance) for smooth filter-width transition.
    // Cold start (N≤1.5): purely spatial.  Mature (N≥12): purely temporal.
    float trust = smoothstep(VAR_FILTER_HISTORY_BEGIN, VAR_FILTER_HISTORY_END, cN);
    if (cN <= 1.5) trust = 0.0;

    float outSigma = mix(sqrt(max(spatialVar, 0.0)), sqrt(max(temporalVar, 0.0)), trust);
    sm_tile[cy][cx].estVar = outSigma * outSigma;
    sm_tile[cy][cx].estVar = max(sanitizeNonnegative(sm_tile[cy][cx].estVar), 0.0);

    memoryBarrierShared();
    barrier();

    // ---- Phase 5b: 3×3 Gaussian blur on variance (inner pixels only) ----
    // Simulates temporal-jitter diffusion to suppress isolated dark-spot artifacts.
    // Only samples within [HALO, HALO+15] are valid — halo cells never ran Phase 5.
    float blurredVar = 0.0;
    float blurW = 0.0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int sx = int(cx) + kx, sy = int(cy) + ky;
            if (sx < int(HALO) || sx > int(HALO) + 15) continue;
            if (sy < int(HALO) || sy > int(HALO) + 15) continue;
            if (!sm_tile[sy][sx].valid) continue;
            float w = exp(-0.5 * float(kx * kx + ky * ky)); // σ=1 Gaussian
            blurredVar += w * sm_tile[sy][sx].estVar;
            blurW += w;
        }
    }
    float estVar = (blurW > 1e-10) ? (blurredVar / blurW) : sm_tile[cy][cx].estVar;
    estVar = max(sanitizeNonnegative(estVar), 0.0);

    // ---- Phase 6: output ----
    imageStore(colorimg3, ivec2(gid), vec4(ctr.p, encodeNormal(centerN)));
    imageStore(colorimg4, ivec2(gid), uvec4(
        packHalf2x16(clamp(outAlice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(outAlice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
        packHalf2x16(clamp(outAlice.CoCg,        vec2(-65504.0), vec2(65504.0))),
        floatBitsToUint(estVar)));
}
