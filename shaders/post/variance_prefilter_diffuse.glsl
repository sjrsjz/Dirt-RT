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
// Preserve the sanitized (pre-canonicalization) center state for output.
// Only the 16x16 interior is written/read; halo CoCg is never consumed.
// Expected worst-case shared allocation is 29376 bytes (below 32 KiB).
shared vec4 sm_output_alice_y[16][16];
shared vec2 sm_output_cocg[16][16];

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

float lightFieldPopVarCanonical(vec4 meanState, float m2) {
    return max(2.0 * m2 - dot(meanState, meanState), 0.0);
}

// Unbiased sample variance → estimator variance of the temporal mean.
// Var(mean) = V_pop_biased / (N-1).  Returns 0 for N ≤ 1.5 (cold start).
float temporalEstVar(vec4 meanState, float m2, float N) {
    if (N <= 1.5) return 0.0;
    return lightFieldPopVarCanonical(meanState, m2) / (N - 1.0);
}

// ---------------------------------------------------------------------------
// Kernel & geometry weights
// ---------------------------------------------------------------------------

const float VAR_KERNEL_DENOM = max(
        VAR_FILTER_KERNEL_SIGMA * VAR_FILTER_KERNEL_SIGMA, 1e-6);
const float VAR_KERNEL_1D[4] = {
    1.0,
    exp(-0.5 / VAR_KERNEL_DENOM),
    exp(-2.0 / VAR_KERNEL_DENOM),
    exp(-4.5 / VAR_KERNEL_DENOM)
    };

const float VAR_BLUR_1D[2] = { 1.0, 0.6065306597 };

// ---------------------------------------------------------------------------
// Swap-buffer load
// ---------------------------------------------------------------------------

void loadSwapSample(uvec2 xy, out vec4 aliceY, out vec4 outputAliceY,
    out vec2 outputCoCg, out float m2, out float w) {
    AliceEncoding alice;
    readDiffuseSwap(xy, alice, w, m2);
    alice = sanitizeAlice(alice);
    outputAliceY = alice.aliceY;
    outputCoCg = alice.CoCg;
    aliceY = canonicalAliceY(outputAliceY);
    w = clamp(sanitizeNonnegative(w), 1.0, float(TEMPORAL_MAX_HISTORY));
    m2 = canonicalMeanY2(aliceY, m2);
}

// ===========================================================================
// Main
// ===========================================================================

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texMax = ivec2(resolution) - 1;
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO);

    // ---- Phase 1: load geometry + light in one 2D cooperative traversal ----
    for (uint r = lid.y; r < SM_H; r += 16u) {
        for (uint c = lid.x; c < SM_W; c += 16u) {
            ivec2 gc = tileOrigin + ivec2(c, r);
            ivec2 clamped = clamp(gc, ivec2(0), texMax);
            vec3 pos;
            float mask;
            readDiffuseGeo(uvec2(clamped), pos, mask);

            bool valid = mask > 0.5 && all(equal(gc, clamped));
            sm_tile[r][c].p = pos;
            sm_tile[r][c].valid = valid;
            sm_tile[r][c].aliceY = vec4(0.0);
            sm_tile[r][c].meanY2 = 0.0;
            sm_tile[r][c].weight = 1.0;
            sm_tile[r][c].estVar = 0.0;

            if (valid) {
                vec4 outputAliceY;
                vec2 outputCoCg;
                loadSwapSample(uvec2(gc), sm_tile[r][c].aliceY,
                    outputAliceY, outputCoCg, sm_tile[r][c].meanY2,
                    sm_tile[r][c].weight);

                bool interior = r >= HALO && r < HALO + 16u
                        && c >= HALO && c < HALO + 16u;
                if (interior) {
                    sm_output_alice_y[r - HALO][c - HALO] = outputAliceY;
                    sm_output_cocg[r - HALO][c - HALO] = outputCoCg;
                }
            }
        }
    }
    barrier();

    // ---- Phase 2: bounds & sky ----
    uint cx = lid.x + HALO, cy = lid.y + HALO;
    TileSample ctr = sm_tile[cy][cx];
    bool inBounds = all(lessThan(gid, uvec2(resolution)));
    bool isActive = inBounds && ctr.valid;

    vec3 centerN = vec3(0.0);
    AliceEncoding outAlice;
    outAlice.aliceY = vec4(0.0);
    outAlice.CoCg = vec2(0.0);

    if (isActive) {
        // ---- Phase 3: center normal & temporal variance ----
        float _r, _pr;
        int _it;
        readGeo1(GEO_N_NORMALS, gid, centerN, _r, _it, _pr);

        outAlice.aliceY = sm_output_alice_y[lid.y][lid.x];
        outAlice.CoCg = sm_output_cocg[lid.y][lid.x];
        vec4 cState = ctr.aliceY;
        float cN = ctr.weight;
        float temporalVar = temporalEstVar(cState, ctr.meanY2, cN);

        // Center-only geometry work is invariant across all 49 taps.
        float centerDistance = max(length(ctr.p), 0.01);
        float resolutionY = max(resolution.y, 1.0);
        float footprintDistance = max(centerDistance, resolutionY * 1e-4);
        float invGeometryScale = resolutionY / max(
                    VAR_FILTER_POSITION_PARAM * footprintDistance,
                    resolutionY * 1e-6);
        float centerPlaneDistance = dot(ctr.p, centerN);

        // ---- Phase 4: spatially-pooled variance ----
        // Pool neighbor moments with mass_i = wSpatial_i · N_i.
        // V_pool captures both within-history and between-history variation.
        float sumMass = 0.0, sumSqW = 0.0;
        vec4 sumState = vec4(0.0);
        float sumM2 = 0.0;

        for (int ky = -3; ky <= 3; ++ky) {
            for (int kx = -3; kx <= 3; ++kx) {
                int sx = int(cx) + kx;
                int sy = int(cy) + ky;
                TileSample s = sm_tile[sy][sx];
                if (!s.valid) continue;

                float planeDist = abs(dot(s.p, centerN) - centerPlaneDistance);
                float kernel = VAR_KERNEL_1D[abs(kx)] * VAR_KERNEL_1D[abs(ky)];
                float wS = kernel * exp2(-planeDist * invGeometryScale);
                float mass = wS * s.weight;

                sumMass += mass;
                sumSqW += mass * wS; // Σ N_i·w_i² (not (N_i·w_i)²)
                sumState += mass * s.aliceY;
                sumM2 += mass * s.meanY2;
            }
        }

        float spatialVar = temporalVar; // fallback

        if (sumMass > 1e-8) {
            float invSumMass = 1.0 / sumMass;
            vec4 poolState = sumState * invSumMass;
            float poolM2 = canonicalMeanY2(poolState, sumM2 * invSumMass);
            float poolPopVar = lightFieldPopVarCanonical(poolState, poolM2);

            // N_eff = (Σ w)² / Σ w²  for weighted independent samples
            float N_eff = (sumMass * sumMass) / max(sumSqW, 1e-12);
            if (N_eff > 1.01) poolPopVar *= N_eff / (N_eff - 1.0);

            // Spatial neighbors estimate per-sample variance, not extra center samples
            spatialVar = poolPopVar / cN;
        }

        // ---- Phase 5: spatial→temporal blend ----
        // Blend sigma (not variance) for a smooth filter-width transition.
        float trust = smoothstep(VAR_FILTER_HISTORY_BEGIN, VAR_FILTER_HISTORY_END, cN);
        if (cN <= 1.5) trust = 0.0;

        float outSigma = mix(sqrt(max(spatialVar, 0.0)),
                sqrt(max(temporalVar, 0.0)), trust);
        sm_tile[cy][cx].estVar = sanitizeNonnegative(outSigma * outSigma);
    } else {
        // No invocation may return before the workgroup-wide blur barrier.
        sm_tile[cy][cx].estVar = 0.0;
    }

    barrier();

    if (!inBounds) return;
    if (!ctr.valid) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        imageStore(colorimg4, ivec2(gid), uvec4(0u));
        return;
    }

    // ---- Phase 5b: 3×3 Gaussian blur on variance (inner pixels only) ----
    // Simulates temporal-jitter diffusion to suppress isolated dark-spot artifacts.
    // Only samples within [HALO, HALO+15] are valid — halo cells never ran Phase 5.
    float blurredVar = 0.0;
    float blurW = 0.0;
    int minKx = max(-1, -int(lid.x));
    int maxKx = min(1, 15 - int(lid.x));
    int minKy = max(-1, -int(lid.y));
    int maxKy = min(1, 15 - int(lid.y));
    for (int ky = minKy; ky <= maxKy; ++ky) {
        for (int kx = minKx; kx <= maxKx; ++kx) {
            int sx = int(cx) + kx, sy = int(cy) + ky;
            if (!sm_tile[sy][sx].valid) continue;
            float w = VAR_BLUR_1D[abs(kx)] * VAR_BLUR_1D[abs(ky)];
            blurredVar += w * sm_tile[sy][sx].estVar;
            blurW += w;
        }
    }
    float estVar = sanitizeNonnegative(blurredVar / blurW);

    // ---- Phase 6: output ----
    imageStore(colorimg3, ivec2(gid), vec4(ctr.p, encodeNormal(centerN)));
    imageStore(colorimg4, ivec2(gid), uvec4(
            packHalf2x16(clamp(outAlice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(outAlice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(outAlice.CoCg, vec2(-65504.0), vec2(65504.0))),
            floatBitsToUint(estVar)));
}
