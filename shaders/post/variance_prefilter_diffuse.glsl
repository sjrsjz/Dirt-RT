#version 430 core

// ===========================================================================
// Pass swap2_c: Diffuse Variance Filter → Colortex Push (Compute)
// ===========================================================================
// 16×16 workgroups + 3px halo → 22×22 shared tile (7×7 kernel).
// Output: colorimg3 = vec4(pos, oct(normal)), colorimg4 = vec4(packMaxEnt, variance)

layout(local_size_x = 16, local_size_y = 16) in;

#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"

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
    vec4 maxEntY; // mean state: xyz = E[Y·u], w = E[Y]
    float meanY2; // raw second moment E[Y²]
    float weight; // temporal sample count
    float estVar; // post-blend estimator variance (Phase 5 → 3×3 blur input)
    bool valid;
};

const uint SM_AREA = SM_W * SM_H;
// Geometry stays FP32, while MaxEnt and moment state keep the source buffer's
// FP16 encoding in LDS. Decode happens only when a tap is consumed. Validity
// is stored in position.w; variance scratch covers only the 16x16 interior.
shared vec4 sm_position_validity[SM_AREA];
shared uvec2 sm_maxent_y_packed[SM_AREA];
shared uint sm_moments_packed[SM_AREA];
shared float sm_est_var[16u * 16u];
// Shared allocation is 14576 bytes, down from 20384 bytes.

// ---------------------------------------------------------------------------
// Sanitization & canonicalisation
// ---------------------------------------------------------------------------

MaxEntEncoding sanitizeMaxEnt(MaxEntEncoding a) {
    if (any(isnan(a.maxEntY)) || any(isinf(a.maxEntY))) a.maxEntY = vec4(0.0);
    if (any(isnan(a.CoCg)) || any(isinf(a.CoCg))) a.CoCg = vec2(0.0);
    return a;
}

float sanitizeNonnegative(float x) {
    if (isnan(x) || isinf(x)) return 0.0;
    return max(x, 0.0);
}

// Project stored f16 state onto the realizable cone |v| ≤ ω.
vec4 canonicalMaxEntY(vec4 s) {
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

MaxEntEncoding decodeSwapMaxEnt(uvec4 packedLight) {
    MaxEntEncoding maxent;
    maxent.maxEntY = vec4(unpackHalf2x16(packedLight.x),
        unpackHalf2x16(packedLight.y));
    maxent.CoCg = unpackHalf2x16(packedLight.z);
    return sanitizeMaxEnt(maxent);
}

void encodeTileLight(uvec4 packedLight, out uvec2 packedMaxEntY,
    out uint packedMoments) {
    MaxEntEncoding maxent = decodeSwapMaxEnt(packedLight);
    vec2 sourceMoments = unpackHalf2x16(packedLight.w);
    float w = clamp(sanitizeNonnegative(sourceMoments.x), 1.0,
        float(TEMPORAL_MAX_HISTORY));
    float m2 = sourceMoments.y * sourceMoments.y;
    vec4 maxEntY = canonicalMaxEntY(maxent.maxEntY);
    m2 = canonicalMeanY2(maxEntY, m2);
    packedMaxEntY = uvec2(packHalf2x16(maxEntY.xy),
        packHalf2x16(maxEntY.zw));
    packedMoments = packHalf2x16(vec2(w,
        min(sqrt(max(m2, 0.0)), 65504.0)));
}

vec4 decodeTileMaxEntY(uvec2 packedMaxEntY) {
    return vec4(unpackHalf2x16(packedMaxEntY.x),
        unpackHalf2x16(packedMaxEntY.y));
}

vec2 decodeTileMoments(uint packedMoments) {
    vec2 weightRootM2 = unpackHalf2x16(packedMoments);
    return vec2(weightRootM2.y * weightRootM2.y, weightRootM2.x);
}

void loadTileSample(uint index, ivec2 gc, ivec2 texMax) {
    ivec2 clamped = clamp(gc, ivec2(0), texMax);
    vec3 pos;
    float mask;
    readDiffuseGeo(uvec2(clamped), pos, mask);

    bool valid = mask > 0.5 && all(equal(gc, clamped));
    sm_position_validity[index] = vec4(pos, valid ? 1.0 : 0.0);
    sm_maxent_y_packed[index] = uvec2(0u);
    sm_moments_packed[index] = 0u;
    if (valid) {
        encodeTileLight(readDiffuseSwapRaw(uvec2(gc)),
            sm_maxent_y_packed[index], sm_moments_packed[index]);
    }
}

// ===========================================================================
// Main
// ===========================================================================

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    uvec2 lid = gl_LocalInvocationID.xy;
    ivec2 texMax = ivec2(resolution) - 1;
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO);

    // ---- Phase 1: each lane owns its center; lanes cooperate on the halo ----
    // Keeping the center output in registers removes 6 KiB of LDS without
    // introducing a second global read.
    uint cx = lid.x + HALO, cy = lid.y + HALO;
    uint centerIndex = cy * SM_W + cx;
    ivec2 centerCoord = ivec2(gid);
    ivec2 centerClamped = clamp(centerCoord, ivec2(0), texMax);
    vec3 centerPos;
    float centerMask;
    readDiffuseGeo(uvec2(centerClamped), centerPos, centerMask);
    bool centerValid = centerMask > 0.5 &&
        all(equal(centerCoord, centerClamped));
    sm_position_validity[centerIndex] =
        vec4(centerPos, centerValid ? 1.0 : 0.0);
    sm_maxent_y_packed[centerIndex] = uvec2(0u);
    sm_moments_packed[centerIndex] = 0u;

    MaxEntEncoding outMaxEnt;
    outMaxEnt.maxEntY = vec4(0.0);
    outMaxEnt.CoCg = vec2(0.0);
    if (centerValid) {
        uvec4 centerLight = readDiffuseSwapRaw(gid);
        outMaxEnt = decodeSwapMaxEnt(centerLight);
        encodeTileLight(centerLight, sm_maxent_y_packed[centerIndex],
            sm_moments_packed[centerIndex]);
    }

    uint tid = gl_LocalInvocationIndex;
    for (uint i = tid; i < SM_AREA; i += 256u) {
        uint tx = i % SM_W;
        uint ty = i / SM_W;
        bool interior = tx >= HALO && tx < HALO + 16u &&
            ty >= HALO && ty < HALO + 16u;
        if (!interior)
            loadTileSample(i, tileOrigin + ivec2(tx, ty), texMax);
    }
    barrier();

    // ---- Phase 2: bounds & sky ----
    vec4 ctrPositionValidity = sm_position_validity[centerIndex];
    vec4 ctrMaxEntY = decodeTileMaxEntY(sm_maxent_y_packed[centerIndex]);
    vec2 ctrMoments = decodeTileMoments(sm_moments_packed[centerIndex]);
    bool inBounds = all(lessThan(gid, uvec2(resolution)));
    bool isActive = inBounds && ctrPositionValidity.w > 0.5;

    vec3 centerN = vec3(0.0);

    if (isActive) {
        // ---- Phase 3: center normal & temporal variance ----
        float _r, _pr;
        int _it;
        readGeo1(GEO_N_NORMALS, gid, centerN, _r, _it, _pr);

        vec4 cState = ctrMaxEntY;
        float cN = ctrMoments.y;
        float temporalVar = temporalEstVar(cState, ctrMoments.x, cN);

        // Center-only geometry work is invariant across all 49 taps.
        float centerDistance = max(length(ctrPositionValidity.xyz), 0.01);
        float resolutionY = max(resolution.y, 1.0);
        float footprintDistance = max(centerDistance, resolutionY * 1e-4);
        float invGeometryScale = resolutionY / max(
                    VAR_FILTER_POSITION_PARAM * footprintDistance,
                    resolutionY * 1e-6);
        float centerPlaneDistance = dot(ctrPositionValidity.xyz, centerN);

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
                uint sampleIndex = uint(sy) * SM_W + uint(sx);
                vec4 samplePositionValidity =
                    sm_position_validity[sampleIndex];
                if (samplePositionValidity.w <= 0.5) continue;
                vec2 sampleMoments =
                    decodeTileMoments(sm_moments_packed[sampleIndex]);

                float planeDist = abs(dot(samplePositionValidity.xyz,
                    centerN) - centerPlaneDistance);
                float kernel = VAR_KERNEL_1D[abs(kx)] * VAR_KERNEL_1D[abs(ky)];
                float wS = kernel * exp2(-planeDist * invGeometryScale);
                float mass = wS * sampleMoments.y;

                sumMass += mass;
                sumSqW += mass * wS; // Σ N_i·w_i² (not (N_i·w_i)²)
                sumState += mass *
                    decodeTileMaxEntY(sm_maxent_y_packed[sampleIndex]);
                sumM2 += mass * sampleMoments.x;
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
        sm_est_var[lid.y * 16u + lid.x] =
            sanitizeNonnegative(outSigma * outSigma);
    } else {
        // No invocation may return before the workgroup-wide blur barrier.
        sm_est_var[lid.y * 16u + lid.x] = 0.0;
    }

    barrier();

    if (!inBounds) return;
    if (ctrPositionValidity.w <= 0.5) {
        imageStore(colorimg3, ivec2(gid), vec4(0.0));
        // Preserve the diffuse-light invalid convention through the packed
        // path: the raw FP32 variance word is negative for sky/no surface.
        imageStore(colorimg4, ivec2(gid), uvec4(0u, 0u, 0u,
            floatBitsToUint(-1.0)));
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
            uint sampleIndex = uint(sy) * SM_W + uint(sx);
            if (sm_position_validity[sampleIndex].w <= 0.5) continue;
            float w = VAR_BLUR_1D[abs(kx)] * VAR_BLUR_1D[abs(ky)];
            uint varianceIndex = uint(int(lid.y) + ky) * 16u +
                uint(int(lid.x) + kx);
            blurredVar += w * sm_est_var[varianceIndex];
            blurW += w;
        }
    }
    float estVar = sanitizeNonnegative(blurredVar / blurW);

    // ---- Phase 6: output ----
    imageStore(colorimg3, ivec2(gid),
        vec4(ctrPositionValidity.xyz, encodeNormal(centerN)));
    imageStore(colorimg4, ivec2(gid), uvec4(
            packHalf2x16(clamp(outMaxEnt.maxEntY.xy, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(outMaxEnt.maxEntY.zw, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(outMaxEnt.CoCg, vec2(-65504.0), vec2(65504.0))),
            floatBitsToUint(estVar)));
}
