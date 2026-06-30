#version 430 compatibility

// ===========================================================================
// Pass swap2_c: Diffuse Variance Filter → Colortex Push (Compute)
// ===========================================================================
// Replaces fragment shader swap2.glsl with a compute shader using:
//   - 16x16 workgroups + 2px halo = 20x20 shared memory tile
//   - Precomputed raw ALICE variance loaded into shared memory
//   - 5x5 geometry-aware bilateral variance filter
//   - imageStore output to colorimg3 (geometry) and colorimg4 (SH+variance)

layout(local_size_x = 16, local_size_y = 16) in;
#define DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

/* RENDERTARGETS: 3,4 */

uniform vec2 resolution;

// --- Output images (colorimgN = writable colortex in compute shaders) ---
layout(rgba32f) uniform writeonly image2D colorimg3;
layout(rgba32f) uniform writeonly image2D colorimg4;

// --- Kernel constants ---
const float hw[3] = float[](1.0, 0.66667, 0.44444); // B-spline 5x5 kernel

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
};
shared TileSample sm_tile[SM_H][SM_W];

// --- Helpers ---

SH sanitizeSH(SH sh) {
    if (any(isnan(sh.shY)) || any(isinf(sh.shY))) sh.shY = vec4(0.0);
    if (any(isnan(sh.CoCg)) || any(isinf(sh.CoCg))) sh.CoCg = vec2(0.0);
    return sh;
}

float sanitizeVariance(float v) {
    if (isnan(v) || isinf(v)) return 0.0;
    return max(v, 0.0);
}

// Unpack swap SH + weight from unified SSBO, return raw ALICE variance.
float computeRawVariance(uint idx) {
    UnifiedDiffuseElement e = diffuseIllumiantionBuffer.data[idx];

    mediump vec2 shY_xy = unpackHalf2x16(floatBitsToUint(e.swap_shY_xy));
    mediump vec2 shY_zw = unpackHalf2x16(floatBitsToUint(e.swap_shY_zw));

    vec4 shY = clamp(vec4(shY_xy, shY_zw), vec4(-10000), vec4(10000));
    mediump vec2 w_v = unpackHalf2x16(floatBitsToUint(e.swap_w_v));
    float weight = w_v.x;

    if (any(isnan(shY)) || any(isinf(shY))) shY = vec4(0.0);
    if (isnan(weight) || isinf(weight))        weight = 0.0;

    return sanitizeVariance(alice_estimator_variance(shY, max(weight, 1.0)));
}

float varianceGeometryWeight(
    vec3 centerPos, vec3 centerNormal,
    vec3 samplePos, vec3 sampleNormal
) {
    float nd = clamp(dot(centerNormal, sampleNormal), 0.0, 1.0);
    float wNormal = pow(nd, VAR_FILTER_NORMAL_POWER);

    float distToCam = max(length(centerPos - camPos), 0.01);
    float pixelFootprint = max(distToCam / max(resolution.y, 1.0), 1e-4);

    float planeDist = abs(dot(samplePos - centerPos, centerNormal));
    float depthTerm = planeDist / max(VAR_FILTER_POSITION_PARAM * pixelFootprint, 1e-6);

    float wDepth = exp(-depthTerm);
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

    for (uint i = threadIdx; i < 400u; i += 256u) {
        uint row = i / 20u;
        uint col = i % 20u;

        ivec2 gc = ivec2(gl_WorkGroupID.xy * 16u) - ivec2(HALO) + ivec2(col, row);
        ivec2 clamped = clamp(gc, ivec2(0), texSize - ivec2(1));
        uint loadIdx = getIdx(uvec2(clamped));

        UnifiedDiffuseElement e = diffuseIllumiantionBuffer.data[loadIdx];
        float d = denoiseBuffer.data[loadIdx].distance;

        TileSample s;
        s.dist  = d;
        s.px    = e.px;
        s.py    = e.py;
        s.pz    = e.pz;
        s.oct_n = e.oct_n;
        s.rawVar = (d > -0.5) ? computeRawVariance(loadIdx) : 0.0;

        sm_tile[row][col] = s;
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
        imageStore(colorimg4, ivec2(gid), vec4(0.0));
        return;
    }

    // =========================================================================
    // Phase 3: Unpack center SH for output (not for variance — that's precomputed)
    // =========================================================================
    uint idx = getIdx(uvec2(clamp(ivec2(gid), ivec2(0), texSize - ivec2(1))));
    UnifiedDiffuseElement ce = diffuseIllumiantionBuffer.data[idx];
    mediump vec2 c_shY_xy = unpackHalf2x16(floatBitsToUint(ce.swap_shY_xy));
    mediump vec2 c_shY_zw = unpackHalf2x16(floatBitsToUint(ce.swap_shY_zw));
    mediump vec2 c_CoCg   = unpackHalf2x16(floatBitsToUint(ce.swap_CoCg));
    mediump vec2 c_w_v    = unpackHalf2x16(floatBitsToUint(ce.swap_w_v));

    SH outSH;
    outSH.shY  = vec4(c_shY_xy, c_shY_zw);
    outSH.CoCg = c_CoCg;
    outSH = sanitizeSH(outSH);

    float centerVariance = sanitizeVariance(
        alice_estimator_variance(outSH.shY, max(c_w_v.x, 1.0))
    );

    // =========================================================================
    // Phase 4: 5x5 geometry-aware bilateral variance filter
    // =========================================================================
    vec3 centerPos    = vec3(centerTile.px, centerTile.py, centerTile.pz);
    vec3 centerNormal = decodeNormal(centerTile.oct_n);

    float sumVar = 0.0;
    float sumW   = 0.0;

    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int sx = int(cx) + kx;
            int sy = int(cy) + ky;

            TileSample s = sm_tile[sy][sx];
            if (s.dist < -0.5) continue;

            vec3 samplePos    = vec3(s.px, s.py, s.pz);
            vec3 sampleNormal = decodeNormal(s.oct_n);

            float wKernel = hw[abs(kx)] * hw[abs(ky)];
            float wGeom = varianceGeometryWeight(centerPos, centerNormal, samplePos, sampleNormal);
            float w = wKernel * wGeom;

            sumVar += w * s.rawVar;
            sumW   += w;
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
    // Phase 5: Write outputs
    // =========================================================================
    // colortex3: pos.xyz + oct-encoded normal (matches old swap2 geometry layout)
    imageStore(colorimg3, ivec2(gid), vec4(centerPos, centerTile.oct_n));
    // colortex4: packed SH + filtered variance (matches old swap2 light_sample layout)
    imageStore(colorimg4, ivec2(gid), vec4(packSH(outSH), filteredVariance));
}
