#version 430 compatibility

// ===========================================================================
// Pass: 镜面虚拟距离曲率校正 (NRD Curvature Correction) — 共享内存优化版
// ===========================================================================
// 管线位置: ray0.rgen 之后, temporal_reflect/refract 之前.
//
// NRD 薄透镜曲率修正:
//   κ = dot(N_neighbor - N_center, edge) / |edge|²  (法线方向导数)
//   d_focused = d / (2 * κ * d + 1)                    (薄透镜公式)
//
// 共享内存 tile: 18×18 (16 + 1px halo), 每线程 ~1.3 次 SSBO 读取
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform vec2 resolution;

// ===========================================================================
// 可调参数
// ===========================================================================

#ifndef CURVATURE_CORRECTION_STRENGTH
#define CURVATURE_CORRECTION_STRENGTH 1.0
#endif

#ifndef CURVATURE_MAX_CORRECTION_FRACTION
#define CURVATURE_MAX_CORRECTION_FRACTION 0.25
#endif

#ifndef CURVATURE_DEPTH_DISCONTINUITY
#define CURVATURE_DEPTH_DISCONTINUITY 0.1
#endif

#ifndef CURVATURE_MIN_VALID_NEIGHBORS
#define CURVATURE_MIN_VALID_NEIGHBORS 3
#endif

// ===========================================================================
// 共享内存 tile — 16×16 工作组的 1px halo 扩展
// ===========================================================================

#define TILE_SIZE 16u
#define HALO 1u
#define SM_W (TILE_SIZE + 2u * HALO)  // 18
#define SM_H (TILE_SIZE + 2u * HALO)  // 18

struct TileSample {
    float dist;
    vec3 normal;
};

shared TileSample sm_tile[SM_H][SM_W];

// ===========================================================================
// 薄透镜 & 曲率核心
// ===========================================================================

float applyThinLens(float hitDist, float curvature) {
    return hitDist / max(2.0 * curvature * hitDist + 1.0, 1e-8);
}

void main() {
    // =========================================================================
    // Phase 1: cooperative load 到共享内存 (round-robin)
    // =========================================================================
    {
        uint tid = gl_LocalInvocationID.y * TILE_SIZE + gl_LocalInvocationID.x;
        uint total = SM_W * SM_H; // 324

        for (uint i = tid; i < total; i += TILE_SIZE * TILE_SIZE) {
            uint row = i / SM_W;
            uint col = i % SM_W;

            ivec2 gc = ivec2(gl_WorkGroupID.xy * TILE_SIZE) - ivec2(HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), ivec2(resolution) - 1);
            uint loadIdx = getIndex(uvec2(clamped));

            TileSample s;
            s.dist = denoiseBuffer.data[loadIdx].distance;
            s.normal = denoiseBuffer.data[loadIdx].macroNormal;
            sm_tile[row][col] = s;
        }
    }
    memoryBarrierShared();
    barrier();

    // =========================================================================
    // Phase 2: 逐像素曲率计算与虚拟距离修正
    // =========================================================================

    uvec2 pix = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    uint idx = getIndex(pix);

    // ---- 读取中心 (从共享内存, 无 SSBO 读取) --------------------------------
    uint cx = gl_LocalInvocationID.x + HALO;
    uint cy = gl_LocalInvocationID.y + HALO;

    TileSample c = sm_tile[cy][cx];
    float d_c = c.dist;
    if (d_c < -0.5) return;
    vec3 N_c = c.normal;
    if (any(isnan(N_c))) return;

    // ---- 读取 4 邻域 (全从共享内存) -----------------------------------------
    TileSample sl = sm_tile[cy][cx - 1];
    TileSample sr = sm_tile[cy][cx + 1];
    TileSample st = sm_tile[cy - 1][cx];
    TileSample sb = sm_tile[cy + 1][cx];

    // ---- 深度不连续检查 & 有效邻域计数 ---------------------------------------
    float discThreshold = CURVATURE_DEPTH_DISCONTINUITY * max(d_c, 0.01);

    bool vl = sl.dist > -0.5 && abs(sl.dist - d_c) < discThreshold;
    bool vr = sr.dist > -0.5 && abs(sr.dist - d_c) < discThreshold;
    bool vt = st.dist > -0.5 && abs(st.dist - d_c) < discThreshold;
    bool vb = sb.dist > -0.5 && abs(sb.dist - d_c) < discThreshold;

    int validN = int(vl) + int(vr) + int(vt) + int(vb);
    if (validN < CURVATURE_MIN_VALID_NEIGHBORS) return;

    // ---- 世界空间像素步长 ---------------------------------------------------
    float tanHalfFovY = 1.0 / max(abs(rtProjection[1][1]), 1e-4);
    float pixelSizeW = 2.0 * d_c * tanHalfFovY / max(resolution.y, 1.0);

    // ---- 相机世界空间切向基 -------------------------------------------------
    vec3 camRight = normalize(vec3(rtModelView[0][0], rtModelView[0][1], rtModelView[0][2]));
    vec3 camUp    = normalize(vec3(rtModelView[1][0], rtModelView[1][1], rtModelView[1][2]));

    // ---- 方向曲率 ----------------------------------------------------------
    // 中心差分: 无效邻域用中心法线替代 (贡献为零)
    vec3 N_r = vr ? sr.normal : N_c;
    vec3 N_l = vl ? sl.normal : N_c;
    vec3 N_t = vt ? st.normal : N_c;
    vec3 N_b = vb ? sb.normal : N_c;

    float dn_dx = dot(N_r - N_l, camRight);
    float dn_dy = dot(N_b - N_t, camUp);
    float curvature = 0.5 * (dn_dx + dn_dy) / max(2.0 * pixelSizeW, 1e-8);
    curvature *= CURVATURE_CORRECTION_STRENGTH;

    // ---- 修正 reflect buffer -----------------------------------------------
    {
        float d = reflectIlluminationBuffer.data[idx].virtualProjDist;
        if (d > 0.0 && d < 0.5 * VPROJDIST_SKY) {
            float dCorr = applyThinLens(d, curvature);
            float lo = d * (1.0 - CURVATURE_MAX_CORRECTION_FRACTION);
            float hi = d * (1.0 + CURVATURE_MAX_CORRECTION_FRACTION);
            dCorr = clamp(dCorr, lo, hi);
            dCorr = max(dCorr, 0.0);
            reflectIlluminationBuffer.data[idx].virtualProjDist = dCorr;
        }
    }

    // ---- 修正 refract buffer -----------------------------------------------
    {
        float d = refractIlluminationBuffer.data[idx].virtualProjDist;
        if (d > 0.0 && d < 0.5 * VPROJDIST_SKY) {
            float dCorr = applyThinLens(d, curvature);
            float lo = d * (1.0 - CURVATURE_MAX_CORRECTION_FRACTION);
            float hi = d * (1.0 + CURVATURE_MAX_CORRECTION_FRACTION);
            dCorr = clamp(dCorr, lo, hi);
            dCorr = max(dCorr, 0.0);
            refractIlluminationBuffer.data[idx].virtualProjDist = dCorr;
        }
    }
}
