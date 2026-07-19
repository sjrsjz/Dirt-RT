#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/lighting/alice.glsl"

// ===========================================================================
// Pass 300 CS: SVGF 空间滤波器 (计算着色器变体) — 前 3 级 à‑trous (R0=1,2,4)
// 修改版: Bures 距离 + 能量感知权重
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D colortex3;
uniform sampler2D colortex4;

layout(rgba32f) uniform image2D colorimg4;

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

#ifndef PHI_ENERGY
#define PHI_ENERGY 1.0
#endif

// ---------------------------------------------------------------------------
// 共享内存
// ---------------------------------------------------------------------------
#define TILE_SIZE (16 + 2 * R0)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

void unpackLightSampleSM(uint tile_idx, out vec3 pos, out vec3 normal,
    out AliceEncoding encoded, out float variance) {
    vec4 geom = sm_geometry[tile_idx];
    vec4 light = sm_light[tile_idx];
    pos = geom.xyz;
    normal = decodeNormal(geom.w);
    encoded = unpackAlice(light.x, light.y, light.z);
    variance = light.w;
}

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uvec2 local_id = gl_LocalInvocationID.xy;
    uint local_idx = gl_LocalInvocationIndex;
    uvec2 group_id = gl_WorkGroupID.xy;

    ivec2 texSize = textureSize(colortex3, 0);

    ivec2 tile_origin = ivec2(group_id * 16u) - ivec2(R0);

    // =========================================================================
    // Phase 1: 协作加载 tile 到共享内存
    // =========================================================================
    for (uint i = local_idx; i < uint(TILE_AREA); i += 256u) {
        uint tx = i % uint(TILE_SIZE);
        uint ty = i / uint(TILE_SIZE);
        ivec2 gc = tile_origin + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);

        sm_geometry[i] = texelFetch(colortex3, cc, 0);

        if (gc == cc) {
            sm_light[i] = texelFetch(colortex4, cc, 0);
        } else {
            sm_light[i] = vec4(0.0, 0.0, 0.0, -1.0); // 天空标记
        }
    }

    barrier();
    memoryBarrierShared();

    // =========================================================================
    // Phase 2: 中心像素有效性检查与预计算
    // =========================================================================
    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    uint cx = local_id.x + uint(R0);
    uint cy = local_id.y + uint(R0);
    uint center_idx = cy * uint(TILE_SIZE) + cx;

    if (sm_light[center_idx].w < 0.0) return;

    vec3 center_pos, center_normal;
    AliceEncoding center_alice;
    float center_var_est;
    unpackLightSampleSM(center_idx, center_pos, center_normal, center_alice, center_var_est);

    center_var_est = max(center_var_est, 1e-12);

    // ---- 预计算中心像素的统计特征 -----------------------------------------
    vec4 c_enc = center_alice.aliceY;
    float c_len_v = length(c_enc.xyz);
    float c_omega = c_enc.w;
    float c_kappa = alice_kappa(c_len_v, c_omega);

    float c_var_omega_est = alice_radial_est_var_from_scalar(center_var_est, c_kappa);
    float c_inv_sqrt_var = inversesqrt(max(center_var_est, 1e-5));
    float c_inv_sqrt_var_omega = inversesqrt(max(c_var_omega_est, 1e-5));

    #if ENABLE_GAUSSIAN_FILTER == 1
    float geomValid = float(c_omega >= 0.0);
    center_alice.aliceY.w = abs(c_omega);
    c_omega = center_alice.aliceY.w;
    #else
    const float geomValid = 1.0;
    #endif

    float dist_to_cam = max(length(center_pos), 0.001);
    float inv_pixel_footprint = 1.0 / (SVGF_POSITION_PARAM
                * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0;
    float sumVarEnergy = center_var_est;
    AliceEncoding accumAlice = center_alice;

    // 3×3 网格采样核 (预计算权重, 归一化偏移 → 单循环)
    // 权重 = hw[|dx|] * hw[|dy|]; hw[0]=1.0, hw[1]=0.66667
    const vec3 GRID_3x3[8] = {
        vec3(-1, -1, 0.44445), vec3(-1, 0, 0.66667), vec3(-1, 1, 0.44445),
        vec3(0, -1, 0.66667), vec3(0, 1, 0.66667),
        vec3(1, -1, 0.44445), vec3(1, 0, 0.66667), vec3(1, 1, 0.44445),
        };

    // =========================================================================
    // Phase 3: 单循环采样 — 全部从共享内存读取
    // =========================================================================
    for (int k = 0; k < 8; k++) {
        int dx = int(GRID_3x3[k].x);
        int dy = int(GRID_3x3[k].y);
        float w_kernel = GRID_3x3[k].z;

        uint sx = cx + uint(dx * R0);
        uint sy = cy + uint(dy * R0);
        uint sample_idx = sy * uint(TILE_SIZE) + sx;

        if (sm_light[sample_idx].w < 0.0) continue;

        // ---- 解包邻域样本 (共享内存读取) ------------------------------
        vec3 sample_world_pos, sample_normal;
        AliceEncoding sample_alice;
        float sample_var_est;
        unpackLightSampleSM(sample_idx, sample_world_pos, sample_normal,
            sample_alice, sample_var_est);

        sample_alice.aliceY.w = abs(sample_alice.aliceY.w);

        // ---- 几何权重 (不变) --------------------------------------------
        vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
        float depthTerm = abs(dot(delta, center_normal));
        float w_geometry = SVGF_NORMAL_POWER * (1.0 - dot(center_normal, sample_normal))
                + depthTerm * geomValid;

        // ---- Bures 距离 + 能量感知 -------------------------------------
        vec4 s_enc = sample_alice.aliceY;
        float s_len_v = length(s_enc.xyz);
        float s_kappa = alice_kappa(s_len_v, s_enc.w);

        float d_bures_sq = alice_bures_distance_sq(c_enc, c_kappa, s_enc, s_kappa);

        float z_bures = sqrt(d_bures_sq) * c_inv_sqrt_var;

        float delta_omega = c_omega - s_enc.w;
        float z_energy = abs(delta_omega) * c_inv_sqrt_var_omega * PHI_ENERGY;

        float w_luma = SVGF_PHI_L * sqrt(z_bures * z_bures + z_energy * z_energy);

        // ---- 组合权重 -------------------------------------------------
        float w0 = w_kernel *  exp2(-(w_geometry + w_luma) * LOG2_E);

        // ---- 累积 ------------------------------------------------------
        accumulate_alice(accumAlice, sample_alice, w0);
        sumWeight += w0;
        sumVarEnergy += w0 * w0 * sample_var_est;
    }

    float inv_sumWeight = 1.0 / sumWeight;
    accumAlice = scale_alice(accumAlice, inv_sumWeight);

    #if ENABLE_GAUSSIAN_FILTER == 1
    #ifndef FINAL_DENOISE_PASS
    accumAlice.aliceY.w *= (2.0 * geomValid - 1.0);
    #endif
    #endif

    float varEnergyOut = sumVarEnergy * inv_sumWeight * inv_sumWeight;
    imageStore(colorimg4, pix, vec4(packAlice(accumAlice), varEnergyOut));
}
