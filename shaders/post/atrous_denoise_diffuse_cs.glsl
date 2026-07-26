#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/alice.glsl"

// ===========================================================================
// Pass 300 CS: 前 3 级 à‑trous (3×3 扩张网格, R0=1,2,4)
// Bures 距离 + 能量感知权重
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D colortex3;
uniform sampler2D colortex4;

layout(rgba32f) uniform image2D colorimg4;

// ---------------------------------------------------------------------------
// 共享内存
// ---------------------------------------------------------------------------
#define TILE_SIZE (16 + 2 * R0)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];
shared vec2 sm_std[TILE_AREA]; // precomputed eigen_std (σ_⊥, σ_∥) per tile pixel

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

void unpackLightSampleSM(uint tile_idx, out vec3 pos,
    out AliceEncoding encoded, out float variance) {
    vec4 geom = sm_geometry[tile_idx];
    vec4 light = sm_light[tile_idx];
    pos = geom.xyz;
    // geom.w is now surfaceMask — normal not needed for ALICE edge-stopping
    encoded = unpackAlice(light.x, light.y, light.z);
    variance = light.w;
}

// à-trous 分数阶方差传播指数
float relevant_power(const float R) {
    if (R <= 1.0f) {
        return 2.0f;
    }
    const float R2 = R * R;
    const float rho = exp(-1.5f * R2 / (R2 - 1.0f));
    const float p_exact = 2.0f - ATROUS_GAMMA * (log(1.0f + 8.0f * rho) / 2.197224577f);
    return max(1.0f, p_exact);
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
            vec4 light = texelFetch(colortex4, cc, 0);
            sm_light[i] = light;

            // 在共享内存加载阶段立即预计算 eigen_std，
            // 消除 atrous 采样循环中 alice_eigen_std 的冗余 sqrt
            AliceEncoding pre_enc = unpackAlice(light.x, light.y, light.z);
            vec4 pre_y = pre_enc.aliceY;
            float pre_omega = abs(pre_y.w);
            float pre_len = length(pre_y.xyz);
            float pre_kappa = alice_kappa(pre_len, pre_omega);
            sm_std[i] = alice_eigen_std(pre_omega, pre_kappa);
        } else {
            sm_light[i] = vec4(0.0, 0.0, 0.0, -1.0); // 天空标记
            sm_std[i] = vec2(0.0);
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

    const float power = relevant_power(R0 * 2); // 乘 2 是因为方差估计是给下一级用的

    vec3 center_pos;
    AliceEncoding center_alice;
    float center_var_est;
    unpackLightSampleSM(center_idx, center_pos, center_alice, center_var_est);

    center_var_est = max(center_var_est, 1e-12);

    // ---- 从共享内存解码中心法线（colortex3.w = oct(centerNormal)）-------
    vec3 center_normal = decodeNormal(sm_geometry[center_idx].w);

    // ---- 预计算中心像素的统计特征 -----------------------------------------
    vec4 c_enc = center_alice.aliceY;
    vec2 c_std = sm_std[center_idx]; // 共享内存加载阶段已预计算 eigen_std

    float c_inv_var = 1.0 / max(center_var_est, 4e-9);

    float dist_to_cam = max(length(center_pos), 0.001);
    float inv_pixel_footprint = 1.0 / (ATROUS_POSITION_PARAM
                * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0;
    float sumVarEnergy = center_var_est;
    AliceEncoding accumAlice = center_alice;

    // 3×3 网格采样核 (小核 R0=1,2,4, 权重预计算)
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

        uint sx = cx + uint(dx * R0);
        uint sy = cy + uint(dy * R0);
        uint sample_idx = sy * uint(TILE_SIZE) + sx;

        if (sm_light[sample_idx].w < 0.0) continue;

        vec3 sample_world_pos;
        AliceEncoding sample_alice;
        float sample_var_est;
        unpackLightSampleSM(sample_idx, sample_world_pos,
            sample_alice, sample_var_est);

        sample_alice.aliceY.w = abs(sample_alice.aliceY.w);

        vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
        float w_geometry = abs(dot(delta, center_normal));

        vec3 s_v = sample_alice.aliceY.xyz;
        vec2 s_std = sm_std[sample_idx]; // 共享内存预计算的 eigen_std

        float d_bures_sq = alice_bures_distance_sq_precomputed(c_enc.xyz, c_std, s_v, s_std);
        float w_luma = ATROUS_PHI_L * R0 * d_bures_sq * c_inv_var;

        const float w_kernel = GRID_3x3[k].z;
        float w0 = w_kernel * exp(-w_geometry) / (1 + w_luma);

        accumulate_alice(accumAlice, sample_alice, w0);
        sumWeight += w0;

        sumVarEnergy += pow(w0, power) * sample_var_est;
    }

    float inv_sumWeight = 1.0 / sumWeight;
    accumAlice = scale_alice(accumAlice, inv_sumWeight);

    float varEnergyOut = sumVarEnergy * pow(inv_sumWeight, power);
    imageStore(colorimg4, pix, vec4(packAlice(accumAlice), varEnergyOut));
}
