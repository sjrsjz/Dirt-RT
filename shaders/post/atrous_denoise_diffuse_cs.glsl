#version 430 core
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
uniform usampler2D colortex4;

layout(rgba32ui) uniform uimage2D colorimg4;

// ---------------------------------------------------------------------------
// 共享内存
// ---------------------------------------------------------------------------
#define TILE_SIZE (16 + 2 * R0)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
// Alice/CoCg retain their source FP16 representation; variance remains the
// original FP32 word. Bures standard deviations are sufficient at FP16
// precision, while inverse vector length is cheaper to reconstruct per tap
// than to reserve another four LDS bytes for every tile sample.
shared uvec4 sm_light_packed[TILE_AREA];
shared uint sm_stddev_packed[TILE_AREA];
// 36 bytes/sample. Worst case (R0=4): 576 * 36 = 20736 bytes,
// down from 32256 bytes and below the three-workgroup 64-KiB threshold.

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

vec2 makeBuresStddev(vec4 encoded, float omega) {
    float len_v_sq = dot(encoded.xyz, encoded.xyz);
    float kappa = alice_kappa(sqrt(len_v_sq), omega);
    return alice_eigen_std(omega, kappa);
}

void unpackDiffuseTileLight(uint tileIndex, out AliceEncoding alice,
    out float variance) {
    uvec4 light = sm_light_packed[tileIndex];
    alice.aliceY = vec4(unpackHalf2x16(light.x),
        unpackHalf2x16(light.y));
    alice.CoCg = unpackHalf2x16(light.z);
    variance = uintBitsToFloat(light.w);
}

float buresDistanceSqSM(vec3 center_v, vec2 center_stddev,
    float center_inv_len_v_sq, float center_trace, float center_anisotropy,
    vec3 sample_v, vec2 sample_stddev, float sample_inv_len_v_sq) {
    float dot_v = dot(center_v, sample_v);
    float c_sq = min(1.0, dot_v * dot_v
                * center_inv_len_v_sq * sample_inv_len_v_sq);

    vec2 sample_stddev_sq = sample_stddev * sample_stddev;
    float sample_trace = 2.0 * sample_stddev_sq.x + sample_stddev_sq.y;
    float sample_anisotropy = sample_stddev_sq.y - sample_stddev_sq.x;
    float cross_ab = center_stddev.x * sample_stddev.y
            + center_stddev.y * sample_stddev.x;
    float cross_2d = sqrt(max(0.0, cross_ab * cross_ab
                    + c_sq * center_anisotropy * sample_anisotropy));

    vec3 delta_v = center_v - sample_v;
    float mean_distance_sq = dot(delta_v, delta_v);
    float cross_trace = center_stddev.x * sample_stddev.x + cross_2d;
    return max(0.0, mean_distance_sq + center_trace + sample_trace
            - 2.0 * cross_trace);
}

// à-trous 分数阶方差传播指数
#if STEP == 1
#define ATROUS_POWER_COEFFICIENT 0.3339015144
#elif STEP == 2
#define ATROUS_POWER_COEFFICIENT 0.4375201036
#elif STEP == 3
#define ATROUS_POWER_COEFFICIENT 0.4592464660
#elif STEP == 4
#define ATROUS_POWER_COEFFICIENT 0.4644479501
#elif STEP == 5
#define ATROUS_POWER_COEFFICIENT 0.4657344365
#elif STEP == 6
#define ATROUS_POWER_COEFFICIENT 0.4660551979
#endif

// ---------------------------------------------------------------------------
// 主函数
// ---------------------------------------------------------------------------

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uvec2 local_id = gl_LocalInvocationID.xy;
    uvec2 group_id = gl_WorkGroupID.xy;

    ivec2 texSize = textureSize(colortex3, 0);

    ivec2 tile_origin = ivec2(group_id * 16u) - ivec2(R0);

    // =========================================================================
    // Phase 1: 协作加载 tile 到共享内存
    // =========================================================================
    for (uint ty = local_id.y; ty < uint(TILE_SIZE); ty += 16u) {
        for (uint tx = local_id.x; tx < uint(TILE_SIZE); tx += 16u) {
            uint i = ty * uint(TILE_SIZE) + tx;
            ivec2 gc = tile_origin + ivec2(tx, ty);
            ivec2 cc = clamp(gc, ivec2(0), texSize - 1);

            if (gc == cc) {
                uvec4 light = texelFetch(colortex4, cc, 0);
                sm_light_packed[i] = light;
                if (uintBitsToFloat(light.w) < 0.0) {
                    // Sky carries a negative variance sentinel. Do not unpack
                    // Alice or evaluate Bures eigenvalues for dead samples.
                    sm_geometry[i] = vec4(0.0);
                    sm_stddev_packed[i] = 0u;
                } else {
                    vec4 aliceY = vec4(unpackHalf2x16(light.x),
                        unpackHalf2x16(light.y));
                    vec2 stddev = makeBuresStddev(aliceY, abs(aliceY.w));
                    sm_geometry[i] = texelFetch(colortex3, cc, 0);
                    sm_stddev_packed[i] = packHalf2x16(clamp(stddev,
                        vec2(0.0), vec2(65504.0)));
                }
            } else {
                sm_geometry[i] = vec4(0.0);
                sm_light_packed[i] = uvec4(0u, 0u, 0u,
                    floatBitsToUint(-1.0));
                sm_stddev_packed[i] = 0u;
            }
        }
    }

    barrier();

    // =========================================================================
    // Phase 2: 中心像素有效性检查与预计算
    // =========================================================================
    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    uint cx = local_id.x + uint(R0);
    uint cy = local_id.y + uint(R0);
    uint center_idx = cy * uint(TILE_SIZE) + cx;

    if (uintBitsToFloat(sm_light_packed[center_idx].w) < 0.0) return;

    const float power = max(1.0, 2.0
                - ATROUS_GAMMA * ATROUS_POWER_COEFFICIENT);
    const float variance_mix = 2.0 - exp2(2.0 - power);

    AliceEncoding center_alice;
    vec3 center_pos = sm_geometry[center_idx].xyz;
    float center_var_raw;
    unpackDiffuseTileLight(center_idx, center_alice, center_var_raw);
    float center_var_est = max(center_var_raw, 1e-16);

    // ---- 从共享内存解码中心法线（colortex3.w = oct(centerNormal)）-------
    vec3 center_normal = decodeNormal(sm_geometry[center_idx].w);

    // ---- 预计算中心像素的统计特征 -----------------------------------------
    vec4 c_enc = center_alice.aliceY;
    vec2 c_stddev = unpackHalf2x16(sm_stddev_packed[center_idx]);
    vec2 c_stddev_sq = c_stddev * c_stddev;
    float c_trace = 2.0 * c_stddev_sq.x + c_stddev_sq.y;
    float c_anisotropy = c_stddev_sq.y - c_stddev_sq.x;
    float c_len_v_sq = dot(c_enc.xyz, c_enc.xyz);
    float c_inv_len_v_sq = c_len_v_sq > 1e-16
        ? 1.0 / c_len_v_sq : 0.0;

    float resolution_y = float(resolution_global.y);
    float dist_to_cam = max(length(center_pos), 0.001);
    float inv_pixel_footprint = resolution_y / (ATROUS_POSITION_PARAM
                * max(dist_to_cam, resolution_y * 0.00001));
    float center_plane_distance = dot(center_pos, center_normal);

    // ---- 初始化累积器 ----------------------------------------------------
    float sumWeight = 1.0;
    vec2 sumVarEnergy = vec2(center_var_est);

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

        AliceEncoding sample_alice;
        vec3 sample_world_pos = sm_geometry[sample_idx].xyz;
        float sample_var_est;
        unpackDiffuseTileLight(sample_idx, sample_alice,
            sample_var_est);
        if (sample_var_est < 0.0) continue;
        float w_geometry = abs(dot(sample_world_pos, center_normal)
                    - center_plane_distance) * inv_pixel_footprint;

        vec3 s_v = sample_alice.aliceY.xyz;
        float s_len_v_sq = dot(s_v, s_v);
        float s_inv_len_v_sq = s_len_v_sq > 1e-16
            ? 1.0 / s_len_v_sq : 0.0;
        float d_bures_sq = buresDistanceSqSM(c_enc.xyz, c_stddev,
                c_inv_len_v_sq, c_trace, c_anisotropy, s_v,
                unpackHalf2x16(sm_stddev_packed[sample_idx]),
                s_inv_len_v_sq);
        float w_luma = ATROUS_PHI_L * d_bures_sq / (center_var_est + sample_var_est);

        const float w_kernel = GRID_3x3[k].z;
        float w0 = w_kernel * exp(-w_geometry - w_luma);

        accumulate_alice(accumAlice, sample_alice, w0);
        sumWeight += w0;
        float weighted_var = w0 * sample_var_est;
        sumVarEnergy += vec2(weighted_var, w0 * weighted_var);
    }

    float inv_sumWeight = 1.0 / sumWeight;
    accumAlice = scale_alice(accumAlice, inv_sumWeight);

    float varEnergyOut = mix(sumVarEnergy.x, sumVarEnergy.y, variance_mix)
            * pow(inv_sumWeight, power);
    imageStore(colorimg4, pix, uvec4(
            packHalf2x16(clamp(accumAlice.aliceY.xy, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(accumAlice.aliceY.zw, vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(accumAlice.CoCg, vec2(-65504.0), vec2(65504.0))),
            floatBitsToUint(varEnergyOut)));
}
