#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"

// ===========================================================================
// Pass 300 CS: SVGF 空间滤波器 (计算着色器变体) — 前 3 级 à‑trous (R0=1,2,4)
// ===========================================================================
// 参考: Schied et al., "Spatiotemporal Variance-Guided Filtering", HPG 2017
//
// 设计动机:
//   300.glsl 为 fragment shader, 每线程通过 texelFetch 读取 18 个 vec4
//   (2 中心 + 16 邻域)。前 3 级 R0 ≤ 4, 邻域与邻近线程高度重叠 —
//   16×16 workgroup 内约 75–90% 的纹理读取是冗余的。
//
//   本 CS 变体使用共享内存 (LDS) 将每组所需纹理数据一次协作加载,
//   所有线程随后从共享内存读取, 将全局纹理读取次数降低约 3–6×。
//
// 共享内存使用量 (16×16 工作组, 256 线程):
//   R0=1: 18×18 tile →  324 vec4(geom) +  324 vec4(light) = 10.1 KB
//   R0=2: 20×20 tile →  400 vec4(geom) +  400 vec4(light) = 12.5 KB
//   R0=4: 24×24 tile →  576 vec4(geom) +  576 vec4(light) = 18.0 KB
//   所有尺寸均小于典型 GPU 的 32–64 KB LDS 限制。
//
// 管线:
//   composite50  R0=1  → dispatch CS (R0=1)
//   composite52  R0=2  → dispatch CS (R0=2)
//   composite53  R0=4  → dispatch CS (R0=4)
//
// 注意:
//   - STEP ≥ 4 (R0 ≥ 8) 的旋转抖动逻辑不在此 CS 中 (步长太大, LDS 收益递减)
//   - STEP = 6 的 out_light_sample_blurred 输出不在此 CS 中
// ===========================================================================

// ---- 工作组配置 — 16×16 = 256 线程, 良好占用率 --------------------------------
layout(local_size_x = 16, local_size_y = 16) in;

// ---- 输入纹理 (只读) -------------------------------------------------------
// Iris/OptiFine: compute shader 读取 colortex 使用 sampler2D + texelFetch
// colortex3: 几何信息 (worldPos.xyz + encodedNormal)
uniform sampler2D colortex3;

// colortex4: 光照样本 (ALICE SH + variance)
// 注意: colortex 自动双缓冲 — sampler 读取的是上一 pass 的输出, 不会产生 RAW 冲突
uniform sampler2D colortex4;

// ---- 输出图像 --------------------------------------------------------------
// Iris/OptiFine: compute shader 写入 colortex 使用 colorimgN + imageStore
// 对应 RENDERTARGETS: 4 — 滤波后的光照样本
layout(rgba32f) uniform image2D colorimg4;

// ---- 共享内存 — 缓存 tile 内的几何 + 光照数据 ----------------------------
// TILE_SIZE = 16 + 2*R0  (R0 由预处理器定义, 与 300.glsl 一致)
// 天空掩码复用 sm_light[].w (方差): >= 0 为有效像素, < 0 为天空
#define TILE_SIZE (16 + 2 * R0)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];

// ===========================================================================
// 辅助函数 — 从共享内存解包光照样本 (与 300.glsl unpackLightSample 语义一致)
// ===========================================================================

void unpackLightSampleSM(uint tile_idx, out vec3 pos, out vec3 normal, out SH sh, out float variance) {
    vec4 geom  = sm_geometry[tile_idx];
    vec4 light = sm_light[tile_idx];
    pos      = geom.xyz;
    normal   = decodeNormal(geom.w);
    sh       = unpackSH(light.x, light.y, light.z);
    variance = light.w;
}

// ===========================================================================
// 主函数
// ===========================================================================

void main() {
    // ---- 工作项标识 --------------------------------------------------------
    ivec2 pix       = ivec2(gl_GlobalInvocationID.xy);
    uvec2 local_id  = gl_LocalInvocationID.xy;
    uint  local_idx = gl_LocalInvocationIndex;
    uvec2 group_id  = gl_WorkGroupID.xy;

    ivec2 texSize = textureSize(colortex3, 0);

    // ---- Tile 在全局坐标中的原点 (左上角, 含 R0 边框) -----------------------
    // WorkGroup 处理的像素范围: [group_id*16, (group_id+1)*16 - 1]
    // 加上 R0 边框以覆盖 3×3 à‑trous 核的所有邻域
    ivec2 tile_origin = ivec2(group_id * 16u) - ivec2(R0);

    // =========================================================================
    // Phase 1: 协作加载 tile 到共享内存
    // =========================================================================
    // 256 线程按 round-robin 分摊 TILE_AREA 个像素的加载任务。
    // 每个线程加载 ceil(TILE_AREA / 256) 个元素。
    // 越界像素的方差写入负值作为天空标记 (方差合法值 ≥ 0)

    for (uint i = local_idx; i < uint(TILE_AREA); i += 256u) {
        uint  tx = i % uint(TILE_SIZE);
        uint  ty = i / uint(TILE_SIZE);
        ivec2 gc = tile_origin + ivec2(tx, ty);          // 全局坐标
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);     // 钳制到有效范围

        sm_geometry[i] = texelFetch(colortex3, cc, 0);

        if (gc == cc) {
            sm_light[i] = texelFetch(colortex4, cc, 0);
        } else {
            // 越界像素: 几何数据无关紧要, 方差标记为负 → 天空
            sm_light[i] = vec4(0.0, 0.0, 0.0, -1.0);
        }
    }

    barrier();
    memoryBarrierShared();

    // =========================================================================
    // Phase 2: 中心像素有效性检查
    // =========================================================================
    // 中心像素在 tile 内的坐标偏移 = R0 (因为 tile 比 workgroup 多了 R0 的边框)
    uint cx = local_id.x + uint(R0);
    uint cy = local_id.y + uint(R0);
    uint center_idx = cy * uint(TILE_SIZE) + cx;

    // 跳过越界像素 (workgroup 边界超出图像尺寸)
    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    // 跳过天空像素 — 方差负值为天空 mask (由 swap2 写入)
    if (sm_light[center_idx].w < 0.0) return;

    // ---- 解包中心像素 ------------------------------------------------------
    vec3 center_pos, center_normal;
    SH center_sh;
    float center_var_est;
    unpackLightSampleSM(center_idx, center_pos, center_normal, center_sh, center_var_est);

    // 像素的世界空间 footprint, 用于距离无关的深度边缘停止
    float dist_to_cam = max(length(center_pos - camPos), 0.001);
    float inv_pixel_footprint = 1.0 / (SVGF_POSITION_PARAM * max(dist_to_cam / float(resolution_global.y), 0.00001));

    // ---- 初始化累积器 (中心像素权重 = 1) -----------------------------------
    float sumWeight    = 1.0;
    float sumVarEnergy = center_var_est;
    SH accumulatedSH   = center_sh;

    // B‑样条权重核 (中心 1.0, 十字 0.66667)
    float hw[2] = float[](1.0, 0.66667);

    // =========================================================================
    // Phase 3: 3×3 à‑trous 采样循环 — 全部从共享内存读取
    // =========================================================================
    // 对于 R0 ≤ 4 (前 3 级), 不使用旋转抖动; 直接轴对齐步进
    // (旋转抖动仅在 STEP ≥ 4 即 R0 ≥ 8 时启用, 由 300.glsl 处理)

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            // 邻域在 tile 内的坐标 — 步长为 R0
            uint sx = cx + uint(i * R0);
            uint sy = cy + uint(j * R0);
            uint sample_idx = sy * uint(TILE_SIZE) + sx;

            // ---- 快速天空检查 (共享内存, ~1 周期延迟) -----------------------
            if (sm_light[sample_idx].w < 0.0) continue;

            // ---- 贴图空间权重 ---------------------------------------------
            float w_kernel = hw[abs(i)] * hw[abs(j)];

            // ---- 解包邻域样本 (共享内存读取) ------------------------------
            vec3 sample_world_pos, sample_normal;
            SH sample_sh;
            float sample_var_est;
            unpackLightSampleSM(sample_idx, sample_world_pos, sample_normal, sample_sh, sample_var_est);

            // ---- 深度权重 -------------------------------------------------
            // 采样点到中心平面的垂直距离, 归一化到屏幕空间
            vec3 delta = (sample_world_pos - center_pos) * inv_pixel_footprint;
            float depthTerm = abs(dot(delta, center_normal));

            // ---- 几何权重 -------------------------------------------------
            float w_geometry = SVGF_NORMAL_POWER * (1.0 - dot(center_normal, sample_normal)) + depthTerm;

            // ---- 亮度权重 -------------------------------------------------
            // 方差预滤波使得 sigma2 不再导致降噪器崩溃
            float sigma2 = max(center_var_est, 1e-8);
            float delta_energy = length(center_sh.shY.xyz - sample_sh.shY.xyz);
            float w_luma = SVGF_PHI_L * delta_energy * inversesqrt(sigma2);

            // ---- 组合权重 -------------------------------------------------
            float w0 = w_kernel * (1.0 + w_luma) * exp2(-(w_geometry + w_luma) * LOG2_E);

            // ---- 累积加权样本 ---------------------------------------------
            accumulate_SH(accumulatedSH, sample_sh, w0);
            sumWeight    += w0;
            sumVarEnergy += w0 * w0 * sample_var_est;
        }
    }

    // ---- 时空方差混合归一化 -------------------------------------------------
    float inv_sumWeight = 1.0 / sumWeight;
    accumulatedSH = scaleSH(accumulatedSH, inv_sumWeight);
    float varEnergyOut = sumVarEnergy * inv_sumWeight * inv_sumWeight;

    // ---- 输出: 压缩光照样本 -------------------------------------------------
    // Iris/OptiFine: compute shader 写入 colortex 使用 imageStore + colorimgN
    imageStore(colorimg4, pix, vec4(packSH(accumulatedSH), varEnergyOut));
}
