#version 430 compatibility

// ===========================================================================
// Pass 100 CS: 漫反射时域累积 (Diffuse Temporal Accumulation)
// ===========================================================================
//
// 本版本引入 ALICE 时域标量方差：
//   variance = tr(Cov(X))
//
// 约定:
//   diffuseIlluminationData.weight = temporal effective weight / N_eff
//   diffuseIlluminationData.variance = temporal raw trace variance = tr(Cov(X))
//   estimator variance = variance / max(weight, 1.0)
//
// 注意:
//   这里维护的是 raw variance，不是已经除以 N 的 estimator variance。
//
// 统计量直接写入 diffuseIlluminationData，随 WriteDiffuse 一并回写，
// 不再需要 extInfoBuffer 间接缓冲区。
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

#include "/lib/lighting/alice.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

uniform vec2 resolution;

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

#define NORMAL_PARAM TEMPORAL_NORMAL_PARAM
#define POSITION_PARAM TEMPORAL_POSITION_PARAM

// 历史最大有效样本数。
#ifndef TEMPORAL_MAX_HISTORY
#define TEMPORAL_MAX_HISTORY 32.0
#endif

// 重投影置信度幂。
#ifndef TEMPORAL_CONFIDENCE_POWER
#define TEMPORAL_CONFIDENCE_POWER 1.0
#endif

// 历史有效权重低于该值时重置
#ifndef TEMPORAL_HISTORY_MIN_WEIGHT
#define TEMPORAL_HISTORY_MIN_WEIGHT 1e-4
#endif

// ---------------------------------------------------------------------------
// AABB 钳制参数 (Axis-Aligned Bounding Box Clamping)
// 在 6D ALICE 嵌入空间 (v.xyz + omega + CoCg) 中线性缩放钳制，防止拖影
// 多通道统一等比缩放: 保持 v/omega 比例 → 无方向伪影
// ---------------------------------------------------------------------------

// 启用 AABB 钳制 (设为 0 则回退到原始 EMA 混合)
#ifndef TEMPORAL_AABB_ENABLE
#define TEMPORAL_AABB_ENABLE 1
#endif

// AABB 邻域半径: 1 = 3×3, 2 = 5×5
#ifndef TEMPORAL_AABB_NEIGHBOR_RADIUS
#define TEMPORAL_AABB_NEIGHBOR_RADIUS 2
#endif

// AABB 扩展系数: 将邻域 min/max 范围按比例放大
// 由于仅 8 个 1 SPP 邻域样本，min/max 严重低估真实范围，需要较大值补偿
#ifndef TEMPORAL_AABB_EXPAND
#define TEMPORAL_AABB_EXPAND 2.0
#endif

// AABB 方差引导扩展系数: 乘以 ALICE 理论 σ (sqrt(Var_scalar))
// 3.0 ≈ 3-sigma，极度宽容，确保统计波动不触发钳制
#ifndef TEMPORAL_AABB_SIGMA_SCALE
#define TEMPORAL_AABB_SIGMA_SCALE 3.0
#endif

// AABB 最小绝对范围 — 暗区保护的最后防线
// 即使 extent≈0 且 σ≈0，也保证这么多绝对扩展，防止暗部信号被钳死
#ifndef TEMPORAL_AABB_MIN_EXTENT
#define TEMPORAL_AABB_MIN_EXTENT 1.0
#endif

// AABB 全局缩放: 觉得整体偏激进/偏保守时优先调此项
#ifndef TEMPORAL_AABB_BOX_SCALE
#define TEMPORAL_AABB_BOX_SCALE 1.0
#endif

// AABB 有效邻域像素最低数量，低于此值则跳过钳制
#ifndef TEMPORAL_AABB_MIN_VALID_NEIGHBORS
#define TEMPORAL_AABB_MIN_VALID_NEIGHBORS 2
#endif

// ---------------------------------------------------------------------------
// AABB 共享内存 Tile — cooperative load 消除邻域 SSBO 冗余读取
// ---------------------------------------------------------------------------
#define TILE_SIZE 16u
#if TEMPORAL_AABB_ENABLE
#define AABB_HALO uint(TEMPORAL_AABB_NEIGHBOR_RADIUS)
#define AABB_SM_W (TILE_SIZE + 2u * AABB_HALO)
#define AABB_SM_H (TILE_SIZE + 2u * AABB_HALO)

struct AABBTileSample {
    float dist;
    vec4 aliceY;
    vec2 CoCg;
};
shared AABBTileSample sm_aabbTile[AABB_SM_H][AABB_SM_W];
#endif

// ---------------------------------------------------------------------------
// 全局变量
// ---------------------------------------------------------------------------

vec3 prevScreenPos;
float info_distance;
uint idx;

DiffuseIlluminationWriteData current_data;
diffuseIlluminationData out_data;

float output_weight = 0.0;

// ===========================================================================
// 工具函数
// ===========================================================================

bool notInRange(vec2 p) {
    return clamp(p, vec2(0.0), vec2(1.0)) != p;
}

bool notInRange3(vec3 p) {
    return
        p.x < 0.0 || p.x > 1.0 ||
        p.y < 0.0 || p.y > 1.0 ||
        p.z < 0.0 || p.z > 1.0;
}

float sanitizeFloatNonNegative(float x) {
    // NaN 情况下 x >= 0.0 为 false
    if (!(x >= 0.0)) return 0.0;
    if (x > 1e20) return 0.0;
    return x;
}

float sanitizeWeight(float w) {
    w = sanitizeFloatNonNegative(w);
    return min(w, TEMPORAL_MAX_HISTORY);
}

// ---------------------------------------------------------------------------
// SVGF 风格的边缘停止权重函数
// ---------------------------------------------------------------------------

float svgfNormalWeight(vec3 centerNormal, vec3 normal, float distance) {
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    return exp2(-POSITION_PARAM * LOG2_E * abs(dot(pixelPos - centerPos, normal)));
}

// ---------------------------------------------------------------------------
// 重投影
// ---------------------------------------------------------------------------

vec3 cameraDelta;

// 重投影: 全光线追踪推导矩阵 (单源一致, 零 Iris 混合)
vec3 reproject(vec3 pos_rel) {
    vec3 prevPlayerPos = pos_rel + cameraDelta;
    vec4 clipPos = rtPrevProjection * rtPrevModelView * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
}

// ===========================================================================
// AABB 钳制辅助函数 (6D ALICE 嵌入空间: aliceY.xyzw + CoCg)
//
// 策略: 计算完整 6D AABB, 取所有越界通道中最保守的缩放因子,
//       统一等比缩放整个 aliceY + CoCg, 保持 |v|/omega 不变
//
// 使用共享内存 tile: cooperative load 消除邻域 SSBO 冗余读取,
// 每像素最多 O(N²) SSBO 读 → O(1).
// ===========================================================================

#if TEMPORAL_AABB_ENABLE

// 从共享内存 tile 中直接读取邻域样本, 计算 6D AABB
void computeAABB_CS(
                 out vec4 min_aliceY, out vec4 max_aliceY,
                 out vec2 min_CoCg, out vec2 max_CoCg,
                 out int validCount) {
    uint cx = gl_LocalInvocationID.x + AABB_HALO;
    uint cy = gl_LocalInvocationID.y + AABB_HALO;

    vec4 center_aliceY = current_data.data_swap.aliceY;
    vec2 center_CoCg = current_data.data_swap.CoCg;

    min_aliceY = center_aliceY;
    max_aliceY = center_aliceY;
    min_CoCg = center_CoCg;
    max_CoCg = center_CoCg;
    validCount = 1;

    vec4 sum_aliceY    = center_aliceY;
    vec4 sum_sq_aliceY = center_aliceY * center_aliceY;

    const int radius = TEMPORAL_AABB_NEIGHBOR_RADIUS;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx == 0 && dy == 0) continue;

            int sx = int(cx) + dx;
            int sy = int(cy) + dy;
            AABBTileSample s = sm_aabbTile[sy][sx];

            if (s.dist < -0.5) continue;

            min_aliceY = min(min_aliceY, s.aliceY);
            max_aliceY = max(max_aliceY, s.aliceY);
            min_CoCg = min(min_CoCg, s.CoCg);
            max_CoCg = max(max_CoCg, s.CoCg);
            sum_aliceY    += s.aliceY;
            sum_sq_aliceY += s.aliceY * s.aliceY;
            validCount++;
        }
    }

    // -- 扩张 --
    vec4 extent_aliceY = max_aliceY - min_aliceY;
    vec2 extent_CoCg = max_CoCg - min_CoCg;

    float scalar_variance = alice_variance(center_aliceY);
    float sigma_alice = sqrt(max(0.0, scalar_variance));

    float inv_n = 1.0 / max(float(validCount), 1.0);
    vec4 mean_aliceY = sum_aliceY * inv_n;
    vec4 neighborhood_var = max(vec4(0.0), sum_sq_aliceY * inv_n - mean_aliceY * mean_aliceY);
    float sigma_neighborhood = sqrt(max(0.0,
        max(max(neighborhood_var.x, neighborhood_var.y),
        max(neighborhood_var.z, neighborhood_var.w))));

    float sigma_combined = max(sigma_alice, sigma_neighborhood);
    float expand_sigma = sigma_combined * TEMPORAL_AABB_SIGMA_SCALE;
    float expand_min = TEMPORAL_AABB_MIN_EXTENT;
    float box_scale = TEMPORAL_AABB_BOX_SCALE;

    vec4 expand_aliceY = (extent_aliceY * TEMPORAL_AABB_EXPAND + vec4(expand_sigma + expand_min)) * box_scale;
    vec2 expand_CoCg = (extent_CoCg * TEMPORAL_AABB_EXPAND + vec2(expand_sigma * 0.5 + expand_min)) * box_scale;

    min_aliceY -= expand_aliceY;
    max_aliceY += expand_aliceY;
    min_CoCg -= expand_CoCg;
    max_CoCg += expand_CoCg;
}

#endif // TEMPORAL_AABB_ENABLE

// 6 通道统一线性缩放:
//   对每个越界通道计算 scale = bound / hist,
//   取所有通道中最小的 scale (最保守), 等比应用到全部 6 维
//   保持 v/omega 比例不变 → 无方向伪影
void clampHistoryToAABB(inout AliceEncoding histAlice,
                        vec4 min_aliceY, vec4 max_aliceY,
                        vec2 min_CoCg, vec2 max_CoCg) {
    float scale = 1.0;

    // 宏: 单通道缩放因子 (取 min 以保最保守)
    #define CHK(v, lo, hi) \
        { float _v = (v); if (_v > (hi)) scale = min(scale, (hi) / _v); \
          else if (_v < (lo)) scale = min(scale, (lo) / _v); }

    CHK(histAlice.aliceY.x, min_aliceY.x, max_aliceY.x);
    CHK(histAlice.aliceY.y, min_aliceY.y, max_aliceY.y);
    CHK(histAlice.aliceY.z, min_aliceY.z, max_aliceY.z);
    CHK(histAlice.aliceY.w, min_aliceY.w, max_aliceY.w);
    CHK(histAlice.CoCg.x, min_CoCg.x, max_CoCg.x);
    CHK(histAlice.CoCg.y, min_CoCg.y, max_CoCg.y);

    #undef CHK

    // 安全钳制: 禁止反转符号, 禁止超过 2× 放大
    scale = clamp(scale, 0.0, 2.0);

    histAlice.aliceY  *= scale;
    histAlice.CoCg *= scale;
}

// ===========================================================================
// 3×3 最近邻几何采样 (用于重投影置信度)
//
// 直接从 SSBO 读取整数坐标处的历史几何数据 (pos + normal),
// 避免双线性插值在亚像素几何抖动时引入的虚假不匹配。
// 返回 false 表示该像素无有效几何 (天空/越界)。
// ===========================================================================

bool fetchHistoryGeometry(ivec2 p, out vec3 pos, out vec3 normal) {
    ivec2 clamped_p = clamp(p, ivec2(0), ivec2(resolution_global) - 1);
    uint n_idx = getIndex(uvec2(clamped_p));

    if (denoiseBuffer.data[n_idx].distance < -0.5) return false;

    UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[n_idx];
    pos    = vec3(e.hist_px, e.hist_py, e.hist_pz);
    normal = decodeNormal(e.hist_oct_n);
    return true;
}

// ===========================================================================
// 时域累积核心逻辑 (NN 几何搜索 + AABB 钳制 + EMA 混合)
// ===========================================================================

void MixDiffuse() {
    const float max_history = TEMPORAL_MAX_HISTORY;
    float curWeight = 1.0;

    // -----------------------------------------------------------------------
    // 情况 1: 重投影失败
    // -----------------------------------------------------------------------
    if (notInRange3(prevScreenPos)) {
        output_weight = curWeight;
        out_data.data_swap = current_data.data_swap;
        return;
    }

    // -----------------------------------------------------------------------
    // 情况 2: 2×2 双线性历史采样 + 逐样本置信度 + 距离尺度修正
    // -----------------------------------------------------------------------
    vec2 prevTexelcoord = prevScreenPos.xy * vec2(resolution_global);
    ivec2 prevTexel = ivec2(floor(prevTexelcoord));

    AliceEncoding accumAlice = init_alice();
    float sumWeight = 0.0;
    float maxTapConf = 0.0;
    float accumHistWeight = 0.0;

    for (int i = 0; i < 4; i++) {
        ivec2 sampleTexel = prevTexel + ivec2(i & 1, i >> 1);
        vec2 sampleCoord = vec2(sampleTexel);

        float bilinearWeight = (1.0 - abs(prevTexelcoord.x - sampleCoord.x))
                             * (1.0 - abs(prevTexelcoord.y - sampleCoord.y));

        diffuseIlluminationData tap = fetchDiffuse(sampleTexel);
        vec3 tapPos, tapNormal;
        if (!fetchHistoryGeometry(sampleTexel, tapPos, tapNormal)) continue;

        float tapW = sanitizeWeight(tap.prev_weight);
        if (tapW < TEMPORAL_HISTORY_MIN_WEIGHT) continue;

        // 几何权重: 历史 pos 转当前相机帧后比较
        vec3 tapPosCur = tapPos - cameraDelta;
        float posW = svgfPositionWeight(tapPosCur, current_data.pos,
                                        current_data.normal, info_distance);
        float normW = svgfNormalWeight(tapNormal, current_data.normal, info_distance);
        float conf = posW * normW;

        maxTapConf = max(maxTapConf, conf);
        float w = bilinearWeight * conf + 1e-10;

        // ===================================================================
        // 基于物理尺度的时域累积有效样本数修正
        // d1 = length(tapPos) (上一帧距离)
        // d2 = length(tapPosCur) (当前帧距离)
        // ===================================================================
        float d1_sq = dot(tapPos, tapPos);
        float d2_sq = dot(tapPosCur, tapPosCur);
        
        // 物理缩放因子: (d2 / d1)^2
        float scale = d2_sq / max(d1_sq, 1e-4);
        
        // 工程钳制保护：
        // 1. 防止极近处除零导致有效权重溢出
        // 2. 限制最大放大倍数（如 4.0），防止大跨度镜头拉远时有效历史权重过度膨胀
        scale = clamp(scale, 0.01, 4.0); 

        // 应用修正后的时域历史有效权重
        float correctedTapW = sanitizeWeight(tapW * scale);
        // ===================================================================

        accumulate_alice(accumAlice, tap.data, w);
        sumWeight += w;
        accumHistWeight += w * correctedTapW; // 使用修正后的 N_eff 参与均值混合
    }

    if (sumWeight < 1e-8) {
        output_weight = curWeight;
        out_data.data_swap = current_data.data_swap;
        return;
    }

    AliceEncoding histAlice = scale_alice(accumAlice, 1.0 / sumWeight);
    float histWeight = accumHistWeight / max(sumWeight, 1e-6);

    float confidence = float(info_distance > -0.5) * maxTapConf;
    confidence = pow(clamp(confidence, 0.0, 1.0), TEMPORAL_CONFIDENCE_POWER);

    // -----------------------------------------------------------------------
    // AABB 钳制 (omega + CoCg) → EMA 混合
    // -----------------------------------------------------------------------
    float Wh = histWeight * confidence;
    if (Wh <= TEMPORAL_HISTORY_MIN_WEIGHT) {
        output_weight = min(curWeight, max_history);
        out_data.data_swap = current_data.data_swap;
        return;
    }

    #if TEMPORAL_AABB_ENABLE
    {
        vec4 min_aliceY, max_aliceY;
        vec2 min_CoCg, max_CoCg;
        int validNeighborCount;
        computeAABB_CS(
                    min_aliceY, max_aliceY,
                    min_CoCg, max_CoCg,
                    validNeighborCount);
        if (validNeighborCount >= TEMPORAL_AABB_MIN_VALID_NEIGHBORS) {
            clampHistoryToAABB(histAlice, min_aliceY, max_aliceY, min_CoCg, max_CoCg);
        }
    }
    #endif

    // -----------------------------------------------------------------------
    // EMA 混合
    // -----------------------------------------------------------------------
    float W = Wh + curWeight;
    float currentAlpha = curWeight / max(W, 1e-6);
    output_weight = min(W, max_history);

    if (currentAlpha >= 0.9999) {
        out_data.data_swap = current_data.data_swap;
    } else {
        out_data.data_swap = mix_alice(histAlice, current_data.data_swap, currentAlpha);
    }
}

// ===========================================================================
// 主入口
// ===========================================================================

void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;

    // =========================================================================
    // Phase 1: Cooperative load AABB tile into shared memory
    // 所有线程必须到达 barrier — 在 OOB / sky 提前返回之前完成。
    // =========================================================================
    #if TEMPORAL_AABB_ENABLE
    {
        uint threadIdx = gl_LocalInvocationID.y * TILE_SIZE + gl_LocalInvocationID.x;
        uint totalThreads = TILE_SIZE * TILE_SIZE;
        for (uint i = threadIdx; i < AABB_SM_W * AABB_SM_H; i += totalThreads) {
            uint row = i / AABB_SM_W;
            uint col = i % AABB_SM_W;

            ivec2 gc = ivec2(gl_WorkGroupID.xy * TILE_SIZE) - ivec2(AABB_HALO) + ivec2(col, row);
            ivec2 clamped = clamp(gc, ivec2(0), ivec2(resolution) - 1);
            uint loadIdx = getIndex(uvec2(clamped));

            float d = denoiseBuffer.data[loadIdx].distance;

            AABBTileSample s;
            s.dist = d;
            if (d > -0.5) {
                UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[loadIdx];
                mediump vec2 aliceY_xy = unpackHalf2x16(floatBitsToUint(e.rt_aliceY_xy));
                mediump vec2 aliceY_zw = unpackHalf2x16(floatBitsToUint(e.rt_aliceY_zw));
                s.aliceY = clamp(vec4(aliceY_xy, aliceY_zw), vec4(-10000), vec4(10000));
                s.CoCg = unpackHalf2x16(floatBitsToUint(e.rt_CoCg));
            } else {
                s.aliceY = vec4(0.0);
                s.CoCg = vec2(0.0);
            }

            sm_aabbTile[row][col] = s;
        }
    }
    barrier();
    memoryBarrierShared();
    #endif

    // =========================================================================
    // Phase 2: 逐像素时域累积
    // =========================================================================
    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    idx = getIndex(pix);


    // -----------------------------------------------------------------------
    // 读取当前像素几何与光照
    // -----------------------------------------------------------------------
    info_distance = denoiseBuffer.data[idx].distance;
    current_data = loadDiffuseInput(idx);

    // 默认输出初始化
    out_data.data_swap = current_data.data_swap;
    out_data.data = init_alice();
    out_data.normal = current_data.normal;
    out_data.normal2 = current_data.normal2;
    out_data.pos = current_data.pos;

    output_weight = 1.0;

    if (info_distance < -0.5) {
        out_data.weight = 0.0;
        WriteDiffuse(out_data, ivec2(pix));
        return;
    }

    // -----------------------------------------------------------------------
    // 重投影到上一帧
    // -----------------------------------------------------------------------
    cameraDelta = camPos - prevRaytracingCamPos;
    prevScreenPos = reproject(current_data.pos);

    // -----------------------------------------------------------------------
    // 执行时域累积
    // -----------------------------------------------------------------------
    MixDiffuse();

    out_data.weight = output_weight;
    WriteDiffuse(out_data, ivec2(pix));
}