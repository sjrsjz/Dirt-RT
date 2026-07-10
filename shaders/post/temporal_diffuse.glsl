#version 430 compatibility

// ===========================================================================
// Pass 100: 漫反射时域累积 (Diffuse Temporal Accumulation)
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

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

#include "/lib/lighting/alice.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

uniform sampler2D colortex0;

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
// 输出
// ---------------------------------------------------------------------------

/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 output_data;


// ---------------------------------------------------------------------------
// 全局变量
// ---------------------------------------------------------------------------

vec3 prevScreenPos;
float info_distance;
uint idx;

in vec2 texCoord;

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
// ===========================================================================

// 采样邻域像素的完整 ALICE 数据
bool sampleNeighborAlice(ivec2 pix, out vec4 aliceY, out vec2 CoCg) {
    ivec2 clamped_pix = clamp(pix, ivec2(0), ivec2(resolution_global) - 1);
    uint n_idx = getIndex(uvec2(clamped_pix));

    if (denoiseBuffer.data[n_idx].distance < -0.5) {
        aliceY = vec4(0.0);
        CoCg = vec2(0.0);
        return false;
    }

    UnifiedDiffuseElement e = diffuseIlluminationBuffer.data[n_idx];
    mediump vec2 aliceY_xy = unpackHalf2x16(floatBitsToUint(e.rt_aliceY_xy));
    mediump vec2 aliceY_zw = unpackHalf2x16(floatBitsToUint(e.rt_aliceY_zw));
    aliceY = clamp(vec4(aliceY_xy, aliceY_zw), vec4(-10000), vec4(10000));
    CoCg = unpackHalf2x16(floatBitsToUint(e.rt_CoCg));
    return true;
}

// 从当前帧邻域计算 6D AABB
void computeAABB(ivec2 pix,
                 out vec4 min_aliceY, out vec4 max_aliceY,
                 out vec2 min_CoCg, out vec2 max_CoCg,
                 out int validCount) {
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

            ivec2 neighbor_pix = pix + ivec2(dx, dy);
            vec4 n_aliceY; vec2 n_CoCg;
            if (sampleNeighborAlice(neighbor_pix, n_aliceY, n_CoCg)) {
                min_aliceY = min(min_aliceY, n_aliceY);
                max_aliceY = max(max_aliceY, n_aliceY);
                min_CoCg = min(min_CoCg, n_CoCg);
                max_CoCg = max(max_CoCg, n_CoCg);
                sum_aliceY    += n_aliceY;
                sum_sq_aliceY += n_aliceY * n_aliceY;
                validCount++;
            }
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
    // 情况 2: 2×2 双线性历史采样 + 逐样本置信度 (参考 TAA 模式)
    //
    // 用 2×2 双线性插值直接读历史数据, 每样本计算几何置信度;
    // 颜色与置信度在同一循环中加权累积, 避免分离的 sampleDiffuse +
    // 3×3 NN 搜索造成的失配.
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

        accumulate_alice(accumAlice, tap.data, w);
        sumWeight += w;
        accumHistWeight += w * tapW;
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
        computeAABB(ivec2(gl_FragCoord.xy),
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
    uvec2 pix = uvec2(gl_FragCoord.xy);
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
        WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
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
    WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
}