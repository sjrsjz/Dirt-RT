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

diffuseIlluminationBufferDataW current_data;
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
// 时域累积核心逻辑 (无方差追踪 — 仅维护有效样本权重)
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
    // 情况 2: 正常重投影，采样历史
    // -----------------------------------------------------------------------
    vec2 prev_screen =
        prevScreenPos.xy * vec2(textureSize(colortex0, 0));

    diffuseIlluminationData histData = sampleDiffuse(prev_screen);

    // 历史权重
    float histWeight = sanitizeWeight(histData.prev_weight);

    // -----------------------------------------------------------------------
    // 重投影置信度
    // -----------------------------------------------------------------------
    vec3 histPosCur = histData.pos - cameraDelta;
    float pos_weight = svgfPositionWeight(
        histPosCur,
        current_data.pos,
        current_data.normal,
        info_distance
    );

    float normal_weight = svgfNormalWeight(
        histData.normal,
        current_data.normal,
        info_distance
    );

    float confidence =
        float(info_distance > -0.5) *
        pos_weight *
        normal_weight;

    confidence = pow(
        clamp(confidence, 0.0, 1.0),
        TEMPORAL_CONFIDENCE_POWER
    );

    // -----------------------------------------------------------------------
    // 简单 EMA 混合 (无方差追踪)
    // Wh = 历史有效权重, Wc = 当前权重 (=1), currentAlpha = Wc/(Wh+Wc)
    // -----------------------------------------------------------------------
    float Wh = histWeight * confidence;

    // 历史无效，直接重置到当前帧
    if (Wh <= TEMPORAL_HISTORY_MIN_WEIGHT) {
        output_weight = min(curWeight, max_history);
        out_data.data_swap = current_data.data_swap;
        return;
    }

    float W = Wh + curWeight;
    float currentAlpha = curWeight / max(W, 1e-6);

    // 限制历史权重
    output_weight = min(W, max_history);

    // -----------------------------------------------------------------------
    // 混合完整 SH / ALICE 数据
    // -----------------------------------------------------------------------
    if (currentAlpha >= 0.9999) {
        out_data.data_swap = current_data.data_swap;
    } else {
        out_data.data_swap = mix_SH(
            histData.data,
            current_data.data_swap,
            currentAlpha
        );
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
    out_data.data = init_SH();
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