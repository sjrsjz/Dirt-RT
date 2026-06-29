#version 430 compatibility

// ===========================================================================
// Pass 100: 漫反射时域累积 (Diffuse Temporal Accumulation)
// ===========================================================================
//
// 本版本引入 ALICE 时域标量方差：
//   variance = tr(Cov(X))
//
// 约定:
//   diffuseIllumiantionData.weight = temporal effective weight / N_eff
//   diffuseIllumiantionData.variance = temporal raw trace variance = tr(Cov(X))
//   estimator variance = variance / max(weight, 1.0)
//
// 注意:
//   这里维护的是 raw variance，不是已经除以 N 的 estimator variance。
//
// 统计量直接写入 diffuseIllumiantionData，随 WriteDiffuse 一并回写，
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
uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;

uniform float near;
uniform float far;
uniform vec2 resolution;
uniform int worldTime;

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------

#define NORMAL_PARAM TEMPORAL_NORMAL_PARAM
#define POSITION_PARAM TEMPORAL_POSITION_PARAM

// 历史最大有效样本数。
#ifndef TEMPORAL_MAX_HISTORY
#define TEMPORAL_MAX_HISTORY 1000.0
#endif

// 当前帧 ALICE intrinsic variance 注入比例。
// 1.0 = 完全使用 ALICE 理论 tr(Cov(X))
// 0.0 = 当前帧视为一个裸观测样本，只由跨帧 Welford 估计经验方差
#ifndef TEMPORAL_CURRENT_INTRINSIC_VAR_SCALE
#define TEMPORAL_CURRENT_INTRINSIC_VAR_SCALE 1.0
#endif

// 重投影置信度幂。
#ifndef TEMPORAL_CONFIDENCE_POWER
#define TEMPORAL_CONFIDENCE_POWER 0.25
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

diffuseIllumiantionBufferData current_data;
diffuseIllumiantionData out_data;

float output_weight = 0.0;
float output_variance = 0.0;

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
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

// ---------------------------------------------------------------------------
// 重投影
// ---------------------------------------------------------------------------

vec3 reproject(vec3 screenPos) {
    vec4 tmp = gbufferProjectionInverse * vec4(screenPos * 2.0 - 1.0, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 playerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = playerPos + cameraPosition;

    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);

    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

vec3 reproject2(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);

    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

// ---------------------------------------------------------------------------
// ALICE 安全方差
//
// alice_variance(encoded) 理论要求 encoded.w >= length(encoded.xyz)。
// 压缩、滤波、半精度写回后可能出现 omega < |v|，这里做锥投影保护。
// ---------------------------------------------------------------------------

vec4 alice_project_cone_safe(vec4 encoded) {
    float lenV = length(encoded.xyz);
    encoded.w = max(encoded.w, lenV);
    return encoded;
}

float alice_trace_cov_x_safe(vec4 encoded) {
    encoded = alice_project_cone_safe(encoded);

    if (encoded.w < 1e-8) {
        return 0.0;
    }

    return sanitizeFloatNonNegative(alice_variance(encoded));
}

// ===========================================================================
// Vector Trace Welford / Chan Merge
// ===========================================================================
//
// 维护对象:
//   X ∈ R^3
//   mean = E[X] = v
//   rawTraceVar = tr(Cov(X))
//   weight = effective sample count
//
// 历史:
//   histMeanV
//   histRawTraceVar
//   histWeight
//
// 当前 batch:
//   curMeanV
//   curRawTraceVar
//   curWeight
//
// 合并公式:
//   delta = curMeanV - histMeanV
//   W = Wh + Wc
//   mean = histMeanV + delta * Wc / W
//   M2 = M2h + M2c + |delta|^2 * Wh * Wc / W
//   rawTraceVar = M2 / W
//
// 注意:
//   rawTraceVar 不是 estimator variance。
//   estimator variance = rawTraceVar / weight。
// ===========================================================================

void mergeTraceCovWelford(
    vec3 histMeanV,
    float histRawTraceVar,
    float histWeight,

    vec3 curMeanV,
    float curRawTraceVar,
    float curWeight,

    float confidence,
    float maxHistory,

    out float outWeight,
    out float outRawTraceVar,
    out float outCurrentAlpha
) {
    histWeight = sanitizeWeight(histWeight);
    histRawTraceVar = sanitizeFloatNonNegative(histRawTraceVar);

    curWeight = max(sanitizeFloatNonNegative(curWeight), 1e-4);
    curRawTraceVar = sanitizeFloatNonNegative(curRawTraceVar);

    confidence = clamp(confidence, 0.0, 1.0);

    // 历史有效样本数经过重投影置信度衰减
    float Wh = histWeight * confidence;
    float Wc = curWeight;

    // 历史无效，直接重置到当前帧
    if (Wh <= TEMPORAL_HISTORY_MIN_WEIGHT) {
        outWeight = min(Wc, maxHistory);
        outRawTraceVar = curRawTraceVar;
        outCurrentAlpha = 1.0;
        return;
    }

    float W = Wh + Wc;
    float invW = 1.0 / max(W, 1e-6);

    vec3 delta = curMeanV - histMeanV;

    // 用于混合完整 SH/ALICE 数据
    outCurrentAlpha = Wc * invW;

    float M2h = histRawTraceVar * Wh;
    float M2c = curRawTraceVar * Wc;

    float M2 =
        M2h +
        M2c +
        dot(delta, delta) * (Wh * Wc * invW);

    // 限制历史权重，但保持 raw variance 不变
    if (W > maxHistory) {
        float scale = maxHistory / W;
        W = maxHistory;
        M2 *= scale;
    }

    outWeight = W;
    outRawTraceVar = sanitizeFloatNonNegative(M2 / max(W, 1e-6));
}

// ===========================================================================
// 时域累积核心逻辑
// ===========================================================================

void MixDiffuse() {
    const float max_history = TEMPORAL_MAX_HISTORY;

    // 当前帧 ALICE 状态
    // 约定：shY.xyz = v, shY.w = omega
    vec4 curY = alice_project_cone_safe(current_data.data_swap.shY);

    // 当前帧 raw trace variance = tr(Cov(X))
    float curRawTraceVar =
        TEMPORAL_CURRENT_INTRINSIC_VAR_SCALE *
        alice_trace_cov_x_safe(curY);

    // 当前帧作为一个新 batch，默认权重为 1。
    float curWeight = 1.0;

    // -----------------------------------------------------------------------
    // 情况 1: 重投影失败
    // -----------------------------------------------------------------------
    if (notInRange3(prevScreenPos)) {
        output_weight = curWeight;
        output_variance = curRawTraceVar;
        out_data.data_swap = current_data.data_swap;
        return;
    }

    // -----------------------------------------------------------------------
    // 情况 2: 正常重投影，采样历史
    // -----------------------------------------------------------------------
    vec2 prev_screen =
        prevScreenPos.xy * vec2(textureSize(colortex0, 0));

    diffuseIllumiantionData histData = sampleDiffuse(prev_screen);

    vec4 histY = alice_project_cone_safe(histData.data.shY);

    // 历史权重与历史 raw trace variance
    float histWeight = sanitizeWeight(histData.prev_weight);
    float histRawTraceVar = sanitizeFloatNonNegative(histData.prev_variance);

    // -----------------------------------------------------------------------
    // 重投影置信度
    // -----------------------------------------------------------------------
    float pos_weight = svgfPositionWeight(
        histData.pos,
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
    // Welford / Chan 合并 raw trace variance
    // -----------------------------------------------------------------------
    float outWeightLocal;
    float outRawTraceVarLocal;
    float currentAlpha;

    mergeTraceCovWelford(
        histY.xyz,
        histRawTraceVar,
        histWeight,

        curY.xyz,
        curRawTraceVar,
        curWeight,

        confidence,
        max_history,

        outWeightLocal,
        outRawTraceVarLocal,
        currentAlpha
    );

    // -----------------------------------------------------------------------
    // 混合完整 SH / ALICE 数据
    //
    // 注意:
    //   Welford 统计量只维护 v 的 trace variance。
    //   但是光照本身仍然在线性嵌入空间中整体混合。
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

    // 输出统计量
    output_weight = outWeightLocal;
    output_variance = outRawTraceVarLocal;
}

// ===========================================================================
// 主入口
// ===========================================================================

void main() {
    uvec2 pix = uvec2(gl_FragCoord.xy);
    idx = getIdx(pix);


    // -----------------------------------------------------------------------
    // 读取当前像素几何与光照
    // -----------------------------------------------------------------------
    info_distance = denoiseBuffer.data[idx].distance;
    current_data = diffuseIllumiantionBuffer.data[idx];

    // 默认输出初始化
    out_data.data_swap = current_data.data_swap;
    out_data.data = init_SH();
    out_data.normal = current_data.normal;
    out_data.normal2 = current_data.normal2;
    out_data.pos = current_data.pos;

    output_weight = 1.0;
    output_variance = 0.0;

    // -----------------------------------------------------------------------
    // 天空 / 无效几何
    //
    // 建议对天空写 weight = 0，避免未来帧重投影误采到天空历史。
    // -----------------------------------------------------------------------
    if (info_distance < -0.5) {
        out_data.weight = 0.0;
        out_data.variance = 0.0;

        WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));

        return;
    }

    // -----------------------------------------------------------------------
    // 重投影到上一帧
    // -----------------------------------------------------------------------
    prevScreenPos = reproject2(current_data.pos);

    // -----------------------------------------------------------------------
    // 执行时域累积
    // -----------------------------------------------------------------------
    MixDiffuse();

    // -----------------------------------------------------------------------
    // 回写时域统计量 (随 diffuse 数据一起走，无需额外缓冲区)
    //
    // out_data.weight = N_eff = temporal effective weight
    // out_data.variance = raw trace variance = tr(Cov(X))
    //
    // 后续 estimator variance = variance / max(weight, 1.0)
    // -----------------------------------------------------------------------
    out_data.weight = output_weight;
    out_data.variance = output_variance;

    // -----------------------------------------------------------------------
    // 写出 diffuse 数据
    // -----------------------------------------------------------------------
    WriteDiffuse(out_data, ivec2(gl_FragCoord.xy));
}