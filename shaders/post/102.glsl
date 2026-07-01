#version 430 compatibility

// ===========================================================================
// Pass 102: 折射时域累积 (Refract Temporal Accumulation)
// ===========================================================================
// 管线位置: 在光线追踪生成折射样本后，与上一帧历史混合
//
// 与 100.glsl (漫反射) / 101.glsl (反射) 的关键区别:
//   - 折射的 mixWeight 用于追踪折射率变化 (穿过不同材质)
//   - 附加 refractWeight 一致性检查 (来自 denoiseBuffer)
//   - 使用 svgfPositionWeight 辅助验证位置一致性
//   - NORMAL_PARAM=64 — 中间严厉度 (折射表面通常不如镜面敏感)
//
// 混合公式: 与反射相同的 EMA 模式
//   new = old + (current - old) / prevW
// ===========================================================================

#define REFRACT_BUFFER_MIN
// 仅写 swap_color (color/lpos/lnormal 由 swap7 写),
// 避免 fragment 内 sampler 读 + imageStore 写同一纹理的 UB (原 102 已验证可行).

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

in vec2 texCoord;

uniform sampler2D colortex0;

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

// 法线相似度权重指数
const float NORMAL_PARAM = 64.0;

// 位置/深度差异灵敏度
const float POSITION_PARAM = 64.0;

const float LUMINANCE_PARAM = 4.0;

// ---------------------------------------------------------------------------
// 边缘停止权重函数
// ---------------------------------------------------------------------------

// 法线权重: 基于法线矢量差的指数衰减 (替代夹角余弦方式)
float svgfNormalWeight(vec3 centerNormal, vec3 normal) {
    return clamp(exp2(-7.21347520 * length(centerNormal - normal)), 0.0, 1.0);
}

// 位置权重: 平面距离衰减 (注: "1 + 0*exp(...)" 当前退化为常数 1)
float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    return exp2(-POSITION_PARAM * LOG2_E * abs(dot(pixelPos - centerPos, normal)
               * (1.0 + 0.0 * exp2(-0.18033688 * distance))));
}

// ---------------------------------------------------------------------------
// 重投影函数
// ---------------------------------------------------------------------------

vec3 reproject(vec3 screenPos) {
    vec4 tmp = gbufferProjectionInverse * vec4(screenPos * 2.0 - 1.0, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 playerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = playerPos + cameraPosition;
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return (prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5);
}

vec3 reproject2(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------

vec3 prevScreenPos;
float info_distance;
uint idx_l;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

vec3IllumiantionData data3;  // 当前像素的折射光照数据 (来自 SSBO)

// ===========================================================================
// 时域混合 (Refract)
// ===========================================================================
void MixRefract() {
    // ---- 重投影失败: 重置历史 --------------------------------------------
    if (notInRange(prevScreenPos.xy)) {
        data3.weight = 1.0;
        return;
    }

    vec3IllumiantionData data = sampleRefract(prevScreenPos.xy * textureSize(colortex0, 0));

    // 位置权重: 主命中点在 data3.normal 平面上的距离差 (对重投影亚像素误差鲁棒)
    float posWeight = svgfPositionWeight(data.pos, data3.pos, data3.normal, info_distance);
    // req 6: 虚拟投射距离 (hit distance) 变化时衰减累积 — vprojdist (= length(normal))
    float hitWeight = exp2(-4.0 * LOG2_E * abs(length(data.normal) - length(data3.normal))
                         / max(length(data3.normal), 0.1));
    float s = exp2(-0.36067376 * abs(denoiseBuffer.data[idx_l].refractWeight - data.mixWeight))
            * float(denoiseBuffer.data[idx_l].distance > -0.5)
            * svgfNormalWeight(data.normal, data3.normal)
            * posWeight * hitWeight;

    float prevW = data.weight;
    // 历史权重上限 = ACCUMULATION_LENGTH (折射比反射更容易变化，所以限制更紧)
    prevW = max(1.0, min(prevW * s + 1.0, ACCUMULATION_LENGTH));

    // EMA 混合
    data3.data_swap = data.data + (data3.data_swap - data.data) / prevW;
    data3.weight = prevW;
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    idx = getIdx(uvec2(gl_FragCoord.xy));

    info_distance = denoiseBuffer.data[idx].distance;

    // 从 SSBO (SpecularRTElement) 重建当前帧折射数据: normal = R*vprojdist
    unpackSpecularRT(refractIllumiantionBuffer.data[idx], data3.pos, data3.normal, data3.data_swap);
    data3.data = vec3(0.0);
    data3.weight = 0.0;
    data3.prev_weight = 0.0;
    data3.mixWeight = 0.0;

    // ---- 天空 / 无效几何: 重置权重后直接写出 -----------------------------
    if (info_distance < -0.5) {
        data3.weight = 1.0;
        WriteRefract(data3, ivec2(gl_FragCoord.xy));
        return;
    }

    // ---- 重投影到上一帧 (主命中点: 定位同一折射表面点) ------------------
    prevScreenPos = reproject2(data3.pos);
    idx_l = getIdx(uvec2(prevScreenPos.xy * textureSize(colortex0, 0)));

    // ---- 执行时域混合 ----------------------------------------------------
    MixRefract();

    // 控制字段由 swap7 写出 (REFRACT_BUFFER_MIN 仅写 swap_color).
    WriteRefract(data3, ivec2(gl_FragCoord.xy));
}
