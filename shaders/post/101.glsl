#version 430 compatibility

// ===========================================================================
// Pass 101: 镜面反射时域累积 (Reflect Temporal Accumulation)
// ===========================================================================
// 管线位置: 在光线追踪生成镜面反射样本后，与上一帧历史混合
//
// 与 100.glsl (漫反射) 的区别:
//   - 反射信号是 vec3 颜色而非 SH 结构 → 使用更简单的时域混合
//   - 法线权重更激进 (NORMAL_PARAM=512)，因为反射对法线方向极其敏感
//   - 使用 reproject2 直接重投影 (与 100.glsl 相同的路径)
//
// 注意: 此 pass 曾是性能瓶颈 (与 200.glsl 合计占用 ~1/4~1/3 的帧时间)，
//       但 200.glsl 目前已禁用 (return; 截断)，因此仅剩本 pass 的开销。
// ===========================================================================

#define REFLECT_BUFFER_MIN

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

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

// 法线权重指数 — 极高值确保反射仅在法线几乎完全一致时才混合
const float NORMAL_PARAM = 1.0;

// 位置/深度差异灵敏度
const float POSITION_PARAM = 32.0;

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

vec3 prevScreenPos; // 重投影后的上一帧屏幕坐标
float info_distance; // 当前像素的光线追踪距离
uint idx_l; // 重投影像素索引
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

vec3IllumiantionData data2; // 当前像素的反射光照数据 (来自 SSBO)

// ===========================================================================
// 时域混合 (Reflect)
// ===========================================================================
// 与漫反射的 SH 混合不同，反射使用简单的 vec3 EMA:
//   new = old + (current - old) / prevW
// 其中 prevW 是受重投影置信度调制的历史累积权重
void MixReflect() {
    if (notInRange(prevScreenPos.xy)) {
        data2.weight = 1.0;
        return;
    }

    vec3IllumiantionData data = sampleReflect(prevScreenPos.xy * textureSize(colortex0, 0));

    // ---- GGX 方向相容性权重 ----
    float roughness = denoiseBuffer.data[idx].roughness;
    float alpha = roughness * roughness; // GGX α
    float cosTheta = abs(dot(normalize(data.normal), normalize(data2.normal))); // H_prev · H_curr
    float tanThetaSq = max((0.99995 - cosTheta * cosTheta) / (1e-5 + cosTheta * cosTheta), 0.0);
    float ggxWeight = exp(-0.0001 * tanThetaSq / (alpha * alpha)); // GGX 分布形状

    // 可选位置权重（几何法线平面距离）
    UnifiedDiffuseElement _de = diffuseIllumiantionBuffer.data[idx];
    vec3 geoNormal = vec3(_de.n2x, _de.n2y, _de.n2z);
    float posWeight = exp(-POSITION_PARAM * abs(dot(data.pos - data2.pos, geoNormal)));

    // 综合置信度
    float s = float(denoiseBuffer.data[idx].distance > -0.5)
            * ggxWeight * posWeight;

    float prevW = data.prev_weight;
    prevW = min(prevW * s + 1.0, 3 * ACCUMULATION_LENGTH);

    data2.data_swap = data.data + (data2.data_swap - data.data) / prevW;
    data2.weight = prevW;
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    idx = getIdx(uvec2(gl_FragCoord.xy));

    info_distance = denoiseBuffer.data[idx].distance;
    data2 = reflectIllumiantionBuffer.data[idx];

    // ---- 天空 / 无效几何: 重置权重后直接写出 -----------------------------
    if (info_distance < -0.5) {
        data2.weight = 1.0;
        data2.mixWeight = 0.0;
        WriteReflect(data2, ivec2(gl_FragCoord.xy));
        return;
    }

    // ---- 重投影到上一帧 --------------------------------------------------
    prevScreenPos = reproject2(data2.pos);
    idx_l = getIdx(uvec2(prevScreenPos.xy * textureSize(colortex0, 0) + 0.5));

    // ---- 执行时域混合 ----------------------------------------------------
    MixReflect();

    WriteReflect(data2, ivec2(gl_FragCoord.xy));
}
