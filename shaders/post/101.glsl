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

// 法线权重指数 — 极高值确保反射仅在法线几乎完全一致时才混合
const float NORMAL_PARAM = 512.0;

// 位置/深度差异灵敏度
const float POSITION_PARAM = 32.0;

const float LUMINANCE_PARAM = 4.0;

// ---------------------------------------------------------------------------
// 边缘停止权重函数
// ---------------------------------------------------------------------------

// 改进的法线权重: 除了夹角余弦外，还考虑法线长度的倒数比来检测 bump/细节变化
float svgfNormalWeight(vec3 centerNormal, vec3 normal, float d) {
    // 法线幅度比: 如果两个法线的长度差异很大 (如平坦 vs 高细节) → 降低权重
    float f = (1e3 + length(centerNormal)) / (1e3 + length(normal));
    f = max(f, 1.0 / f) - 1.0;
    float A = exp(-250.0 * abs(f));   // 幅度差异惩罚
    // 方向 + 幅度联合: A 惩罚长度差异，pow 惩罚方向差异
    return min(1.0, A * clamp(pow(max(dot(normalize(centerNormal), normalize(normal)), 0.0),
                                   NORMAL_PARAM), 0.0, 1.0));
}

// 位置权重: 距离衰减随 distance 增大而减小 (远处像素宽容度更高)
float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)
               * (10.0 / (1.0 + 10.0 * distance))));
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

vec3 prevScreenPos;          // 重投影后的上一帧屏幕坐标
float info_distance;         // 当前像素的光线追踪距离
uint idx_l;                  // 重投影像素索引
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

vec3IllumiantionData data2;  // 当前像素的反射光照数据 (来自 SSBO)

// ===========================================================================
// 时域混合 (Reflect)
// ===========================================================================
// 与漫反射的 SH 混合不同，反射使用简单的 vec3 EMA:
//   new = old + (current - old) / prevW
// 其中 prevW 是受重投影置信度调制的历史累积权重
void MixReflect() {
    // ---- 重投影失败: 重置历史 --------------------------------------------
    if (notInRange(prevScreenPos.xy)) {
        data2.weight = 1.0;
        return;
    }

    vec3IllumiantionData data = sampleReflect(prevScreenPos.xy * textureSize(colortex0, 0));

    // 重投影置信度: 几何有效性 × 法线一致性
    float s = float(denoiseBuffer.data[idx].distance > -0.5)
            * svgfNormalWeight(data.normal, data2.normal, info_distance);
    // 注: 位置权重被注释掉了，因为反射表面的位置关系被法线一致性充分捕获

    // 对置信度进行 gamma 调整: pow(s, 0.05) ≈ 在 s 接近 1 时才充分信任
    // 这有效阻止了反射在边界处的"拖影"
    s = pow(s, 0.05);

    float prevW = data.prev_weight;
    // 历史权重增量更新: 每次成功混合 +1, 上限 = 10 * ACCUMULATION_LENGTH
    prevW = max(1.0, min(prevW * s + 1.0, 10.0 * ACCUMULATION_LENGTH));

    // EMA 混合
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
