#version 430 compatibility

// ===========================================================================
// Pass 101: 镜面反射时域累积 (Reflect Temporal Accumulation)
// ===========================================================================
// 管线位置: 在光线追踪生成镜面反射样本后，与上一帧历史混合
//
// 与 100.glsl (漫反射) 的区别:
//   - 反射信号是 vec3 颜色而非 AliceEncoding 结构 → 使用更简单的时域混合
//   - 法线权重更激进 (NORMAL_PARAM=512)，因为反射对法线方向极其敏感
//   - 使用 reproject 直接重投影 (与 100.glsl 相同的路径)
//
// 注意: 此 pass 曾是性能瓶颈 (与 200.glsl 合计占用 ~1/4~1/3 的帧时间)，
//       但 200.glsl 目前已禁用 (return; 截断)，因此仅剩本 pass 的开销。
// ===========================================================================

#define REFLECT_BUFFER_MIN
// 仅写 swap_color (color/lpos/lnormal 由 swap5 写),
// 避免 fragment 内 sampler 读 + imageStore 写同一纹理的 UB (原 101 已验证可行).

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

in vec2 texCoord;

uniform sampler2D colortex0;

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

vec3 cameraDelta; // camPos - prevRaytracingCamPos (光线追踪源)

// 重投影: 全光线追踪推导矩阵 (单源一致, 零 Iris 混合, 零大数相消)
vec3 reproject(vec3 pos_rel) {
    vec3 prevPlayerPos = pos_rel + cameraDelta;
    vec4 clipPos = rtPrevProjection * rtPrevModelView * vec4(prevPlayerPos, 1.0);
    vec3 ndc = clipPos.xyz / clipPos.w;
    return ndc * 0.5 + 0.5;
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

vec3IlluminationData data2; // 当前像素的反射光照数据 (来自 SSBO)

// ===========================================================================
// 时域混合 (Reflect)
// ===========================================================================
// 与漫反射的 AliceEncoding 混合不同，反射使用简单的 vec3 EMA:
//   new = old + (current - old) / prevW
// 其中 prevW 是受重投影置信度调制的历史累积权重
void MixReflect() {
    if (notInRange(prevScreenPos.xy)) {
        data2.weight = 1.0;
        return;
    }

    // 2×2 双线性历史采样 + 逐样本置信度 (参考 TAA 模式)
    vec2 prevTexelcoord = prevScreenPos.xy * vec2(resolution_global);
    ivec2 prevTexel = ivec2(floor(prevTexelcoord));

    float roughness = denoiseBuffer.data[idx].roughness;
    float alpha = roughness * roughness;
    vec3 curVirtual = data2.pos + data2.normal;
    float curVproj = max(length(data2.normal), 0.001);

    vec3 accumColor = vec3(0.0);
    float sumWeight = 0.0;
    float maxTapConf = 0.0;
    float accumPrevWeight = 0.0;

    for (int i = 0; i < 4; i++) {
        ivec2 sampleTexel = prevTexel + ivec2(i & 1, i >> 1);
        vec3 samplePos, sampleNormal;

        vec3IlluminationData tap = fetchReflect(sampleTexel);
        if (!fetchReflectHistoryGeometry(sampleTexel, samplePos, sampleNormal)) continue;
        if (tap.prev_weight < 1e-4) continue;
        if (tap.weight < 0.0) continue; // sky

        vec2 sampleCoord = vec2(sampleTexel);
        float bw = (1.0 - abs(prevTexelcoord.x - sampleCoord.x))
                 * (1.0 - abs(prevTexelcoord.y - sampleCoord.y));

        // GGX 方向相容性
        float cosTheta = abs(dot(normalize(sampleNormal), normalize(data2.normal)));
        float tanThetaSq = max((0.999999 - cosTheta * cosTheta) / (1e-9 + cosTheta * cosTheta), 0.0);
        float ggxConf = exp2(-0.00014426950 * tanThetaSq / (alpha * alpha));

        // 虚拟击中点位置权重
        vec3 samplePosCur = samplePos - cameraDelta;
        vec3 sampleVirtual = samplePosCur + sampleNormal;
        float posConf = exp2(-POSITION_PARAM * LOG2_E * length(sampleVirtual - curVirtual) / curVproj);

        float conf = ggxConf * posConf;
        maxTapConf = max(maxTapConf, conf);
        float w = bw * conf + 1e-10;

        vec3 tc = tap.data;
        if (any(isnan(tc)) || any(isinf(tc))) continue;
        accumColor += tc * w;
        sumWeight += w;
        accumPrevWeight += w * tap.prev_weight;
    }

    if (sumWeight < 1e-8) {
        data2.weight = 1.0;
        return;
    }

    vec3 blendColor = accumColor / sumWeight;
    float prevW = accumPrevWeight / max(sumWeight, 1e-6);
    float s = float(denoiseBuffer.data[idx].distance > -0.5) * maxTapConf;

    prevW = min(prevW * s + 1.0, ACCUMULATION_LENGTH);

    data2.data_swap = blendColor + (data2.data_swap - blendColor) / prevW;
    if (any(isnan(data2.data_swap))) data2.data_swap = vec3(0.0);
    data2.weight = prevW;
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    idx = getIndex(uvec2(gl_FragCoord.xy));

    info_distance = denoiseBuffer.data[idx].distance;

    // 从 SSBO (SpecularRTElement) 重建当前帧反射数据: normal = R*virtualProjDist
    unpackSpecularRT(reflectIlluminationBuffer.data[idx], data2.pos, data2.normal, data2.data_swap);
    data2.data = vec3(0.0);
    data2.weight = 0.0;
    data2.prev_weight = 0.0;
    data2.mixWeight = 0.0;

    // ---- 天空 / 无效几何: 重置权重后直接写出 -----------------------------
    if (info_distance < -0.5) {
        data2.weight = 1.0;
        data2.mixWeight = 0.0;
        WriteReflect(data2, ivec2(gl_FragCoord.xy));
        return;
    }

    // ---- 重投影到上一帧 (主命中点: 定位同一反射表面点) ------------------
    cameraDelta = camPos - prevRaytracingCamPos;
    prevScreenPos = reproject(data2.pos);
    idx_l = getIndex(uvec2(prevScreenPos.xy * textureSize(colortex0, 0) + 0.5));

    // ---- 执行时域混合 ----------------------------------------------------
    MixReflect();

    // 控制字段通过 swap_color.w 传递 (REFLECT_BUFFER_MIN 仅写 swap_color).
    // flip / prev_weight / lpos / lnormal 由 swap5 写出.
    data2.mixWeight = denoiseBuffer.data[idx].reflectWeight;
    WriteReflect(data2, ivec2(gl_FragCoord.xy));
}
