#version 430 compatibility

// ===========================================================================
// Pass 101: 镜面反射时域累积 (Reflect Temporal Accumulation) — NRD RELAX 双路径
// ===========================================================================
// 管线位置: 在光线追踪生成镜面反射样本后，与上一帧历史混合。
//
// 双路径 (NRD RELAX 镜面时域累积，curvature=0):
//   SMB (表面运动): reproject(反射表面点 pos)。
//       posConf (反射点位移) + ggxConf (法线相容) → 运动抗拖影。
//   VMB (虚拟运动): reproject(pos + viewDir·vproj) — NRD RELAX virtual motion。
//       viewDir = normalize(pos) = eye→surface 方向, vproj = 反射投射距离。
//       静止相机 → viewDir·vproj 与 pos 共视线 → VMB=SMB (无漂移)。
//       运镜 → 视线延长点 tracking 反射内容 (平面反遮挡 + posConf 验证)。
//   vha = Dfactor(NoV,rough) × VMB置信 × motionFactor × SMB回退。
//       Dfactor: 光滑镜面→高→VMB; 粗糙→低→SMB。
//       motionFactor = smoothstep(表面视差): 静止→0→纯SMB; 运镜→1→允许VMB。
//
// EMA: prevW = min(prevW·s + 1, ACCUMULATION_LENGTH), s = 最优 tap 置信度。
// 数据契约不变: REFLECT_BUFFER_MIN 仅写 swap_color, 控制字段由 swap5 写。
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

// 反射点位移灵敏度 (SMB 抗拖影 + VMB 内容匹配): 位移越大置信度越低
const float POSITION_PARAM = 32.0;

// NRD 反遮挡阈值 (frustumSize 比例): 平面距离超过此值视为不同表面
const float DISOCCLUSION_THRESHOLD = 0.02;

// ---------------------------------------------------------------------------
// NRD 辅助
// ---------------------------------------------------------------------------

// NRD _NRD_GetSpecularDominantFactor (G2): 镜面主导占比, smooth/掠射→高→偏 VMB
float GetSpecularDominantFactor(float NoV, float roughness) {
    float a = 0.298475 * log(39.4115 - 39.0029 * roughness);
    float f = pow(clamp(1.0 - NoV, 0.0, 1.0), 10.8649) * (1.0 - a) + a;
    return clamp(f, 0.0, 1.0);
}

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

float info_distance; // 当前像素的主射线距离
uint idx; // 当前像素 SSBO 索引

vec3 surfaceN; // 反射表面法线 (G-buffer macroNormal)
float disocclusionThreshold; // NRD 平面距离阈值 (世界单位)

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}

vec3IlluminationData data2; // 当前像素的反射光照数据 (来自 SSBO)

// ---------------------------------------------------------------------------
// 单路径评估: 在 prevUV 处 2×2 双线性采样历史
//   useGgx = true (SMB): tapConf = posConf × ggxConf (同表面点, R 应一致 → ggx 抗拖影)
//   useGgx = false (VMB): tapConf = posConf (不同镜面点 R 必然不同, 仅用内容匹配)
// posConf = 反射命中点位移: SMB 运动→0 (抗拖影); VMB 追踪→~1 (内容匹配)
// 输出 found / result / Wnew (EMA 权重) / maxTapConf (最优 tap 置信度)
// ---------------------------------------------------------------------------
void evalPath(vec2 prevUV, bool useGgx,
    vec3 pos, vec3 normal, vec3 N, float roughness,
    vec3 curVirtual, float vproj, vec3 curColor,
    out bool found, out vec3 result, out float Wnew, out float maxTapConf) {
    found = false;
    result = curColor;
    Wnew = 1.0;
    maxTapConf = 0.0;
    if (notInRange(prevUV)) return;

    vec2 prevTexelcoord = prevUV * vec2(resolution_global);
    ivec2 prevTexel = ivec2(floor(prevTexelcoord));

    float alpha = roughness * roughness + 1e-6; // GGX α²
    vec3 R = normalize(normal); // 当前反射方向 (ggx 用)

    vec3 accumColor = vec3(0.0);
    float sumWeight = 0.0;
    float accumPrevWeight = 0.0;

    for (int i = 0; i < 4; i++) {
        ivec2 sampleTexel = prevTexel + ivec2(i & 1, i >> 1);
        vec3 samplePos, sampleNormal;

        vec3IlluminationData tap = fetchReflect(sampleTexel);
        if (!fetchReflectHistoryGeometry(sampleTexel, samplePos, sampleNormal)) continue;
        if (tap.prev_weight < 1e-4) continue;
        if (tap.weight < 0.0) continue; // sky

        vec3 samplePosCur = samplePos - cameraDelta; // 历史表面点 → 当前相机系

        // NRD 硬反遮挡: 当前表面点与历史表面点需共面 (平面距离)
        float planeDist = abs(dot(pos - samplePosCur, surfaceN));
        if (planeDist >= disocclusionThreshold) continue;

        vec2 sampleCoord = vec2(sampleTexel);
        float bw = (1.0 - abs(prevTexelcoord.x - sampleCoord.x))
                * (1.0 - abs(prevTexelcoord.y - sampleCoord.y));

        // 反射命中点位移置信度 (SMB 抗拖影 / VMB 内容匹配)
        vec3 sampleVirtual = samplePosCur + sampleNormal; // 历史反射命中点 Q_P
        float posConf = exp2(-POSITION_PARAM * LOG2_E * length(sampleVirtual - curVirtual) / vproj);

        float tapConf = posConf;
        if (useGgx) {
            // GGX 方向相容性 (仅 SMB: 同表面点 R 应一致)
            float cosTheta = abs(dot(normalize(sampleNormal), R));
            float tanThetaSq = max((0.99999 - cosTheta * cosTheta) / (1e-9 + cosTheta * cosTheta), 0.0);
            float ggxConf = exp2(-0.0001 * tanThetaSq / (alpha * alpha));
            tapConf *= ggxConf;
        }

        maxTapConf = max(maxTapConf, tapConf);
        float w = bw * tapConf + 1e-10;

        vec3 tc = tap.data;
        if (any(isnan(tc)) || any(isinf(tc))) continue;
        accumColor += tc * w;
        sumWeight += w;
        accumPrevWeight += w * tap.prev_weight;
    }

    if (sumWeight < 1e-8) return; // 无有效 tap

    found = true;
    vec3 blendColor = accumColor / sumWeight;
    float prevW = accumPrevWeight / max(sumWeight, 1e-6);
    float s = float(denoiseBuffer.data[idx].distance > -0.5) * maxTapConf;
    prevW = min(prevW * s + 1.0, ACCUMULATION_LENGTH);
    result = blendColor + (curColor - blendColor) / prevW;
    if (any(isnan(result))) result = curColor;
    Wnew = prevW;
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

    cameraDelta = camPos - prevRaytracingCamPos;

    // ---- 当前像素反射几何 / 材质 -----------------------------------------
    vec3 pos = data2.pos;
    vec3 normal = data2.normal; // R · vproj
    float vproj = max(length(normal), 0.001);
    vec3 curColor = data2.data_swap; // 当前帧 raw RT (含噪)
    vec3 N = denoiseBuffer.data[idx].macroNormal;
    float roughness = denoiseBuffer.data[idx].roughness;
    vec3 curVirtual = pos + normal; // 当前反射命中点 Q_C
    float posLen = max(length(pos), 0.001);
    vec3 viewDir = pos / posLen; // eye → surface (相机在原点)

    surfaceN = N;
    // frustumSize ≈ dist · 2·tan(fovY/2), tan(fovY/2) = 1/rtProjection[1][1] (jitter 不影响 y 缩放)
    float frustumScale = 2.0 / max(abs(rtProjection[1][1]), 1e-4);
    disocclusionThreshold = DISOCCLUSION_THRESHOLD * posLen * frustumScale;

    // ---- SMB: 重投影反射表面点 -------------------------------------------
    bool smbFound;
    vec3 smbResult;
    float smbWnew, smbConf;
    evalPath(reproject(pos).xy, true, pos, normal, N, roughness,
        curVirtual, vproj, curColor, smbFound, smbResult, smbWnew, smbConf);

    // ---- VMB: 重投影视线方向延长点 (NRD RELAX virtual motion) --------------
    // NRD: prevVirtualWorldPos = prevWorldPos + normalize(currentViewVector)·hitDistFocused
    // 在相机相对系中 (curvature=0): VMB sample = reproject(pos + viewDir·vproj)
    // 静止相机 → viewDir·vproj 与 pos 共视线 → VMB=SMB (无漂移)。
    // 运镜 → 视线延长点与表面点分离 → VMB 追踪反射内容。
    bool vmbFound = false;
    vec3 vmbResult = curColor;
    float vmbWnew = 1.0, vmbConf = 0.0;
    bool reflectHitSky = (vproj > 0.5 * VPROJDIST_SKY);
    if (!reflectHitSky) {
        vec3 vmbUV3 = reproject(pos + viewDir * vproj);
        if (!any(isnan(vmbUV3)) && !any(isinf(vmbUV3))) {
            evalPath(vmbUV3.xy, false, pos, normal, N, roughness,
                curVirtual, vproj, curColor, vmbFound, vmbResult, vmbWnew, vmbConf);
        }
    }

    // ---- virtualHistoryAmount (SMB ↔ VMB 混合) ---------------------------
    vec3 V = -viewDir; // surface → eye (VMB 用了 viewDir=eye→surface)
    float NoV = abs(dot(N, V));
    float Dfactor = GetSpecularDominantFactor(NoV, roughness);

    float vha = vmbFound ? (Dfactor * vmbConf) : 0.0;
    // SMB 回退: SMB 更可信时偏向 SMB
    vha *= (smbConf > 1e-6) ? clamp(vmbConf / smbConf, 0.0, 1.0) : 1.0;
    // 运镜门控: 静止相机 VMB→0 (VMB=SMB 同视线, vha 无关; 但保持干净)
    float parallaxPx = length((reproject(pos).xy - texCoord) * vec2(resolution_global));
    float motionFactor = smoothstep(0.5, 1.5, parallaxPx);
    vha *= motionFactor;
    vha = clamp(vha, 0.0, 1.0);

    vec3 result = mix(smbResult, vmbResult, vha);
    float Wnew = mix(smbWnew, vmbWnew, vha);
    if (any(isnan(result))) result = curColor;

    // ---- 写出 (REFLECT_BUFFER_MIN: 仅 swap_color) ------------------------
    data2.data_swap = result;
    data2.weight = Wnew;
    // 控制字段通过 swap_color.w 传递; flip / prev_weight / lpos / lnormal 由 swap5 写出.
    data2.mixWeight = denoiseBuffer.data[idx].reflectWeight;
    WriteReflect(data2, ivec2(gl_FragCoord.xy));
}
