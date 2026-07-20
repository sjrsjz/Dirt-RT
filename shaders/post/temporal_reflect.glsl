#version 430 compatibility

// ===========================================================================
// Pass 101 CS: 镜面反射时域累积 (Reflect Temporal Accumulation) — NRD RELAX 双路径
// ===========================================================================
// 管线位置: 在光线追踪生成镜面反射样本后，与上一帧历史混合。
//
// 双路径 (NRD RELAX 镜面时域累积，curvature_correct_specular 预处理):
//   SMB (表面运动): reproject(反射表面点 pos)。
//       posConf (反射点位移) + ggxConf (法线相容) → 运动抗拖影。
//   VMB (虚拟运动): reproject(pos + viewDir·vproj) — NRD RELAX virtual motion。
//       viewDir = normalize(pos) = eye→surface 方向, vproj = 曲率修正后反射投射距离。
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
// 避免 SSBO 读写的潜在冲突 (转为纯计算管线后 UV 边界由 dispatch 保证).

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------

uniform vec2 resolution;

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

// ---------------------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------------------

float info_distance; // 当前像素的主射线距离

vec3 surfaceN; // 反射表面法线 (G-buffer macroNormal)
float disocclusionThreshold; // NRD 平面距离阈值 (世界单位)

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
}
// ---------------------------------------------------------------------------
// 优化后的单路径评估 (基于波瓣相似度与健壮的 EMA 更新)
// ---------------------------------------------------------------------------
void evalPath(vec2 prevUV, bool useGgx,
    vec3 pos, vec3 R_cur, vec3 N, float roughness,
    vec3 curVirtual, float vproj, vec3 curColor,
    out bool found, out vec3 result, out float Wnew, out float maxTapConf) {
    found = false;
    result = curColor;
    Wnew = 1.0;
    maxTapConf = 0.0;
    if (notInRange(prevUV)) return;

    vec2 prevTexelcoord = prevUV * vec2(resolution_global);
    ivec2 prevTexel = ivec2(floor(prevTexelcoord));

    // alpha 是粗糙度的平方，代表 GGX NDF 的方差 (Variance)
    float alpha = roughness * roughness;

    // 视觉光路总长度 (用于稳健地归一化位置误差)
    float opticalDepth = max(length(pos) + vproj, 0.2);

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

        vec3 samplePosCur = samplePos - cameraDelta;

        // 1. NRD 硬反遮挡: 当前表面点与历史表面点需共面
        float planeDist = abs(dot(pos - samplePosCur, surfaceN));
        if (planeDist >= disocclusionThreshold) continue;

        // 双线性权重
        vec2 sampleCoord = vec2(sampleTexel);
        float bw = (1.0 - abs(prevTexelcoord.x - sampleCoord.x))
                * (1.0 - abs(prevTexelcoord.y - sampleCoord.y));

        // 2. 虚拟位置置信度 (基于统计假设检验 p-value 严格推导)
        vec3 sampleVirtual = samplePosCur + sampleNormal;
        float distDiff = length(sampleVirtual - curVirtual);

        // alpha2 = alpha^2 = roughness^4 (表示反射分布的统计方差尺度)
        float alpha2 = alpha * alpha;

        // 引入 2.0 因子 (来自卡方分布自由度为2的推导)
        // 设定 1e-5 的下限，防止极光滑镜面时分母除零崩溃
        float varianceScale = 2.0 * opticalDepth * opticalDepth * max(alpha2, 1e-4);

        // 标准的 p-value 概率公式 (2D 高斯差分平方模长在卡方分布下的显著性检验)
        float posConf = exp(-(distDiff * distDiff) / varianceScale);

        float tapConf = posConf;

        // 3. GGX 波瓣相似度 (VNDF Lobe Similarity)
        if (useGgx) {
            vec3 R_prev = normalize(sampleNormal);
            float R_dot = clamp(dot(R_cur, R_prev), 0.0, 1.0);

            // 将 alpha 限制在下限，防止高光镜面时的除零/梯度爆炸
            // 粗糙度越大 (lobe越宽)，容忍度越高；越光滑，要求 R 越一致。
            float lobeSigma = max(alpha, 0.01);

            // 这是一个近似两个 vMF 分布 (球形高斯) 的对数相似度度量
            float ggxConf = exp(-REFLECT_GGX_CONFIDENCE * (1.0 - R_dot) / lobeSigma);
            tapConf *= ggxConf;
        }

        maxTapConf = max(maxTapConf, tapConf);
        float w = bw * tapConf + 1e-6; // 防止权重为0

        vec3 tc = tap.data;
        if (any(isnan(tc)) || any(isinf(tc))) continue;

        accumColor += tc * w;
        sumWeight += w;
        accumPrevWeight += w * tap.prev_weight;
    }

    if (sumWeight < 1e-8) return;

    found = true;
    vec3 blendColor = accumColor / sumWeight;

    // -------------------------------------------------------------
    // 4. 重构的 EMA 历史权重更新逻辑
    // -------------------------------------------------------------
    float historyW = accumPrevWeight / sumWeight;

    // 天空判定
    float validHit = float(info_distance > -0.5);
    float confidence = validHit * maxTapConf;

    // 正确的做法：用置信度动态限制“当前帧允许的最大历史长度”
    // 如果 confidence 为 0.9，允许的最大历史就是 32 * 0.9 = 28.8 帧，
    // 而不是每一帧都对历史做乘法导致指数衰减！
    float allowedLength = ACCUMULATION_LENGTH * confidence;

    // 限制历史权重不超标，并 +1 加入当前帧
    float prevW = min(historyW, allowedLength) + 1.0;

    // 标准的 EMA 混合公式: 历史权重占比 = (prevW - 1) / prevW
    // 如果置信度极低，prevW 会变成 1.0，当前帧颜色的权重就会是 1.0
    float blendAlpha = 1.0 / max(prevW, 1.0);

    result = mix(blendColor, curColor, blendAlpha);

    if (any(isnan(result))) result = curColor;
    Wnew = prevW;
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    vec2 texCoord = (vec2(pix) + 0.5) / vec2(resolution);

    { vec3 _pos; readGeo0(GEO_N_GEO, pix, _pos, info_distance); }

    // 从 SSBO (SpecularRTElement) 重建当前帧反射数据: normal = R*virtualProjDist
    float vproj;
    vec3IlluminationData curr_sample; // 当前像素的反射光照数据 (来自 SSBO)

    unpackSpecularRT_Refl(pix, curr_sample.pos, curr_sample.normal, curr_sample.data_swap, vproj);
    curr_sample.data = vec3(0.0);
    curr_sample.weight = 0.0;
    curr_sample.prev_weight = 0.0;

    // ---- 天空 / 无效几何: 重置权重后直接写出 -----------------------------
    if (info_distance < -0.5) {
        curr_sample.weight = 1.0;
        WriteReflect(curr_sample, ivec2(pix));
        return;
    }

    cameraDelta = camPos - prevRaytracingCamPos;

    // ---- 当前像素反射几何 / 材质 -----------------------------------------
    vec3 pos = curr_sample.pos;
    vec3 normal = curr_sample.normal; // R
    vec3 curColor = curr_sample.data_swap; // 当前帧 raw RT (含噪)
    int _illumType; float roughness;
    vec3 N;
    float _pr; readGeo1(GEO_N_NORMALS, pix, N, roughness, _illumType, _pr);
    vec3 curVirtual = pos + normal * vproj; // 当前反射命中点 Q_C
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
    float virtualMotionRoughnessWeight = smoothstep(0.0, 0.3, roughness);
    vha *= (1.0 - virtualMotionRoughnessWeight);

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
    curr_sample.data_swap = result;
    curr_sample.weight = Wnew;
    WriteReflect(curr_sample, ivec2(pix));
}
