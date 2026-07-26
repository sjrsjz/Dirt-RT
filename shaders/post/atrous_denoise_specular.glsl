#version 430 compatibility

// ===========================================================================
// Pass 301: 镜面反射/折射 NRD 风格屏幕空间降噪 (à-trous, fragment 变体 R0≥8)
// ===========================================================================
// 基于 NRD RELAX À-trous 的正确镜面降噪实现:
//   1. 视线方向权重 (view-dependent specular)
//   2. 镜面波瓣法线权重 (考虑法线和视线)
//   3. 粗糙度权重 (正确参数化)
//   4. 表面几何权重 (平面距离)
//   5. 击中距离权重 (虚拟投射距离)
//   6. 亮度权重 (方差归一化)
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // f16(R,G)|f16(B,roughness)|f16(variance,virtualProjDist)|oct(H)

// Poisson 圆盘采样表 (NRD) — 预计算采样偏移 + 高斯核权重
// .xy = 归一化采样偏移, .z = length(.xy), .w = 高斯核权重 exp(-z^2/2)
const vec4 POISSON_8[8] = {
    vec4( -0.4706069, -0.4427112, +0.6461146, +0.81170 ),
    vec4( -0.9057375, +0.3003471, +0.9542373, +0.63422 ),
    vec4( -0.3487388, +0.4037880, +0.5335386, +0.86734 ),
    vec4( +0.1023042, +0.6439373, +0.6520134, +0.80847 ),
    vec4( +0.5699277, +0.3513750, +0.6695386, +0.79925 ),
    vec4( +0.2939128, -0.1131226, +0.3149309, +0.95161 ),
    vec4( +0.7836658, -0.4208784, +0.8895339, +0.67328 ),
    vec4( +0.1564120, -0.8198990, +0.8346850, +0.70589 )
};

// NRD-style 粗糙度权重参数 (返回 (a, -b) 用于 ComputeWeight)
vec2 GetRoughnessWeightParams(float roughness, float fraction) {
    const float sensitivity = 0.03; // NRD_ROUGHNESS_SENSITIVITY
    float a = 1.0 / mix(sensitivity, 1.0, clamp(roughness * fraction, 0.0, 1.0));
    float b = roughness * a;
    return vec2(a, -b);
}

// NRD ComputeWeight (指数版本)
float ComputeWeight(float x, float px, float py) {
    const float scale = 3.0; // NRD_EXP_WEIGHT_DEFAULT_SCALE
    float arg = -scale * abs(x * px + py);
    // ExpApprox for negative x: 1 / (x*x - x + 1)
    return 1.0 / (arg * arg - arg + 1.0);
}

// NRD 镜面波瓣半角的 tan 值
float GetSpecLobeTanHalfAngle(float roughness, float percentOfVolume) {
    roughness = clamp(roughness, 0.0, 1.0);
    percentOfVolume = clamp(percentOfVolume, 0.0, 1.0);
    return roughness * roughness * percentOfVolume / (1.0 - percentOfVolume + 1e-6);
}

// NRD GetSpecMagicCurve: 镜面→0, 粗糙→1
float GetSpecMagicCurve(float roughness) {
    return 1.0 - exp2(-30.0 * roughness * roughness);
}

// NRD ComputeExponentialWeight: exp2(-scale * |x·a + b|)
float ComputeExponentialWeight(float x, float a, float b) {
    return exp2(-3.0 * abs(x * a + b));
}

// NRD 击中距离权重参数 (返回 (a, -b) 用于 ComputeExponentialWeight):
// 未收敛 (低 accumSpeed) → 极严格 (保护信号形状);
// 已收敛 (高 accumSpeed, 低粗糙度) → 严格 (防止亮物体泄漏到反射中)
vec2 GetHitDistanceWeightParams(float hitDist, float nonLinearAccumSpeed, float roughness) {
    float smc = GetSpecMagicCurve(roughness);
    float norm = mix(0.0005, 1.0, min(nonLinearAccumSpeed, smc));
    float a = 1.0 / max(norm, 1e-8);
    float b = hitDist * a;
    return vec2(a, -b);
}

// NRD 镜面法线权重参数 (À-trous 版本)
vec2 GetNormalWeightParams_ATrous(float roughness, float lobeAngleFraction, float lobeAngleSlack) {
    // 主参数: 锥角
    float angle = atan(GetSpecLobeTanHalfAngle(roughness, lobeAngleFraction));
    angle += lobeAngleSlack;
    angle = min(angle, 1.5708); // min(angle, PI/2)

    float f = 0.9; // 简化版本，没有历史长度和置信度放松
    return vec2(angle, f);
}

// NRD 镜面法线权重 (À-trous 版本) - 关键: 同时考虑法线和视线
float GetSpecularNormalWeight_ATrous(vec2 params, vec3 n0, vec3 n, vec3 v0, vec3 v) {
    float cosaN = dot(n0, n);
    float cosaV = dot(v0, v);
    float cosa = min(cosaN, cosaV); // 取最严格的
    float a = acos(clamp(cosa, -1.0, 1.0)); // AcosApprox
    a = smoothstep(0.0, params.x, a);
    return clamp(1.0 - a * params.y, 0.0, 1.0);
}

// 简化版法线权重参数 (仅用于角度)
float GetNormalWeightParam2(float angleFraction) {
    float angle = atan(GetSpecLobeTanHalfAngle(1.0, angleFraction));
    angle = 1.0 / max(angle, 0.001);
    return angle;
}

// 平面距离权重 (À-trous 版本)
float GetPlaneDistanceWeight_Atrous(vec3 centerWorldPos, vec3 centerNormal, vec3 sampleWorldPos, float threshold) {
    float distanceToCenterPointPlane = abs(dot(sampleWorldPos - centerWorldPos, centerNormal));
    return distanceToCenterPointPlane < threshold ? 1.0 : 0.0;
}


/* RENDERTARGETS: 4 */
layout(location = 0) out vec4 color;

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    ivec2 texSize = textureSize(colortex3, 0);

    vec4 centerGeom = texelFetch(colortex3, pix, 0);
    vec4 centerLight = texelFetch(colortex4, pix, 0);

    vec3 cPos, cR, cRad, cH;
    float cRough, cVar, cVproj;
    unpackSpecularSample(PackedLightSample(centerGeom, centerLight),
        cPos, cR, cRad, cRough, cVar, cVproj, cH);

    // 天空 (variance<0): 透传
    if (cVar < 0.0) {
        color = centerLight;
        return;
    }

    // ---- NRD-style 权重参数计算 ----

    // 1. 中心视线向量 (关键!)
    vec3 centerV = -normalize(cPos);

    // 2. 粗糙度权重参数
    float roughnessFraction = 0.15; // gRoughnessFraction, 可调
    vec2 roughnessWeightParams = GetRoughnessWeightParams(cRough, roughnessFraction);

    // 3. 镜面法线权重参数 (完整版 - 考虑视线)
    float specularLobeAngleFraction = 0.5; // gLobeAngleFraction
    float specularLobeAngleSlack = 0.3; // gSpecLobeAngleSlack
    vec2 specularNormalWeightParams = GetNormalWeightParams_ATrous(
        cRough, specularLobeAngleFraction, specularLobeAngleSlack);

    // 4. 简化版法线权重参数 (仅角度，用于粗糙表面)
    float diffuseLobeAngleFraction = 0.5; // 简化版使用
    float specularNormalWeightParamSimplified = GetNormalWeightParam2(diffuseLobeAngleFraction);

    // 5. 几何平面距离阈值
    float depthThreshold = 0.1 * max(length(cPos), 0.01); // gDepthThreshold

    // 6. 亮度权重 (方差归一化)
    float centerLuminance = luma(cRad);
    float specularPhiLIlluminationInv = 1.0 / max(1e-4, 4.0 * sqrt(cVar)); // gSpecPhiLuminance

    // 7. 击中距离权重 (NRD): 防止不同深度的反射内容相互泄漏
    //    未收敛 → 极严格 (保护信号形状); 已收敛镜面 → 严格 (防止亮物体泄漏)
    float nonLinearAccumSpeed = 0.2; // 中等收敛估计
    vec2 hitDistanceWeightParams = GetHitDistanceWeightParams(cVproj, nonLinearAccumSpeed, cRough);
    float smc = GetSpecMagicCurve(cRough);
    float minHitDistWeight = 0.1 * smc; // NRD: gMinHitDistanceWeight * smc

    // 累积器 (中心像素权重 = 0.44198^2, NRD 3x3 高斯核心)
    const float centerWeight = 0.44198 * 0.44198;
    float sumW = centerWeight;
    vec3 sumRadiance = cRad * centerWeight;
    float sumVariance = cVar * centerWeight * centerWeight;

    // ---- Poisson 圆盘采样 (NRD, STEP>=4) -----------------------------------
    float theta = 2.0 * PI * fract(rand(vec2(pix)) + R0 * 0.6180339887498949);
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0 * 1.75;

    for (int k = 0; k < 8; k++) {
        vec4 ps = POISSON_8[k];
        ivec2 samplePos = pix + ivec2(round(rotM * ps.xy));

        if (samplePos.x < 0 || samplePos.y < 0 ||
                samplePos.x >= texSize.x || samplePos.y >= texSize.y) {
            continue;
        }

        vec4 sG = texelFetch(colortex3, samplePos, 0);
        vec4 sL = texelFetch(colortex4, samplePos, 0);
        vec3 sPos, sR, sRad, sH;
        float sRough, sVar, sVproj;
        unpackSpecularSample(PackedLightSample(sG, sL),
            sPos, sR, sRad, sRough, sVar, sVproj, sH);

        if (sVar < 0.0) continue; // 天空

        float w_kernel = ps.w; // 预计算: exp(-z^2/2)

            // 1. 几何权重 (平面距离)
            float geometryW = GetPlaneDistanceWeight_Atrous(cPos, cH, sPos, depthThreshold);
            geometryW *= w_kernel;

            if (geometryW < 1e-4) continue;

            // 2. 样本视线向量 (NRD 关键: 添加放松以减少视线相关性拒绝)
            const float roughnessEdgeStoppingRelaxation = 0.3; // gRoughnessEdgeStoppingRelaxation
            vec3 sampleV = -normalize(sPos + roughnessEdgeStoppingRelaxation * cPos);

            // 3. 镜面法线权重 (完整版: 同时考虑法线和视线差异)
            float normalWSpecular = GetSpecularNormalWeight_ATrous(
                specularNormalWeightParams, cH, sH, centerV, sampleV);

            // 4. 简化版法线权重 (仅角度，作为后备)
            float angles = acos(clamp(dot(cH, sH), -1.0, 1.0));

            // 5. 粗糙度权重
            float roughnessWSpecular = ComputeWeight(sRough, roughnessWeightParams.x, roughnessWeightParams.y);

            // 6. 组合镜面权重
            float wSpecular = geometryW * normalWSpecular * roughnessWSpecular;

            if (wSpecular < 1e-4) continue;

            // 7. 亮度权重
            float sampleLuminance = luma(sRad);
            float specularLuminanceW = abs(centerLuminance - sampleLuminance) * specularPhiLIlluminationInv;
            specularLuminanceW = min(2.0, specularLuminanceW); // gSpecMaxLuminanceRelativeDifference
            wSpecular *= exp(-specularLuminanceW);

            // 8. 击中距离权重 (NRD): 不同反射深度的样本不混合
            float hitDistW = mix(minHitDistWeight, 1.0,
                ComputeExponentialWeight(sVproj, hitDistanceWeightParams.x, hitDistanceWeightParams.y));
            wSpecular *= hitDistW;

            // 9. 累积
            sumW += wSpecular;
            sumRadiance += sRad * wSpecular;
            sumVariance += sVar * wSpecular * wSpecular;
    }

    // ---- 归一化输出 ----
    vec3 filteredRadiance = sumRadiance / max(sumW, 1e-8);
    float filteredVariance = sumVariance / max(sumW * sumW, 1e-8);

    if (any(isnan(filteredRadiance))) filteredRadiance = vec3(0.0);

    // 输出: 仅颜色+方差变化, roughness/virtualProjDist/H 透传
    color = packSpecularSample(cPos, cR, filteredRadiance, cRough, filteredVariance, cVproj, cH).data1;
}
