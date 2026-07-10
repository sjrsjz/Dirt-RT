#version 430 compatibility
#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ===========================================================================
// Pass 301 CS: 镜面 NRD 风格降噪 (计算着色器变体, 前 3 级 à-trous R0=1,2,4)
// ===========================================================================
// 基于 NRD RELAX À-trous 的正确镜面降噪实现:
//   1. 视线方向权重 (view-dependent specular)
//   2. 镜面波瓣法线权重 (考虑法线和视线)
//   3. 粗糙度权重 (正确参数化)
//   4. 表面几何权重 (平面距离)
//   5. 击中距离权重 (虚拟投射距离)
//   6. 亮度权重 (方差归一化)
// ===========================================================================

layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // f16(R,G)|f16(B,roughness)|f16(variance,virtualProjDist)|oct(H)

layout(rgba32f) uniform image2D colorimg4;

#define HALO R0
#define TILE_SIZE (16 + 2 * HALO)
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

shared vec4 sm_geometry[TILE_AREA];
shared vec4 sm_light[TILE_AREA];

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

// 从共享内存解包镜面样本
void unpackSpecularSampleSM(uint tile_idx, out vec3 pos, out vec3 R, out vec3 radiance,
    out float roughness, out float variance, out float virtualProjDist, out vec3 H) {
    PackedLightSample s;
    s.data0 = sm_geometry[tile_idx];
    s.data1 = sm_light[tile_idx];
    unpackSpecularSample(s, pos, R, radiance, roughness, variance, virtualProjDist, H);
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uvec2 lid = gl_LocalInvocationID.xy;
    uint local_idx = gl_LocalInvocationIndex;
    uvec2 group_id = gl_WorkGroupID.xy;
    ivec2 texSize = textureSize(colortex3, 0);

    ivec2 tile_origin = ivec2(group_id * 16u) - ivec2(HALO);

    // ---- Phase 1: 协作加载 tile 到共享内存 ----
    for (uint i = local_idx; i < uint(TILE_AREA); i += 256u) {
        uint tx = i % uint(TILE_SIZE);
        uint ty = i / uint(TILE_SIZE);
        ivec2 gc = tile_origin + ivec2(tx, ty);
        ivec2 cc = clamp(gc, ivec2(0), texSize - 1);
        if (gc == cc) {
            sm_geometry[i] = texelFetch(colortex3, cc, 0);
            sm_light[i] = texelFetch(colortex4, cc, 0);
        } else {
            // 越界: 天空 mask (variance<0)
            sm_geometry[i] = vec4(0.0);
            sm_light[i] = vec4(0.0, 0.0, pack2HalfClamped(-1.0, 0.0), 0.0);
        }
    }

    barrier();
    memoryBarrierShared();

    if (pix.x >= texSize.x || pix.y >= texSize.y) return;

    uint cx = lid.x + uint(HALO);
    uint cy = lid.y + uint(HALO);
    uint center_idx = cy * uint(TILE_SIZE) + cx;

    vec3 cPos, cR, cRad, cH;
    float cRough, cVar, cVproj;
    unpackSpecularSampleSM(center_idx, cPos, cR, cRad, cRough, cVar, cVproj, cH);

    // 天空: 早退, 保留 swap4 写入的 mask
    if (cVar < 0.0) return;

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

    // 累积器 (中心像素权重 = 0.44198^2, NRD 3x3 高斯核心)
    const float centerWeight = 0.44198 * 0.44198;
    float sumW = centerWeight;
    vec3 sumRadiance = cRad * centerWeight;
    float sumVariance = cVar * centerWeight * centerWeight;

    // ---- Phase 2: 3x3 À-trous 滤波 ----
    const float kernelWeightGaussian3x3[2] = float[2](0.44198, 0.27901);

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            if (xx == 0 && yy == 0) continue; // 跳过中心

            int sx = int(cx) + xx * R0;
            int sy = int(cy) + yy * R0;
            if (sx < 0 || sy < 0 || sx >= int(TILE_SIZE) || sy >= int(TILE_SIZE)) continue;
            uint sample_idx = uint(sy) * uint(TILE_SIZE) + uint(sx);

            vec3 sPos, sR, sRad, sH;
            float sRough, sVar, sVproj;
            unpackSpecularSampleSM(sample_idx, sPos, sR, sRad, sRough, sVar, sVproj, sH);

            if (sVar < 0.0) continue; // 天空

            // 高斯核
            float kernel = kernelWeightGaussian3x3[abs(xx)] * kernelWeightGaussian3x3[abs(yy)];

            // 1. 几何权重 (平面距离)
            float geometryW = GetPlaneDistanceWeight_Atrous(cPos, cH, sPos, depthThreshold);
            geometryW *= kernel;

            if (geometryW < 1e-4) continue;

            // 2. 样本视线向量 (NRD 关键: 添加放松以减少视线相关性拒绝)
            const float roughnessEdgeStoppingRelaxation = 0.3; // gRoughnessEdgeStoppingRelaxation
            vec3 sampleV = -normalize(sPos + roughnessEdgeStoppingRelaxation * cPos);

            // 3. 镜面法线权重 (完整版: 同时考虑法线和视线差异)
            float normalWSpecular = GetSpecularNormalWeight_ATrous(
                specularNormalWeightParams, cH, sH, centerV, sampleV);

            // 4. 简化版法线权重 (仅角度，作为后备)
            float angles = acos(clamp(dot(cH, sH), -1.0, 1.0));
            float normalWSpecularSimplified = ComputeWeight(angles, specularNormalWeightParamSimplified, 0.0);

            // 5. 粗糙度权重
            float roughnessWSpecular = ComputeWeight(sRough, roughnessWeightParams.x, roughnessWeightParams.y);

            // 6. 组合镜面权重 (根据粗糙度选择完整或简化版本)
            // 对于光滑表面 (低粗糙度)，使用完整的视线相关权重
            // 对于粗糙表面，使用简化版本
            bool useFullSpecularWeight = true; // gRoughnessEdgeStoppingEnabled
            float wSpecular = geometryW * (useFullSpecularWeight ?
                (normalWSpecular * roughnessWSpecular) : normalWSpecularSimplified);

            if (wSpecular < 1e-4) continue;

            // 7. 亮度权重
            float sampleLuminance = luma(sRad);
            float specularLuminanceW = abs(centerLuminance - sampleLuminance) * specularPhiLIlluminationInv;
            specularLuminanceW = min(2.0, specularLuminanceW); // gSpecMaxLuminanceRelativeDifference
            wSpecular *= exp(-specularLuminanceW);

            // 8. 累积
            sumW += wSpecular;
            sumRadiance += sRad * wSpecular;
            sumVariance += sVar * wSpecular * wSpecular;
        }
    }

    // ---- Phase 3: 归一化输出 ----
    vec3 filteredRadiance = sumRadiance / max(sumW, 1e-8);
    float filteredVariance = sumVariance / max(sumW * sumW, 1e-8);

    if (any(isnan(filteredRadiance))) filteredRadiance = vec3(0.0);

    imageStore(colorimg4, pix, packSpecularSample(cPos, cR, filteredRadiance, cRough, filteredVariance, cVproj, cH).data1);
}
