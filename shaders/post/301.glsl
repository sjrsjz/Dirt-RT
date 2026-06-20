#version 430 compatibility

// ===========================================================================
// Pass 301: 屏幕空间模糊滤波器 (Screen-Space Blur Filter)
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

// ---------------------------------------------------------------------------
// Uniform 输入
// ---------------------------------------------------------------------------
uniform sampler2D colortex3; // geometry: pos.xyz + encoded normal
uniform sampler2D colortex4; // light_sample: radiance.xyz + packed(weight, roughness)

// ---------------------------------------------------------------------------
// 可调参数
// ---------------------------------------------------------------------------
const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 1.0;

// ---------------------------------------------------------------------------
// 边缘停止 / 权重函数
// ---------------------------------------------------------------------------
float svgfNormalWeight(vec3 centerNormal, vec3 normal, float S) {
    return pow(max(dot(centerNormal, normal), 0.0), S);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

float computeAnisotropicAxisScale(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    vec3 x = an * B - bn * A;
    return abs(bn) * sqrt(max(1.0 - an * an, 0.0)) / max(0.01, dot(x, x));
}

float GetRoughnessWeight(float roughness0, float roughness) {
    float norm = roughness0 * roughness0 * 0.99 + 0.01;
    float w = abs(roughness0 - roughness) * (1.0 / norm);
    return clamp(1.0 - w, 0.0, 1.0);
}

/* RENDERTARGETS: 4 */
layout(location = 0) out vec4 color;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));

    // ---- 跳过无效像素 (天空/未命中) --------------------------------------
    bufferData info_ = denoiseBuffer.data[idx];
    if (info_.distance < -0.5) return;

    ivec2 pix = ivec2(gl_FragCoord.xy);

    // 几何法线 (用于构建反射平面)
    vec3 geoNormal = diffuseIllumiantionBuffer.data[idx].normal2;

    // ---- 解包中心像素 ----------------------------------------------------
    vec4 centerGeom = texelFetch(colortex3, pix, 0);
    vec4 centerLight = texelFetch(colortex4, pix, 0);
    vec3 centerPos, centerNormal, centerRadiance;
    float centerWeight, centerRoughness;
    unpackSpecularSample(PackedLightSample(centerGeom, centerLight),
        centerPos, centerNormal, centerRadiance,
        centerWeight, centerRoughness);

    // 使用正确的 hit distance 作为深度
    float depth = info_.distance;

    // ---- 各向异性轴计算 (基于反射平面) ------------------------------------
    vec3 planeN = -reflect(centerNormal, geoNormal);
    vec3 viewDir = cross(camX_global, camY_global);
    float axis_A = 0.75 + max(computeAnisotropicAxisScale(viewDir, camX_global, planeN), 0.0);
    float axis_B = 0.75 + max(computeAnisotropicAxisScale(viewDir, camY_global, planeN), 0.0);
    axis_A *= axis_A;
    axis_B *= axis_B;

    // ---- 动态模糊因子 (用深度取代原来错误的权重) ---------------------------
    float blur_factor = (1.0 - exp(-0.25 * depth)) / 3.0;
    float normal_factor = (1.0 - exp(-0.1 * depth)) * NORMAL_PARAM;

    // ---- à-trous 采样 ----------------------------------------------------
    ivec2 samplePos;
    ivec2 texSize = textureSize(colortex3, 0);

    float theta = 2.0 * PI * rand(vec2(pix + 11 + R0));

    #if STEP != 1
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    vec3 A = vec3(0.0); // 累积加权颜色
    float w = 0.0; // 累积总权重

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            #if STEP == 1
            samplePos = pix + ivec2(i, j);
            #else
            samplePos = pix + ivec2(rotM * vec2(i, j));
            #endif
            if (samplePos.x < 0 || samplePos.y < 0 ||
                    samplePos.x >= texSize.x || samplePos.y >= texSize.y) {
                continue;
            }

            if (denoiseBuffer.data[getIdx(uvec2(samplePos))].distance < -0.5) continue;

            // ---- 解包邻域样本 ----------------------------------------------------
            vec4 sampleGeom = texelFetch(colortex3, samplePos, 0);
            vec4 sampleLight = texelFetch(colortex4, samplePos, 0);
            vec3 samplePosW, sampleNormal, sampleRadiance;
            float sampleWeight, sampleRoughness;
            unpackSpecularSample(PackedLightSample(sampleGeom, sampleLight),
                samplePosW, sampleNormal, sampleRadiance,
                sampleWeight, sampleRoughness);

            float rW = GetRoughnessWeight(centerRoughness, sampleRoughness);

            // 各向异性深度权重
            float w1 = exp(-blur_factor * (axis_A * i * i + axis_B * j * j));

            // 位置/平面距离权重
            float w_pos = exp(-POSITION_PARAM * abs(dot(centerPos - samplePosW, centerNormal)));

            // 法线一致性权重
            float w_norm = svgfNormalWeight(centerNormal, sampleNormal, normal_factor);

            // 边界裁剪与组合
            float w0 = rW * w_pos * w_norm * w1
                    * float(samplePos == clamp(samplePos, vec2(0), texSize));

            A += sampleRadiance * w0;
            w += w0;
        }
    }

    // ---- 中心像素 (权重 = 1) -----------------------------------------------
    A += centerRadiance;
    w += 1.0;

    if (any(isnan(A))) A = vec3(0.0);
    // ---- 输出：重新打包为 light_sample 格式 --------------------------------
    float packedWR = pack2Half(centerWeight, centerRoughness);
    color = vec4(A / max(w, 0.01), packedWR);
}
