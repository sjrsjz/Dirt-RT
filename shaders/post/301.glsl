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
    if (info_.distance < -0.5) {
        // 天空像素: 写入负值 roughness 作为天空 mask
        // colortex4 格式: f16(R,G)|f16(B,roughness)|weight|spare
        color = vec4(0.0, uintBitsToFloat(packHalf2x16(vec2(0.0, -1.0))), 0.0, 0.0);
        return;
    }

    ivec2 pix = ivec2(gl_FragCoord.xy);

    // 几何法线 (用于构建反射平面)
    vec3 geoNormal = decodeNormal(diffuseIllumiantionBuffer.data[idx].oct_n2);

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
    float blur_factor  = (1.0 - exp2(-0.36067376 * depth)) / 3.0;   // exp(-0.25*depth) → exp2
    float normal_factor = (1.0 - exp2(-0.14426950 * depth)) * NORMAL_PARAM; // exp(-0.1*depth) → exp2

    // exp → exp2: 将 LOG2_E 折叠进循环不变量, 避免内层循环重复乘
    float blur_factor2   = blur_factor   * LOG2_E;
    float pos_param2     = POSITION_PARAM * LOG2_E;  // = 1.442695
    float normal_factor2 = normal_factor * LOG2_E;

    // ---- à-trous 采样 ----------------------------------------------------
    ivec2 samplePos;
    ivec2 texSize = textureSize(colortex3, 0);

    #if STEP >= 4
    // 旋转抖动 — 仅大步长启用, 避免网格伪影
    float theta = 2.0 * PI * rand(vec2(pix + 11 + R0));
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    vec3 A = vec3(0.0); // 累积加权颜色
    float w = 0.0; // 累积总权重

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            #if STEP >= 4
            samplePos = pix + ivec2(rotM * vec2(i, j));
            #else
            // 小步长 (≤3): 轴对齐 à‑trous, 无需旋转抖动
            samplePos = pix + R0 * ivec2(i, j);
            #endif
            if (samplePos.x < 0 || samplePos.y < 0 ||
                    samplePos.x >= texSize.x || samplePos.y >= texSize.y) {
                continue;
            }

            // ---- 解包邻域样本 ----------------------------------------------------
            vec4 sampleGeom = texelFetch(colortex3, samplePos, 0);
            vec4 sampleLight = texelFetch(colortex4, samplePos, 0);
            vec3 samplePosW, sampleNormal, sampleRadiance;
            float sampleWeight, sampleRoughness;
            unpackSpecularSample(PackedLightSample(sampleGeom, sampleLight),
                samplePosW, sampleNormal, sampleRadiance,
                sampleWeight, sampleRoughness);

            // 天空检查 — roughness < 0 复用作天空 mask
            if (sampleRoughness < 0.0) continue;

            float rW = GetRoughnessWeight(centerRoughness, sampleRoughness);

            // 单次 exp2: 各向异性深度 + 位置平面距离 + 法线 (LOG2_E 已折叠入常量)
            // 法线权重 pow(dot,S) 用 exp2(-S2·(1-dot)) 逼近 (SVGF 标准近似)
            float w0 = rW * exp2(-(blur_factor2 * (axis_A * i * i + axis_B * j * j)
                                 + pos_param2 * abs(dot(centerPos - samplePosW, centerNormal))
                                 + normal_factor2 * (1.0 - dot(centerNormal, sampleNormal))))
                    * float(samplePos == clamp(samplePos, vec2(0), texSize));

            A += sampleRadiance * w0;
            w += w0;
        }
    }

    // ---- 中心像素 (权重 = 1) -----------------------------------------------
    A += centerRadiance;
    w += 1.0;

    if (any(isnan(A))) A = vec3(0.0);
    // ---- 输出: f16(R,G) | f16(B,roughness) | weight | spare ---------------
    vec3 filteredRadiance = A / max(w, 0.01);
    color = vec4(
        uintBitsToFloat(packHalf2x16(filteredRadiance.rg)),
        uintBitsToFloat(packHalf2x16(vec2(filteredRadiance.b, centerRoughness))),
        centerWeight,
        0.0
    );
}
