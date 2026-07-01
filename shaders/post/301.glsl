#version 430 compatibility

// ===========================================================================
// Pass 301: 镜面反射/折射 NRD 风格屏幕空间降噪 (à-trous, fragment 变体 R0≥8)
// ===========================================================================
// 仅读 colortex3 (pos.xyz + oct(R)) + colortex4 (f16 RGB | roughness | variance |
// vprojdist | oct(H)), 无 SSBO. 虚拟平面法线 = H (GGX 主半向量); 各向异性由入射面
// cross(V,R) 决定; 模糊尺度由虚拟投射距离 vprojdist 驱动; 亮度权重由预计算方差引导.
// 每采样点仅一次 exp2 (各向异性+位置+法线+亮度权重合并, LOG2_E 折叠进常量).
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // f16(R,G)|f16(B,roughness)|f16(variance,vprojdist)|oct(H)

// 可调参数
const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 1.0;

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

float luma3(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

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

    // 视线 / 入射面 (替代旧 geoNormal 反射平面)
    vec3 V = -normalize(cPos);
    vec3 planeN = cross(V, cR);
    vec3 viewDir = cross(camX_global, camY_global);
    float axis_A = 0.75 + max(computeAnisotropicAxisScale(viewDir, camX_global, planeN), 0.0);
    float axis_B = 0.75 + max(computeAnisotropicAxisScale(viewDir, camY_global, planeN), 0.0);
    axis_A *= axis_A;
    axis_B *= axis_B;

    // 模糊尺度由虚拟投射距离驱动 (取代旧 info_.distance)
    float depth = cVproj;
    float blur_factor   = (1.0 - exp2(-0.36067376 * depth)) / 3.0;        // exp(-0.25*depth) → exp2
    float normal_factor = (1.0 - exp2(-0.14426950 * depth)) * NORMAL_PARAM; // exp(-0.1*depth) → exp2

    // exp → exp2: LOG2_E 折叠进循环不变量
    float blur_factor2   = blur_factor   * LOG2_E;
    float pos_param2     = POSITION_PARAM * LOG2_E;
    float normal_factor2 = normal_factor * LOG2_E;
    float luma_phi2      = SVGF_PHI_L * LOG2_E * inversesqrt(max(cVar, 1e-8));
    float cLuma = luma3(cRad);

    #if STEP >= 4
    // 旋转抖动 — 仅大步长启用, 避免网格伪影
    float theta = 2.0 * PI * rand(vec2(pix + 11 + R0));
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    vec3 A = cRad;            // 中心像素 (权重 = 1)
    float w = 1.0;
    float varEnergy = cVar;   // 方差传播 (Σa²·var / (Σa)²)
    ivec2 samplePos;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            #if STEP >= 4
            samplePos = pix + ivec2(rotM * vec2(i, j));
            #else
            samplePos = pix + R0 * ivec2(i, j);
            #endif
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

            float rW = GetRoughnessWeight(cRough, sRough);

            // 单次 exp2: 各向异性 + 位置平面(⊥H) + 法线(H) + 亮度(方差引导)
            float w0 = rW * exp2(-(blur_factor2 * (axis_A * float(i * i) + axis_B * float(j * j))
                                 + pos_param2 * abs(dot(cPos - sPos, cH))
                                 + normal_factor2 * (1.0 - dot(cH, sH))
                                 + luma_phi2 * abs(cLuma - luma3(sRad))));

            A += sRad * w0;
            w += w0;
            varEnergy += w0 * w0 * sVar;
        }
    }

    if (any(isnan(A))) A = vec3(0.0);
    vec3 filtered = A / max(w, 0.01);
    float outVar = varEnergy / max(w * w, 1e-8);

    // 输出: 仅颜色+方差变化, roughness/vprojdist/H 透传
    color = packSpecularSample(cPos, cR, filtered, cRough, outVar, cVproj, cH).data1;
}
