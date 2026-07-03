#version 430 compatibility

// ===========================================================================
// Pass 301: 镜面反射/折射 NRD 风格屏幕空间降噪 (à-trous, fragment 变体 R0≥8)
// ===========================================================================
// NRD 虚拟追踪权重: 比较反射方向(R)的 GGX lobe 相似度 + 虚拟击中距离(virtualProjDist),
// 而非屏幕空间表面位置/H 向量. 平整镜面上表面属性一致但反射深度可差千米,
// 必须追踪"镜子里的虚像"而非镜子表面.
// ===========================================================================

#include "/lib/constants.glsl"
#include "/lib/common.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform sampler2D colortex3; // (pos.xyz, oct(R))
uniform sampler2D colortex4; // f16(R,G)|f16(B,roughness)|f16(variance,virtualProjDist)|oct(H)

float GetRoughnessWeight(float roughness0, float roughness) {
    float norm = roughness0 * roughness0 * SPEC_ROUGH_NORM_A + SPEC_ROUGH_NORM_B;
    float w = abs(roughness0 - roughness) * (1.0 / norm);
    return clamp(1.0 - w, 0.0, 1.0);
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

    // === NRD 镜面降噪权重 ===
    // 核心思想: 追踪反射光线的虚拟击中点, 而非镜子表面本身.
    // 平整镜面上相邻像素的表面坐标/法线几乎一致, 但反射内容可能深度差异极大.
    // 因此位置/法线权重必须基于反射后的虚拟世界, 而非屏幕空间表面.
    vec3 V = -normalize(cPos);
    vec3 macroNormal = normalize(V + cR);  // 宏观几何法线 (V+R ∥ surface normal)

    // GGX 波瓣宽度: α² = roughness⁴, 控制 R 向量可接受的偏差范围
    float alpha = max(cRough * cRough, SPEC_MIN_ALPHA);
    float alpha2 = alpha * alpha;
    float lobe_param2 = SPEC_BLUR_BOOST / (alpha2 * SPEC_LOBE_DIVISOR) * LOG2_E;
    float hit_dist_param2 = SPEC_BLUR_BOOST * SPEC_HIT_DIST_SENS * LOG2_E;
    float surf_pos_param2 = SPEC_BLUR_BOOST * SPEC_SURF_PARAM * LOG2_E;

    // 亮度权重: 粗糙表面放宽拒绝
    float luma_phi2 = SPEC_BLUR_BOOST * SVGF_PHI_L * LOG2_E * inversesqrt(max(cVar, 1e-8)) / (1.0 + cRough * SPEC_LUMA_ROUGH_SOFT);
    float cLuma = luma(cRad);

    #if STEP >= 4
    float theta = 2.0 * PI * rand(vec2(pix + R0));
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    vec3 A = cRad;            // 中心像素 (权重 = 1)
    float w = 1.0;
    float varEnergy = cVar;
    ivec2 samplePos;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            #if STEP >= 4
            samplePos = pix + ivec2(round(rotM * vec2(i, j)));
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

            // ---- NRD 镜面专属权重 ----
            // 1. 表面连续性 (辅助, 仅防跨物体泄漏)
            float surfDist = abs(dot(cPos - sPos, macroNormal));
            float w_surf = exp2(-surf_pos_param2 * surfDist);

            // 2. 高光波瓣余弦相似度: cR 与 sR 的夹角必须在 GGX lobe 宽度内
            float R_dot_R = max(dot(cR, sR), 0.0);
            float w_lobe = exp2(-(1.0 - R_dot_R) * lobe_param2);

            // 3. 虚拟击中距离: 相对误差 (abs(a-b))/(a+b+eps), 尺度无关
            float hitDistDiff = abs(cVproj - sVproj);
            float hitDistSum = cVproj + sVproj + 1e-5;
            float w_hitDist = exp2(-(hitDistDiff / hitDistSum) * hit_dist_param2);

            // 4. 亮度方差引导
            float w_luma = exp2(-luma_phi2 * abs(cLuma - luma(sRad)));

            float w0 = rW * w_surf * w_lobe * w_hitDist * w_luma;

            A += sRad * w0;
            w += w0;
            varEnergy += w0 * w0 * sVar;
        }
    }

    if (any(isnan(A))) A = vec3(0.0);
    vec3 filtered = A / max(w, 0.01);
    float outVar = varEnergy / max(w * w, 1e-8);

    // 输出: 仅颜色+方差变化, roughness/virtualProjDist/H 透传
    color = packSpecularSample(cPos, cR, filtered, cRough, outVar, cVproj, cH).data1;
}
