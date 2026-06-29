#version 430 compatibility
#define DIFFUSE_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4 */

layout(location = 0) out vec4 geometry;
layout(location = 1) out vec4 light_sample;

uniform vec2 resolution;

PackedLightSample packLightSample(vec3 pos, vec3 normal, SH sh, float variance) {
    PackedLightSample sample_data;
    sample_data.data0 = vec4(pos, encodeNormal(normal));
    sample_data.data1 = vec4(packSH(sh), variance);
    return sample_data;
}

#ifndef VAR_FILTER_NORMAL_POWER
#define VAR_FILTER_NORMAL_POWER SVGF_NORMAL_POWER
#endif

#ifndef VAR_FILTER_POSITION_PARAM
#define VAR_FILTER_POSITION_PARAM SVGF_POSITION_PARAM
#endif

// 0 = 标准 3x3 平均方差滤波
// 1 = 保守模式：只允许提升方差，不允许降低中心方差
#define VAR_FILTER_CONSERVATIVE 0

SH sanitizeSH(SH sh) {
    if (any(isnan(sh.shY)) || any(isinf(sh.shY))) {
        sh.shY = vec4(0.0);
    }

    if (any(isnan(sh.CoCg)) || any(isinf(sh.CoCg))) {
        sh.CoCg = vec2(0.0);
    }

    return sh;
}

float sanitizeVariance(float v) {
    if (isnan(v) || isinf(v)) return 0.0;
    return max(v, 0.0);
}

float fetchRawAliceVariance(ivec2 coord) {
    diffuseIllumiantionData d = fetchDiffuse(coord);

    SH sh;
    sh.shY  = d.data_swap.shY;
    sh.CoCg = d.data_swap.CoCg;
    sh = sanitizeSH(sh);

    return sanitizeVariance(alice_estimator_variance(sh.shY, d.weight));
}

float varianceGeometryWeight(
    vec3 centerPos,
    vec3 centerNormal,
    vec3 samplePos,
    vec3 sampleNormal
) {
    // normal bilateral
    float nd = clamp(dot(centerNormal, sampleNormal), 0.0, 1.0);
    float wNormal = pow(nd, VAR_FILTER_NORMAL_POWER);

    // plane-distance bilateral
    float distToCam = max(length(centerPos - camPos), 0.01);
    float pixelFootprint = max(distToCam / max(resolution.y, 1.0), 1e-4);

    float planeDist = abs(dot(samplePos - centerPos, centerNormal));
    float depthTerm = planeDist / max(VAR_FILTER_POSITION_PARAM * pixelFootprint, 1e-6);

    float wDepth = exp(-depthTerm);

    return wNormal * wDepth;
}

float filterVariance3x3(
    ivec2 pix,
    vec3 centerPos,
    vec3 centerNormal,
    float centerVariance
) {
    ivec2 texSize = ivec2(resolution) - ivec2(1);

    // 5x5 B-spline-like kernel
    float hw[3] = float[](1.0, 0.66667, 0.44444);

    float sumVar = 0.0;
    float sumW   = 0.0;

    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            ivec2 q = clamp(pix + ivec2(x, y), ivec2(0), texSize);
            uint qidx = getIdx(uvec2(q));

            // 如果 denoiseBuffer 中天空/无命中是 distance < -0.5，就跳过
            if (denoiseBuffer.data[qidx].distance < -0.5) {
                continue;
            }

            UnifiedDiffuseElement _e = diffuseIllumiantionBuffer.data[qidx];
            vec3 samplePos    = vec3(_e.px, _e.py, _e.pz);
            vec3 sampleNormal = vec3(_e.nx, _e.ny, _e.nz);

            float wKernel = hw[abs(x)] * hw[abs(y)];
            float wGeom = varianceGeometryWeight(
                centerPos,
                centerNormal,
                samplePos,
                sampleNormal
            );

            float w = wKernel * wGeom;

            float v = fetchRawAliceVariance(q);

            sumVar += w * v;
            sumW   += w;
        }
    }

    float filteredVariance = centerVariance;

    if (sumW > 1e-8) {
        filteredVariance = sumVar / sumW;
    }

#if VAR_FILTER_CONSERVATIVE
    // 保守模式：防止方差 guide 被滤得过低。
    filteredVariance = max(filteredVariance, centerVariance);
#endif

    return sanitizeVariance(filteredVariance);
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint idx = getIdx(uvec2(pix));

    // 获取中心点几何与基础数据
    UnifiedDiffuseElement _ce = diffuseIllumiantionBuffer.data[idx];
    vec3 centerNormal = vec3(_ce.nx, _ce.ny, _ce.nz);
    vec3 centerPos    = vec3(_ce.px, _ce.py, _ce.pz);
    diffuseIllumiantionData centerData = fetchDiffuse(pix);

    // 使用原始未过滤的光照数据
    SH outSH;
    outSH.shY  = centerData.data_swap.shY;
    outSH.CoCg = centerData.data_swap.CoCg;
    outSH = sanitizeSH(outSH);

    // 原始 ALICE 方差
    float rawVariance = sanitizeVariance(
        alice_estimator_variance(outSH.shY, centerData.weight)
    );

    // 3x3 几何感知方差滤波
    float variance = filterVariance3x3(
        pix,
        centerPos,
        centerNormal,
        rawVariance
    );

    PackedLightSample outSample = packLightSample(
        centerPos,
        centerNormal,
        outSH,
        variance
    );

    geometry = outSample.data0;
    light_sample = outSample.data1;
}