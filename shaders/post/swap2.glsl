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

PackedLightSample packLightSample(vec3 pos, vec3 normal, SH sh, float weight, float variance) {
    PackedLightSample sample_data;
    sample_data.data0 = vec4(pos, encodeNormal(normal));
    sample_data.data1 = vec4(packSH(sh), pack2Half(weight, variance));
    return sample_data;
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint idx = getIdx(uvec2(pix));

    // 获取中心点几何与基础数据
    vec3 centerNormal = diffuseIllumiantionBuffer.data[idx].normal;
    vec3 centerPos    = diffuseIllumiantionBuffer.data[idx].pos;
    diffuseIllumiantionData centerData = fetchDiffuse(pix);

    // 使用原始未过滤的光照数据（不进行萤火虫拦截）
    SH outSH;
    outSH.shY  = centerData.data_swap.shY;
    outSH.CoCg = centerData.data_swap.CoCg;
    if (any(isnan(outSH.shY)))  outSH.shY  = vec4(0);
    if (any(isnan(outSH.CoCg))) outSH.CoCg = vec2(0);

    float variance = alice_estimator_variance(outSH.shY, centerData.weight);
    variance = max(variance, outSH.shY.w * outSH.shY.w * 0.00005); // 根据光子能量约束方差最小值

    PackedLightSample outSample = packLightSample(centerPos, centerNormal, outSH, centerData.weight, variance);
    geometry = outSample.data0;
    light_sample = outSample.data1;
}