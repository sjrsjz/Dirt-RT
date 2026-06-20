#version 430 compatibility

// ===========================================================================
// Pass swap6: 折射缓冲写入 (Refract Buffer Write)
// ===========================================================================
// 管线位置: 在折射时域累积 (102.glsl) 后，将数据写入 colortex 供后续使用
// ===========================================================================

#define REFRACT_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/denoise.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4 */
layout(location = 0) out vec4 geometry;
layout(location = 1) out vec4 light_sample;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp = fetchRefract(ivec2(gl_FragCoord.xy));
    PackedLightSample packed_sample = packSpecularSample(
            refractIllumiantionBuffer.data[idx].pos,
            refractIllumiantionBuffer.data[idx].normal,
            tmp.data_swap,
            tmp.weight,
            denoiseBuffer.data[idx].roughness
        );
    geometry = packed_sample.data0;
    light_sample = packed_sample.data1;
}
