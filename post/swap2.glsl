#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5,6 */

layout(location = 0) out vec4 diffuseNormal;
layout(location = 1) out vec4 diffusePos;
layout(location = 2) out vec4 shY;
layout(location = 3) out vec4 CoCg;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));

    diffuseNormal.xyz = diffuseIllumiantionBuffer.data[idx].normal;
    diffusePos.xyz = diffuseIllumiantionBuffer.data[idx].pos;
    shY=diffuseIllumiantionBuffer.data[idx].data_swap.shY;
    CoCg.xy=diffuseIllumiantionBuffer.data[idx].data_swap.CoCg;
}
