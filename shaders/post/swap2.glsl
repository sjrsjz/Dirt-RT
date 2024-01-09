#version 430 compatibility
#define DIFFUSE_BUFFER
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
    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_FragCoord.xy));
    diffuseNormal.xyz = diffuseIllumiantionBuffer.data[idx].normal;
    diffusePos.xyz = diffuseIllumiantionBuffer.data[idx].pos;
    shY = tmp.data_swap.shY;
    CoCg.xy = tmp.data_swap.CoCg;
    CoCg.z = tmp.weight;
}
