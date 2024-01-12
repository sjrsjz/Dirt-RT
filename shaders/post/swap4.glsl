#version 430 compatibility
#define REFLECT_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5 */

layout(location = 0) out vec4 reflectNormal;
layout(location = 1) out vec4 reflectPos;
layout(location = 2) out vec4 color;

void main() {

    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp=fetchReflect(ivec2(gl_FragCoord.xy));
    reflectNormal.xyz = reflectIllumiantionBuffer.data[idx].normal;
    reflectPos.xyz = reflectIllumiantionBuffer.data[idx].pos;
    //reflectNormal.xyz = tmp.normal;
    //reflectPos.xyz = tmp.pos;
    color.xyz=tmp.data_swap;
    color.w = denoiseBuffer.data[idx].reflectWeight;
}
