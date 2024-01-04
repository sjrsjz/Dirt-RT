#version 430 compatibility
#define REFRACT_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"


uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5 */

layout(location = 0) out vec4 refractNormal;
layout(location = 1) out vec4 refractPos;
layout(location = 2) out vec4 color;

void main() {
    
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp=fetchRefract(ivec2(gl_FragCoord.xy));
    refractNormal.xyz = tmp.normal;
    refractPos.xyz = tmp.pos;
    color.xyz=tmp.data_swap;
}
