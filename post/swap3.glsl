#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos

uniform sampler2D colortex5;
uniform sampler2D colortex6;

/* RENDERTARGETS: 0 */

layout(location = 0) out vec4 fragColor;


void main() {

    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    diffuseIllumiantionBuffer.data[idx].data_swap.shY=texelFetch(colortex5,ivec2(gl_FragCoord.xy),0);
    diffuseIllumiantionBuffer.data[idx].data_swap.CoCg=texelFetch(colortex6,ivec2(gl_FragCoord.xy),0).xy;
}
