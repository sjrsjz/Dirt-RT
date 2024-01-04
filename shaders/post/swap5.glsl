#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;

/* RENDERTARGETS: 0 */

layout(location = 0) out vec4 fragColor;


void main() {
    uint idx=getIdx(uvec2(gl_FragCoord.xy));

    reflectIllumiantionBuffer.data[idx].data_swap=texelFetch(colortex5,ivec2(gl_FragCoord.xy),0).xyz;
    reflectIllumiantionBuffer.data[idx].lnormal =texelFetch(colortex3,ivec2(gl_FragCoord.xy),0).xyz;
    reflectIllumiantionBuffer.data[idx].lpos = texelFetch(colortex4,ivec2(gl_FragCoord.xy),0).xyz;
}
