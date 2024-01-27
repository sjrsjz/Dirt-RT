#version 430 compatibility
layout(local_size_x = 16,local_size_y = 16) in;
#define DIFFUSE_BUFFER
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
uniform sampler2D colortex6;

/* RENDERTARGETS: 0 */

//layout(location = 0) out vec4 fragColor;


void main() {
    diffuseIllumiantionData tmp=sampleDiffuse(vec2(gl_GlobalInvocationID.xy));
    if (any(isnan(tmp.data_swap.shY))) tmp.data_swap.shY = vec4(0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0);
    tmp.data=tmp.data_swap;

    tmp.data_swap.shY=texelFetch(colortex5,ivec2(gl_GlobalInvocationID.xy),0);
    tmp.data_swap.CoCg=texelFetch(colortex6,ivec2(gl_GlobalInvocationID.xy),0).xy;
    tmp.normal =texelFetch(colortex3,ivec2(gl_GlobalInvocationID.xy),0).xyz;
    tmp.pos = texelFetch(colortex4,ivec2(gl_GlobalInvocationID.xy),0).xyz;

    //tmp.weight=length(tmp.data.shY)*100;
    WriteDiffuse(tmp,ivec2(gl_GlobalInvocationID.xy));
}
