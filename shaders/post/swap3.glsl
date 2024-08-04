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
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    diffuseIllumiantionData tmp=fetchDiffuse(pix);
    if (any(isnan(tmp.data_swap.shY))) tmp.data_swap.shY = vec4(0);
    if (any(isnan(tmp.data_swap.CoCg))) tmp.data_swap.CoCg = vec2(0);
    tmp.data=tmp.data_swap;

    tmp.data_swap.shY=texelFetch(colortex5,pix,0);
    tmp.data_swap.CoCg=texelFetch(colortex6,pix,0).xy;
    //uint idx = getIdx(uvec2(gl_GlobalInvocationID.xy));
    vec4 tmp_=texelFetch(colortex4,pix,0);
    tmp.normal =texelFetch(colortex3,pix,0).xyz;
    tmp.pos = tmp_.xyz;
    //tmp.weight=length(tmp.data.shY)*100;
    WriteDiffuse(tmp,pix);
}
