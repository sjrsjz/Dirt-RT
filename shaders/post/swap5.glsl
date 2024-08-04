#version 430 compatibility
layout(local_size_x = 16,local_size_y = 16) in;
#define REFLECT_BUFFER
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

//layout(location = 0) out vec4 fragColor;


void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    uint idx=getIdx(uvec2(gl_GlobalInvocationID.xy));
    vec3IllumiantionData tmp=fetchReflect(pix);
    if (any(isnan(tmp.data_swap))) tmp.data_swap = vec3(0);
    tmp.data=tmp.data_swap;
    tmp.data_swap=texelFetch(colortex5,pix,0).xyz;
    tmp.normal =texelFetch(colortex3,pix,0).xyz;
    tmp.pos = texelFetch(colortex4,pix,0).xyz;
    tmp.mixWeight=denoiseBuffer.data[idx].reflectWeight;
    WriteReflect(tmp,pix);
}
