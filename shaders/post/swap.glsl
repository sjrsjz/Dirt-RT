#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos

uniform sampler2D colortex0;

/* RENDERTARGETS: 0 */

layout(location = 0) out vec4 fragColor;

void main() {
    //return;
    
    //uint idx = getIdx(uvec2(gl_FragCoord.xy));

    //diffuseIllumiantionBuffer.data[idx].data_swap = diffuseIllumiantionBuffer.data[idx].data;

    //reflectIllumiantionBuffer.data[idx].data_swap = reflectIllumiantionBuffer.data[idx].data;

    //refractIllumiantionBuffer.data[idx].data_swap = refractIllumiantionBuffer.data[idx].data;
   
}
