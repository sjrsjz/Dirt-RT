#version 430 compatibility

//#define USE_NOISE_TEXTURE
layout(local_size_x = 16,local_size_y = 16) in;
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/light_color.glsl"

const ivec3 workGroups = ivec3(64, 32, 1);

void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    setSkyVars();
    GenSky(SunLight_global,MoonLight_global,lightDir_global,camPos,p);
}