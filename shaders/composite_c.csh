#version 430 compatibility


layout(local_size_x = 16,local_size_y = 16) in;
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/light_color.glsl"

const ivec3 workGroups = ivec3(64, 32, 1);

void main(){
    BlurSkyY(ivec2(gl_GlobalInvocationID.xy));
}