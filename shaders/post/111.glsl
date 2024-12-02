#version 430 compatibility
#define DIFFUSE_BUFFER_MIN
layout(local_size_x = 16,local_size_y = 16) in;
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"
/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex6Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;

const bool colortex0Clear = true;
const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

void main() {
    return;
    uint idx = getIdx(uvec2(gl_GlobalInvocationID.xy));
    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_GlobalInvocationID.xy));
    vec2 prev_data = extInfoBuffer.data[idx];
    tmp.weight = prev_data.x;
    tmp.variance = prev_data.y;
    WriteDiffuse(tmp, ivec2(gl_GlobalInvocationID.xy));
}
