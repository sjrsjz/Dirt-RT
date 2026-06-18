#version 430 compatibility
#define DIFFUSE_BUFFER_MIN
#include "/lib/buffers/frame_data.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;

const bool colortex0Clear = true;
const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

/* RENDERTARGETS: 2 */
layout(location = 0) out vec4 Emission;
uniform sampler2D extInfoBuffer_Sampler;
void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    Emission = vec4(denoiseBuffer.data[idx].emission, 0);

    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_FragCoord.xy));
    vec4 prev_data = texelFetch(extInfoBuffer_Sampler, ivec2(gl_FragCoord.xy), 0);
    tmp.weight = prev_data.x;
    tmp.variance = prev_data.y;
    WriteDiffuse(tmp, ivec2(gl_FragCoord.xy));
}
