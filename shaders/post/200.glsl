#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

layout(local_size_x = 8,local_size_y = 8) in;

/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex6Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;


const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

/* RENDERTARGETS: 0 */

void main() {
    return;
    
    uint idx = getIdx(uvec2(gl_GlobalInvocationID.xy));

    if (denoiseBuffer.data[idx].distance < -0.5) {
        return;
    }
    /*SH tmp=diffuseIllumiantionBuffer.data[idx].data_swap;
    
    if (any(isnan(tmp.shY))) tmp.shY = vec4(0);
    if (any(isnan(tmp.CoCg))) tmp.CoCg = vec2(0);
    diffuseIllumiantionBuffer.data[idx].data = tmp;
    vec3 tmp2=reflectIllumiantionBuffer.data[idx].data_swap;
    if (any(isnan(tmp2))) tmp2 = vec3(0);
    reflectIllumiantionBuffer.data[idx].data = tmp2;
    
    tmp2 = refractIllumiantionBuffer.data[idx].data_swap;
    if (any(isnan(tmp2))) tmp2 = vec3(0);
    refractIllumiantionBuffer.data[idx].data = tmp2;*/
}
