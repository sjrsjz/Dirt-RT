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
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;
uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;

uniform float near;
uniform float far;
uniform vec2 resolution;
uniform int worldTime;
/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex6Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;


const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = false;
const bool colortex8Clear = false;
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
const bool colortex7Clear = false;
const bool colortex8Clear = false;
*/

/* RENDERTARGETS: 0 */

layout(location = 0) out vec4 fragColor;

void main() {
    //return;
    
    uint idx = getIdx(uvec2(gl_FragCoord.xy));

    if (denoiseBuffer.data[idx].distance < -0.5) {
        diffuseIllumiantionBuffer.data[idx].weight = 0;
        reflectIllumiantionBuffer.data[idx].weight = 0;
        refractIllumiantionBuffer.data[idx].weight = 0;
        return;
    }

    diffuseIllumiantionBuffer.data[idx].data = diffuseIllumiantionBuffer.data[idx].data_swap;
    if (any(isnan(diffuseIllumiantionBuffer.data[idx].data.shY))) diffuseIllumiantionBuffer.data[idx].data.shY = vec4(0);
    if (any(isnan(diffuseIllumiantionBuffer.data[idx].data.CoCg))) diffuseIllumiantionBuffer.data[idx].data.CoCg = vec2(0);
    
    reflectIllumiantionBuffer.data[idx].data = reflectIllumiantionBuffer.data[idx].data_swap;
    if (any(isnan(reflectIllumiantionBuffer.data[idx].data))) reflectIllumiantionBuffer.data[idx].data = vec3(0);
    
    refractIllumiantionBuffer.data[idx].data = refractIllumiantionBuffer.data[idx].data_swap;
    if (any(isnan(refractIllumiantionBuffer.data[idx].data))) refractIllumiantionBuffer.data[idx].data = vec3(0);
}
