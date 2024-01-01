#version 430
#include "/lib/buffers/frame_data.glsl"
//#ifndef VULKANITE
//#include "/lib/no_vulkanite.glsl"
//#elif
#include "/lib/constants.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"


/*
//const int colortex4Format = RGBA32F;
//const bool colortex4Clear = false;
*/

in vec2 texCoord;
uniform float near;
uniform float far;
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex8;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex9;
uniform sampler2D depthtex0;
/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    #ifdef SRR_
    const int scale = 2;
    #else
    const int scale = 1;
    #endif
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    vec4 entity = gBuffer.data[idx].color;
    bool mask = texture(depthtex0, texCoord).x != 1 && gBuffer.data[idx].depth.x <= gBuffer.data[idx].depth.y;
    //vec4 scene = avgExposure*texture(colortex0, texCoord);
    vec4 scene = avgExposure*texture(colortex0, texCoord);
    gBuffer.data[idx].depth = vec2(10);
    fragColor = vec4(pow(ACESFilm(mask ? entity.rgb : scene.rgb), vec3(1 / 2.2)), 1);
}

//#endif
