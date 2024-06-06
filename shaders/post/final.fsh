#version 430
#include "/lib/buffers/frame_data.glsl"
//#ifndef VULKANITE
//#include "/lib/no_vulkanite.glsl"
//#elif
#include "/lib/constants.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"



in vec2 texCoord;
uniform float near;
uniform float far;
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;
uniform sampler2D depthtex0;
/* RENDERTARGETS: 0,8,9 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragData;
layout(location = 2) out vec4 fragData2;

#define HDR

void main() {
    #ifdef SRR_
    const int scale = 2;
    #else
    const int scale = 1;
    #endif
    vec4 entity = texture(colortex7, texCoord);
    bool mask = texture(colortex8, texCoord).w < texture(colortex9, texCoord).w;////&& gBuffer.data[idx].depth.x <= gBuffer.data[idx].depth.y;
    vec4 data = texture(colortex0, texCoord);
#ifdef HDR
    float luminance = dot(data.xyz, vec3(0.2126, 0.7152, 0.0722));
    //vec4 scene = vec4(vec3(clamp(HDR_AB_global.x*luminance+HDR_AB_global.y,avgExposure*0.75,avgExposure*10)),1)*data;

    vec4 scene = vec4(vec3(clamp(HDR_AB_global.x/(luminance+100)+HDR_AB_global.y,0,avgExposure*5)),1)*data;


#else
    vec4 scene = data*avgExposure;
#endif
    fragColor.xyz = mask ? mix(scene.rgb,entity.rgb,entity.a) : scene.rgb;//vec4(pow(ACESFilm(mask ? mix(scene.rgb,entity.rgb,entity.a) : scene.rgb), vec3(1 / 2.2)), 1);
    
}

//#endif
