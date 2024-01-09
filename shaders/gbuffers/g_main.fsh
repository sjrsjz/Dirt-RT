#version 430
#include "/lib/light_color.glsl"
#include "/lib/buffers/frame_data.glsl"
uniform sampler2D gtexture;
uniform mat4 gbufferModelViewInverse;
in vec2 texCoord;
in vec3 normal;
in vec3 position;
uniform sampler2D colortex9;
/*
const int depthtex0Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;
const int colortex9Format = RGBA32F;
const int colortex8Clear = true;
const int colortex9Clear = true;
*/
/* RENDERTARGETS: 7,9 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragCoord;

void main() {
    fragCoord=gl_FragCoord;//vec4(0,0,gl_FragCoord.w,texture(colortex9,texCoord).w);
    if (texture(gtexture, texCoord).a < 0.01) {
        discard;
    }
    uint idx=getIdx(uvec2(gl_FragCoord.xy));
    //bool k=gBuffer.data[idx].depth.x>gl_FragCoord.z;

    float intensity=dot(lightDir_global,normal);
    vec3 sun=vec3(10);
    vec3 final=clamp(sun*intensity,0.2,1);
    fragColor=pow(texture(gtexture, texCoord),vec4(2.2,2.2,2.2,1))*vec4(final,1);
    
    /*if(k){
        gBuffer.data[idx].color =pow(texture(gtexture, texCoord),vec4(2.2,2.2,2.2,1));
        gBuffer.data[idx].color*=vec4(final,1);
        gBuffer.data[idx].color.xyz=clamp(gBuffer.data[idx].color.xyz,0,1);
        gBuffer.data[idx].depth.x=gl_FragCoord.z;

    }*/
}
