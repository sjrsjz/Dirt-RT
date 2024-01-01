#version 430
#include "/lib/light_color.glsl"
#include "/lib/buffers/frame_data.glsl"
uniform sampler2D gtexture;
uniform mat4 gbufferModelViewInverse;
in vec2 texCoord;
in vec3 normal;
in vec3 position;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;
void main() {
    if (texture(gtexture, texCoord).a < 0.01) {
        discard;
    }
    uint idx=getIdx(uvec2(gl_FragCoord.xy));
    bool k=gBuffer.data[idx].depth.x>gl_FragCoord.z;

    float intensity=dot(lightDir_global,normal);
    vec3 sun=vec3(10);//+0*getSkyColor(vec3(10),vec3(1),position,lightDir,lightDir)*-lightDir.y;
    vec3 final=clamp(sun*intensity,0.2,1);
    if(k){
        gBuffer.data[idx].color =pow(texture(gtexture, texCoord),vec4(2.2,2.2,2.2,1));
        gBuffer.data[idx].color*=vec4(final,1);
        gBuffer.data[idx].color.xyz=clamp(gBuffer.data[idx].color.xyz,0,1);
        gBuffer.data[idx].depth.x=gl_FragCoord.z;

    }
}
