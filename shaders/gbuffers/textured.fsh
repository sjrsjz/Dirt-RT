#version 430
#include "/lib/buffers/frame_data.glsl"
uniform sampler2D gtexture;
uniform sampler2D colortex9;
in vec2 texCoord;
/*
const int colortex4Format = RGBA32F;
const int colortex8Clear = false;
*/
/* RENDERTARGETS: 7,8 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragCoord;
void main() {
    fragCoord=gl_FragCoord;//vec4(0,0,texture(colortex9,texCoord).z,gl_FragCoord.w);
    if (texture(gtexture,texCoord).a < 0.01) {
        discard;
    }
    
    //discard;
    //uint idx=getIdx(uvec2(gl_FragCoord.xy));
    //gBuffer.data[idx].depth.y=min(gl_FragCoord.z,gBuffer.data[idx].depth.y);

}
