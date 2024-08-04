#version 430
#include "/lib/buffers/frame_data.glsl"
uniform sampler2D gtexture;
uniform sampler2D colortex9;
in vec2 texCoord;
/*
const int colortex4Format = RGBA32F;
const bool colortex8Clear = true;
*/
/* RENDERTARGETS: 7,8 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragCoord;
void main() {
    fragCoord=vec4(gl_FragCoord.xy,min(gl_FragCoord.z,texture(colortex9,texCoord).z),gl_FragCoord.w);
    //fragCoord=gl_FragCoord;//vec4(0,0,texture(colortex9,texCoord).z,gl_FragCoord.w);
    if (texture(gtexture,texCoord).a < 0.01) {
        discard;
    }
    
    //discard;
    //uint idx=getIdx(uvec2(gl_FragCoord.xy));
    //gBuffer.data[idx].depth.y=min(gl_FragCoord.z,gBuffer.data[idx].depth.y);

}
