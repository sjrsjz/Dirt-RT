#version 430
#include "/lib/buffers/frame_data.glsl"
uniform sampler2D gtexture;
uniform sampler2D colortex5;
in vec2 texCoord;
/*
const int colortex4Format = RGBA32F;
const int colortex8Clear = false;
*/
/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;
void main() {
    if (texture(gtexture,texCoord).a < 0.01) {
        discard;
    }
    uint idx=getIdx(uvec2(gl_FragCoord.xy));
    gBuffer.data[idx].depth.y=min(gl_FragCoord.z,gBuffer.data[idx].depth.y);

}
