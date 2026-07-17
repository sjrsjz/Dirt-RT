#version 430
#include "/lib/buffers/frame_data.glsl"
uniform sampler2D gtexture;
uniform sampler2D colortex9;
in vec2 texCoord;

/* RENDERTARGETS: 8 */
layout(location = 0) out vec4 fragCoord;
void main() {
    fragCoord=vec4(gl_FragCoord.xy,min(gl_FragCoord.z,texture(colortex9,texCoord).z),gl_FragCoord.w);
    if (texture(gtexture,texCoord).a < 0.01) {
        discard;
    }
}
