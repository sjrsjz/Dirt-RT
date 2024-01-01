#version 430 compatibility
#include "/lib/buffers/frame_data.glsl"

out vec2 texCoord;
void main() {
    /*if (gl_VertexID == 0) {
        //time_global+=frameTimeCounter;
    }*/
    gl_Position = ftransform();
    texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
}
