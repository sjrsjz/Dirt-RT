#version 430 compatibility

out vec2 texCoordRaw;

void main() {
    gl_Position = ftransform();
    texCoordRaw = gl_MultiTexCoord0.xy;
}