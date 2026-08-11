#version 430 core

in vec3 vaPosition;
in vec2 vaUV0;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 textureMatrix;

out vec2 texCoord;

void main() {
    gl_Position = projectionMatrix * modelViewMatrix *
        vec4(vaPosition, 1.0);
    texCoord = (textureMatrix * vec4(vaUV0, 0.0, 1.0)).xy;
}
