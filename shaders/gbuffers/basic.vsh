#version 430 core

in vec3 vaPosition;
in vec4 vaColor;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec4 color;

void main() {
    gl_Position = projectionMatrix * modelViewMatrix *
        vec4(vaPosition, 1.0);
    color = vaColor;
}
