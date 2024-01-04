#version 430 compatibility

out vec4 color;

void main() {
    gl_Position = ftransform();

    color = gl_Color;
}
