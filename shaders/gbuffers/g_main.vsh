#version 430 core
in vec3 vaPosition;
in vec3 vaNormal;
in vec2 vaUV0;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 textureMatrix;
uniform mat3 normalMatrix;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform vec3 shadowLightPosition;
out vec2 texCoord;
out vec3 normal;
out vec3 viewPos;  // view-space position for linear depth in fragment shader
void main() {
    vec4 viewPosition = modelViewMatrix * vec4(vaPosition, 1.0);
    gl_Position = projectionMatrix * viewPosition;
    normal = (normalMatrix * vaNormal) * mat3(gbufferModelView);
    texCoord = (textureMatrix * vec4(vaUV0, 0.0, 1.0)).xy;
    // modelViewMatrix contains per-draw chunk/entity transforms. Using only
    // gbufferModelView drops that translation and corrupts the overlay depth.
    viewPos = viewPosition.xyz;
}
