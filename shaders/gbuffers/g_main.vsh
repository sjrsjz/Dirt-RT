#version 430 compatibility
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform vec3 shadowLightPosition;
out vec2 texCoord;
out vec3 normal;
out vec3 position;
out vec3 viewPos;  // view-space position for linear depth in fragment shader
void main() {

    gl_Position = ftransform();
    position=gl_Vertex.xyz;
    normal=(gl_NormalMatrix *gl_Normal)*mat3(gbufferModelView);
    texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    viewPos = (gbufferModelView * gl_Vertex).xyz;  // distance from camera = length(viewPos)
}
