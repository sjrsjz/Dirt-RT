#ifndef FRAME_DATA_GLSL
#define FRAME_DATA_GLSL
#include "/lib/constants.glsl"

layout(std140,set=3, binding = 1) buffer frameData {
    float avgExposure;
    vec3 SunLight_global;
    vec3 MoonLight_global;
    vec3 lightDir_global;
    vec3 camPos;
    int frame_id;
    float time_global;
    float dTime_global;
    vec3 camX_global;
    vec3 camY_global;
    float rainStrength_global;
    float wetStrength_global;
    float wetness_global;
    int world_type_global;
};
struct gBufferData{
    vec4 pos;
    vec4 normal;
    vec4 color;
    vec4 coord;
    vec2 depth;
};

layout(std430,set=3, binding = 2) buffer GBuffer{
    gBufferData data[];
}gBuffer;

mat3x4 mixAB(mat3x4 A,mat3x4 B,float x){
    return A+(B-A)*x;
}
float mixAB(float A,float B,float x){
    return A+(B-A)*x;
}

#endif // FRAME_DATA_GLSL