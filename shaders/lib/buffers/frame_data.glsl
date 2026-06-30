#ifndef FRAME_DATA_GLSL
#define FRAME_DATA_GLSL
#include "/lib/constants.glsl"

layout(std140, set = 3, binding = 1) buffer frameData {
    float avgExposure;
    float div_avgExposure;
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
    uvec2 resolution_global;

    mat4 gbufferPreviousProjection_global;
    mat4 gbufferPreviousModelView_global;
    vec3 previousCameraPosition_global;
};

#endif // FRAME_DATA_GLSL
