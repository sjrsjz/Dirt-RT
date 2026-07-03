#ifndef FRAME_DATA_GLSL
#define FRAME_DATA_GLSL
#include "/lib/constants.glsl"

layout(std140, set = 3, binding = 1) buffer FrameData {
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

    vec3 prevRaytracingCamPos;   // 光线追踪相机位置 (上一帧: exposure.glsl 写, 供时域 cameraDelta)

    // 光线追踪推导矩阵 (单源, 无 Iris 混合, 供重投影)
    mat4 rtModelView, rtPrevModelView;      // 世界→视图旋转 (当前帧 / 前帧)
    mat4 rtProjection, rtPrevProjection;    // 非对称视锥投影 (含 TAA jitter 偏移)
};

#endif // FRAME_DATA_GLSL
