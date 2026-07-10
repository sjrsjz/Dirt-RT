#ifndef FRAME_DATA_GLSL
#define FRAME_DATA_GLSL
#include "/lib/constants.glsl"

layout(std430, set = 3, binding = 1) buffer FrameData {
    // 光线追踪推导矩阵 (单源, 无 Iris 混合, 供重投影)
    mat4 rtModelView, rtPrevModelView;      // 世界→视图旋转 (当前帧 / 前帧)
    mat4 rtProjection, rtPrevProjection;    // 非对称视锥投影 (含 TAA jitter 偏移)

    vec3 lightDir_global;
    float avgExposure;
    vec3 camPos;
    float div_avgExposure;
    vec3 camX_global;
    float time_global;
    vec3 camY_global;
    float dTime_global;
    vec3 prevRaytracingCamPos;   // 光线追踪相机位置 (上一帧: auto_exposure.glsl 写, 供时域 cameraDelta)
    float rainStrength_global;
    uvec2 resolution_global;
    int world_type_global;
    int frame_id;
    float wetStrength_global;
    float wetness_global;
};

#endif // FRAME_DATA_GLSL
