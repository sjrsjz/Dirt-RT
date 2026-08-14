#ifndef FRAME_DATA_GLSL
#define FRAME_DATA_GLSL
#include "/lib/constants.glsl"

layout(std430, set = 3, binding = 1) buffer FrameData {
    // 光线追踪推导矩阵 (单源, 无 Iris 混合, 供重投影)
    mat4 rtModelView;                       // 当前帧世界→视图旋转

    // Precomposed once by the primary-visibility ray pass. Reprojection
    // shaders must not rebuild P * MV independently for every pixel.
    mat4 rtViewProjection, rtPrevViewProjection;
    mat4 rtInverseViewProjection;

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
    // Camera-side refractive medium used by the primary reflection lobe.
    // Keeping this in FrameData lets the deferred MaxEnt decoder evaluate the
    // exact same dielectric Fresnel branch as the ray-generation pass.
    uint eye_medium_global;
    float wetStrength_global;
    float wetness_global;
    // One atomic increment per participating bloom workgroup. composite79
    // clears it before bloom, including on the first loaded frame.
    uint bloomCompletedGroups;
    // Exposure is split into the fast optical pupil response and the slower
    // retinal/neural response. Their product is avgExposure.
    float pupilExposure;
    float neuralExposure;
    // RT projection coefficients (P00, P11, P20, P21). Cached when ray0
    // constructs the projection so post passes can recover view-ray angles
    // without performing a matrix inverse per pixel.
    vec4 rtProjectionParams;
};

#endif // FRAME_DATA_GLSL
