#version 430 core

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

#include "/lib/buffers/frame_data.glsl"
#include "/lib/common.glsl"
#include "/lib/settings.glsl"
#include "/lib/constants.glsl"
#include "/lib/post_processing/tonemap.glsl"

uniform sampler2D colortex1;
uniform int frameCounter;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform float wetness;
uniform float viewWidth;
uniform float viewHeight;
uniform vec3 sunPosition;
uniform mat4 gbufferModelViewInverse;

const int NUM_SAMPLES = 128;

// 曝光曲线: f(x) = ln(k + e^x) - ln(1 + k)
// 应用于 tonemap 之前作为暗部压缩
#include "/lib/post_processing/exposure_response.glsl"

// One sample per invocation; two balanced reductions replace the serial
// texture loop and its invocation-local 128-element array.
shared vec2 exposureReduction[NUM_SAMPLES];

vec2 exposureReduce(vec2 value) {
    uint lane = gl_LocalInvocationIndex;
    exposureReduction[lane] = value;
    barrier();
    for (uint stride = uint(NUM_SAMPLES) / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            exposureReduction[lane] += exposureReduction[lane + stride];
        barrier();
    }
    return exposureReduction[0];
}

void main() {
    uint lane = gl_LocalInvocationIndex;
    float i = float(lane);
    float theta = 2.3999632 * i;
    float radius = sqrt((i + 0.5) / float(NUM_SAMPLES));
    vec2 uv = vec2(0.5) + vec2(cos(theta), sin(theta)) * radius * 0.45;
    float logLuma = log2(max(luma(textureLod(colortex1, uv, 0.0).rgb), 1e-10));
    vec2 moments = exposureReduce(vec2(logLuma, logLuma * logLuma)) / float(NUM_SAMPLES);
    float mean = moments.x;
    // A negative second central moment is roundoff, not positive variance.
    float variance = max(moments.y - mean * mean, 1e-4);
    float tolerance = max(float(AUTO_EXPOSURE_OUTLIER_TOLERANCE), 1e-4);
    float delta = logLuma - mean;
    float weight = exp2(-delta * delta * (LOG2_E / (tolerance * variance)));
    // All lanes must finish reading the first reduction before reusing it.
    barrier();
    vec2 weighted = exposureReduce(vec2(logLuma * weight, weight));
    if (lane != 0u) return;
    float final_log_luma = weighted.x / max(weighted.y, 1e-5);

    // --- Delta time & state updates ---
    dTime_global = max(frameTimeCounter - time_global, 0.0);
    time_global = frameTimeCounter;
    float mix0 = exp2(-0.09016844 * dTime_global);
    rainStrength_global = rainStrength;
    wetStrength_global = mix(rainStrength, wetStrength_global, mix0);
    wetness_global = mix(wetness, wetness_global, mix0);
    resolution_global = uvec2(viewWidth, viewHeight);

    float currentLuma = exp2(final_log_luma);

    // Pupil aperture is only the fast optical component of adaptation. It must
    // not clamp total exposure: doing that with a 2 mm lower limit forces a
    // 0.25x exposure floor and makes a clear outdoor scene about 20x brighter
    // than the previous 0.0125x safety floor. The slower neural component
    // supplies the remaining range while pupil limits still affect transients.
    const float MIN_TOTAL_EXPOSURE = 1.25e-2;
    const float MAX_TOTAL_EXPOSURE = 10.0;
    float targetExposure = clamp(
        calculateExposure(currentLuma),
        MIN_TOTAL_EXPOSURE,
        MAX_TOTAL_EXPOSURE);
    float targetPupilExposure = calculatePupilExposure(currentLuma);
    float targetNeuralExposure = targetExposure / max(targetPupilExposure, 1e-20);

    pupilExposure = ensurePositive(pupilExposure, targetPupilExposure);
    neuralExposure = ensurePositive(neuralExposure, targetNeuralExposure);

    if (frameCounter <= 1) {
        pupilExposure = targetPupilExposure;
        neuralExposure = targetNeuralExposure;
    } else {
        float pupilSpeed = (targetPupilExposure < pupilExposure) ? 6.0 : 1.5;
        float neuralSpeed = (targetNeuralExposure < neuralExposure) ? 1.2 : 0.35;
        pupilExposure = adaptExposureState(
            pupilExposure, targetPupilExposure, pupilSpeed);
        neuralExposure = adaptExposureState(
            neuralExposure, targetNeuralExposure, neuralSpeed);
    }
    avgExposure = clamp(
        pupilExposure * neuralExposure,
        MIN_TOTAL_EXPOSURE,
        MAX_TOTAL_EXPOSURE);
    div_avgExposure = 1.0 / max(avgExposure, 1e-20);

    // --- Save camera matrices ---
    prevRaytracingCamPos = camPos;     // 光线追踪相机 (供下一帧时域 cameraDelta)
    // 从视空间转为世界空间的太阳方向 (供下一帧时域 sunDir)
    vec3 lightDir = mat3(gbufferModelViewInverse) * sunPosition;
    // 然后侧转30度，模拟太阳路径旋转 (绕 X 轴旋转)
    float angle = radians(SUN_PATH_ROTATION);
    mat3 rotationMatrix = mat3(1.0, 0.0, 0.0,
                                   0.0, cos(angle), sin(angle),
                                   0.0, -sin(angle), cos(angle));
    lightDir_global = normalize(rotationMatrix * lightDir);
}
