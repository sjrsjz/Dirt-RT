#version 430 compatibility

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
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
float applyExposureCurve(float x, float k) {
    if (k <= 0.0) return x;
    return log(k + exp(x)) - log(1.0 + k);
}

// 曝光曲线导数: f'(x) = e^x / (k + e^x) = 1 / (1 + k·e^(-x))
float exposureCurveDerivative(float x, float k) {
    if (k <= 0.0) return 1.0;
    return 1.0 / (1.0 + k * exp(-x));
}

// 牛顿迭代求逆: 找到 linear_L 使得 tonemap(curve(linear_L)) = expected
// 建模完整管线: linear → exposure_curve → tonemap → sRGB
float solveBaseline(float expected) {
    float k = EXPOSURE_CURVE_K;
    float x = expected;

    for (int i = 0; i < 8; i++) {
        float curved = applyExposureCurve(x, k);
        float f_x = TonyMcMapface_LumaApprox(curved) - expected;

        // 链式法则: d/dx tonemap(curve(x)) = tonemap'(curve) · curve'(x)
        float t_deriv = TonyMcMapface_LumaApprox_Deriv(curved);
        float c_deriv = exposureCurveDerivative(x, k);
        float f_prime_x = t_deriv * c_deriv;

        x = x - f_x / max(f_prime_x, 1e-6);
    }
    return max(x, 0.0);
}

// 将用户自定义的 sRGB 纸白亮度转换为线性空间
float convertPaperWhiteToLinear(float sRGB_paperWhite) {
    if (sRGB_paperWhite <= 0.04045) {
        return sRGB_paperWhite / 12.92;
    } else {
        return pow((sRGB_paperWhite + 0.055) / 1.055, 2.4);
    }
}

float calculateExposure(float avgLuminance) {
    float baseline = solveBaseline(convertPaperWhiteToLinear(DISPLAY_PAPER_WHITE_LUMINANCE));
    float baseExposure = baseline / max(avgLuminance, 1e-20);    
    float exposureCorrection = pow(100.0 / DISPLAY_MAX_LUMINANCE, 0.7);
    return baseExposure * exposureCorrection;
}

// Estimate the physical pupil response from adapting luminance. The
// Stanley-Davies form uses a 20 degree adapting field (100*pi square degrees).
// Shader radiance is converted to photometric luminance with the standard
// daylight efficacy used by calibrated HDR radiance data.
float calculatePupilExposure(float sceneLuminance) {
    const float REFERENCE_PUPIL_DIAMETER_MM = 4.0;
    const float DAYLIGHT_LUMINOUS_EFFICACY = 179.0;
    const float ADAPTING_FIELD_AREA_DEG2 = 100.0 * PI;

    float diameterA = clamp(float(PUPIL_MIN_DIAMETER_MM), 1.0, 10.0);
    float diameterB = clamp(float(PUPIL_MAX_DIAMETER_MM), 1.0, 10.0);
    float minDiameter = min(diameterA, diameterB);
    float maxDiameter = max(diameterA, diameterB);

    float adaptingLuminance = max(sceneLuminance * DAYLIGHT_LUMINOUS_EFFICACY, 1e-8);
    float x = pow(adaptingLuminance * ADAPTING_FIELD_AREA_DEG2 / 846.0, 0.41);
    float diameter = 7.75 - 5.75 * x / (x + 2.0);
    diameter = clamp(diameter, minDiameter, maxDiameter);

    float diameterRatio = diameter / REFERENCE_PUPIL_DIAMETER_MM;
    return diameterRatio * diameterRatio;
}

float adaptExposureState(float currentValue, float targetValue, float speed) {
    return exp2(mix(
        log2(max(currentValue, 1e-20)),
        log2(max(targetValue, 1e-20)),
        1.0 - exp2(-max(dTime_global, 0.0) * speed * LOG2_E)
    ));
}

void main() {
    // --- Delta time & state updates ---
    dTime_global = frameTimeCounter - time_global;
    time_global = frameTimeCounter;
    float mix0 = exp2(-0.09016844 * dTime_global);
    rainStrength_global = rainStrength;
    wetStrength_global = mix(wetStrength_global, rainStrength, mix0);
    wetness_global = mix(wetness_global, wetness, mix0);
    resolution_global = uvec2(viewWidth, viewHeight);

    float logLumas[NUM_SAMPLES];
    float sum_log = 0.0;
    float sum_sq_log = 0.0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float theta = 2.3999632 * float(i); 
        float r = sqrt(float(i) + 0.5) / sqrt(float(NUM_SAMPLES));
        vec2 uv = vec2(0.5) + vec2(cos(theta), sin(theta)) * r * 0.45; 
        
        vec3 c = texture(colortex1, uv).rgb;
        float luma = max(luma(c), 1e-10);
        float logL = log2(luma);
        
        logLumas[i] = logL;
        sum_log += logL;
        sum_sq_log += logL * logL;
    }

    float mean = sum_log / float(NUM_SAMPLES);
    float variance = max(abs(sum_sq_log / float(NUM_SAMPLES) - mean * mean), 1e-4); 

    float weighted_sum_log = 0.0;
    float total_weight = 0.0;

    float tolerance = 2.0; 

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float diff = logLumas[i] - mean;
        
        float pdf_weight = exp2(-(diff * diff) / (tolerance * variance) * LOG2_E);
        
        weighted_sum_log += logLumas[i] * pdf_weight;
        total_weight += pdf_weight;
    }

    float final_log_luma = weighted_sum_log / max(total_weight, 1e-5);
    
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
    vec3 lightDir = normalize((vec3(gbufferModelViewInverse * vec4(normalize(sunPosition), 0.0))));
    // 然后侧转30度，模拟太阳路径旋转 (绕 X 轴旋转)
    float angle = radians(SUN_PATH_ROTATION);
    mat3 rotationMatrix = mat3(1.0, 0.0, 0.0,
                                   0.0, cos(angle), sin(angle),
                                   0.0, -sin(angle), cos(angle));
    lightDir_global = normalize(rotationMatrix * lightDir);
}
