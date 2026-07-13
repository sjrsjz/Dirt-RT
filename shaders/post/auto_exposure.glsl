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

const int NUM_SAMPLES = 128;

// 使用牛顿迭代从期望的线性亮度计算需要输入的线性亮度基准
float solveBaseline(float expected) {
    float x = expected; // 初始猜测
    for (int i = 0; i < 5; i++) {
        float f_x = TonyMcMapface_LumaApprox(x) - expected;
        float f_prime_x = (TonyMcMapface_LumaApprox(x + 1e-5) - TonyMcMapface_LumaApprox(x - 1e-5)) / (2e-5);
        x = x - f_x / max(f_prime_x, 1e-6); // 避免除以零
    }
    return max(x, 0.0); // 确保非负
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
        float luma = max(luma(c), 1e-4);
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

    float targetExposure = clamp(calculateExposure(currentLuma), 1e-10, 10.0);

    if (frameCounter <= 1) {
        avgExposure = targetExposure;
    } else {
        float adaptSpeed = (targetExposure < avgExposure) ? 3.0 : 0.8;
        avgExposure = exp2(mix(
            log2(avgExposure),
            log2(targetExposure),
            1.0 - exp2(-dTime_global * adaptSpeed * LOG2_E)
        ));
    }
    div_avgExposure = 1.0 / max(avgExposure, 1e-6);

    // --- Save camera matrices ---
    prevRaytracingCamPos = camPos;     // 光线追踪相机 (供下一帧时域 cameraDelta)
}