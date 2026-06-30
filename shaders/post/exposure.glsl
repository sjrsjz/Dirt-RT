#version 430 compatibility

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

#include "/lib/buffers/frame_data.glsl"
#include "/lib/colors.glsl"
#include "/lib/settings.glsl"
#include "/lib/constants.glsl"

uniform sampler2D colortex1;
uniform int frameCounter;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform float wetness;
uniform float viewWidth;
uniform float viewHeight;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform vec3 cameraPosition;

const int NUM_SAMPLES = 33;

void main() {
    // --- Delta time & state updates ---
    dTime_global = frameTimeCounter - time_global;
    time_global = frameTimeCounter;
    float mix0 = exp(-0.0625 * dTime_global);
    rainStrength_global = rainStrength;
    wetStrength_global = wetStrength_global * mix0 + rainStrength * (1.0 - mix0);
    wetness_global = wetness_global * mix0 + wetness * (1.0 - mix0);
    resolution_global = uvec2(viewWidth, viewHeight);

    float logLumas[NUM_SAMPLES];
    float sum_log = 0.0;
    float sum_sq_log = 0.0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float theta = 2.3999632 * float(i); 
        float r = sqrt(float(i) + 0.5) / sqrt(float(NUM_SAMPLES));
        vec2 uv = vec2(0.5) + vec2(cos(theta), sin(theta)) * r * 0.45; 
        
        vec3 c = texture(colortex1, uv).rgb * ExposureS;
        float luma = max(luminance(c), 1e-4);
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
        
        float pdf_weight = exp(-(diff * diff) / (tolerance * variance));
        
        weighted_sum_log += logLumas[i] * pdf_weight;
        total_weight += pdf_weight;
    }

    float final_log_luma = weighted_sum_log / max(total_weight, 1e-5);
    
    float currentLuma = exp2(final_log_luma);

    float targetExposure = clamp(calculateExposure(currentLuma), 0.00025, 25.0);

    if (frameCounter <= 1) {
        avgExposure = targetExposure;
    } else {
        float adaptSpeed = (targetExposure < avgExposure) ? 3.0 : 0.8;
        avgExposure = exp(mix(
            log(avgExposure),
            log(targetExposure),
            1.0 - exp(-dTime_global * adaptSpeed)
        ));
    }
    div_avgExposure = 1.0 / max(avgExposure, 1e-6);

    // --- Save camera matrices ---
    gbufferPreviousModelView_global = gbufferModelView;
    gbufferPreviousProjection_global = gbufferProjection;
    previousCameraPosition_global = cameraPosition;
}