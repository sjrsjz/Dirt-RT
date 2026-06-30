#version 430 compatibility

// ===========================================================================
// Pass exposure: Auto-Exposure (Compute)
// ===========================================================================
// Single-thread compute shader: 1x1x1 workgroup, one invocation.
// Samples colortex1 at 13 metering points, robust-weighted luminance,
// temporal smoothing, writes frameData UBO.

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

struct Sample {
    vec2 position;
    float weight;
};

const Sample samples[13] = Sample[](
    Sample(vec2(0.5, 0.5), 0.1),
    Sample(vec2(0.35, 0.5), 0.1),
    Sample(vec2(0.65, 0.5), 0.1),
    Sample(vec2(0.5, 0.35), 0.1),
    Sample(vec2(0.5, 0.65), 0.1),
    Sample(vec2(0.25, 0.5), 0.07),
    Sample(vec2(0.75, 0.5), 0.07),
    Sample(vec2(0.5, 0.25), 0.07),
    Sample(vec2(0.5, 0.75), 0.07),
    Sample(vec2(0.25, 0.25), 0.07),
    Sample(vec2(0.75, 0.25), 0.07),
    Sample(vec2(0.25, 0.75), 0.07),
    Sample(vec2(0.75, 0.75), 0.07)
);

void main() {
    // --- Delta time & rain/wetness smoothing ---
    dTime_global = frameTimeCounter - time_global;
    time_global = frameTimeCounter;
    float mix0 = exp(-0.0625 * dTime_global);
    rainStrength_global = rainStrength;
    wetStrength_global = wetStrength_global * mix0 + rainStrength * (1.0 - mix0);
    wetness_global = wetness_global * mix0 + wetness * (1.0 - mix0);
    resolution_global = uvec2(viewWidth, viewHeight);

    // --- First pass: gather metering samples ---
    vec3 sumX = vec3(0.0);
    vec3 sumX2 = vec3(0.0);
    vec3 sampleC[13];

    float w = 0.0;
    for (int i = 0; i < 13; i++) {
        vec3 c = ExposureS * texture(colortex1, samples[i].position).rgb;
        sampleC[i] = c;
        w += samples[i].weight;
        sumX += c * samples[i].weight;
        sumX2 += c * c * samples[i].weight;
    }
    sumX /= w;
    sumX2 /= w;

    // --- Second pass: robust outlier rejection ---
    vec3 sigma2 = 2.0 * (sumX2 - sumX * sumX + 1e-3);
    vec3 w3 = vec3(0.0);
    sumX2 = vec3(0.0);

    for (int i = 0; i < 13; i++) {
        vec3 weight = exp(-(sampleC[i] - sumX) * (sampleC[i] - sumX) / sigma2);
        w3 += weight;
        sumX2 += sampleC[i] * weight;
    }

    float luminanceSum = luminance(sumX2 / (w3 + 0.00001));

    // --- Temporal smoothing of exposure ---
    float exposure = clamp(calculateExposure(luminanceSum), 0.00025, 25.0);
    dTime_global *= 0.5;

    if (frameCounter <= 1) {
        avgExposure = exposure;
    } else {
        avgExposure = exp(mix(
            log(avgExposure),
            log(exposure),
            1.0 - exp(-dTime_global)
        ));
    }
    div_avgExposure = 1.0 / avgExposure;

    // --- Save camera matrices for temporal reprojection ---
    gbufferPreviousModelView_global = gbufferModelView;
    gbufferPreviousProjection_global = gbufferProjection;
    previousCameraPosition_global = cameraPosition;
}
