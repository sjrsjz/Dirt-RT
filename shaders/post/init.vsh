#version 430 compatibility

#include "/lib/buffers/frame_data.glsl"
#include "/lib/colors.glsl"
#include "/lib/settings.glsl"
#include "/lib/constants.glsl"
uniform sampler2D colortex0;
uniform int frameCounter;
uniform int worldTime;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform float wetness;
uniform float viewWidth;
uniform float viewHeight;
out vec2 texCoord;

struct Sample {
    vec2 position;
    float weight;
};

const Sample[] samples = Sample[](
        Sample(vec2(0.5, 0.5), 0.2),
        Sample(vec2(0.35, 0.5), 0.1),
        Sample(vec2(0.65, 0.5), 0.1),
        Sample(vec2(0.5, 0.35), 0.1),
        Sample(vec2(0.5, 0.65), 0.1),
        Sample(vec2(0.25, 0.5), 0.07),
        Sample(vec2(0.75, 0.5), 0.07),
        Sample(vec2(0.5, 0.25), 0.07),
        Sample(vec2(0.5, 0.75), 0.07),
        Sample(vec2(0.25, 0.25), 0.03),
        Sample(vec2(0.75, 0.25), 0.03),
        Sample(vec2(0.25, 0.75), 0.03),
        Sample(vec2(0.75, 0.75), 0.03)
    );

void main() {
    if (gl_VertexID == 0) {
        dTime_global = frameTimeCounter - time_global;
        time_global = frameTimeCounter;
        float mix0 = exp(-0.0625*dTime_global);
        rainStrength_global = rainStrength;
        wetStrength_global = wetStrength_global * mix0 + rainStrength * (1 - mix0);
        wetness_global = wetness_global * mix0 + wetness * (1 - mix0);
        resolution_global = uvec2(viewWidth,viewHeight);
/*
        #if defined(HELL)
        world_type_global=World_HELL;
        #elif defined(THE_END)
        world_type_global=World_THE_END;
        #else
        world_type_global=World_OVERWORLD;
        #endif
*/  
        return;
        float luminanceSum = 0.0;
        for (int i = 0; i < samples.length(); i++) {
            #ifdef SRR
            luminanceSum += luminance(texture(colortex0, samples[i].position * 0.5).rgb) * samples[i].weight;
            #else
            luminanceSum += luminance(texture(colortex0, samples[i].position).rgb) * samples[i].weight;
            #endif
        }
        float exposure = clamp(calculateExposure(luminanceSum), 0.1, 25.0);
        if (frameCounter <= 1) {
            avgExposure = exposure;
        } else {
            avgExposure = exp(mix(
                        log(avgExposure),
                        log(exposure),
                        1 - exp(-dTime_global)
                    ));
        }
    }

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    texCoord = gl_MultiTexCoord0.xy;
}
