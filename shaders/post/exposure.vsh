#version 430 compatibility

#include "/lib/buffers/frame_data.glsl"
#include "/lib/colors.glsl"
#include "/lib/settings.glsl"
#include "/lib/constants.glsl"

#include "/lib/hdr.glsl"

uniform sampler2D colortex1;
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
    if (gl_VertexID == 0) {
        dTime_global = frameTimeCounter - time_global;
        time_global = frameTimeCounter;
        float mix0 = exp(-0.0625 * dTime_global);
        rainStrength_global = rainStrength;
        wetStrength_global = wetStrength_global * mix0 + rainStrength * (1 - mix0);
        wetness_global = wetness_global * mix0 + wetness * (1 - mix0);
        resolution_global = uvec2(viewWidth, viewHeight);
        /*
                #if defined(HELL)
                world_type_global=World_HELL;
                #elif defined(THE_END)
                world_type_global=World_THE_END;
                #else
                world_type_global=World_OVERWORLD;
                #endif
        /*
        #if defined(HELL)
        world_type_global=World_HELL;
        #elif defined(THE_END)
        world_type_global=World_THE_END;
        #else
        world_type_global=World_OVERWORLD;
        #endif
*/
        float luminanceSum = 0.0;
        vec3 sumX = vec3(0);
        vec3 sumX2 = vec3(0);
        vec3 sampleC[samples.length()];


        /*float sum_X_n = 0.0;
        float sum_X_n2 = 0.0;
        float sum_X_n3 = 0.0;
        float sum_X_n4 = 0.0;
        */

        float sum_X = 0.0;
        float sum_X2 = 0.0;
        float sum_div = 0.0;
        float sum_div_2 = 0.0;
        float sum_div2 = 0.0;
        float C=10;

        float w = 0;
        for (int i = 0; i < samples.length(); i++) {
            vec3 c = ExposureS * texture(colortex1, samples[i].position).rgb;
            sampleC[i] = c;
            w += samples[i].weight;
            sumX += c * samples[i].weight;
            sumX2 += c * c * samples[i].weight;
        }
        sumX /= w;
        sumX2 /= w;
        vec3 sigma2 = 3 * (sumX2 - sumX * sumX + 1e-3);
        vec3 w3 = vec3(0);
        sumX2 = vec3(0);
        for (int i = 0; i < samples.length(); i++) {
            vec3 weight = exp(-(sampleC[i] - sumX) * (sampleC[i] - sumX) / sigma2);
            w3 += weight;
            sumX2 += sampleC[i] * weight;
            float X_n = luminance(sampleC[i]);
            /*sum_X_n += X_n;
            sum_X_n2 += X_n * X_n;
            sum_X_n3 += X_n * X_n * X_n;
            sum_X_n4 += X_n * X_n * X_n * X_n;*/
            sum_X += X_n;
            sum_X2 += X_n * X_n;
            sum_div += X_n /(C + X_n);
            sum_div_2 += X_n * X_n / (C + X_n);
            sum_div2 += X_n * X_n / (C + X_n) / (C + X_n);
        }
        float A, B;
        //linear_HDR_AB(sum_X_n, sum_X_n2, sum_X_n3, sum_X_n4, A, B);
        div_HDR_AB(sum_X, sum_X2, sum_div, sum_div_2, sum_div2, A, B);
        B=max(B, 0.0);
        A=max(A, 0.0);
        luminanceSum = luminance(sumX2 / (w3 + 0.001));
        luminanceSum = pow(luminanceSum, 0.75*exp(-luminanceSum*0.1)+0.25);
        float exposure = clamp(calculateExposure(luminanceSum), 0.0025, 25.0);
        if (frameCounter <= 1) {
            avgExposure = exposure;
            HDR_AB_global = vec2(0, avgExposure);
        } else {
            avgExposure = exp(mix(
                        log(avgExposure),
                        log(exposure),
                        1 - exp(-dTime_global)
                    ));
            HDR_AB_global = mix(HDR_AB_global, vec2(A,B), 1 - exp(-dTime_global));
        }
        div_avgExposure = 1 / avgExposure;
    }

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    texCoord = gl_MultiTexCoord0.xy;
}
