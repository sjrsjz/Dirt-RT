#version 430
#include "/lib/common.glsl"

in vec2 texCoord;
#ifdef PASS0
uniform sampler2D colortex0;
/* RENDERTARGETS: 6 */
layout(location = 0) out vec4 fragColor;
#else
uniform sampler2D colortex0;
uniform sampler2D colortex6;
/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 fragColor;
#endif

void main() {
    // 高斯金字塔下采样
    #define R 2
    vec3 sumX = vec3(0);
    float w0 = 0;
    for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
            float w = exp(-(i * i + j * j) * 0.5);
            #ifdef PASS0
            sumX += w * texelFetch(colortex0, ivec2(gl_FragCoord.xy + ivec2(i, j)), 0).rgb;
            #else
            sumX += w * texelFetch(colortex6, ivec2(gl_FragCoord.xy + ivec2(i, j)), 0).rgb;
            #endif
            w0 += w;
        }
    }
    #ifdef PASS0
    fragColor.rgb = sumX / w0;
    #else
    fragColor.rgb = mix(sumX / w0, texture(colortex0, texCoord).rgb, 0.5);
    #endif
    if (any(isnan(fragColor.rgb))) fragColor.rgb = vec3(0);
}
