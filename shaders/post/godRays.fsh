#version 430

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex8;
uniform sampler2D colortex9;

/* RENDERTARGETS: 2 */
layout(location = 0) out vec4 fragColor;

void main() {
    //return;
    const int sampleN = 16;
    vec3 sumX = vec3(0);
    float w0 = 0;
    vec2 texSize = textureSize(colortex2, 0);
    for (int k = -sampleN; k <= sampleN; k++) {
        int i=k;//int(sign(k)*pow(abs(k),1.25));
        #if STEP==1
        float w = exp(-i * i * 0.015) * step(-0.5,denoiseBuffer.data[getIdx(uvec2(gl_FragCoord.xy+vec2(i, 0)))].distance) *
            float(clamp(gl_FragCoord.xy + vec2(i, 0), vec2(0), texSize) == gl_FragCoord.xy + vec2(i , 0));
        sumX += texelFetch(colortex2, ivec2(gl_FragCoord.xy + vec2(i, 0)), 0).xyz * w;
        #else
        float w = exp(-i * i * 0.015)*step(-0.5,denoiseBuffer.data[getIdx(uvec2(gl_FragCoord.xy+vec2(0, i)))].distance) *
        float(clamp(gl_FragCoord.xy + vec2(0, i), vec2(0), texSize) == gl_FragCoord.xy + vec2(0, i));
        sumX += texelFetch(colortex2, ivec2(gl_FragCoord.xy + vec2(0, i)), 0).xyz * w;
        #endif
        w0 += w;
    }
    sumX /= max(w0, 1e-3);
    if (any(isnan(sumX))) sumX = vec3(0);
    fragColor.xyz = sumX;
    #if STEP==2
    denoiseBuffer.data[getIdx(uvec2(gl_FragCoord.xy))].emission=sumX;
    #endif
}
