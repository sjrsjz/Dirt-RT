#version 430
#include "/lib/common.glsl"

in vec2 texCoord;
uniform sampler2D colortex0;
uniform sampler2D colortex1;

/* RENDERTARGETS: 1 */
layout(location = 0) out vec4 fragColor;

float kernel(float x) {
    return exp(-7.5 * pow(x + 0.00125, 0.25) - 1.5 * pow(x, 0.025)) * sqrt(1 - x * x);
}

void main() {
    //return;
    #ifdef SRR_
    if (texCoord.x > 0.5 || texCoord.y > 0.5) return;
    #endif
    const int sampleN = 32;
    vec3 sumX = vec3(0);
    float angleOffset = rand(texCoord - 400) * 32 * PI;
    const float angleShift = 2 * PI / sampleN;
    float R = rand(texCoord * 100 + 300);
    
    float weight = kernel(R) / kernel(0);

    R *= 0.25;
    mat2 rotM = mat2(cos(angleShift), sin(angleShift), -sin(angleShift), cos(angleShift));
    vec2 v = vec2(cos(angleOffset), sin(angleOffset)) * R * textureSize(colortex1, 0).x;
    vec3 sumX_2 = vec3(0), sumX2 = vec3(0);
    vec3 s[sampleN];
    for (int i = 0; i < sampleN; i++) {
        v *= rotM;
        vec3 A = texelFetch(colortex0, ivec2(gl_FragCoord.xy + v), 0).xyz;
        sumX += A;
        sumX2 += A * A;
        s[i] = A;
    }
    vec3 validN3 = vec3(0);
    sumX/=sampleN;
    for (int i = 0; i < sampleN; i++) {
        vec3 w = exp(-pow(abs(s[i].xyz - sumX), vec3(0.3)));
        sumX_2 += s[i].xyz * w;
        validN3 += w;
    }
    sumX_2 /= max(validN3, 1e-5);
    fragColor.xyz = sumX_2 * weight;
    if (any(isnan(fragColor.xyz))) fragColor.xyz = vec3(0);
}
