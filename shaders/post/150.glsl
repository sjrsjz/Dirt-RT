#version 430 compatibility
#define DIFFUSE_BUFFER_MIN
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos

in vec2 texCoord;


/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex6Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;

const bool colortex0Clear = true;
const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

const float NORMAL_PARAM = 4.0;
const float POSITION_PARAM = 4.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal) {
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    // Modified to check for distance from the center plane
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

/* RENDERTARGETS: 5,6 */

layout(location = 0) out vec4 shY;
layout(location = 1) out vec4 CoCg;

vec3 prevScreenPos;
bufferData info_;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
}
uniform sampler2D colortex0;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex6;

// variance estimation
float updateVariance(SH M_n, float D_n, SH X_nplus1, float w) { // w is the weight of the history average
    vec2 diff_CoCg = X_nplus1.CoCg - M_n.CoCg;
    vec4 diff_shY = X_nplus1.shY - M_n.shY;
    return w*(D_n + (dot(diff_CoCg,diff_CoCg)+dot(diff_shY,diff_shY))/(1+w))/(1+w);
}


SH fetchSH(ivec2 coord) {
    SH sh;
    sh.shY = texelFetch(colortex5, coord, 0);
    sh.CoCg = texelFetch(colortex6, coord, 0).xy;
    return sh;
}

void writeSH(SH sh, ivec2 coord) {
    shY = sh.shY;
    CoCg = vec4(sh.CoCg, texelFetch(colortex6, coord, 0).zw);
}

void MixDiffuse() {
    SH centerSH = fetchSH(ivec2(gl_FragCoord.xy));
    const int S = 2;
    float w = 0;

    ivec2 texSize = ivec2(textureSize(colortex0, 0));
    SH avgSH = init_SH();
    float D=0;

    vec3 centerNormal = texelFetch(colortex3, ivec2(gl_FragCoord.xy), 0).xyz;
    vec3 centerPos = texelFetch(colortex4, ivec2(gl_FragCoord.xy), 0).xyz;

    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            if (i == 0 && j == 0) continue;
            ivec2 pix = ivec2(gl_FragCoord.xy) + ivec2(i, j);
            uint idx2 = getIdx(uvec2(pix));
            SH sample1 = fetchSH(pix);

            vec3 sampleNormal = texelFetch(colortex3, pix, 0).xyz;
            vec3 samplePos = texelFetch(colortex4, pix, 0).xyz;

            float w0 = float(denoiseBuffer.data[idx2].distance > -0.5) * svgfNormalWeight(sampleNormal, centerNormal) * svgfPositionWeight(samplePos, centerPos, centerNormal);
            w0 *= float(!notInRange(pix));
            D = updateVariance(avgSH, D, sample1, 1.0/max(1e-1,w0));
            accumulate_SH(avgSH, sample1, w0);
            w += w0;
        }
    }
    //D = max(D, 0.001);
    avgSH = scaleSH(avgSH, 1 / (w + 0.01));
    
    float diff = dot(centerSH.shY - avgSH.shY, centerSH.shY - avgSH.shY) + dot(centerSH.CoCg - avgSH.CoCg, centerSH.CoCg - avgSH.CoCg);
    float sigma = 10*(D);
    if (diff > sigma) {
        centerSH = avgSH;
    }

    writeSH(centerSH, ivec2(gl_FragCoord.xy));
}



void main() {
    texSize = vec2(textureSize(colortex0, 0));
    MixDiffuse();
}
