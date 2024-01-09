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

const bool colortex0Clear = false;
const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = false;
const bool colortex8Clear = false;
*/

const float NORMAL_PARAM = 128.0;
const float POSITION_PARAM = 16.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal) {
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    // Modified to check for distance from the center plane
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

/* RENDERTARGETS: 2 */

layout(location = 0) out vec4 Emission;

vec3 prevScreenPos;
bufferData info_;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
}

void MixDiffuse() {
    diffuseIllumiantionData center = sampleDiffuse(gl_FragCoord.xy);
    vec3 c0 = project_SH_irradiance(center.data_swap, center.normal2);
    const int S = 0;
    float w = 0;

    vec3 sumX = vec3(0);
    vec3 sumX2 = vec3(0);
    float w_M[2 * S + 1][2 * S + 1];
    uint idx_M[2 * S + 1][2 * S + 1];
    vec3 c00[2 * S + 1][2 * S + 1];

    //ivec2 max_ij=ivec2(0);
    //float maxL=-1;
    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            float a = 1; //exp(-0.25 * (i * i + j * j));
            uint idx2 = getIdx(uvec2(gl_FragCoord.xy + vec2(i, j)));
            diffuseIllumiantionData sample1 = fetchDiffuse(ivec2(gl_FragCoord.xy + vec2(i, j)));
            vec3 c1 = project_SH_irradiance(sample1.data_swap, sample1.normal2);
            c00[i + S][j + S] = c1;
            float w0 = float(denoiseBuffer.data[idx2].distance > -0.5) * svgfNormalWeight(sample1.normal, center.normal) * svgfPositionWeight(sample1.pos, center.pos, center.normal) * a;
            w_M[i + S][j + S] = w0;
            sumX += c1 * w0;
            sumX2 += c1 * c1 * w0;
            float L = luma(c1);
            w += w0;
        }
    }
    w = 1 / max(w, 1e-3);
    sumX *= w;
    sumX2 *= w;
    vec3 sigma = 0.25 / (2 * max(abs(sumX2 - sumX * sumX), 0.001));

    SH sumX_ = init_SH();
    w = 0;
    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            diffuseIllumiantionData sample1 = fetchDiffuse(ivec2(gl_FragCoord.xy + vec2(i, j)));
            vec3 c1 = project_SH_irradiance(sample1.data_swap, sample1.normal2);
            vec3 dc = c1 - sumX;
            vec3 p = exp(-pow(dot(dc, dc), 1.25) * sigma);
            float w0 = w_M[i + S][j + S] * luma(p * vec3(greaterThan(p, vec3(0.))));
            accumulate_SH(sumX_, sample1.data_swap, w0);
            w += w0;
        }
    }

    center.data_swap = scaleSH(sumX_, 1 / max(w, 0.01));
    WriteDiffuse(center, ivec2(gl_FragCoord.xy));
}

void main() {
    //return;
    idx = getIdx(uvec2(gl_FragCoord.xy));
    info_ = denoiseBuffer.data[idx];
    Emission = vec4(info_.emission, 0);
    //return;
    /*if (info_.distance < -0.5) {
            return;
    /*if (info_.distance < -0.5) {
        return;
    }*/
    // MixDiffuse();
}
