#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos

in vec2 texCoordRaw;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;
uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;

uniform float near;
uniform float far;
uniform vec2 resolution;
uniform int worldTime;

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

vec3 reproject(vec3 screenPos) {
    vec4 tmp = gbufferProjectionInverse * vec4(screenPos * 2.0 - 1.0, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 playerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = playerPos + cameraPosition;
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return (prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5);
}

vec3 reproject2(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}


/* RENDERTARGETS: 2,8 */

layout(location = 0) out vec4 Emission;
layout(location = 1) out vec4 Variance;

vec3 prevScreenPos;
bufferData info_;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
    //a;
}

/*void MixDiffuse() {
    diffuseIllumiantionData center = diffuseIllumiantionBuffer.data[idx];
    vec3 c0 = decodeSH(center.data_swap, center.normal2);
    const int S = 3;
    float w = 0;
    mat3x4 sumX = mat3x4(0);
    float w_M[2 * S + 1][2 * S + 1];
    uint idx_M[2 * S + 1][2 * S + 1];
    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            float a = exp(-0.1 * (i * i + j * j));
            uint idx2 = getIdx(uvec2(gl_FragCoord.xy + vec2(i, j)));
            diffuseIllumiantionData sample1 = diffuseIllumiantionBuffer.data[idx2];
            float w0 = float(denoiseBuffer.data[idx2].distance > -0.5) * svgfNormalWeight(sample1.normal, diffuseIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(sample1.pos, diffuseIllumiantionBuffer.data[idx].pos, diffuseIllumiantionBuffer.data[idx].normal) * a;
            w_M[i + S][j + S] = w0;
            idx_M[i + S][j + S] = idx;
            sumX += sample1.data_swap * w0;
            w += w0;
        }
    }
    sumX /= max(w, 0.001);
    vec3 c2 = decodeSH(sumX, center.normal2);
    sumX = mat3x4(0);
    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            uint idx2 = idx_M[i + S][j + S];
            diffuseIllumiantionData sample1 = diffuseIllumiantionBuffer.data[idx2];
            vec3 c1 = decodeSH(sample1.data_swap - sumX, sample1.normal2);
            float w0 = w_M[i + S][j + S] * exp(- 0.25*luma(pow(abs(c1), vec3(0.3))));
            sumX += sample1.data_swap * w0;
            w += w0;
        }
    }

    diffuseIllumiantionBuffer.data[idx].data_swap = sumX / max(w, 0.01);
}*/


void MixDiffuse() {
    diffuseIllumiantionData center = diffuseIllumiantionBuffer.data[idx];
    vec3 c0 = project_SH_irradiance(center.data_swap, center.normal2);
    const int S = 1;
    float w = 0;
    
    vec3 sumX=vec3(0);
    vec3 sumX2=vec3(0);
    float w_M[2 * S + 1][2 * S + 1];
    uint idx_M[2 * S + 1][2 * S + 1];
    vec3 c00[2 * S + 1][2 * S + 1];
    
    //ivec2 max_ij=ivec2(0);
    //float maxL=-1;
    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            float a = 1;//exp(-0.25 * (i * i + j * j));
            uint idx2 = getIdx(uvec2(gl_FragCoord.xy + vec2(i, j)));
            diffuseIllumiantionData sample1 = diffuseIllumiantionBuffer.data[idx2];
            vec3 c1 = project_SH_irradiance(sample1.data_swap, sample1.normal2);
            c00[i + S][j + S]=c1;
            float w0 = float(denoiseBuffer.data[idx2].distance > -0.5) * svgfNormalWeight(sample1.normal, diffuseIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(sample1.pos, diffuseIllumiantionBuffer.data[idx].pos, diffuseIllumiantionBuffer.data[idx].normal) * a;
            w_M[i + S][j + S] = w0;
            idx_M[i + S][j + S] = idx;
            //sumX += sample1.data_swap * w0;
            
            sumX+=c1*w0;
            sumX2+=c1*c1*w0;
            float L=luma(c1);
            //max_ij=L>maxL?ivec2(i,j):max_ij;
            //maxL=max(maxL,L);
            w += w0;
        }
    }
    sumX /= max(w, 0.001);
    sumX2 /= max(w, 0.001);
    vec3 sigma=0.25/(2*max(abs(sumX2-sumX*sumX),0.001));
    Variance.xyz=sigma;
    
    SH sumX_ = init_SH();
    w=0;
    for (int i = -S; i <= S; i++) {
        for (int j = -S; j <= S; j++) {
            uint idx2 = idx_M[i + S][j + S];
            diffuseIllumiantionData sample1 = diffuseIllumiantionBuffer.data[idx2];
            vec3 c1 = project_SH_irradiance(sample1.data_swap, sample1.normal2);
            vec3 dc=c1-sumX;
            vec3 p=exp(-pow(dot(dc,dc),1.25)*sigma);
            float w0 = w_M[i + S][j + S] * luma(p*vec3(greaterThan(p,vec3(0.))));
            accumulate_SH(sumX_, sample1.data_swap , w0);
            w += w0;
        }
    }

    diffuseIllumiantionBuffer.data[idx].data_swap =scaleSH(sumX_ , 1/max(w, 0.01));
}



void MixReflect() {
    if (notInRange(prevScreenPos.xy)) {
        reflectIllumiantionBuffer.data[idx].weight = 1;
        return;
    }
    vec3IllumiantionData data = sampleReflect(prevScreenPos.xy * texSize);

    float s = float(denoiseBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].distance > -0.5) * svgfNormalWeight(data.lnormal, reflectIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(data.lpos, reflectIllumiantionBuffer.data[idx].pos, reflectIllumiantionBuffer.data[idx].normal);

    float prevW = data.weight;
    prevW = max(1, min(prevW * s + 1, ACCUMULATION_LENGTH));

    reflectIllumiantionBuffer.data[idx].data_swap = data.data - (data.data_swap - reflectIllumiantionBuffer.data[idx].data_swap) / prevW;
    reflectIllumiantionBuffer.data[idx].weight = prevW;
}

void MixRefract() {
    if (notInRange(prevScreenPos.xy)) {
        refractIllumiantionBuffer.data[idx].weight = 1;
        return;
    }
    vec3IllumiantionData data = sampleRefract(prevScreenPos.xy * texSize);

    float s = float(denoiseBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].distance > -0.5) * svgfNormalWeight(data.lnormal, refractIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(data.lpos, refractIllumiantionBuffer.data[idx].pos, refractIllumiantionBuffer.data[idx].normal);

    float prevW = data.weight;
    prevW = max(1, min(prevW * s + 1, ACCUMULATION_LENGTH));

    refractIllumiantionBuffer.data[idx].data_swap = data.data - (data.data_swap - refractIllumiantionBuffer.data[idx].data_swap) / prevW;
    refractIllumiantionBuffer.data[idx].weight = prevW;
}

void main() {
    
    idx = getIdx(uvec2(gl_FragCoord.xy));
    
    
    info_ = denoiseBuffer.data[idx];
    Emission=vec4(info_.emission,0);
    //return;
    if (info_.distance < -0.5) {
        return;
    }
    MixDiffuse();
}
