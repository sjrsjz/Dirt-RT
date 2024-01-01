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

const float NORMAL_PARAM = 1.0;
const float POSITION_PARAM = 1.0;
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


/* RENDERTARGETS: 0 */

layout(location = 0) out vec4 fragColor;

vec3 prevScreenPos;
bufferData info_;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
}

/*void MixSample() {
    denoiseBuffer.data[idx].lastSample=denoiseBuffer.data[idx].currSample;
    return;
    if (notInRange(prevScreenPos.xy)) {
        denoiseBuffer.data[idx].lastSample = vec4(0);
        return;
    }
    denoiseBuffer.data[idx].lastSample=denoiseBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].currSample;
}*/

void MixDiffuse() {
    if (notInRange(prevScreenPos.xy)) {
        diffuseIllumiantionBuffer.data[idx].weight = 1;
        //float k = luma(decodeSH(diffuseIllumiantionBuffer.data[idx].data_swap, diffuseIllumiantionBuffer.data[idx].normal2));
        //diffuseIllumiantionBuffer.data[idx].sumX = k;
        //diffuseIllumiantionBuffer.data[idx].sumX2 = k * k;
        return;
    }
    
    diffuseIllumiantionData data = sampleDiffuse(prevScreenPos.xy * texSize);

    float s = float(denoiseBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].distance > -0.5) * svgfNormalWeight(data.lnormal, diffuseIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(data.lpos, diffuseIllumiantionBuffer.data[idx].pos, diffuseIllumiantionBuffer.data[idx].normal);
    s = (min(1, s + 0.875) - 0.875)*8;
    float prevW = data.weight;
    prevW = max(1, min(prevW * s + 1, ACCUMULATION_LENGTH*10));

    //float k = abs(luma(ACESFilm(avgExposure*decodeSH(diffuseIllumiantionBuffer.data[idx].data_swap,diffuseIllumiantionBuffer.data[idx].normal2))));
    //diffuseIllumiantionBuffer.data[idx].sumX=data.lsumX+(k-data.lsumX)/prevW;
    //diffuseIllumiantionBuffer.data[idx].sumX2=data.lsumX2+(k*k-data.lsumX2)/prevW;

    diffuseIllumiantionBuffer.data[idx].data_swap = mix_SH(data.data,diffuseIllumiantionBuffer.data[idx].data_swap,1/prevW);
    diffuseIllumiantionBuffer.data[idx].weight = prevW;
}

void MixReflect() {
    if (notInRange(prevScreenPos.xy)) {
        reflectIllumiantionBuffer.data[idx].weight = 1;
        return;
    }
    vec3IllumiantionData data = sampleReflect(prevScreenPos.xy * texSize);

    float s = float(reflectIllumiantionBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].distance > -0.5) * svgfNormalWeight(data.lnormal, reflectIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(data.lpos, reflectIllumiantionBuffer.data[idx].pos, reflectIllumiantionBuffer.data[idx].normal);
    s=float(reflectIllumiantionBuffer.data[idx].distance < -0.5&&reflectIllumiantionBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].distance < -0.5)*(1-s)+s;
    s = (min(1, s + 0.875) - 0.875)*8;
    float prevW = data.weight;
    prevW = max(1, min(prevW * s + 1, ACCUMULATION_LENGTH*10));

    reflectIllumiantionBuffer.data[idx].data_swap = data.data - (data.data - reflectIllumiantionBuffer.data[idx].data_swap) / prevW;
    reflectIllumiantionBuffer.data[idx].weight = prevW;
}

void MixRefract() {
    if (notInRange(prevScreenPos.xy)) {
        refractIllumiantionBuffer.data[idx].weight = 1;
        return;
    }
    vec3IllumiantionData data = sampleRefract(prevScreenPos.xy * texSize);

    float s = float(refractIllumiantionBuffer.data[getIdx(uvec2(prevScreenPos.xy * texSize))].distance > -0.5) * svgfNormalWeight(data.lnormal, refractIllumiantionBuffer.data[idx].normal) * svgfPositionWeight(data.lpos, refractIllumiantionBuffer.data[idx].pos, refractIllumiantionBuffer.data[idx].normal);

    s = (min(1, s + 0.875) - 0.875)*8;
    float prevW = data.weight;
    prevW = max(1, min(prevW * s + 1, ACCUMULATION_LENGTH*10));

    refractIllumiantionBuffer.data[idx].data_swap = data.data - (data.data - refractIllumiantionBuffer.data[idx].data_swap) / prevW;
    refractIllumiantionBuffer.data[idx].weight = prevW;
}

void main() {
   
    idx = getIdx(uvec2(gl_FragCoord.xy));

    info_ = denoiseBuffer.data[idx];

    if (info_.distance < -0.5) {
        diffuseIllumiantionBuffer.data[idx].weight = 0;
        reflectIllumiantionBuffer.data[idx].weight = 0;
        refractIllumiantionBuffer.data[idx].weight = 0;
        return;
    }
    texSize = textureSize(colortex0, 0);
    prevScreenPos = reproject2(diffuseIllumiantionBuffer.data[idx].pos);
    //MixSample();
    MixDiffuse();
    MixReflect();
    MixRefract();

}
