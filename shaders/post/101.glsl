#version 430 compatibility

#define REFLECT_BUFFER_MIN


#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos

in vec2 texCoord;

uniform sampler2D colortex0;


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

float svgfNormalWeight(vec3 centerNormal, vec3 normal, float distance) {
    return exp(-5*abs(length(centerNormal)-length(normal)))*clamp(pow(max(dot(normalize(centerNormal),normalize(normal)), 0.0), NORMAL_PARAM*(0.25*0+8+0*8*exp(-0.125*distance))),0,1);
//    clamp(exp(-5*length(centerNormal-normal)),0.,1.);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    // Modified to check for distance from the center plane
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)*(0.25+16*exp(-0.25*distance))));
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
float info_distance;
uint idx_l;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
    
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


vec3IllumiantionData data2;


void MixReflect() {
    if (notInRange(prevScreenPos.xy)) {
        data2.weight = 1;
        return;
    }
    
    vec3IllumiantionData data = sampleReflect(prevScreenPos.xy*textureSize(colortex0,0));

    float s = exp(-abs(denoiseBuffer.data[idx_l].reflectWeight-data.mixWeight))*float(denoiseBuffer.data[idx_l].distance > -0.5)* svgfNormalWeight(data.normal, data2.normal,info_distance)
             * svgfPositionWeight(data.pos, data2.pos, data2.normal,info_distance);
    s = pow((min(1, s + 0.56) - 0.56)/0.56,0.25);
    
    float prevW = data.weight;
    prevW = max(1, min(prevW * s + 1, ACCUMULATION_LENGTH*5));

    data2.data_swap = data.data+(data2.data_swap-data.data)/prevW;
    data2.weight = prevW;
}

void main() {
   //严重消耗性能，与200.glsl一同占据用时的1/4~1/3
    
    idx = getIdx(uvec2(gl_FragCoord.xy));

    info_distance = denoiseBuffer.data[idx].distance;
    data2=reflectIllumiantionBuffer.data[idx];
    if (info_distance < -0.5) {

        data2.weight = 1;
        data2.mixWeight = 0;
        WriteReflect(data2,ivec2(gl_FragCoord.xy));

        return;
    }
    prevScreenPos = reproject2(data2.pos);
    idx_l=getIdx(uvec2(prevScreenPos.xy*textureSize(colortex0,0)));

    MixReflect();
   // data2.weight=10;
    WriteReflect(data2,ivec2(gl_FragCoord.xy));
}
