#version 430 compatibility

#define DIFFUSE_BUFFER_MIN

layout(local_size_x = 16,local_size_y = 16) in;
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos




uniform sampler2D colortex0;
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

const bool colortex0Clear = true;
const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

const float NORMAL_PARAM = 2.0;
const float POSITION_PARAM = 256.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal, float distance) { 
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM*(0.25+4*exp(-0.25*distance)));
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    // Modified to check for distance from the center plane
    return max(exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)*(0.1+64*exp(-0.25*distance))))*2,0);
}

vec3 reproject(vec3 screenPos) {
    vec4 tmp = gbufferProjectionInverse * vec4(screenPos * 2.0 - 1.0, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 playerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 worldPos = playerPos + cameraPosition;
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}

vec3 reproject2(vec3 worldPos) {
    vec3 prevPlayerPos = worldPos - previousCameraPosition;
    vec3 prevViewPos = (gbufferPreviousModelView * vec4(prevPlayerPos, 1.0)).xyz;
    vec4 prevClipPos = gbufferPreviousProjection * vec4(prevViewPos, 1.0);
    return prevClipPos.xyz / prevClipPos.w * 0.5 + 0.5;
}



vec3 prevScreenPos;
float info_distance;
uint idx_l;
vec2 texSize;
uint idx;
vec3 curr_rd;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
    
}

diffuseIllumiantionData data1;

// variance estimation
float updateVariance(SH M_n, float D_n, SH X_nplus1, float w) { // w is the weight of the history average
    vec2 diff_CoCg = X_nplus1.CoCg - M_n.CoCg;
    vec4 diff_shY = X_nplus1.shY - M_n.shY;
    return (D_n*w + (dot(diff_CoCg,diff_CoCg)+dot(diff_shY,diff_shY)))/(1 + (w+1)*w);
}


void MixDiffuse() {
    if (notInRange(prevScreenPos.xy)) {
        data1.weight = 1;
        return;
    }
    diffuseIllumiantionData data = sampleDiffuse(prevScreenPos.xy*textureSize(colortex0,0));

    float prev_dot = denoiseBuffer.data[idx].last_rd_dot_n;
    float curr_dot = -dot(data1.normal, curr_rd);
    float s0 = exp(-16*max(0,curr_dot-prev_dot));// 当夹角变小时，说明历史信息不可信
    denoiseBuffer.data[idx].last_rd_dot_n = curr_dot;

    float s =  s0 *float(denoiseBuffer.data[idx_l].distance > -0.5) * svgfPositionWeight(data.pos, data1.pos, data1.normal,info_distance) * svgfNormalWeight(data.normal, data1.normal,info_distance) ;
    s = pow((min(1, s + 0.5) - 0.5)/0.5,0.125);
    //s = (min(1, s + 0.875) - 0.875)*8;
    float prevW = data.weight;
    prevW *= s;
    data1.variance = updateVariance(data1.data, data1.variance, data.data, prevW);
    prevW = clamp(prevW + 1,1,ACCUMULATION_LENGTH);

    data1.data_swap = mix_SH(data.data,data1.data_swap,1/prevW);

    data1.weight = prevW;
}


void main() {
   //严重消耗性能，与200.glsl一同占据用时的1/4~1/3
    //vec2 texCoord=vec2(gl_GlobalInvocationID.xy)/(textureSize(colortex0,0));
    idx = getIdx(uvec2(gl_GlobalInvocationID.xy));

    info_distance = denoiseBuffer.data[idx].distance;
    curr_rd = normalize(denoiseBuffer.data[idx].rd);
    data1=diffuseIllumiantionBuffer.data[idx];
    if (info_distance < -0.5) {
        data1.weight = 1;
        WriteDiffuse(data1,ivec2(gl_GlobalInvocationID.xy));
        return;
    }
    prevScreenPos = reproject2(data1.pos);
    idx_l=getIdx(uvec2(prevScreenPos.xy*textureSize(colortex0,0)));
    MixDiffuse();
    WriteDiffuse(data1,ivec2(gl_GlobalInvocationID.xy));
}
