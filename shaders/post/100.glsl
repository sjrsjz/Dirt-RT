#version 430 compatibility

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

//layout(local_size_x = 16,local_size_y = 16) in;
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
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;

const bool colortex0Clear = true;
const bool colortex1Clear = false;
const bool colortex2Clear = false;

const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

const float NORMAL_PARAM = 32.0;
const float POSITION_PARAM = 256.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal, float distance) { 
    return pow(max(dot(centerNormal, normal), 0.0), NORMAL_PARAM*(0.25+4*exp(-0.25*distance)));
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal, float distance) {
    // Modified to check for distance from the center plane
    return exp(-(pow(POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)/sqrt(distance)),4)));
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

in vec2 texCoord;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), vec2(1)) != p;
    
}

diffuseIllumiantionBufferData data1;
diffuseIllumiantionData out_data;

// variance estimation
float updateVariance(SH M_n, float D_n, SH X_nplus1, float w) { // w is the weight of the history average
    vec2 diff_CoCg = X_nplus1.CoCg - M_n.CoCg;
    vec4 diff_shY = X_nplus1.shY - M_n.shY;
    float w1 = 1.0 / (1.0 + w);
    return w*(D_n*w + (dot(diff_CoCg,diff_CoCg)+dot(diff_shY,diff_shY))*w1)*w1*w1;
}

float output_weight = 0;
float output_variance = 0;

void MixDiffuse() {
    if (notInRange(prevScreenPos.xy)) {
        return;
    }
    vec2 prev_screen = prevScreenPos.xy * textureSize(colortex0,0);
    diffuseIllumiantionData data = sampleDiffuse(prev_screen);

    float pos_weight = 0;
    for(int i=-1;i<=1;i++){
        for(int j=-1;j<=1;j++){
            vec2 offset = vec2(i,j);
            vec3 pos = sampleDiffusePos(prev_screen + offset);
            float w = svgfPositionWeight(pos, data1.pos, data1.normal,info_distance);
            //pos_weight += w*w;
            pos_weight = max(pos_weight, w);
        }

    }
    //pos_weight = sqrt(pos_weight/9);  // 实际上这玩意成了一种几何边缘检测，也许可以用来阻止降噪器在几何边缘失效的问题

    //diffuseIllumiantionData data = sampleDiffuse(prevScreenPos.xy*textureSize(colortex0,0)-0.5);
    


    float s =  float(denoiseBuffer.data[idx].distance > -0.5)*pos_weight;
                  //* svgfPositionWeight(data.pos, data1.pos, data1.normal,info_distance);
    float prevW = data.prev_weight;
    prevW *= s;

    output_variance = min(100, updateVariance(data1.data_swap, data.prev_variance, data.data, prevW));
    prevW = clamp(prevW + 1,1,max(ACCUMULATION_LENGTH,50*pow(output_variance*avgExposure,-0.125)));

    out_data.data_swap = mix_SH(data.data,data1.data_swap,1/prevW);

    output_weight = prevW;
}
/* RENDERTARGETS: 5 */
layout(location = 0) out vec4 output_data;

layout(rgba32f) uniform image2D extInfoBuffer;

void main() {
    // if(gl_FragCoord.x > resolution.x/2 || gl_FragCoord.y > resolution.y/2) {
    //     return;
    // }


    // uvec2 pix = uvec2(gl_FragCoord.xy * 2);
    uvec2 pix = uvec2(gl_FragCoord.xy);
    
    idx = getIdx(pix);

    info_distance = denoiseBuffer.data[idx].distance;
    curr_rd = normalize(denoiseBuffer.data[idx].rd);
    data1 = diffuseIllumiantionBuffer.data[idx];

    //
    // accumulate_SH(data1.data_swap, diffuseIllumiantionBuffer.data[getIdx(uvec2(gl_FragCoord.xy) * 2 + uvec2(0,1))].data_swap, 1.);
    // accumulate_SH(data1.data_swap, diffuseIllumiantionBuffer.data[getIdx(uvec2(gl_FragCoord.xy) * 2 + uvec2(1,0))].data_swap, 1.);
    // accumulate_SH(data1.data_swap, diffuseIllumiantionBuffer.data[getIdx(uvec2(gl_FragCoord.xy) * 2 + uvec2(1,1))].data_swap, 1.);
    // data1.data_swap = scaleSH(data1.data_swap, 1.0/4.0);

    out_data.data_swap = data1.data_swap;
    out_data.data = init_SH();
    out_data.normal = data1.normal;
    out_data.normal2 = data1.normal2;
    out_data.pos = data1.pos;
    output_weight = 1;
    output_variance = 0;
    if (info_distance < -0.5) {
        WriteDiffuse(out_data,ivec2(gl_FragCoord.xy));
        imageStore(extInfoBuffer,ivec2(gl_FragCoord.xy),vec4(output_weight,output_variance,0,0));
        return;
    }
    prevScreenPos = reproject2(data1.pos);
    //prevScreenPos = reproject2(data1.pos) * vec3(0.5,0.5,1);
    
    idx_l=getIdx(uvec2(prevScreenPos.xy*textureSize(colortex0,0)+0.5));
    MixDiffuse();

    imageStore(extInfoBuffer,ivec2(gl_FragCoord.xy),vec4(output_weight,output_variance,0,0));

    WriteDiffuse(out_data,ivec2(gl_FragCoord.xy));

}
