#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos
uniform sampler2D colortex0;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex6;



/*
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA16F;
const int colortex6Format = RGBA16F;
*/

const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;

float S_tex_w;
const float NORMAL_PARAM = 1;//64.0;
const float POSITION_PARAM = 16.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal) {
    return max(dot(centerNormal, normal), 0.0);
}

/* RENDERTARGETS: 5,6 */

layout(location = 0) out vec4 shY;
layout(location = 1) out vec4 CoCg;


float updateVariance(SH M_n, float D_n, SH X_nplus1, float w) { // w is the weight of the history average
    vec2 diff_CoCg = X_nplus1.CoCg - M_n.CoCg;
    vec4 diff_shY = X_nplus1.shY - M_n.shY;
    float w1 = 1.0 / (1.0 + w);
    return w*(D_n*w + (dot(diff_CoCg,diff_CoCg)+dot(diff_shY,diff_shY))*w1)*w1*w1;
}

void main() {
    //shY=texelFetch(colortex5,ivec2(gl_FragCoord.xy),0);
    //CoCg=texelFetch(colortex6,ivec2(gl_FragCoord.xy),0);
    //return;
    uint idx = getIdx(uvec2(gl_FragCoord.xy));

    bufferData info_ = denoiseBuffer.data[idx];
    if (info_.distance < -0.5) {
        return;
    }

    SH A = init_SH();

    mediump float w = 1;
    ivec2 pix=ivec2(gl_FragCoord.xy);
    lowp vec3 centerNormal = texelFetch(colortex3, pix, 0).xyz;
    vec4 tmp_=texelFetch(colortex4, pix, 0);
    vec3 centerPos = tmp_.xyz;

    ivec2 samplePos;

    SH centerSH;// avg_SH;

    ivec2 texSize = textureSize(colortex3, 0) - 1;
    centerSH.shY = texelFetch(colortex5, pix, 0);
    vec4 tex = texelFetch(colortex6, pix, 0);
    
    centerSH.CoCg = tex.xy;
    mediump float scale = 0.025 * clamp(tex.z + 0.5*tex.z*tex.z, 0, 100)  / (1 + 5 * tex.w)* float(R0>2) * pow(R0,2) * sqrt(avgExposure) * abs(dot(centerNormal,info_.rd));
    
    S_tex_w = tex.w;
    mediump float D = 0;

    SH avg_SH = centerSH;
#if STEP == 1
    #define A_ 6
    #define B_ 3
#else
    #define A_ 4
    #define B_ 2
#endif

    #define totalIterations (A_ + 1) * (A_ + 1)
    #if STEP == 1
    mediump float sum_D=0;
    #endif
 
    mediump float pos_scale = POSITION_PARAM/(max(1,0.005*info_.distance));
 

    ivec2 rand_offset = ivec2(round(rand(vec2(pix))*2-1),round(rand(vec2(pix+10))*2-1));


    SH tmp;
    for (int i = -B_; i <= B_; i++) {
        for (int j = -B_; j <= B_; j++) {
            if (i==0 && j==0) {
                continue;
            }
            samplePos = pix + R0 * ivec2(i,j) + rand_offset;
            
            tmp.shY = texelFetch(colortex5, samplePos, 0);
            vec4 CoCgWV = texelFetch(colortex6, samplePos, 0);
            tmp.CoCg = CoCgWV.xy;
            
            mediump vec4 delta_shY = tmp.shY - centerSH.shY;

            float d = denoiseBuffer.data[getIdx(uvec2(samplePos))].distance;
            
            vec3 sampleNormal = texelFetch(colortex3, samplePos, 0).xyz;

#if STEP == 1
            mediump float k = abs(dot(texelFetch(colortex4, samplePos, 0).xyz - centerPos, centerNormal)) + 0.3;
            mediump float w1 = exp(-sqrt(R0)*0.025*(i*i+j*j) - k * k * pos_scale - scale * dot(delta_shY,delta_shY)/ (tex.w) * tex.z);
#else
            mediump float k = abs(dot(texelFetch(colortex4, samplePos, 0).xyz - centerPos, centerNormal)) + 0.3;
            mediump float w1 = exp(-sqrt(R0)*0.025*(i*i+j*j) - k * k * pos_scale - scale * dot(delta_shY,delta_shY));
#endif

            mediump float w0 = max(dot(centerNormal, sampleNormal),0)
                    * w1
                    * float(samplePos == clamp(samplePos, vec2(0), texSize)) * step(-0.5, d);
            D = updateVariance(avg_SH, D, tmp, w);
            accumulate_SH(avg_SH, tmp, w0);
            w += w0;
        
        #if STEP == 1
            sum_D += CoCgWV.w * w0;
        #endif
        }
    }
#if STEP == 1
    //accumulate_SH(avg_SH, centerSH, D * sqrt(avgExposure) * 10.25);
    //w += D * sqrt(avgExposure) * 10.25;
#endif
    //avg_SH = centerSH;
    //w = 1;
    #if STEP == 1
    tex.w = max(tex.w  + (sum_D)/(w+1e-3), 10 * D);
    #else
    tex.w = (tex.w + D) * 0.5;

    #endif

    if (any(isnan(avg_SH.shY))) avg_SH.shY = vec4(0);
    if (any(isnan(avg_SH.CoCg))) avg_SH.CoCg = vec2(0);
    avg_SH = scaleSH(avg_SH, 1 / w);

    shY = avg_SH.shY;
    CoCg = vec4(avg_SH.CoCg, tex.zw);
}
