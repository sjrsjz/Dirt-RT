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
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA32F;


const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
/*

const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA32F;


const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
/*

const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA32F;


const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
/*

const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA32F;


const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
*/
float S_tex_w;
const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 64.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal) {
    return pow(max(dot(centerNormal, normal), 0.0), clamp(S_tex_w,1,NORMAL_PARAM));
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    // Modified to check for distance from the center plane
    return exp(-clamp(S_tex_w,1,POSITION_PARAM) * abs(dot(pixelPos - centerPos, normal)));
}

/* RENDERTARGETS: 5,6 */

layout(location = 0) out vec4 shY;
layout(location = 1) out vec4 CoCg;

float K(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    float ab = dot(A, B);
    vec3 x = an * B - bn * A;
    return abs(bn) * sqrt(1 - an * an) / max(0.01, dot(x, x));
}

// variance estimation
float updateVariance(SH M_n, float D_n, SH X_nplus1, float w) { // w is the weight of the history average
    vec2 diff_CoCg = X_nplus1.CoCg - M_n.CoCg;
    vec4 diff_shY = X_nplus1.shY - M_n.shY;
    float _1_div_w = 1 / (1 + w);
    return (D_n + (dot(diff_CoCg,diff_CoCg)+dot(diff_shY,diff_shY))*_1_div_w)*(1-_1_div_w);
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
    const float st[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };
    float w = 0;
    ivec2 pix=ivec2(gl_FragCoord.xy);
    vec3 centerNormal = texelFetch(colortex3, pix, 0).xyz;
    vec4 tmp_=texelFetch(colortex4, pix, 0);
    vec3 centerPos = tmp_.xyz;

    ivec2 samplePos;
    samplePos.x = pix.x - R0;

    SH centerSH;// avg_SH;

    ivec2 texSize = textureSize(colortex3, 0) - 1;
    centerSH.shY = texelFetch(colortex5, pix, 0);
    vec4 tex = texelFetch(colortex6, pix, 0);
    
    centerSH.CoCg = tex.xy;
//    float centerW = clamp(tex.z*0.5, 0, 5)+clamp((tex.z-5)*0.5, 0, 10); 

    float centerW = clamp(tex.z*0.25, 0, 2.5)+pow(clamp((tex.z-5), 0, 30),0.75); 
    //float weight = 0;
    //float scale = centerW/(2+6*exp(-0.1*avgExposure)/(0.01+avgExposure)+tex.w * (avgExposure)) * (0.5 * log(R0)+1)/(0.25+0.75*pow(abs(dot(info_.rd,centerNormal)),1));// centerW *35 / (1+tex.w)*tex.z/(20+tex.z) / (1+avgExposure) ;//(100/(tex.z+1)+1000*tex.w);// * sqrt(avgExposure) / (30 / (0.5*tex.w+1) + tmp_.w * avgExposure );
    
    float scale = centerW/(1+10*exp(-0.1*avgExposure)/(0.01+avgExposure)+0.025*tex.w * (avgExposure)) * (pow(R0,0.75)+1)/(0.5+0.5*pow(abs(dot(info_.rd,centerNormal)),1));// centerW *35 / (1+tex.w)*tex.z/(20+tex.z) / (1+avgExposure) ;//(100/(tex.z+1)+1000*tex.w);// * sqrt(avgExposure) / (30 / (0.5*tex.w+1) + tmp_.w * avgExposure );
    
    S_tex_w = tex.w;
    float D = 0;


    SH avg_SH = centerSH;
    for (int i = 0; i <= 2; i++) {
        samplePos.y = pix.y - R0;
        for (int j = 0; j <= 2; j++) {
            if (i == 1 && j == 1) {
                samplePos.y += R0;
                continue;
            }
            SH tmp;
            tmp.shY = texelFetch(colortex5, samplePos, 0);
            vec4 C0 = texelFetch(colortex6, samplePos, 0);
            tmp.CoCg = C0.xy;
            vec4 delta_shY = tmp.shY - centerSH.shY;
            vec2 delta_CoCg = tmp.CoCg - centerSH.CoCg;
            float delta = st[i][j] / (1 + scale * (sqrt(dot(delta_shY, delta_shY) + dot(delta_CoCg, delta_CoCg))));
            //weight += delta;
            float w0 = svgfNormalWeight(centerNormal, texelFetch(colortex3, samplePos, 0).xyz)
                    * svgfPositionWeight(centerPos, texelFetch(colortex4, samplePos, 0).xyz, centerNormal)
                    * float(samplePos == clamp(samplePos, vec2(0), texSize)) * step(-0.5, denoiseBuffer.data[getIdx(uvec2(samplePos))].distance) * delta;
            D = updateVariance(avg_SH, D, tmp, (1+w)/(w0+1e-2));
            accumulate_SH(avg_SH, tmp, w0);
            w += w0;
            samplePos.y += R0;
        }
        samplePos.x += R0;
    }
    //D=100*(D);
    avg_SH.shY-=centerSH.shY;
    avg_SH.CoCg-=centerSH.CoCg;

    //D *= avgExposure;
    //tex.w = max(tex.w,1.25*sqrt(D));//tex.w*0.25 + D*0.75;//+(1*D-tex.w)*exp(-max(tex.z,0)*0.0);//*0.5+tex.w * 0.5;
    
    tex.w = max(tex.w,1*sqrt(D));//tex.w*0.25 + D*0.75;//+(1*D-tex.w)*exp(-max(tex.z,0)*0.0);//*0.5+tex.w * 0.5;
    
    float w0 = 4 /(1+10*tmp_.w*avgExposure);
    accumulate_SH(avg_SH, centerSH, w0);
    w += w0;

    if (any(isnan(avg_SH.shY))) avg_SH.shY = vec4(0);
    if (any(isnan(avg_SH.CoCg))) avg_SH.CoCg = vec2(0);
    avg_SH = scaleSH(avg_SH, 1 / (w + 0.001));

    shY = avg_SH.shY;
    CoCg = vec4(avg_SH.CoCg, tex.zw);

    //CoCg.w = tex.w+weight;//10 * luma(project_SH_irradiance(tmp0, diffuseIllumiantionBuffer.data[idx].normal)) / avgExposure * pow(2, R0);
}
