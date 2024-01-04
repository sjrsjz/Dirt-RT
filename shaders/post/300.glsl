#version 430 compatibility
#define DIFFUSE_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

//2,3,4,5,6,7,8,9

//2:pos

uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;
uniform sampler2D colortex6;


/*
const int colortex0Format = RGBA32F;
const int colortex1Format = RGBA32F;
const int colortex2Format = RGBA32F;
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA32F;
const int colortex6Format = RGBA32F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;


const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
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


/* RENDERTARGETS: 0,5,6 */

layout(location = 0) out vec4 variance;
layout(location = 1) out vec4 shY;
layout(location = 2) out vec4 CoCg;


float K(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    float ab = dot(A, B);
    vec3 x = an * B - bn * A;
    return abs(bn) * sqrt(1 - an * an) / max(0.01, dot(x, x));
}

void main() {
    //shY=texelFetch(colortex5,ivec2(gl_FragCoord.xy),0);
    //CoCg=texelFetch(colortex6,ivec2(gl_FragCoord.xy),0);
    //return;
        //WriteDiffuse(fetchDiffuse(ivec2(gl_FragCoord.xy)),ivec2(gl_FragCoord.xy));
    uint idx = getIdx(uvec2(gl_FragCoord.xy));

    bufferData info_ = denoiseBuffer.data[idx];
    if (info_.distance < -0.5) {
        return;
    }

    SH A = init_SH();
    float s[3] = { 1, 2, 1 };
    float t[3] = { 1, 2, 1 };

    float w = 0;
    vec3 centerNormal=texelFetch(colortex3,ivec2(gl_FragCoord.xy),0).xyz;
    vec3 centerPos=texelFetch(colortex4,ivec2(gl_FragCoord.xy),0).xyz;
    s[0] = 0.5+K(cross(camX_global, camY_global), camX_global, centerNormal);
    t[0] = 0.5+K(cross(camX_global, camY_global), camY_global, centerNormal);
    s[2] = s[0];
    t[2] = t[0];

    
    //float isNotBackground[9];
    //uint idx_M[9];
    ivec2 samplePos;
    samplePos.x=int(gl_FragCoord.x-R0);
    //int k=0;
    SH centerSH;
    centerSH.shY=texelFetch(colortex5,ivec2(gl_FragCoord.xy),0);
    centerSH.CoCg=texelFetch(colortex6,ivec2(gl_FragCoord.xy),0).xy;
    float centerW=min(pow(fetchDiffuse(ivec2(gl_FragCoord.xy)).weight,1.5)*0.003,0.2)*pow(R0,0.875);
    for (int i = 0; i <= 2; i++) {
        samplePos.y=int(gl_FragCoord.y-R0);
        for (int j = 0; j <= 2; j++) {
            uint idx2=getIdx(uvec2(samplePos));
            float w1 =s[i] * t[j]* step(-0.5,denoiseBuffer.data[idx2].distance);
            
            //idx_M[k]=idx2;
            
            SH tmp;
            tmp.shY=texelFetch(colortex5,samplePos,0);
            tmp.CoCg=texelFetch(colortex6,samplePos,0).xy;
            float dL=centerW*sqrt((dot(tmp.shY-centerSH.shY,tmp.shY-centerSH.shY)+dot(tmp.CoCg-centerSH.CoCg,tmp.CoCg-centerSH.CoCg)));
            float w0 = svgfNormalWeight(centerNormal, texelFetch(colortex3,(samplePos),0).xyz)
                     * svgfPositionWeight(centerPos, texelFetch(colortex4,samplePos,0).xyz, centerNormal)
                     * exp(-dL)
                     * w1;

            
            accumulate_SH(A,tmp,w0);
            //isNotBackground[k] = w1;
            w += w0;
            //k++;
            samplePos.y+=R0;
        }
        samplePos.x+=R0;
    }
    if (any(isnan(A.shY))) A.shY = vec4(0);
    if (any(isnan(A.CoCg))) A.CoCg = vec2(0);
    SH tmp=scaleSH(A , 1/max(w, 0.01));
    shY=tmp.shY;
    CoCg.xy=tmp.CoCg;

}
