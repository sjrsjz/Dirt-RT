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

const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 64.0;
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

    SH centerSH;
    ivec2 texSize = textureSize(colortex3, 0) - 1;
    centerSH.shY = texelFetch(colortex5, pix, 0);
    vec4 tex = texelFetch(colortex6, pix, 0);
    centerSH.CoCg = tex.xy;

    float centerW = clamp(tex.z / 16, 0, 4); //  pow(1.25,R0*clamp(tex.z / 32 - 1, 0, 2)*0.05) * (2 - abs(dot(denoiseBuffer.data[idx].rd,centerNormal))) * clamp(tex.z / 16, 1, 2);
    centerW += 0.1 * centerW * centerW;
    float weight = 0;
    float scale = centerW * 0.075 * tex.w * sqrt(avgExposure) / (0.35/(0.05*tex.w+1) + sqrt(tmp_.w)*avgExposure);
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
            float delta = exp(-scale * sqrt(sqrt(dot(delta_shY, delta_shY) + dot(delta_CoCg, delta_CoCg))));
            delta *= st[i][j];
            weight += delta;
            float w0 = svgfNormalWeight(centerNormal, texelFetch(colortex3, samplePos, 0).xyz)
                    * svgfPositionWeight(centerPos, texelFetch(colortex4, samplePos, 0).xyz, centerNormal)
                    * float(samplePos == clamp(samplePos, vec2(0), texSize)) * step(-0.5, denoiseBuffer.data[getIdx(uvec2(samplePos))].distance) * delta;

            accumulate_SH(A, tmp, w0);
            w += w0;
            samplePos.y += R0;
        }
        samplePos.x += R0;
    }

    tex.w = tex.w * 0.75 + 0.25 * weight;
    float w0 = 4 * step(-0.5, denoiseBuffer.data[getIdx(uvec2(pix))].distance);
    accumulate_SH(A, centerSH, w0);
    w += w0;

    if (any(isnan(A.shY))) A.shY = vec4(0);
    if (any(isnan(A.CoCg))) A.CoCg = vec2(0);
    SH tmp0 = scaleSH(A, 1 / max(w, 0.00001));

    shY = tmp0.shY;
    CoCg = vec4(tmp0.CoCg, tex.zw);

    //CoCg.w = tex.w+weight;//10 * luma(project_SH_irradiance(tmp0, diffuseIllumiantionBuffer.data[idx].normal)) / avgExposure * pow(2, R0);
}
