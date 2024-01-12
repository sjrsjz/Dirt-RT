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
    float s[3] = { 1, 2, 1 };
    float t[3] = { 1, 2, 1 };

    float w = 0;
    vec3 centerNormal = texelFetch(colortex3, ivec2(gl_FragCoord.xy), 0).xyz;
    vec3 centerPos = texelFetch(colortex4, ivec2(gl_FragCoord.xy), 0).xyz;
    s[0] = 0.75 + K(cross(camX_global, camY_global), camX_global, centerNormal);
    t[0] = 0.75 + K(cross(camX_global, camY_global), camY_global, centerNormal);
    s[2] = s[0];
    t[2] = t[0];
    s[1] = 2 * clamp(pow(texelFetch(colortex6, ivec2(gl_FragCoord.xy), 0).z, 0.25), 1, 50);
    t[1] = s[1];

    ivec2 samplePos;
    samplePos.x = int(gl_FragCoord.x - R0);

    SH centerSH;
    ivec2 texSize = textureSize(colortex3, 0);
    centerSH.shY = texelFetch(colortex5, ivec2(gl_FragCoord.xy), 0);
    vec3 tex = texelFetch(colortex6, ivec2(gl_FragCoord.xy), 0).xyz;
    centerSH.CoCg = tex.xy;
    float centerW = min(pow(max(tex.z-5,0), 1.25) * 0.002, 0.2) * pow(R0, 0.875) * 0.2;
    for (int i = 0; i <= 2; i++) {
        samplePos.y = int(gl_FragCoord.y - R0);
        for (int j = 0; j <= 2; j++) {
            uint idx2 = getIdx(uvec2(samplePos));
            float w1 = s[i] * t[j] * step(-0.5, denoiseBuffer.data[idx2].distance);

            SH tmp;
            tmp.shY = texelFetch(colortex5, samplePos, 0);
            tmp.CoCg = texelFetch(colortex6, samplePos, 0).xy;
            float dL = centerW * (dot(tmp.shY - centerSH.shY, tmp.shY - centerSH.shY) + dot(tmp.CoCg - centerSH.CoCg, tmp.CoCg - centerSH.CoCg));
            float w0 = svgfNormalWeight(centerNormal, texelFetch(colortex3, samplePos, 0).xyz)
                    * svgfPositionWeight(centerPos, texelFetch(colortex4, samplePos, 0).xyz, centerNormal)
                    * exp(-dL)
                    * w1 * float(samplePos == clamp(samplePos, vec2(0), texSize));

            accumulate_SH(A, tmp, w0);
            w += w0;
            samplePos.y += R0;
        }
        samplePos.x += R0;
    }
    if (any(isnan(A.shY))) A.shY = vec4(0);
    if (any(isnan(A.CoCg))) A.CoCg = vec2(0);
    SH tmp = scaleSH(A, 1 / max(w, 0.01));
    shY = tmp.shY;
    CoCg.z = texelFetch(colortex6, ivec2(gl_FragCoord.xy), 0).z;
    CoCg.xy = tmp.CoCg;
}
