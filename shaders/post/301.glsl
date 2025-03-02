#version 430 compatibility

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;

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
const int colortex3Format = RGBA32F;
const int colortex4Format = RGBA32F;
const int colortex5Format = RGBA16F;
const int colortex6Format = RGBA16F;
const int colortex7Format = RGBA32F;
const int colortex8Format = RGBA32F;


const bool colortex1Clear = false;
const bool colortex2Clear = false;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = true;
const bool colortex8Clear = true;
*/

const float NORMAL_PARAM = 8.0;
const float POSITION_PARAM = 1.0;
const float LUMINANCE_PARAM = 4.0;

float svgfNormalWeight(vec3 centerNormal, vec3 normal, float S) {
    return pow(max(dot(centerNormal, normal), 0.0), S);
}

float svgfPositionWeight(vec3 centerPos, vec3 pixelPos, vec3 normal) {
    // Modified to check for distance from the center plane
    return exp(-POSITION_PARAM * abs(dot(pixelPos - centerPos, normal)));
}

/* RENDERTARGETS: 5 */

layout(location = 0) out vec4 color;

vec3 prevScreenPos;
bufferData info_;
vec2 texSize;
uint idx;

bool notInRange(vec2 p) {
    return clamp(p, vec2(0), texSize) != p;
}

float K(vec3 B, vec3 A, vec3 n) {
    float an = dot(A, n);
    float bn = dot(B, n);
    float ab = dot(A, B);
    vec3 x = an * B - bn * A;
    return abs(bn) * sqrt(1 - an * an) / max(0.01, dot(x, x));
}
float GetRoughnessWeight(float roughness0, float roughness)
{
    float norm = roughness0 * roughness0 * 0.99 + 0.01;
    float w = abs(roughness0 - roughness) * (1.0 / norm);
    return clamp(1.0 - w, 0.0, 1.0);
}
void main() {
    //color=texelFetch(colortex5,ivec2(gl_FragCoord.xy),0);
    //return;
    idx = getIdx(uvec2(gl_FragCoord.xy));

    info_ = denoiseBuffer.data[idx];
    if (info_.distance < -0.5) {
        return;
    }

    vec3 geoNormal = diffuseIllumiantionBuffer.data[idx].normal2; // 表面法线

    vec3 A = vec3(0);

    mediump float w = 0;
    ivec2 pix = ivec2(gl_FragCoord.xy);
    mediump vec4 centerNormal_ = texelFetch(colortex3, pix, 0);
    mediump vec3 centerNormal = normalize(centerNormal_.xyz);
    mediump float second_ray_distance = length(centerNormal_.xyz);

    vec4 centerPos = texelFetch(colortex4, pix, 0);
    mediump vec4 centerColor = texelFetch(colortex5, pix, 0);

    vec3 planeN = -reflect(centerNormal, geoNormal);
    float axis_A = 0.75 + max(K(cross(camX_global, camY_global), camX_global, planeN), 0);
    float axis_B = 0.75 + max(K(cross(camX_global, camY_global), camY_global, planeN), 0);
    axis_A *= axis_A;
    axis_B *= axis_B;


    float blur_factor = (1 - exp(- 0.25 * centerPos.w)) / (1.0 + 0.25 * second_ray_distance);

    float normal_factor = (1 - exp(- 0.1 * centerPos.w)) * NORMAL_PARAM;

    ivec2 samplePos;
    ivec2 texSize = textureSize(colortex3, 0);

    float theta = 2 * PI * rand(vec2(pix + 11 + R0));

    #if STEP != 1
    mat2 rotM = mat2(cos(theta), -sin(theta), sin(theta), cos(theta)) * R0;
    #endif

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;
            }
            #if STEP == 1
            samplePos = pix + ivec2(i, j);
            #else
            samplePos = pix + ivec2(rotM * vec2(i, j));
            #endif
            vec4 c=texelFetch(colortex5, samplePos, 0);
            float rW = GetRoughnessWeight(centerColor.w, c.w);
            vec4 B=texelFetch(colortex3, samplePos, 0);
            mediump float w1 = exp(- blur_factor * (axis_A*i*i+axis_B*j*j));
            mediump float w0 = rW * exp(- POSITION_PARAM * abs(dot(centerPos.xyz - texelFetch(colortex4, samplePos, 0).xyz, centerNormal)))
                    * svgfNormalWeight(centerNormal, normalize(B.xyz), normal_factor)
                    * w1 * float(samplePos == clamp(samplePos, vec2(0), texSize));
            A += c.xyz * w0;
            w += w0;
        }
    }
    mediump float w0 = 1;//(1 + min(0.125 * centerPos.w,16) + clamp(centerPos.w*0.1, 0, 8)*0.25) * centerNormal_.w;
    A += centerColor.xyz * w0;
    w += w0;

    if (any(isnan(A))) A = vec3(0);
    color = vec4(A / max(w , 0.01), centerColor.w);
}
