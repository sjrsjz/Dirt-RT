
#include "/lib/constants.glsl"
#include "/lib/common.glsl"

struct bufferData {
    vec3 macroNormal;
    vec3 light;
    vec3 albedo;
    vec3 albedo2;
    float distance;
    vec3 absorption;
    vec3 emission;
    vec3 rd;
    int illumiantionType;
    //vec4 currSample;
    //vec4 lastSample;
};

layout(std430, set = 3, binding = 0) buffer DenoiseBuffer {
    bufferData data[];
} denoiseBuffer;

struct SH
{
    vec4 shY;
    vec2 CoCg;
};

// Switch to enable or disable the *look* of spherical harmonics lighting.
// Does not affect the performance, just for A/B image comparison.
#define ENABLE_SH 1

vec3 project_SH_irradiance(SH sh, vec3 N)
{
    #if ENABLE_SH
    float d = dot(sh.shY.xyz, N);
    float Y = 2.0 * (1.023326 * d + 0.886226 * sh.shY.w);
    Y = max(Y, 0.0);

    sh.CoCg *= Y * 0.282095 / (sh.shY.w + 1e-6);

    float T = Y - sh.CoCg.y * 0.5;
    float G = sh.CoCg.y + T;
    float B = T - sh.CoCg.x * 0.5;
    float R = B + sh.CoCg.x;

    return max(vec3(R, G, B), vec3(0.0));
    #else
    return sh.shY.xyz;
    #endif
}

SH irradiance_to_SH(vec3 color, vec3 dir)
{
    SH result;

    #if ENABLE_SH
    float Co = color.r - color.b;
    float t = color.b + Co * 0.5;
    float Cg = color.g - t;
    float Y = max(t + Cg * 0.5, 0.0);

    result.CoCg = vec2(Co, Cg);

    float L00 = 0.282095;
    float L1_1 = 0.488603 * dir.y;
    float L10 = 0.488603 * dir.z;
    float L11 = 0.488603 * dir.x;

    result.shY = vec4(L11, L1_1, L10, L00) * Y;
    #else
    result.shY = vec4(color, 0);
    result.CoCg = vec2(0);
    #endif

    return result;
}

vec3 SH_to_irradiance(SH sh)
{
    float Y = sh.shY.w / 0.282095;

    float T = Y - sh.CoCg.y * 0.5;
    float G = sh.CoCg.y + T;
    float B = T - sh.CoCg.x * 0.5;
    float R = B + sh.CoCg.x;

    return max(vec3(R, G, B), vec3(0.0));
}

SH mix_SH(SH a, SH b, float s)
{
    SH result;
    result.shY = mix(a.shY, b.shY, vec4(s));
    result.CoCg = mix(a.CoCg, b.CoCg, vec2(s));
    return result;
}

SH init_SH()
{
    SH result;
    result.shY = vec4(0);
    result.CoCg = vec2(0);
    return result;
}

SH scaleSH(SH A, float x) {
    SH tmp;
    tmp.CoCg = A.CoCg * x;
    tmp.shY = A.shY * x;
    return tmp;
}

void accumulate_SH(inout SH accum, SH b, float scale)
{
    accum.shY += b.shY * scale;
    accum.CoCg += b.CoCg * scale;
}

struct diffuseIllumiantionData {
    SH data;
    SH data_swap;
    vec3 pos;
    vec3 normal;
    vec3 normal2;
    //float sumX;
    //float sumX2;
    //float lsumX;
    //float lsumX2;
    float weight;
};
layout(std140, set = 3, binding = 3) buffer DiffuseIllumiantionDataBuffer {
    diffuseIllumiantionData data[];
} diffuseIllumiantionBuffer;

struct vec3IllumiantionData {
    float distance;
    vec3 data;
    vec3 data_swap;
    vec3 pos;
    vec3 normal;
    float weight;
};

layout(std140, set = 3, binding = 4) buffer ReflectIllumiantionDataBuffer {
    vec3IllumiantionData data[];
} reflectIllumiantionBuffer;

layout(std140, set = 3, binding = 5) buffer RefractIllumiantionDataBuffer {
    vec3IllumiantionData data[];
} refractIllumiantionBuffer;


#if defined(DIFFUSE_BUFFER) || defined(DIFFUSE_BUFFER_MIN)


layout(rgba32f) uniform image2D diffuseIllumiantionData_shY_swap;
layout(rg32f) uniform image2D diffuseIllumiantionData_CoCg_swap;
uniform sampler2D diffuseIllumiantionData_shY_Sampler;
uniform sampler2D diffuseIllumiantionData_CoCg_Sampler;
uniform sampler2D diffuseIllumiantionData_shY_swap_Sampler;
uniform sampler2D diffuseIllumiantionData_CoCg_swap_Sampler;
uniform sampler2D diffuseIllumiantionData_lnormal_Sampler;
uniform sampler2D diffuseIllumiantionData_lpos_Sampler;
#ifndef DIFFUSE_BUFFER_MIN
layout(rgba32f) uniform image2D diffuseIllumiantionData_shY;
layout(rgba32f) uniform image2D diffuseIllumiantionData_CoCg;
layout(rgba32f) uniform image2D diffuseIllumiantionData_lnormal;
layout(rgba32f) uniform image2D diffuseIllumiantionData_lpos;

#endif

diffuseIllumiantionData sampleDiffuse(vec2 p) {
    diffuseIllumiantionData tmp;

    vec4 tmp4 = texture(diffuseIllumiantionData_CoCg_swap_Sampler, p);
    tmp.data_swap.CoCg = tmp4.xy;
    tmp.data_swap.shY = texture(diffuseIllumiantionData_shY_swap_Sampler, p);
    //#ifndef DIFFUSE_BUFFER_MIN
    tmp4 = texture(diffuseIllumiantionData_CoCg_Sampler, p);
    tmp.data.CoCg = tmp4.xy;
    tmp.data.shY = texture(diffuseIllumiantionData_shY_Sampler, p);
    tmp.weight = tmp4.z;
    tmp.normal = texture(diffuseIllumiantionData_lnormal_Sampler, p).xyz;
    tmp.pos = texture(diffuseIllumiantionData_lpos_Sampler, p).xyz;
   // #endif
    return tmp;
}

diffuseIllumiantionData fetchDiffuse(ivec2 p) {
    diffuseIllumiantionData tmp;

    vec4 tmp4 = texelFetch(diffuseIllumiantionData_CoCg_swap_Sampler, p, 0);
    tmp.data_swap.CoCg = tmp4.xy;
    tmp.data_swap.shY = texelFetch(diffuseIllumiantionData_shY_swap_Sampler, p, 0);
    //#ifndef DIFFUSE_BUFFER_MIN
    tmp4 = texelFetch(diffuseIllumiantionData_CoCg_Sampler, p, 0);
    tmp.data.CoCg = tmp4.xy;
    tmp.data.shY = texelFetch(diffuseIllumiantionData_shY_Sampler, p, 0);
    tmp.weight = tmp4.z;

    tmp.normal = texelFetch(diffuseIllumiantionData_lnormal_Sampler, p, 0).xyz;
    tmp.pos = texelFetch(diffuseIllumiantionData_lpos_Sampler, p, 0).xyz;
    //#endif
    return tmp;
}

void WriteDiffuse(diffuseIllumiantionData data, ivec2 p) {

    imageStore(diffuseIllumiantionData_shY_swap, p, data.data_swap.shY);
    imageStore(diffuseIllumiantionData_CoCg_swap, p, vec4(data.data_swap.CoCg, 0, 0));
    #ifndef DIFFUSE_BUFFER_MIN
    imageStore(diffuseIllumiantionData_shY, p, data.data.shY);
    imageStore(diffuseIllumiantionData_CoCg, p, vec4(data.data.CoCg, data.weight, 0));
    imageStore(diffuseIllumiantionData_lpos, p, vec4(data.pos, 0));
    imageStore(diffuseIllumiantionData_lnormal, p, vec4(data.normal, 0));
    #endif
}
#endif

#if defined(REFLECT_BUFFER) || defined(REFLECT_BUFFER_MIN) 

layout(rgba32f) uniform image2D reflectIllumiantionData_swap_color;
uniform sampler2D reflectIllumiantionData_color_Sampler;
uniform sampler2D reflectIllumiantionData_color_swap_Sampler;
uniform sampler2D reflectIllumiantionData_lnormal_Sampler;
uniform sampler2D reflectIllumiantionData_lpos_Sampler;
#ifndef REFLECT_BUFFER_MIN
layout(rgba32f) uniform image2D reflectIllumiantionData_color;
layout(rgba32f) uniform image2D reflectIllumiantionData_lnormal;
layout(rgba32f) uniform image2D reflectIllumiantionData_lpos;
#endif

vec3IllumiantionData sampleReflect(vec2 p) {
    vec3IllumiantionData tmp;
    vec4 tmp4 = texture(reflectIllumiantionData_color_swap_Sampler, p);
    tmp.data_swap = tmp4.xyz;
    #ifndef REFLECT_BUFFER_MIN
    tmp4 = texture(reflectIllumiantionData_color_Sampler, p);
    tmp.data = tmp4.xyz;
    tmp.weight = tmp4.w;
    tmp.normal = texture(reflectIllumiantionData_lnormal_Sampler, p).xyz;
    tmp.pos = texture(reflectIllumiantionData_lpos_Sampler, p).xyz;
    #endif
    return tmp;
}
vec3IllumiantionData fetchReflect(ivec2 p) {
    vec3IllumiantionData tmp;
    vec4 tmp4 = texelFetch(reflectIllumiantionData_color_swap_Sampler, p, 0);
    tmp.data_swap = tmp4.xyz;
    #ifndef REFLECT_BUFFER_MIN
    tmp4 = texelFetch(reflectIllumiantionData_color_Sampler, p, 0);
    tmp.data = tmp4.xyz;
    tmp.weight = tmp4.w;
    tmp.normal = texelFetch(reflectIllumiantionData_lnormal_Sampler, p, 0).xyz;
    tmp.pos = texelFetch(reflectIllumiantionData_lpos_Sampler, p, 0).xyz;
    #endif
    return tmp;
}
void WriteReflect(vec3IllumiantionData data, ivec2 p) {
    imageStore(reflectIllumiantionData_swap_color, p, vec4(data.data_swap, data.weight));
    #ifndef REFLECT_BUFFER_MIN
    imageStore(reflectIllumiantionData_color, p, vec4(data.data, data.weight));
    imageStore(reflectIllumiantionData_lpos, p, vec4(data.pos, 0));
    imageStore(reflectIllumiantionData_lnormal, p, vec4(data.normal, 0));
    #endif
}
#endif

#if defined(REFRACT_BUFFER) || defined(REFRACT_BUFFER_MIN)

layout(rgba32f) uniform image2D refractIllumiantionData_swap_color;
uniform sampler2D refractIllumiantionData_color_Sampler;
uniform sampler2D refractIllumiantionData_color_swap_Sampler;
uniform sampler2D refractIllumiantionData_lnormal_Sampler;
uniform sampler2D refractIllumiantionData_lpos_Sampler;
#ifndef REFRAECT_BUFFER_MIN
layout(rgba32f) uniform image2D refractIllumiantionData_color;
layout(rgba32f) uniform image2D refractIllumiantionData_lnormal;
layout(rgba32f) uniform image2D refractIllumiantionData_lpos;

#endif

vec3IllumiantionData sampleRefract(vec2 p) {
    vec3IllumiantionData tmp;
    vec4 tmp4 = texture(refractIllumiantionData_color_swap_Sampler, p);
    tmp.data_swap = tmp4.xyz;
    #ifndef REFRACT_BUFFER_MIN
    tmp4 = texture(refractIllumiantionData_color_Sampler, p);
    tmp.data = tmp4.xyz;
    tmp.weight = tmp4.w;

    tmp.normal = texture(refractIllumiantionData_lnormal_Sampler, p).xyz;
    tmp.pos = texture(refractIllumiantionData_lpos_Sampler, p).xyz;
    #endif
    return tmp;
}
vec3IllumiantionData fetchRefract(ivec2 p) {
    vec3IllumiantionData tmp;
    vec4 tmp4 = texelFetch(refractIllumiantionData_color_swap_Sampler, p, 0);
    tmp.data_swap=tmp4.xyz;
    #ifndef REFRACT_BUFFER_MIN
    tmp4 = texelFetch(refractIllumiantionData_color_Sampler, p, 0);
    tmp.data = tmp4.xyz;
    tmp.weight = tmp4.w;
    tmp.normal = texelFetch(refractIllumiantionData_lnormal_Sampler, p, 0).xyz;
    tmp.pos = texelFetch(refractIllumiantionData_lpos_Sampler, p, 0).xyz;
    #endif
    return tmp;
}
void WriteRefract(vec3IllumiantionData data, ivec2 p) {
    
    imageStore(refractIllumiantionData_swap_color, p, vec4(data.data_swap, data.weight));
    #ifndef REFRACT_BUFFER_MIN
    imageStore(refractIllumiantionData_color, p, vec4(data.data, data.weight));
    imageStore(refractIllumiantionData_lpos, p, vec4(data.pos, 0));
    imageStore(refractIllumiantionData_lnormal, p, vec4(data.normal, 0));
    #endif
}
#endif
