#include "/lib/constants.glsl"
#include "/lib/common.glsl"

struct bufferData{
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

layout(std430,set=3, binding = 0) buffer DenoiseBuffer{
    bufferData data[];
}denoiseBuffer;



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

    float   T       = Y - sh.CoCg.y * 0.5;
    float   G       = sh.CoCg.y + T;
    float   B       = T - sh.CoCg.x * 0.5;
    float   R       = B + sh.CoCg.x;

    return max(vec3(R, G, B), vec3(0.0));
#else
    return sh.shY.xyz;
#endif
}

SH irradiance_to_SH(vec3 color, vec3 dir)
{
    SH result;

#if ENABLE_SH
    float   Co      = color.r - color.b;
    float   t       = color.b + Co * 0.5;
    float   Cg      = color.g - t;
    float   Y       = max(t + Cg * 0.5, 0.0);

    result.CoCg = vec2(Co, Cg);

    float   L00     = 0.282095;
    float   L1_1    = 0.488603 * dir.y;
    float   L10     = 0.488603 * dir.z;
    float   L11     = 0.488603 * dir.x;

    result.shY = vec4 (L11, L1_1, L10, L00) * Y;
#else
    result.shY = vec4(color, 0);
    result.CoCg = vec2(0);
#endif

    return result;
}

vec3 SH_to_irradiance(SH sh)
{
    float   Y       = sh.shY.w / 0.282095;

    float   T       = Y - sh.CoCg.y * 0.5;
    float   G       = sh.CoCg.y + T;
    float   B       = T - sh.CoCg.x * 0.5;
    float   R       = B + sh.CoCg.x;

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

SH scaleSH(SH A,float x){
    SH tmp;
    tmp.CoCg=A.CoCg*x;
    tmp.shY=A.shY*x;
    return tmp;
}

void accumulate_SH(inout SH accum, SH b, float scale)
{
    accum.shY += b.shY * scale;
    accum.CoCg += b.CoCg * scale;
}

struct diffuseIllumiantionData{
    SH data;
    SH data_swap;
    vec3 pos;
    vec3 normal;
    vec3 normal2;
    vec3 lpos;
    vec3 lnormal;
    //float sumX;
    //float sumX2;
    //float lsumX;
    //float lsumX2;
    float weight;
};
layout(std140,set=3, binding = 3) buffer DiffuseIllumiantionDataBuffer{
    diffuseIllumiantionData data[];
}diffuseIllumiantionBuffer;


struct vec3IllumiantionData{
    float distance;
    vec3 data;
    vec3 data_swap;
    vec3 pos;
    vec3 normal;
    vec3 lpos;
    vec3 lnormal;
    float weight;
};

layout(std140,set=3, binding = 4) buffer ReflectIllumiantionDataBuffer{
    vec3IllumiantionData data[];
}reflectIllumiantionBuffer;

layout(std140,set=3, binding = 5) buffer RefractIllumiantionDataBuffer{
    vec3IllumiantionData data[];
}refractIllumiantionBuffer;

diffuseIllumiantionData blendDiffuse(diffuseIllumiantionData A,diffuseIllumiantionData B,float x){
    diffuseIllumiantionData t;
    //t.data_swap=A.data_swap;
    t.data=mix_SH(A.data,B.data,x);
    t.lpos=mix(A.lpos,B.lpos,x);
    //t.pos=B.pos;
    //t.normal=B.normal;
    t.lnormal=(mix(A.lnormal,B.lnormal,x));
    t.weight=(B.weight-A.weight)*x+A.weight;
    
    //t.lsumX=A.lsumX+(B.lsumX-A.lsumX)*x;
    //t.lsumX2=A.lsumX2+(B.lsumX2-A.lsumX2)*x;
    return t;
}

diffuseIllumiantionData sampleDiffuse(vec2 p){
   // p-=0.5;
    uvec2 p1=uvec2(p);
    vec2 p2=p-p1;
    diffuseIllumiantionData A=diffuseIllumiantionBuffer.data[getIdx(p1)];
    diffuseIllumiantionData B=diffuseIllumiantionBuffer.data[getIdx(p1+uvec2(1,0))];
    diffuseIllumiantionData C=diffuseIllumiantionBuffer.data[getIdx(p1+uvec2(0,1))];
    diffuseIllumiantionData D=diffuseIllumiantionBuffer.data[getIdx(p1+uvec2(1,1))];
    return blendDiffuse(blendDiffuse(A,B,p2.x),blendDiffuse(C,D,p2.x),p2.y);
}

vec3IllumiantionData blendData(vec3IllumiantionData A,vec3IllumiantionData B,float x){
    vec3IllumiantionData t;
    //t.data=B.data;
    t.data=mix(A.data,B.data,x);
    t.lpos=mix(A.lpos,B.lpos,x);
    //t.pos=B.pos;
    //t.normal=B.normal;
    t.lnormal=(mix(A.lnormal,B.lnormal,x));
    t.weight=A.weight+(B.weight-A.weight)*x;

    t.distance=(B.distance-A.distance)*x+A.distance;
    return t;
}

vec3IllumiantionData sampleReflect(vec2 p){
    //p-=0.5;
    uvec2 p1=uvec2(p);
    vec2 p2=p-p1;
    vec3IllumiantionData A=reflectIllumiantionBuffer.data[getIdx(p1)];
    vec3IllumiantionData B=reflectIllumiantionBuffer.data[getIdx(p1+uvec2(1,0))];
    vec3IllumiantionData C=reflectIllumiantionBuffer.data[getIdx(p1+uvec2(0,1))];
    vec3IllumiantionData D=reflectIllumiantionBuffer.data[getIdx(p1+uvec2(1,1))];
    return blendData(blendData(A,B,p2.x),blendData(C,D,p2.x),p2.y);
}

vec3IllumiantionData sampleRefract(vec2 p){
    //p-=0.5;
    uvec2 p1=uvec2(p);
    vec2 p2=p-p1;
    vec3IllumiantionData A=refractIllumiantionBuffer.data[getIdx(p1)];
    vec3IllumiantionData B=refractIllumiantionBuffer.data[getIdx(p1+uvec2(1,0))];
    vec3IllumiantionData C=refractIllumiantionBuffer.data[getIdx(p1+uvec2(0,1))];
    vec3IllumiantionData D=refractIllumiantionBuffer.data[getIdx(p1+uvec2(1,1))];
    return blendData(blendData(A,B,p2.x),blendData(C,D,p2.x),p2.y);
}


