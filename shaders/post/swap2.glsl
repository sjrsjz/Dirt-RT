#version 430 compatibility
#define DIFFUSE_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

uniform sampler2D colortex0;

/* RENDERTARGETS: 3,4,5,6,0 */

layout(location = 0) out vec4 diffuseNormal;
layout(location = 1) out vec4 diffusePos;
layout(location = 2) out vec4 shY;
layout(location = 3) out vec4 CoCg;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_FragCoord.xy));
    diffuseNormal.xyz = diffuseIllumiantionBuffer.data[idx].normal;
    diffusePos = vec4(diffuseIllumiantionBuffer.data[idx].pos,tmp.variance);
    //float c=100/(avgExposure*min(tmp.weight,10));
    //SH tmp0=tmp.data_swap;//irradiance_to_SH(min(project_SH_irradiance(tmp.data_swap,diffuseIllumiantionBuffer.data[idx].normal),c),diffuseIllumiantionBuffer.data[idx].normal);
//    shY = tmp.data_swap.shY;
//    CoCg.xy = tmp.data_swap.CoCg;

    shY = tmp.data_swap.shY;//normalize(tmp.data_swap.shY)*clamp(length(tmp.data_swap.shY),0,0.5/avgExposure*(1+5*tmp.weight));
    CoCg = vec4(tmp.data_swap.CoCg, tmp.weight, 0);
    if (any(isnan(shY))) shY=vec4(0);
    if (any(isnan(CoCg))) CoCg=vec4(0);
    
}
