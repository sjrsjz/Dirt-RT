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

uniform vec2 resolution;
void main() {
    // if(gl_FragCoord.x >= resolution.x/2 || gl_FragCoord.y >= resolution.y/2) {
    //     return;
    // }
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    //uint idx = getIdx(uvec2(gl_FragCoord.xy)*2+1);

    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_FragCoord.xy));
    diffuseNormal.xyz = diffuseIllumiantionBuffer.data[idx].normal;
    if(tmp.weight <= 1+1e-3){
        diffusePos = vec4(diffuseIllumiantionBuffer.data[idx].pos, 1000);
    }
    else{
        diffusePos = vec4(diffuseIllumiantionBuffer.data[idx].pos, tmp.variance * tmp.weight / (tmp.weight - 1));
    }
    
    shY = clamp(tmp.data_swap.shY, 0, 100 / avgExposure);
    CoCg = clamp(vec4(tmp.data_swap.CoCg, tmp.weight, 0), 0, 100 / avgExposure);
    if (any(isnan(shY))) shY=vec4(0);
    if (any(isnan(CoCg))) CoCg=vec4(0);
    
}
