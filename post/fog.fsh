#version 430

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex8;
uniform sampler2D colortex9;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    bufferData data = denoiseBuffer.data[idx];
    diffuseIllumiantionData tmp = diffuseIllumiantionBuffer.data[idx];
    vec3IllumiantionData tmp2 = reflectIllumiantionBuffer.data[idx];
    vec3IllumiantionData tmp3 = refractIllumiantionBuffer.data[idx];

    //float sigma=max(0,diffuseIllumiantionBuffer.data[idx].sumX2-diffuseIllumiantionBuffer.data[idx].sumX*diffuseIllumiantionBuffer.data[idx].sumX);

    //fragColor.xyz=vec3(sigma*0.00125);//decodeSH(tmp.data_swap, tmp.normal2);

    if (data.distance < -0.5) {
        fragColor.xyz = data.absorption * getSkyColor(SunLight_global, MoonLight_global, camPos, data.rd, lightDir_global);
    }
    else
    {
        //fragColor.xyz=vec3(abs(project_SH_irradiance(tmp.data_swap,faceforward(tmp.normal2,tmp.normal2,-tmp.normal))));
        fragColor.xyz = data.absorption * ((project_SH_irradiance(tmp.data_swap,tmp.normal2) + tmp3.data_swap) * data.albedo2 + tmp2.data_swap * data.albedo + data.light) + data.emission;
    }
}
