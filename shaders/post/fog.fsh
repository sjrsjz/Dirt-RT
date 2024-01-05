#version 430
#define DIFFUSE_BUFFER_MIN
#define REFLECT_BUFFER_MIN
#define REFRACT_BUFFER_MIN

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/utils.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/light_color.glsl"

in vec2 texCoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    bufferData data = denoiseBuffer.data[idx];
    diffuseIllumiantionData tmp = fetchDiffuse(ivec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp2=fetchReflect(ivec2(gl_FragCoord.xy));
    vec3IllumiantionData tmp3 = fetchRefract(ivec2(gl_FragCoord.xy));

    if (data.distance < -0.5) {
        fragColor.xyz = data.absorption * getSkyColor(SunLight_global, MoonLight_global, camPos, data.rd, lightDir_global);
    }
    else
    {
        //fragColor.xyz=vec3(1)*tmp3.weight;
        
        //fragColor.xyz=vec3(abs(project_SH_irradiance(tmp.data,faceforward(tmp.normal2,tmp.normal2,-tmp.normal))));
        fragColor.xyz =data.absorption * ((project_SH_irradiance(tmp.data_swap,tmp.normal2) + tmp3.data_swap) * data.albedo2 + tmp2.data_swap * data.albedo + data.light) + data.emission;
    }
}
