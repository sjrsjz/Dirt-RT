#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#define REFLECT_BUFFER_MIN
#define REFRACT_BUFFER_MIN2

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
    
    if (data.distance < -0.5) {
        //setSkyVars();
        //fragColor.xyz = data.absorption * getSkyColor(SunLight_global, MoonLight_global, camPos, data.rd, lightDir_global);
        fragColor.xyz = data.absorption *SampleSky(data.rd) + data.emission;
    }
    else
    {
        ivec2 pix = ivec2(gl_FragCoord.xy);
        diffuseIllumiantionData tmp = fetchDiffuse(pix);
        vec3IllumiantionData tmp2 = fetchReflect(pix);
        vec3IllumiantionData tmp3 = fetchRefract(pix);
        diffuseIllumiantionBuffer.data[idx].weight=tmp.weight;
        //reflectIllumiantionBuffer.data[idx].mixWeight=data.reflectWeight;
        //fragColor.xyz = tmp3.normal;
        //fragColor.xyz = diffuseIllumiantionBuffer.data[idx].normal2;
        //fragColor.xyz = diffuseIllumiantionBuffer.data[idx].normal2;
        
        //fragColor.xyz=vec3(1)*(project_SH_irradiance(tmp.data_swap,diffuseIllumiantionBuffer.data[idx].normal2)) ;
        //fragColor.xyz=vec3(diffuseIllumiantionBuffer.data[idx].weight);//*(50 - exp(-abs(diffuseIllumiantionBuffer.data[idx].weight)*0.1)*47.5);
        //fragColor.xyz=reflectIllumiantionBuffer.data[idx].normal;//vec3(abs(project_SH_irradiance(tmp.data,faceforward(tmp.normal2,tmp.normal2,-tmp.normal))));
        fragColor.xyz = data.absorption * ((project_SH_irradiance(tmp.data_swap,diffuseIllumiantionBuffer.data[idx].normal2) + tmp3.data_swap) * data.albedo2 + tmp2.data_swap * data.albedo + data.light) + data.emission;
    }
}
