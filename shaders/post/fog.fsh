#version 430 compatibility
#define DIFFUSE_BUFFER_MIN2
#define REFLECT_BUFFER_MIN2
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
    uint idx = getIdx(uvec2(gl_FragCoord.xy)); //*0.5
    bufferData data = denoiseBuffer.data[idx];

    if (data.distance < -0.5) {
        //setSkyVars();
        //fragColor.xyz = data.absorption * getSkyColor(SunLight_global, MoonLight_global, camPos, data.rd, lightDir_global);
        fragColor.xyz = data.absorption *SampleSky(data.rd) + data.emission;
        diffuseIllumiantionBuffer.data[idx].data_swap=init_SH();
    }
    else
    {
        ivec2 pix = ivec2(gl_FragCoord.xy); //*0.5
        diffuseIllumiantionData tmp = fetchDiffuse(pix);
        //diffuseIllumiantionData tmp = fetchDiffuse(pix/2);    
        vec3IllumiantionData tmp2 = fetchReflect(pix);
        vec3IllumiantionData tmp3 = fetchRefract(pix);
        prevDiffuseIllumiantionBuffer.data[idx].data_swap = tmp.data_swap;
        prevDiffuseIllumiantionBuffer.data[idx].weight = max(tmp.weight,0);
        
        //fragColor.xyz = (diffuseIllumiantionBuffer.data[idx].normal2);
        //fragColor.xyz = diffuseIllumiantionBuffer.data[idx].normal;
        //fragColor.xyz = reflectIllumiantionBuffer.data[idx].normal;
        
        //fragColor.xyz = sqrt(tmp.variance)*vec3(1);
        
        //fragColor.xyz = (tmp.weight)*vec3(1);
        //fragColor.xyz = tmp2.weight*vec3(0.5);
        //fragColor.xyz = vec3(1) * max(0,dot(tmp.data_swap.shY.xyz,diffuseIllumiantionBuffer.data[idx].normal2));
        
        //fragColor.xyz=vec3(1)*(project_SH_irradiance(tmp.data_swap,diffuseIllumiantionBuffer.data[idx].normal2)) ;
        //fragColor.xyz=abs(tmp.data_swap.shY.xyz) * vec3(1);
        //fragColor.xyz=vec3(1)*max(dot(tmp.data_swap.shY.xyz,diffuseIllumiantionBuffer.data[idx].normal2),0);
        
        //fragColor.xyz=abs(light_sigma(tmp.data_swap)*vec3(1)) ;
        //fragColor.xyz = vec3(1)*(tmp.data_swap.shY.w-length(tmp.data_swap.shY.xyz));
        
        //fragColor.xyz=vec3(diffuseIllumiantionBuffer.data[idx].weight);//*(50 - exp(-abs(diffuseIllumiantionBuffer.data[idx].weight)*0.1)*47.5);
        //fragColor.xyz=vec3(1)*reflectIllumiantionBuffer.data[idx].mixWeight;//vec3(abs(project_SH_irradiance(tmp.data,faceforward(tmp.normal2,tmp.normal2,-tmp.normal))));
        //fragColor.xyz = data.albedo2;
        fragColor.xyz = data.absorption * ((project_SH_irradiance(tmp.data_swap,diffuseIllumiantionBuffer.data[idx].normal2) + tmp3.data_swap) * data.albedo2 + tmp2.data_swap * data.albedo + data.light) + data.emission;
        //fragColor.xyz = tmp2.data_swap * data.albedo;
        //fragColor.xyz = max(-reflect(normalize(reflectIllumiantionBuffer.data[idx].normal),normalize(diffuseIllumiantionBuffer.data[idx].normal2)),0) * vec3(1);
        //fragColor.xyz = vec3(1) * length(reflectIllumiantionBuffer.data[idx].normal);
    }
}
