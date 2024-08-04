#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL
#include "/lib/settings.glsl"
const float EMISSION_INTENSITY = 0.55;

const vec3 HCM_METALS[] = vec3[](
        vec3(0.53123, 0.51236, 0.49583), // Iron
        vec3(0.94423, 0.77610, 0.37340), // Gold
        vec3(0.91230, 0.91385, 0.91968), // Aluminium
        vec3(0.55560, 0.55454, 0.55478), // Chrome
        vec3(0.92595, 0.72090, 0.50415), // Copper
        vec3(0.63248, 0.62594, 0.64148), // Lead
        vec3(0.67885, 0.64240, 0.58841), // Platinum
        vec3(0.96200, 0.94947, 0.92212) // Silver
    );

struct Material {
    vec3 albedo;
    vec3 F0;
    float metallic;
    float roughness;
    float subsurface_scattering;
    vec3 emission;
    vec3 normal;
    float ambientOcclusion;
    float translucent;
    ivec2 block_id;
    vec3 light_texture;
    //vec2 block_texture;
};

float adhesion(vec3 n,vec3 w,vec3 g,float a){
    //w:wind direction
    //g:gravity direction
    //n:surface normal
    //a:roughness
    float tanA=sqrt(max(pow(abs(dot(n,w)),-2)-1,0));
    float tanB=sqrt(max(pow(abs(dot(n,g)),-2)-1,0));
    float a2=a*a;
    float t=sqrt(tanB*tanB+a2);
    return (float(dot(n,g)<0)*2*a2/
        ((1+sqrt(1+a2*tanA*tanA))*(tanB+t)*t));
}

Material getMaterial(vec4 albedo, vec4 normal, vec4 specular, mat3 tbn, float wetStrength, float wetness ,float skylight ,vec3 macroNormal) {
    Material material;

    // Translucency
    material.translucent = albedo.a;

    // Roughness
    material.roughness = (1.0 - specular.r) * (1.0 - specular.r);

    // Normal
    material.normal = normal.xyz * 2.0 - 1.0;
    material.normal.z = sqrt(max(1.0 - dot(material.normal.xy, material.normal.xy), 0.0));
    material.normal = normalize(tbn * material.normal);

    // Albedo, F0 and metallic
    int f0Channel = int(specular.g * 255.0);
    #if 0
    if (f0Channel < 230) {
        if (false) { //material.translucent) {
            material.albedo = (1.0 - albedo.rgb) * albedo.a;
        } else {
            material.albedo = albedo.rgb;
        }
        material.F0 = specular.ggg;
        material.metallic = 0.0;
    } else if (f0Channel < 238) {
        material.albedo = albedo.rgb;
        material.F0 = HCM_METALS[f0Channel - 230];
        material.metallic = 1.0;
    } else {
        material.albedo = vec3(1.0);
        material.F0 = albedo.rgb;
        material.metallic = 1.0;
    }

    #else
    //int porosity = int(specular.b * 255.0);
    float adhesion_ = clamp(adhesion(macroNormal,vec3(0,-1,0),vec3(0,-1,0),material.roughness)+0.25,0,1);
    float mix0 = min(wetStrength*adhesion_*min(skylight/255,1) + wetness*0.15,1); //* porosity/64.0*float(porosity<=64);
    mix0 *= maxWetness;
    material.roughness = max(1 - mix0 * 1.5,0) * material.roughness;
    material.normal=normalize(mix(material.normal,macroNormal,mix0));
    mix0 *= 0.25;
    if (f0Channel < 230) {
        material.F0 = sqrt(f0Channel * albedo.rgb / 229.0);
        material.F0 = mix(material.F0, vec3(1), mix0);
        material.metallic = 0;
        material.albedo = albedo.rgb;//* (1 - material.F0);
    } else if (f0Channel < 238) {
        material.F0 = HCM_METALS[f0Channel - 230];
        material.F0 = mix(material.F0, vec3(1), mix0);
        material.metallic = 0;
        material.albedo = vec3(0);
    } else {
        material.F0 = albedo.rgb;
        material.F0 = mix(material.F0, vec3(1), mix0);
        material.metallic = 0;
        material.albedo = vec3(0);
    }
    material.subsurface_scattering = 0.25;//specular.b;
    #endif

    // Emission
    material.emission = albedo.rgb * (specular.a == 1.0 ? 0.0 : specular.a);

    // Ambient occlusion
    material.ambientOcclusion = 1.0 - normal.b;

    return material;
}
#endif // MATERIAL_GLSL
