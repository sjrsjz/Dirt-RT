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
    vec3 emission;
    vec3 normal;
    float ambientOcclusion;
    float translucent;
    ivec2 block_id;
    vec3 mid_block;
    //vec2 block_texture;
};

Material getMaterial(vec4 albedo, vec4 normal, vec4 specular, mat3 tbn, float wetStrength, float wetness) {
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
    float mix0 = min(wetStrength * max(material.normal.y + 0.15, 0) + wetness, 1) * maxWetness; //* porosity/64.0*float(porosity<=64);
    material.roughness = (1 - mix0) * material.roughness;

    if (f0Channel < 230) {
        material.F0 = f0Channel * albedo.rgb / 229.0;
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

    #endif

    // Emission
    material.emission = albedo.rgb * EMISSION_INTENSITY * (specular.a == 1.0 ? 0.0 : specular.a);

    // Ambient occlusion
    material.ambientOcclusion = 1.0 - normal.b;

    return material;
}
#endif // MATERIAL_GLSL
