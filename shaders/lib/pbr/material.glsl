#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL
#include "/lib/settings.glsl"
const float EMISSION_INTENSITY = 15.0;

// LabPBR 1.3 硬编码金属 F0 查找表 (specular.g ∈ [230,255])
// 230-235: 固定金属, 236-237: 保留, 238-255: 自定义金属 (F0 = albedo.rgb)
vec3 getHardcodedMetalF0(int channel) {
    if (channel == 230) return vec3(0.53123, 0.51236, 0.49583); // Iron
    if (channel == 231) return vec3(0.94423, 0.77610, 0.37340); // Gold
    if (channel == 232) return vec3(0.91230, 0.91385, 0.91968); // Aluminium
    if (channel == 233) return vec3(0.92595, 0.72090, 0.50415); // Copper
    if (channel == 234) return vec3(0.63248, 0.62594, 0.64148); // Lead
    if (channel == 235) return vec3(0.38760, 0.34111, 0.24700); // Silicon
    return vec3(0.04); // 236-237 reserved → fallback dielectric
}

struct Material {
    vec3 albedo;
    vec3 F0;
    float metallic;
    float roughness;
    float subsurface_scattering;
    vec3 emission;
    vec3 macroNormal;
    float ambientOcclusion;
    float translucent;
    ivec2 block_id;
    vec3 light_texture;
    //vec2 block_texture;
};

// Keep a tangent-space normal usable by the geometric surface and by the
// current incident ray.  A normal map can remain in the geometric hemisphere
// yet point behind the viewer at grazing angles.  That makes NoV non-positive,
// zeroes the BSDF, and can turn isolated texels completely black.
//
// Alpha-Piscium applies a 0.5 * geometric NoV floor to the Fresnel cosine.  We
// enforce the same floor on the normal itself so every RT consumer (sampling,
// Fresnel, G-buffer reconstruction and the denoisers) sees one convention.
vec3 constrainMappedNormal(vec3 mappedNormal, vec3 geometryNormal,
    vec3 viewDirection) {
    vec3 N = mappedNormal;
    N *= dot(N, geometryNormal) < 0.0 ? -1.0 : 1.0;

    float NoV = dot(N, viewDirection);
    float minNoV = max(1e-4, 0.5 * max(dot(geometryNormal,
        viewDirection), 0.0));
    if (NoV < minNoV) {
        vec3 viewTangent = N - viewDirection * NoV;
        float tangentLength2 = dot(viewTangent, viewTangent);
        if (tangentLength2 > 1e-8) {
            float tangentScale = sqrt(max(1.0 - minNoV * minNoV, 0.0))
                * inversesqrt(tangentLength2);
            N = viewTangent * tangentScale + viewDirection * minNoV;
        } else {
            N = geometryNormal;
        }
    }
    // All inputs are unit length; the correction above rebuilds N from two
    // orthogonal unit components, so another unconditional normalize is dead.
    return N;
}

float adhesion(vec3 n, vec3 w, vec3 g, float a) {
    float tanA = sqrt(max(pow(abs(dot(n, w)), -2) - 1, 0));
    float tanB = sqrt(max(pow(abs(dot(n, g)), -2) - 1, 0));
    float a2 = a * a;
    float t = sqrt(tanB * tanB + a2);
    return (float(dot(n, g) < 0) * 2 * a2 /
        ((1 + sqrt(1 + a2 * tanA * tanA)) * (tanB + t) * t));
}

Material getMaterial(vec4 albedo, vec4 macroNormal, vec4 specular, mat3 tbn, float wetStrength, float wetness, float skylight, vec3 geometryNormal) {
    Material material;

    // === Translucency === (albedo.a)
    material.translucent = albedo.a;

    // === Roughness: α = (1 - perceptualSmoothness)² === (specular.r)
    float smoothness = specular.r;
    material.roughness = (1.0 - smoothness) * (1.0 - smoothness);

    // === Normal: XY from RG channels, Z reconstructed, B = Material AO === (macroNormal)
    material.macroNormal = macroNormal.xyz * 2.0 - 1.0;
    material.macroNormal.z = sqrt(max(1.0 - dot(material.macroNormal.xy, material.macroNormal.xy), 0.0));
    material.macroNormal = normalize(tbn * material.macroNormal);

    // === Ambient Occlusion === (macroNormal.b)
    // LabPBR 1.3: B = AO directly. 但多数纹理包沿用旧约定 B = 1-AO.
    // 为兼容性保留取反; 若纹理包严格遵循 LabPBR 1.3, 改为 macroNormal.b.
    material.ambientOcclusion = 1.0 - macroNormal.b;

    // === F0, metallic, albedo === (specular.g)
    int f0Channel = int(specular.g * 255.0 + 0.5);

    if (f0Channel < 230) {
        // Dielectric: f0 = val/255 ∈ [0.0, ~0.9]
        float f0 = float(f0Channel) / 255.0;
        material.F0 = vec3(f0);
        material.metallic = 0.0;
        material.albedo = albedo.rgb;
    } else if (f0Channel <= 235) {
        // Hardcoded metals (230-235)
        material.F0 = getHardcodedMetalF0(f0Channel);
        material.metallic = 1.0;
        material.albedo = vec3(0.0);
    } else {
        // Custom metals (238-255) + reserved (236-237)
        // 为简单起见, 238+ 统一走自定义金属路径 (F0 = albedo.rgb)
        material.F0 = albedo.rgb;
        material.metallic = 1.0;
        material.albedo = vec3(0.0);
    }

    // === Porosity / Subsurface Scattering === (specular.b)
    int bChannel = int(specular.b * 255.0 + 0.5);
    float porosity = 0.0;
    if (bChannel <= 64) {
        // Porosity: 0~64 → 0.0~1.0
        porosity = float(bChannel) / 64.0;
        material.subsurface_scattering = 0.0;
    } else {
        // Subsurface Scattering: 65~255 → 0.0~1.0
        material.subsurface_scattering = float(bChannel - 65) / 190.0;
    }

    // === Emission === (specular.a)
    int aChannel = int(specular.a * 255.0 + 0.5);
    if (aChannel < 255) {
        // 0~254: emission intensity, linear
        float emissionRaw = float(aChannel) / 254.0;
        material.emission = albedo.rgb * pow(emissionRaw, 2.0) * EMISSION_INTENSITY;
    } else {
        material.emission = vec3(0.0);
    }

    // === Wetness modulation ===
    float adhesion_ = clamp(adhesion(geometryNormal, vec3(0, -1, 0), vec3(0, -1, 0), material.roughness) + 0.25, 0.0, 1.0);
    float mix0 = min(wetStrength * adhesion_ * min(skylight / 255.0, 1.0) * porosity + wetness * 0.15, 1.0);
    mix0 *= MAX_WETNESS;
    material.roughness = max(1.0 - mix0 * 1.5, 0.0) * material.roughness;
    material.macroNormal = normalize(mix(material.macroNormal, geometryNormal, mix0));
    // Wet dielectric → F0 blends toward 1.0 (thin water film); metals unaffected
    mix0 *= 0.25;
    if (material.metallic < 0.5) {
        material.F0 = mix(material.F0, vec3(1.0), mix0);
    }

    return material;
}
#endif // MATERIAL_GLSL
