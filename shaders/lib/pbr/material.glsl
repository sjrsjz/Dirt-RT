#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL
#include "/lib/settings.glsl"
const float EMISSION_INTENSITY = 15.0;

// LabPBR 1.3 predefined-metal F0 values derived from the standard's complex
// IOR table. Values 238-254 are reserved; as permitted by LabPBR they fall
// back to the same albedo-driven conductor model as value 255.
vec3 getHardcodedMetalF0(int channel) {
    if (channel == 230) return vec3(0.53123, 0.51236, 0.49583); // Iron
    if (channel == 231) return vec3(0.94423, 0.77610, 0.37340); // Gold
    if (channel == 232) return vec3(0.91230, 0.91385, 0.91968); // Aluminium
    if (channel == 233) return vec3(0.55560, 0.55454, 0.55478); // Chrome
    if (channel == 234) return vec3(0.92595, 0.72090, 0.50415); // Copper
    if (channel == 235) return vec3(0.63248, 0.62594, 0.64148); // Lead
    if (channel == 236) return vec3(0.67885, 0.64240, 0.58841); // Platinum
    if (channel == 237) return vec3(0.96200, 0.94947, 0.92212); // Silver
    return vec3(1.0);
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
// Apply a 0.5 * geometric NoV floor to the Fresnel cosine.  We
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

    // LabPBR stores the unoccluded/accessibility factor directly: zero is
    // fully occluded and one is unoccluded. Path tracing deliberately does not
    // multiply it into indirect transport, which would double-count geometry.
    material.ambientOcclusion = macroNormal.b;

    // === F0, metallic, albedo === (specular.g)
    int f0Channel = int(specular.g * 255.0 + 0.5);

    if (f0Channel < 230) {
        // Dielectric: f0 = val/255 ∈ [0.0, ~0.9]
        float f0 = float(f0Channel) / 255.0;
        material.F0 = vec3(f0);
        material.metallic = 0.0;
        material.albedo = albedo.rgb;
    } else if (f0Channel <= 237) {
        // Predefined conductors use albedo as a reflection tint, never as a
        // diffuse lobe.
        material.F0 = getHardcodedMetalF0(f0Channel) * albedo.rgb;
        material.metallic = 1.0;
        material.albedo = vec3(0.0);
    } else {
        // Reserved values 238-254 use the allowed value-255 fallback.
        material.F0 = albedo.rgb;
        material.metallic = 1.0;
        material.albedo = vec3(0.0);
    }

    // === Porosity / Subsurface Scattering === (specular.b)
    int bChannel = int(specular.b * 255.0 + 0.5);
    float porosity = 0.0;
    if (material.metallic > 0.5) {
        // The blue channel is reserved on conductors.
        material.subsurface_scattering = 0.0;
    } else if (bChannel <= 64) {
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
        material.emission = albedo.rgb * emissionRaw * EMISSION_INTENSITY;
    } else {
        material.emission = vec3(0.0);
    }

    // === Wetness modulation ===
    float adhesion_ = clamp(adhesion(geometryNormal, vec3(0, -1, 0), vec3(0, -1, 0), material.roughness) + 0.25, 0.0, 1.0);
    float porousWetness = min(wetStrength * adhesion_
        * min(skylight / 255.0, 1.0) * porosity, 1.0);
    float absorbedWater = porousWetness * MAX_WETNESS;
    float surfaceWetness = min(porousWetness + wetness * 0.15, 1.0)
        * MAX_WETNESS;
    // Porous dielectrics darken as water fills air gaps. A real water-film
    // Fresnel lobe requires a layered BSDF; changing the substrate F0 toward
    // one injects energy and is not a valid substitute.
    if (material.metallic < 0.5)
        material.albedo *= mix(1.0, 0.6, absorbedWater);
    material.roughness = max(1.0 - surfaceWetness * 1.5, 0.0)
        * material.roughness;
    material.macroNormal = normalize(mix(
        material.macroNormal, geometryNormal, surfaceWetness));

    return material;
}
#endif // MATERIAL_GLSL
