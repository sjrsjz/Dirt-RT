#ifndef LIGHTING_SPECULAR_MAXENT_GLSL
#define LIGHTING_SPECULAR_MAXENT_GLSL

#include "/lib/common.glsl"
#include "/lib/buffers/specular_buffer.glsl"
#include "/lib/lighting/specular_maxent_cdf.glsl"

// The reflection buffer stores moments of the proposal-weighted incident
// radiance measure
//
//     dnu(wi) = q_vndf(wi) * Li(wi) dwi .
//
// Consequently its reconstruction must be modulated by
//
//     K(wi) = f_s(wo, wi) * NoL / q_vndf(wi)
//           = F(wo, H) * G2(NoV, NoL) / G1(NoV).
//
// D and 1 / (4 NoV) cancel exactly. Keeping them in the decoder creates an
// avoidable 0/0-conditioned expression as alpha approaches zero.

float specularMaxEntKappa(float rho) {
    // The cubic reciprocal energy closure has E[u] = kappa * axis.
    return clamp(rho, 0.0, 1.0 - 1.1920928955078125e-7);
}

vec3 specularSurfaceFresnel(vec3 wo, vec3 halfVector,
        vec3 Cs, vec2 S, float etaRatio) {
    vec3 conductorF = reflectanceColor(Cs,
        abs(dot(wo, halfVector))).rgb * S.x;
    float dielectricF = fresnel(wo, halfVector, etaRatio);
    return mix(conductorF, vec3(dielectricF), clamp(S.y, 0.0, 1.0));
}

vec3 projectSpecularMaxEnt(SpecularMaxEnt signal,
        vec3 primaryRay, vec3 macroNormal, vec3 geometryNormal,
        float ggxAlpha, vec3 Cs, vec2 S, float etaRatio) {
    signal = sanitizeSpecularMaxEnt(signal);
    float totalY = signal.maxEntY.w;
    if (totalY <= 1e-8)
        return vec3(0.0);

    vec3 totalRgb = specularMaxEntTotalRgb(signal);
    vec3 wo = -normalize(primaryRay);
    if (dot(macroNormal, wo) <= 1e-6)
        return vec3(0.0);

    // A delta event has no finite proposal density; the stored atom is Li.
    if (ggxAlpha <= SPECULAR_DELTA_ALPHA)
        return totalRgb * specularSurfaceFresnel(
            wo, macroNormal, Cs, S, etaRatio);

    float momentLength = length(signal.maxEntY.xyz);
    // rho=0 is the valid isotropic MaxEnt distribution, not an empty signal.
    vec3 axis = momentLength > 1e-10
        ? signal.maxEntY.xyz / momentLength : macroNormal;
    float rho = clamp(momentLength / totalY, 0.0, 1.0);

    float kappa = specularMaxEntKappa(rho);
    vec3 response = ggxQLiCdfResponse(kappa, max(ggxAlpha, 1e-4),
        axis, wo, macroNormal, geometryNormal, etaRatio);
    vec3 conductor = S.x * (Cs * response.z
        + (vec3(1.0) - Cs) * response.y);
    vec3 modulation = mix(conductor, vec3(response.x),
        clamp(S.y, 0.0, 1.0));

    // CoCg is angularly shared. Because YCoCg->RGB is linear, reconstructing
    // RGB per direction and integrating is exactly totalRgb times E[K].
    return totalRgb * modulation;
}

#endif // LIGHTING_SPECULAR_MAXENT_GLSL
