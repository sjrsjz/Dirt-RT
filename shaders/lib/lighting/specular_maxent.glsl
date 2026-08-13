#ifndef LIGHTING_SPECULAR_MAXENT_GLSL
#define LIGHTING_SPECULAR_MAXENT_GLSL

#include "/lib/common.glsl"
#include "/lib/buffers/specular_buffer.glsl"

// The reflection buffer stores moments of the incident-radiance measure
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

const int SPECULAR_MAXENT_QUADRATURE_SAMPLES = 16;

float specularMaxEntKappa(float rho) {
    rho = clamp(rho, 0.0, 1.0 - 1e-6);
    return 3.0 * rho / (2.0 + sqrt(max(4.0 - 3.0 * rho * rho, 1e-12)));
}

vec3 specularMaxEntSample(vec3 axis, float kappa, vec2 xi) {
    float mu;
    if (kappa < 1e-5) {
        mu = 2.0 * xi.x - 1.0;
    } else {
        float inverseCubeMin = pow(1.0 + kappa, -3.0);
        float inverseCubeMax = pow(1.0 - kappa, -3.0);
        float inverseCube = mix(inverseCubeMin, inverseCubeMax, xi.x);
        mu = (1.0 - pow(max(inverseCube, 1e-20), -1.0 / 3.0))
            / kappa;
        mu = clamp(mu, -1.0, 1.0);
    }

    float phi = 2.0 * PI * xi.y;
    float sinTheta = sqrt(max(1.0 - mu * mu, 0.0));
    vec3 tangent = abs(axis.y) < 0.999
        ? normalize(cross(vec3(0.0, 1.0, 0.0), axis))
        : vec3(1.0, 0.0, 0.0);
    vec3 bitangent = cross(axis, tangent);
    return normalize(axis * mu + sinTheta
        * (tangent * cos(phi) + bitangent * sin(phi)));
}

float specularRadicalInverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 specularQuadraturePoint(int sampleIndex) {
    return vec2((float(sampleIndex) + 0.5)
            / float(SPECULAR_MAXENT_QUADRATURE_SAMPLES),
        specularRadicalInverse(uint(sampleIndex)));
}

vec3 specularSurfaceFresnel(vec3 wo, vec3 halfVector,
        vec3 Cs, vec2 S, float etaRatio) {
    vec3 conductorF = reflectanceColor(Cs,
        abs(dot(wo, halfVector))).rgb * S.x;
    float dielectricF = fresnel(wo, halfVector, etaRatio);
    return mix(conductorF, vec3(dielectricF), clamp(S.y, 0.0, 1.0));
}

vec3 specularVndfModulation(vec3 wo, vec3 wi,
        vec3 macroNormal, vec3 geometryNormal, float ggxAlpha,
        vec3 Cs, vec2 S, float etaRatio) {
    float NoV = dot(macroNormal, wo);
    float NoL = dot(macroNormal, wi);
    if (NoV <= 1e-6 || NoL <= 1e-6
            || dot(geometryNormal, wi) <= 0.0)
        return vec3(0.0);

    vec3 halfSum = wo + wi;
    float halfLength2 = dot(halfSum, halfSum);
    if (halfLength2 <= 1e-12)
        return vec3(0.0);
    vec3 halfVector = halfSum * inversesqrt(halfLength2);
    if (dot(macroNormal, halfVector) <= 1e-6
            || dot(wo, halfVector) <= 1e-6)
        return vec3(0.0);

    // This floor exactly matches GGXVNDFNormal/evaluateSpecularBRDF.
    float alpha = max(ggxAlpha, 1e-4);
    float G1V = GGX_G1_standard(NoV, alpha);
    float G2 = GGX_G2_standard(NoV, NoL, alpha);
    return specularSurfaceFresnel(wo, halfVector, Cs, S, etaRatio)
        * (G2 / max(G1V, 1e-8));
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

    // The exact delta branch stores Li, so its limiting modulation is F.
    if (ggxAlpha <= SPECULAR_DELTA_ALPHA)
        return totalRgb * specularSurfaceFresnel(
            wo, macroNormal, Cs, S, etaRatio);

    float momentLength = length(signal.maxEntY.xyz);
    // rho=0 is the valid isotropic MaxEnt distribution, not an empty signal.
    vec3 axis = momentLength > 1e-10
        ? signal.maxEntY.xyz / momentLength : macroNormal;
    float rho = clamp(momentLength / totalY, 0.0, 1.0);

    // A single directional sample ceases to have exactly rho=1 after FP16
    // storage. Treat the representable neighbourhood of the sphere as the
    // directional atom it encodes; this also preserves the alpha->0 limit.
    if (1.0 - rho <= 2e-3)
        return totalRgb * specularVndfModulation(wo, axis,
            macroNormal, geometryNormal, ggxAlpha, Cs, S, etaRatio);

    float kappa = specularMaxEntKappa(rho);
    vec3 integral = vec3(0.0);
    for (int i = 0; i < SPECULAR_MAXENT_QUADRATURE_SAMPLES; ++i) {
        vec3 wi = specularMaxEntSample(axis, kappa,
            specularQuadraturePoint(i));
        integral += specularVndfModulation(wo, wi,
            macroNormal, geometryNormal, ggxAlpha, Cs, S, etaRatio);
    }

    // CoCg is angularly shared. Because YCoCg->RGB is linear, reconstructing
    // RGB per direction and integrating is exactly totalRgb times E[K].
    return totalRgb * (integral
        / float(SPECULAR_MAXENT_QUADRATURE_SAMPLES));
}

#endif // LIGHTING_SPECULAR_MAXENT_GLSL
