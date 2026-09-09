#ifndef DIRT_RT_LIB_PBR_GGX_GLSL
#define DIRT_RT_LIB_PBR_GGX_GLSL

// Isotropic GGX distribution, Smith masking and visible-normal sampling.
#include "/lib/constants.glsl"
#include "/lib/math/random.glsl"
#include "/lib/math/sampling.glsl"
// NoX * (1 + 2*Lambda(NoX)); avoids tan(theta)^2 and its two divisions.
float GGX_smithRoot(float NoX, float alpha) {
    float a2 = alpha * alpha;
    return sqrt(fma(1.0 - a2, NoX * NoX, a2));
}

float GGX_G2(float VoN, float LoN, float a) {
    float v = max(abs(VoN), 1e-6);
    float l = max(abs(LoN), 1e-6);
    float sv = GGX_smithRoot(v, a);
    float sl = GGX_smithRoot(l, a);
    return l * (v + sv) / (l * sv + v * sl);
}

// Shared by the VNDF path sampler and the MaxEnt reconstruction. The finite
// branch is continuous at this boundary because its modulation tends to F.
const float SPECULAR_DELTA_ALPHA = 1e-5;

// Heitz's isotropic GGX visible-normal sampler. viewDirection points from the
// surface toward the previous vertex and must lie in macroNormal's hemisphere.
// Samples D_visible(H | V), not the unconditioned NDF D(H) * NoH.
vec3 GGXVNDFNormal(vec3 macroNormal, vec3 viewDirection,
        float roughness, vec2 xi) {
    vec3 N = normalize(macroNormal);
    vec3 V = normalize(viewDirection);
    if (dot(N, V) < 0.0) N = -N;

    vec3 T, B;
    orthonormalBasis(N, T, B);
    vec3 localV = vec3(dot(V, T), dot(V, B), max(dot(V, N), 1e-6));

    float alpha = max(roughness, 1e-4);
    vec3 stretchedV = normalize(vec3(alpha * localV.xy, localV.z));
    float lensq = dot(stretchedV.xy, stretchedV.xy);
    vec3 T1 = lensq > 1e-12
        ? vec3(-stretchedV.y, stretchedV.x, 0.0) * inversesqrt(lensq)
        : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(stretchedV, T1);

    float radius = sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;
    float t1 = radius * cos(phi);
    float t2 = radius * sin(phi);
    float blend = 0.5 * (1.0 + stretchedV.z);
    t2 = mix(sqrt(max(1.0 - t1 * t1, 0.0)), t2, blend);
    vec3 stretchedH = t1 * T1 + t2 * T2
        + sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0)) * stretchedV;
    vec3 localH = normalize(vec3(
        alpha * stretchedH.xy, max(stretchedH.z, 0.0)));
    return normalize(T * localH.x + B * localH.y + N * localH.z);
}

vec3 GGXVNDFNormal(vec3 macroNormal, vec3 viewDirection,
        float roughness, vec3 pos) {
    return GGXVNDFNormal(macroNormal, viewDirection, roughness, rand2(pos));
}

float GGX_D(float costheta, float a) {
    float NoH = clamp(costheta, 0.0, 1.0);
    float alpha = max(a, 1e-4);
    float a2 = alpha * alpha;
    // Preserve alpha^2 at NoH=1. Subtracting it from 1 first rounds it
    // away in FP32 for narrow lobes, spuriously hitting the denominator floor.
    float b = fma(a2, NoH * NoH, (1.0 - NoH) * (1.0 + NoH));
    return a2 / max(PI * b * b, 1e-20);
}

#endif // DIRT_RT_LIB_PBR_GGX_GLSL
