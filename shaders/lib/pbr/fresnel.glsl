#ifndef DIRT_RT_LIB_PBR_FRESNEL_GLSL
#define DIRT_RT_LIB_PBR_FRESNEL_GLSL

// Color importance and interface Fresnel; eta is eta_i / eta_t.

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

vec4 reflectanceColor(vec3 c, float cosA) {
    vec3 F0 = c + (1.0 - c) * pow5(1.0 - clamp(abs(cosA), 0.0, 1.0));
    return vec4(F0, luma(F0));
}

float mixp(float F, float S) {
    return F * S / max(1 + (S - 1) * F, 1e-5);
}
float fresnel(vec3 v, vec3 n, float rs) {
    if (rs == 1.0) return 0.0;
    vec2 A;
    A.x = clamp(abs(dot(v, n)), 0.0, 1.0);
    A.y = sqrt(max(1 - (1 - A.x * A.x) * (rs * rs), 0));
    A = (A * rs - A.yx) / max(A * rs + A.yx, 1e-4);
    return 0.5 * dot(A, A);
}

// The binary material cases avoid evaluating the unused Fresnel model.
vec3 surfaceFresnel(vec3 wo, vec3 H, vec3 Cs, float Sx,
        float transmissionSelector, float etaRatio) {
    if (transmissionSelector <= 0.0)
        return reflectanceColor(Cs, dot(wo, H)).rgb * Sx;
    float dielectricF = fresnel(wo, H, etaRatio);
    if (transmissionSelector >= 1.0) return vec3(dielectricF);
    return mix(reflectanceColor(Cs, dot(wo, H)).rgb * Sx,
        vec3(dielectricF), transmissionSelector);
}

#endif // DIRT_RT_LIB_PBR_FRESNEL_GLSL
