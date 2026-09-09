#ifndef TONEMAP_GLSL
#define TONEMAP_GLSL

// ===========================================================================
// TonyMcMapface MLP32 Tonemapper Constants
// Architecture: 4 -> 32 -> 3
// Params: 259
// C1 Continuous: SiLU + Softplus
// ===========================================================================

#include "/lib/post_processing/tonemap_coefficients.glsl"

vec4 tony_silu(vec4 x) {
    return x / (vec4(1.0f) + exp(-x));
}

vec3 tony_softplus(vec3 x) {
    // Numerically stable softplus.
    return max(x, vec3(0.0f)) + log(vec3(1.0f) + exp(-abs(x)));
}

vec3 TonyMcMapface_Tiny(vec3 hdrColor) {
    float luma = dot(hdrColor, vec3(0.2126f, 0.7152f, 0.0722f));
    float luma_rein = luma / (1.0f + luma);

    vec4 X = vec4(hdrColor / max(luma, 1e-6f), luma_rein);

    // Consume each hidden group immediately to shorten its live range.
    vec3 scale = W_O0 * tony_silu(W_H0 * X + B_H0);
    scale += W_O1 * tony_silu(W_H1 * X + B_H1);
    scale += W_O2 * tony_silu(W_H2 * X + B_H2);
    scale += W_O3 * tony_silu(W_H3 * X + B_H3);
    scale += W_O4 * tony_silu(W_H4 * X + B_H4);
    scale += W_O5 * tony_silu(W_H5 * X + B_H5);
    scale += W_O6 * tony_silu(W_H6 * X + B_H6);
    scale += W_O7 * tony_silu(W_H7 * X + B_H7);

    scale += B_O;
    scale = tony_softplus(scale);

    return luma_rein * scale;
}

float TonyMcMapface_LumaApprox(float x) {
    if (x <= 1e-20) return 0.0;
    float reinhard = x / (1.0 + x);
    float lnx = log(x);
    float diff = lnx - 2.901905;
    float exponent = -(diff * diff) / 5.355152;
    float residual = (0.185418 / x) * exp(exponent);
    return min(reinhard + residual, 1.0);
}

float TonyMcMapface_LumaApprox_Deriv(float x) {
    if (x <= 1e-20) return 0.0;

    float inv_one_plus_x = 1.0 / (1.0 + x);
    float reinhard = x * inv_one_plus_x;
    
    float lnx = log(x);
    float diff = lnx - 2.901905;
    float exponent = -(diff * diff) / 5.355152;
    float residual = (0.185418 / x) * exp(exponent);

    if (reinhard + residual >= 1.0) {
        return 0.0;
    }

    float d_reinhard = inv_one_plus_x * inv_one_plus_x;
    
    float d_residual = (residual / x) * (diff * -0.3734721 - 1.0);

    return d_reinhard + d_residual;
}

vec3 linear_to_srgb(vec3 linear_color) {
    // 限制在 0.0 - 1.0 范围内
    vec3 clapped = clamp(linear_color, 0.0, 1.0);
    
    // 标准 sRGB OETF 公式
    bvec3 cutoff = lessThanEqual(clapped, vec3(0.0031308));
    vec3 higher = 1.055 * pow(clapped, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = clapped * 12.92;
    
    return mix(higher, lower, cutoff);
}

vec3 apply_shadow_toe(vec3 x, float k) {
    // x: 输入的 HDR 颜色
    // k: 暗部压制系数

    float luma = dot(x, vec3(0.2126f, 0.7152f, 0.0722f));
    float lumaCurved = luma + log((1.0 + k * exp(-luma)) / (1.0 + k));
    x *= lumaCurved / max(luma, 1e-6);
    return x;
}

#endif // TONEMAP_GLSL