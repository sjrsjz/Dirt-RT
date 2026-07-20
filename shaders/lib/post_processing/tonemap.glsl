#ifndef TONEMAP_GLSL
#define TONEMAP_GLSL

vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ===========================================================================
// TonyMcMapface MLP32 Tonemapper Constants
// Architecture: 4 -> 32 -> 3
// Params: 259
// C1 Continuous: SiLU + Softplus
// ===========================================================================

const mat4 W_H0 = mat4(0.17220883f, -0.89687788f, 0.07870596f, -3.57016397f, -0.18302338f, 0.39321777f, 1.37373221f, 1.35187447f, -1.70556712f, 0.17347254f, -0.70345992f, 0.14414795f, -0.14490041f, -5.90188694f, -3.24553657f, -5.97503090f);
const mat4 W_H1 = mat4(0.03309277f, -1.55753350f, -0.30574444f, 0.65191174f, 0.11134207f, 0.17784463f, -1.24469483f, 0.90026641f, 0.01125838f, -1.19826531f, -0.96134710f, -0.37302950f, 0.00013473f, 0.01511161f, 3.73411894f, -3.73947072f);
const mat4 W_H2 = mat4(-1.51333261f, -0.02660427f, 0.21807751f, 0.41123623f, -0.77061456f, -0.08950084f, 0.91506195f, 0.47997180f, -0.18867685f, -0.00902512f, 0.18835203f, -0.13139100f, 4.04136515f, 0.00026118f, -3.08433056f, -1.97823000f);
const mat4 W_H3 = mat4(-0.85396767f, 0.02370699f, 0.13429005f, -0.54744345f, -2.21329093f, -0.44129482f, 0.98073047f, 1.17667592f, -1.52906001f, 0.10070935f, -5.59534788f, -0.08233189f, 5.03818750f, -7.61004400f, -2.77956939f, -3.58878708f);
const mat4 W_H4 = mat4(0.53315401f, -0.08024634f, -1.69016290f, 0.36039159f, 1.99527156f, 1.41596556f, 0.62067997f, 1.22255301f, 0.01113263f, 0.14326632f, -2.57122707f, -0.10082172f, -6.29794598f, -3.02064157f, -0.65179080f, -2.92241049f);
const mat4 W_H5 = mat4(0.04512897f, -0.07269895f, 0.47952944f, -1.53016019f, 0.60040289f, -0.81901938f, 1.15403426f, -0.86586475f, -1.69413018f, -0.12078610f, 0.09762014f, 0.14889823f, -0.66741490f, 1.88922799f, -3.28159046f, -0.29605243f);
const mat4 W_H6 = mat4(-0.13825643f, -2.81689262f, -0.02195428f, -2.20566130f, 0.96907127f, -0.86397815f, -0.07382919f, -1.31941354f, -0.40615675f, 0.05047470f, -0.00748885f, -0.52221060f, 0.08091091f, 1.94926107f, -0.00082508f, 3.50773859f);
const mat4 W_H7 = mat4(-1.48828614f, 0.55854470f, 0.63562036f, 0.45000061f, 0.94871575f, 0.72863239f, -1.58016860f, 1.35044074f, -0.40667894f, -0.85600096f, -0.12810165f, -0.16593079f, -1.71059191f, -1.21011209f, -7.02386618f, -4.53511238f);

const vec4 B_H0 = vec4(-0.03212728f, -0.12741597f, 1.27997160f, 0.89963150f);
const vec4 B_H1 = vec4(-0.15576406f, -0.31193820f, -1.25281036f, 0.11391133f);
const vec4 B_H2 = vec4(-1.84374750f, 0.12492187f, 0.23290648f, 0.45987171f);
const vec4 B_H3 = vec4(-2.62117386f, -0.02498293f, 0.39910612f, 1.46952462f);
const vec4 B_H4 = vec4(0.64687061f, 0.78832901f, 0.99078590f, 0.17632824f);
const vec4 B_H5 = vec4(0.20900369f, -1.06635606f, 0.26272020f, -1.06674540f);
const vec4 B_H6 = vec4(0.02672042f, -1.56759191f, 0.10389147f, -2.22280383f);
const vec4 B_H7 = vec4(0.86010957f, 0.53962404f, -0.52685189f, 0.50813138f);

const mat4x3 W_O0 = mat4x3(0.16297528f, 0.27415916f, -0.12237209f, -0.35404983f, -0.05451852f, 0.63655829f, 0.17715570f, 0.18228154f, -0.60370392f, -0.27463967f, 0.04700392f, -0.02069514f);
const mat4x3 W_O1 = mat4x3(0.00001696f, 0.00002367f, 0.00001452f, -2.42461658f, 0.30889636f, -0.21804303f, -0.00894637f, -0.08882921f, 1.31761467f, -0.32948405f, -0.46498209f, 0.42016876f);
const mat4x3 W_O2 = mat4x3(0.56540394f, -0.02883837f, -0.00774096f, 0.00000196f, 0.00001653f, 0.00000364f, 0.20317355f, -0.12175427f, 1.28268063f, 0.52631706f, -0.25248200f, 0.01564907f);
const mat4x3 W_O3 = mat4x3(0.46311215f, -0.06014687f, 4.36588001f, 0.32702148f, -0.63597095f, 2.27990985f, -0.02087253f, 0.01444990f, -0.63722968f, -0.87991935f, 0.07228399f, 0.05196380f);
const mat4x3 W_O4 = mat4x3(0.23338057f, -0.29369789f, 1.16607642f, -1.09753752f, 0.49664265f, 0.18832590f, 0.50291049f, -0.01166486f, -0.05036644f, 0.03295832f, -0.08245452f, -1.82669497f);
const mat4x3 W_O5 = mat4x3(-0.29230669f, -0.30265021f, -1.41057384f, -0.31403840f, 0.14490381f, -0.08226181f, 0.77149445f, -0.92214715f, 0.70909983f, 0.09952506f, 0.13429405f, -0.24633411f);
const mat4x3 W_O6 = mat4x3(-0.03166604f, 0.25120756f, 0.25722957f, -0.22078298f, -0.15864460f, 0.07456334f, -0.00001994f, -0.00003040f, -0.00001018f, 3.43367553f, 0.44156790f, 0.70023507f);
const mat4x3 W_O7 = mat4x3(-0.67123586f, -0.29137522f, -0.14854398f, -0.02907328f, 0.03320853f, -0.43614498f, 0.23234607f, -0.48140821f, -0.13557516f, 0.29889819f, 1.04827595f, -1.68784535f);

const vec3 B_O = vec3(0.70168769f, 0.39938542f, 0.91512948f);

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

    vec4 h0 = tony_silu(W_H0 * X + B_H0);
    vec4 h1 = tony_silu(W_H1 * X + B_H1);
    vec4 h2 = tony_silu(W_H2 * X + B_H2);
    vec4 h3 = tony_silu(W_H3 * X + B_H3);
    vec4 h4 = tony_silu(W_H4 * X + B_H4);
    vec4 h5 = tony_silu(W_H5 * X + B_H5);
    vec4 h6 = tony_silu(W_H6 * X + B_H6);
    vec4 h7 = tony_silu(W_H7 * X + B_H7);

    vec3 scale =
        W_O0 * h0
      + W_O1 * h1
      + W_O2 * h2
      + W_O3 * h3
      + W_O4 * h4
      + W_O5 * h5
      + W_O6 * h6
      + W_O7 * h7;

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