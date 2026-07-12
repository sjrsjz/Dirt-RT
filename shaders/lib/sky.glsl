#ifndef SKY_GLSL
#define SKY_GLSL
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/common.glsl"

float S_R = 0.0;
float cosD_S = 0.0;
vec3 b_P = vec3(0); // atmosphere thickness
float b_k = 0.; // mix

vec3 Mie = vec3(0);
vec3 Rayleigh = vec3(0);
vec3 b_k0 = vec3(0);
vec3 b_Q = vec3(0); // absorption
vec3 b_g0 = vec3(0); // single scatter

const float P = 30000.;
const float R = 6370000;
const vec3 Sun = vec3(1000);
const vec3 Moon = Sun * 0.0001;
float celestial_strength = 0.0;

void setSkyVars() {
    switch (world_type_global) {
        case WORLD_THE_END:
        S_R = 0.25;
        cosD_S = 1.0 / sqrt(1.0 + S_R * S_R);
        Rayleigh = vec3(5.8e-6, 1.35e-5, 3.31e-5);
        Mie = vec3(luma(Rayleigh));
        b_P = vec3(30000);
        b_k = 0.9;
        break;
        case WORLD_THE_NETHER:
        S_R = 0.05;
        cosD_S = 1.0 / sqrt(1.0 + S_R * S_R);
        Rayleigh = 100.0 * vec3(5.8e-6, 1.35e-4, 1.35e-4);
        Mie = vec3(luma(Rayleigh));
        b_P = vec3(30000);
        b_k = 0.5;
        break;
        default:
        S_R = 0.05;
        cosD_S = 1.0 / sqrt(1.0 + S_R * S_R);
        Rayleigh = vec3(5.8e-6, 1.35e-5, 3.31e-5);
        Mie = vec3(luma(Rayleigh));
        b_P = vec3(30000);
        b_k = 0.1 + rainStrength_global * 0.9;
        break;
    }

    b_k0 = mix(Rayleigh, Mie, b_k);
    b_Q = b_k0 / b_P; // absorption
    b_g0 = mix(Rayleigh, Mie, b_k); // single scatter

    celestial_strength = 4.0 / (5.0 * PI * max(1e-6, 1.0 - cosD_S));
}

float distance_to_surface(float R, float y, float A) {
    return max(sqrt(R * R - y * y * (1.0 - A * A)) - y * A, 0.0);
}

// 分析式大气天空 (含日盘) — NEE 光源采样用
vec3 sampleSky(float pos_y, in vec3 n, in vec3 lightDir) {
    lightDir = -normalize(lightDir);
    vec3 moonDir = -lightDir;
    vec3 n0 = n;

    mediump vec3 b_g0_2 = b_g0 * b_g0;

    mediump float dot_n_L = dot(n, lightDir);
    mediump vec3 tmp_x = 1.0 + b_g0_2 - 2.0 * b_g0 * dot_n_L;
    tmp_x *= tmp_x * tmp_x;
    mediump vec3 g_sun = 3.0 / (8.0 * PI) * (1.0 + dot_n_L * dot_n_L) * (1.0 - b_g0_2) / (2.0 + b_g0_2) * inversesqrt(tmp_x);

    mediump float dot_n_M = dot(n, moonDir);
    mediump vec3 tmp_x_moon = 1.0 + b_g0_2 - 2.0 * b_g0 * dot_n_M;
    tmp_x_moon *= tmp_x_moon * tmp_x_moon;
    mediump vec3 g_moon = 3.0 / (8.0 * PI) * (1.0 + dot_n_M * dot_n_M) * (1.0 - b_g0_2) / (2.0 + b_g0_2) * inversesqrt(tmp_x_moon);

    vec3 t = b_Q * 0.5 * (P - pos_y);

    float n_distance = distance_to_surface(P + R, R + pos_y, n.y);
    float s_distance = distance_to_surface(P + R, R + pos_y, lightDir.y);
    float m_distance = distance_to_surface(P + R, R + pos_y, moonDir.y);

    vec3 sun_intersect = vec3(0.0, R + pos_y, 0.0) - lightDir * s_distance;
    vec3 moon_intersect = vec3(0.0, R + pos_y, 0.0) - moonDir * m_distance;

    vec3 sun_normal = normalize(sun_intersect);
    vec3 moon_normal = normalize(moon_intersect);

    // 太阳散射
    vec3 c_sun = Sun * g_sun;
    c_sun *= abs((exp(-t * n_distance) - exp(-t * s_distance)) / (n.y - lightDir.y)) * max(dot(lightDir, sun_normal), 0.0);

    // 日盘
    float disc_core = 1.0 - (1.0 - cosD_S) * 0.25;
    vec3 disc = celestial_strength * exp(-t * n_distance) * smoothstep(cosD_S, disc_core, dot(n0, lightDir));
    c_sun += Sun * disc;

    // 月亮散射
    vec3 c_moon = Moon * g_moon;
    c_moon *= abs((exp(-t * n_distance) - exp(-t * m_distance)) / (n.y - moonDir.y)) * max(dot(moonDir, moon_normal), 0.0);

    // 月盘
    c_moon += Moon * disc;

    return max(c_sun + c_moon, 0.0);
}

// 无日盘天空 — PT pass 非 NEE 光线天空命中用
vec3 sampleSkyNoSun(float pos_y, in vec3 n, in vec3 lightDir) {
    lightDir = -normalize(lightDir);
    vec3 moonDir = -lightDir;

    mediump vec3 b_g0_2 = b_g0 * b_g0;

    mediump float dot_n_L = dot(n, lightDir);
    mediump vec3 tmp_x = 1.0 + b_g0_2 - 2.0 * b_g0 * dot_n_L;
    tmp_x *= tmp_x * tmp_x;
    mediump vec3 g_sun = 3.0 / (8.0 * PI) * (1.0 + dot_n_L * dot_n_L) * (1.0 - b_g0_2) / (2.0 + b_g0_2) * inversesqrt(tmp_x);

    mediump float dot_n_M = dot(n, moonDir);
    mediump vec3 tmp_x_moon = 1.0 + b_g0_2 - 2.0 * b_g0 * dot_n_M;
    tmp_x_moon *= tmp_x_moon * tmp_x_moon;
    mediump vec3 g_moon = 3.0 / (8.0 * PI) * (1.0 + dot_n_M * dot_n_M) * (1.0 - b_g0_2) / (2.0 + b_g0_2) * inversesqrt(tmp_x_moon);

    vec3 t = b_Q * 0.5 * (P - pos_y);

    float n_distance = distance_to_surface(P + R, R + pos_y, n.y);
    float s_distance = distance_to_surface(P + R, R + pos_y, lightDir.y);
    float m_distance = distance_to_surface(P + R, R + pos_y, moonDir.y);

    vec3 sun_intersect = vec3(0.0, R + pos_y, 0.0) - lightDir * s_distance;
    vec3 moon_intersect = vec3(0.0, R + pos_y, 0.0) - moonDir * m_distance;

    vec3 sun_normal = normalize(sun_intersect);
    vec3 moon_normal = normalize(moon_intersect);

    vec3 c_sun = Sun * g_sun;
    c_sun *= abs((exp(-t * n_distance) - exp(-t * s_distance)) / (n.y - lightDir.y)) * max(dot(lightDir, sun_normal), 0.0);

    vec3 c_moon = Moon * g_moon;
    c_moon *= abs((exp(-t * n_distance) - exp(-t * m_distance)) / (n.y - moonDir.y)) * max(dot(moonDir, moon_normal), 0.0);

    return max(c_sun + c_moon, 0.0);
}

#endif // LIGHT_COLOR_GLSL
