#ifndef DIRT_RT_LIB_MATH_NOISE_GLSL
#define DIRT_RT_LIB_MATH_NOISE_GLSL

// Procedural value noise and its analytic derivatives.
#include "/lib/math/hash.glsl"
float hash(float n)
{
    return fract(cos(n) * 41415.92653);
}

vec4 noised(in vec3 x)
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w * w * (3.0 - 2.0 * w);
    vec3 du = 6.0 * w * (1.0 - w);

    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    float a = hash(n + 0.0);
    float b = hash(n + 1.0);
    float c = hash(n + 157.0);
    float d = hash(n + 158.0);
    float e = hash(n + 113.0);
    float f = hash(n + 114.0);
    float g = hash(n + 270.0);
    float h = hash(n + 271.0);

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    return vec4(k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z,
        du * (vec3(k1, k2, k3) + u.yzx * vec3(k4, k5, k6) + u.zxy * vec3(k6, k4, k5) + k7 * u.yzx * u.zxy));
}

float valueNoise(vec3 position) {
    vec3 p = floor(position);
    vec3 f = fract(position);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(mix(mix(hash11(n + 0.0), hash11(n + 1.0), f.x),
            mix(hash11(n + 157.0), hash11(n + 158.0), f.x), f.y),
        mix(mix(hash11(n + 113.0), hash11(n + 114.0), f.x),
            mix(hash11(n + 270.0), hash11(n + 271.0), f.x), f.y), f.z);
}

vec4 fbm3D(in vec3 x, int n)
{
    const float scale = 1.5;

    float a = 0.0;
    float b = 0.5;
    float f = 1.0;
    vec3 d = vec3(0.0);
    for (int i = 0; i < n; i++)
    {
        vec4 n = noised(f * x * scale);
        a += b * n.x; // accumulate values
        d += b * n.yzw * f * scale; // accumulate derivatives
        b *= 0.5 / (1 + dot(d, d)); // amplitude decrease
        f *= 2; // frequency increase
    }

    return vec4(a, d);
}

#endif // DIRT_RT_LIB_MATH_NOISE_GLSL
