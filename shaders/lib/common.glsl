#ifndef DIRT_RT_LIB_COMMON_GLSL
#define DIRT_RT_LIB_COMMON_GLSL

// Compatibility facade. New modules should include their direct dependencies.

#include "/lib/constants.glsl"
#include "/lib/math/random.glsl"
#include "/lib/math/sampling.glsl"
#include "/lib/math/noise.glsl"
#include "/lib/pbr/fresnel.glsl"
#include "/lib/pbr/ggx.glsl"

vec2 rot(vec2 a, float theta) {
    return a.xx * vec2(cos(theta), sin(theta)) + a.yy * vec2(-sin(theta), cos(theta));
}

float ensurePositive(float x, float defaultValue) {
    return (x <= 0.0 || isnan(x) || isinf(x)) ? defaultValue : x;
}

#endif // DIRT_RT_LIB_COMMON_GLSL
