#ifndef SMOOTHERSTEP_GLSL
#define SMOOTHERSTEP_GLSL

float smootherstep01(float x) {
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}
vec2 smootherstep01(vec2 v) {
    return vec2(smootherstep01(v.x), smootherstep01(v.y));
}

vec3 smootherstep01(vec3 v) {
    return vec3(smootherstep01(v.x), smootherstep01(v.y), smootherstep01(v.z));
}

float evenSmootherstep01(float x) {
    return x * x * x * x * (35.0 + x * (-84.0 + x * (70.0 + x * (-20.0 + x))));
}

vec2 evenSmootherstep01(vec2 v) {
    return vec2(evenSmootherstep01(v.x), evenSmootherstep01(v.y));
}

vec3 evenSmootherstep01(vec3 v) {
    return vec3(evenSmootherstep01(v.x), evenSmootherstep01(v.y), evenSmootherstep01(v.z));
}

#endif // SMOOTHERSTEP_GLSL