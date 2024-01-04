#ifndef UTILS_GLSL
#define UTILS_GLSL

float linearizeDepth(float depth, float near, float far) {
    return (near * far) / (depth * (near - far) + far);
}

#endif // UTILS_GLSL