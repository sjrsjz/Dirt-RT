#ifndef COLORS_GLSL
#define COLORS_GLSL

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float calculateExposure(float avgLuminance) {
    return 1.0 / (9.6 * avgLuminance);
}

#endif // COLORS_GLSL