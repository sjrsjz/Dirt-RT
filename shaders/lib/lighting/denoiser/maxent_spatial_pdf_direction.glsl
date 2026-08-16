#ifndef MAXENT_SPATIAL_PDF_DIRECTION_GLSL
#define MAXENT_SPATIAL_PDF_DIRECTION_GLSL

vec3 denoiserSpatialSafeDirection(vec3 direction, vec3 fallback) {
    float length2 = dot(direction, direction);
    return length2 > 1e-20
        ? direction * inversesqrt(length2) : fallback;
}

// Dominant outgoing direction of the primary GGX VNDF proposal. This matches
// the roughness-dependent dominant-direction approximation used by the
// radiance-cache specular lookup: a delta lobe follows mirror reflection and
// a maximally rough lobe converges to the macro normal.
vec3 denoiserSpatialGgxVndfDominantDirection(vec3 primaryRay,
        vec3 macroNormal, float ggxAlpha) {
    vec3 normal = denoiserSpatialSafeDirection(
        macroNormal, vec3(0.0, 1.0, 0.0));
    vec3 incident = denoiserSpatialSafeDirection(primaryRay, -normal);
    // GGXVNDFNormal applies the same face-forward convention to the macro
    // normal before sampling the visible-normal distribution.
    if (dot(normal, -incident) < 0.0) normal = -normal;
    float perceptualRoughness = sqrt(clamp(ggxAlpha, 0.0, 1.0));
    float dominantFactor = (1.0 - perceptualRoughness)
        * (sqrt(max(1.0 - perceptualRoughness, 0.0))
            + perceptualRoughness);
    vec3 mirrorDirection = reflect(incident, normal);
    return denoiserSpatialSafeDirection(
        mix(normal, mirrorDirection, dominantFactor), normal);
}

float denoiserSpatialPdfDirectionExponent(vec3 centerDirection,
        vec3 sampleDirection) {
    float directionCosine = clamp(dot(centerDirection, sampleDirection),
        -1.0, 1.0);
    return max(MAXENT_SPATIAL_PDF_DIRECTION_SENSITIVITY, 0.0)
        * (1.0 - directionCosine);
}

// Same view-conditioned virtual-projection scale as specular temporal
// reprojection. It is evaluated once during variance preparation and packed
// into the otherwise unused spatial-geometry word.
float denoiserSpatialSpecularVirtualScale(vec3 primaryRay,
        vec3 geometryNormal, float perceptualRoughness) {
    vec3 ray = denoiserSpatialSafeDirection(
        primaryRay, vec3(0.0, 0.0, -1.0));
    vec3 normal = denoiserSpatialSafeDirection(
        geometryNormal, vec3(0.0, 1.0, 0.0));
    float NoV = abs(dot(normal, -ray));
    float roughness = clamp(perceptualRoughness, 0.0, 1.0);
    float a = 0.298475 * log(max(39.4115 - 39.0029 * roughness, 1e-5));
    return clamp(pow(clamp(1.0 - NoV, 0.0, 1.0), 10.8649)
        * (1.0 - a) + a, 0.0, 1.0);
}

#endif // MAXENT_SPATIAL_PDF_DIRECTION_GLSL
