#ifndef MAXENT_DENOISER_VIRTUAL_PROJECTION_GLSL
#define MAXENT_DENOISER_VIRTUAL_PROJECTION_GLSL

#include "/lib/lighting/denoiser/internal_constants.glsl"

vec3 denoiserSpatialSafeDirection(vec3 direction, vec3 fallback) {
    float length2 = dot(direction, direction);
    return length2 > 1e-20
        ? direction * inversesqrt(length2) : fallback;
}

float denoiserSpatialDistanceRejectionScale(
        float centerSurfaceDistance, float resolutionY) {
    resolutionY = max(resolutionY, 1.0);
    float centerDistance = max(centerSurfaceDistance, 0.001);
    float footprintDistance = max(centerDistance, resolutionY * 1e-5);
    return resolutionY / max(
        float(MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE) * footprintDistance,
        resolutionY * 1e-6);
}

float denoiserSpatialAxialDistanceExponent(
        float centerDirectionOffset, vec3 centerDirection,
        vec3 samplePrimaryRay, float sampleSurfaceDistance,
        float rejectionScale) {
    float sampleDirectionOffset = sampleSurfaceDistance
        * dot(centerDirection, samplePrimaryRay);
    return rejectionScale
        * abs(sampleDirectionOffset - centerDirectionOffset);
}

float denoiserSpatialPdfDirectionExponent(vec3 centerDirection, vec3 sampleDirection) {
    return MAXENT_SPATIAL_PDF_DIRECTION_EXPONENT_SCALE
        * (1.0 - clamp(dot(centerDirection, sampleDirection), -1.0, 1.0));
}

// Same view-conditioned virtual-projection scale as specular temporal
// reprojection. It is evaluated once during variance preparation.
float denoiserSpatialSpecularVirtualScale(vec3 primaryRay,
        vec3 geometryNormal, float perceptualRoughness) {
    vec3 ray = denoiserSpatialSafeDirection(
        primaryRay, vec3(0.0, 0.0, -1.0));
    vec3 normal = denoiserSpatialSafeDirection(
        geometryNormal, vec3(0.0, 1.0, 0.0));
    float NoV = abs(dot(normal, -ray));
    float roughness = perceptualRoughness;
    float a = 0.298475 * log(39.4115 - 39.0029 * roughness);
    return clamp(pow(clamp(1.0 - NoV, 0.0, 1.0), 10.8649)
        * (1.0 - a) + a, 0.0, 1.0);
}

// Dominant direction of the view-conditioned GGX reflection PDF. primaryRay
// points from the camera to the first surface, so reflect(primaryRay, N) is
// the outgoing reflection direction. The fitted factor is the same GGX VNDF
// dominant-direction fit used by the virtual projection.
vec3 denoiserSpatialSpecularPdfDirectionFromFactor(vec3 primaryRay,
        vec3 geometryNormal, float dominantFactor) {
    vec3 ray = denoiserSpatialSafeDirection(primaryRay, vec3(0.0, 0.0, -1.0));
    vec3 normal = denoiserSpatialSafeDirection(geometryNormal, vec3(0.0, 1.0, 0.0));
    return denoiserSpatialSafeDirection(mix(normal, reflect(ray, normal), dominantFactor), normal);
}

#endif // MAXENT_DENOISER_VIRTUAL_PROJECTION_GLSL
