#ifndef MAXENT_SPATIAL_VIRTUAL_PROJECTION_GLSL
#define MAXENT_SPATIAL_VIRTUAL_PROJECTION_GLSL

vec3 denoiserSpatialSafeDirection(vec3 direction, vec3 fallback) {
    float length2 = dot(direction, direction);
    return length2 > 1e-20
        ? direction * inversesqrt(length2) : fallback;
}

float denoiserSpatialSurfaceRejectionScale(
        float centerSurfaceDistance, float resolutionY) {
    resolutionY = max(resolutionY, 1.0);
    float centerDistance = max(centerSurfaceDistance, 0.001);
    float footprintDistance = max(centerDistance, resolutionY * 1e-5);
    return resolutionY / max(
        float(MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE) * footprintDistance,
        resolutionY * 1e-6);
}

float denoiserSpatialSurfacePlaneDepthExponent(
        float centerPlaneOffset, vec3 centerSurfaceNormal,
        vec3 samplePrimaryRay, float sampleSurfaceDistance,
        float surfaceRejectionScale) {
    float samplePlaneOffset = sampleSurfaceDistance
        * dot(centerSurfaceNormal, samplePrimaryRay);
    return surfaceRejectionScale
        * abs(samplePlaneOffset - centerPlaneOffset);
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

#endif // MAXENT_SPATIAL_VIRTUAL_PROJECTION_GLSL
