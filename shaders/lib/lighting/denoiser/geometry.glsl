#ifndef MAXENT_DENOISER_GEOMETRY_GLSL
#define MAXENT_DENOISER_GEOMETRY_GLSL

// Canonical denoiser-geometry adapter shared by all spatial signals.
//
// RGBA32UI ABI produced by variance_prepare.glsl:
//   x = floatBitsToUint(first-surface distance), negative means invalid/sky
//   y = oct32 first-surface geometry normal
//   z = oct32 estimated sampling-PDF direction
//   w = half(perceptual roughness) | unused half
// The primary ray is reconstructed from the pixel coordinate instead of being
// stored, leaving both independent directions at full oct32 precision.
//
// The variance-preparation policy owns the signal roughness: diffuse writes
// exactly 1.0, while specular writes sqrt(primary GGX alpha).
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/lighting/denoiser/signal.glsl"

bool denoiserSpatialGeometryWordsValid(uvec4 words) {
    return uintBitsToFloat(words.x) >= 0.0;
}

struct DenoiserSpatialCenterGeometry {
    vec3 geometryNormal;
    vec3 pdfDirection;
    vec3 primaryRay;
    float surfaceDistance;
    float surfacePlaneOffset;
    float ggxAlpha;
};

DenoiserSpatialCenterGeometry denoiserSpatialDecodeCenterGeometry(
        uvec4 words, ivec2 pixel) {
    DenoiserSpatialCenterGeometry geometry;
    geometry.surfaceDistance = uintBitsToFloat(words.x);
    geometry.geometryNormal = decodeNormalU(words.y);
    geometry.pdfDirection = decodeNormalU(words.z);
    geometry.primaryRay = reconstructPrimaryRay(uvec2(pixel));
    geometry.surfacePlaneOffset = geometry.surfaceDistance
        * dot(geometry.geometryNormal, geometry.primaryRay);
    float roughness = unpackHalf2x16(words.w).x;
    geometry.ggxAlpha = roughness * roughness;
    return geometry;
}

void denoiserSpatialDecodeSampleGeometry(uvec4 words, ivec2 pixel,
        out vec3 primaryRay, out vec3 pdfDirection,
        out float surfaceDistance) {
    primaryRay = reconstructPrimaryRay(uvec2(pixel));
    pdfDirection = decodeNormalU(words.z);
    surfaceDistance = uintBitsToFloat(words.x);
}

bool denoiserSpatialTryVirtualWorldPositionFromWords(
        ivec2 pixel, uvec4 signalWords, out vec3 position) {
    if (!denoiserSpatialSignalWordsValid(signalWords)) {
        position = vec3(0.0);
        return false;
    }
    float virtualDistance = unpackHalf2x16(signalWords.w).y;
    position = reconstructPrimaryRay(uvec2(pixel)) * virtualDistance;
    return true;
}

vec3 denoiserSpatialVirtualWorldPositionFromWords(
        ivec2 pixel, uvec4 signalWords, vec3 fallback) {
    vec3 position;
    return denoiserSpatialTryVirtualWorldPositionFromWords(
        pixel, signalWords, position) ? position : fallback;
}

#endif // MAXENT_DENOISER_GEOMETRY_GLSL
