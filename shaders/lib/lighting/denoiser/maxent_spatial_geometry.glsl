#ifndef MAXENT_SPATIAL_GEOMETRY_GLSL
#define MAXENT_SPATIAL_GEOMETRY_GLSL

// Canonical denoiser-geometry adapter shared by all spatial signals.
//
// colortex3 RGBA32UI ABI, produced by maxent_variance_prepare.glsl:
//   x = floatBitsToUint(first-surface distance), negative means invalid/sky
//   y = oct32 first-surface geometry normal
//   z = oct32 primary ray direction
//   w = half(perceptual roughness) | unused half
//
// The variance-preparation policy owns the signal roughness: diffuse writes
// exactly 1.0, while specular writes sqrt(primary GGX alpha).
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_common.glsl"

uniform usampler2D colortex3;

ivec2 denoiserSpatialImageSize() {
    return textureSize(colortex3, 0);
}

uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel) {
    return texelFetch(colortex3, pixel, 0);
}

bool denoiserSpatialGeometryWordsValid(uvec4 words) {
    return uintBitsToFloat(words.x) >= 0.0;
}

struct DenoiserSpatialCenterGeometry {
    vec3 surfaceNormal;
    vec3 primaryRay;
    float surfaceDistance;
    float surfacePlaneOffset;
    float ggxAlpha;
};

DenoiserSpatialCenterGeometry denoiserSpatialDecodeCenterGeometry(
        uvec4 words) {
    DenoiserSpatialCenterGeometry geometry;
    geometry.surfaceDistance = uintBitsToFloat(words.x);
    geometry.surfaceNormal = decodeNormalU(words.y);
    geometry.primaryRay = decodeNormalU(words.z);
    geometry.surfacePlaneOffset = geometry.surfaceDistance
        * dot(geometry.surfaceNormal, geometry.primaryRay);
    float roughness = unpackHalf2x16(words.w).x;
    geometry.ggxAlpha = roughness * roughness;
    return geometry;
}

void denoiserSpatialDecodeSampleGeometry(uvec4 words,
        out vec3 primaryRay, out float surfaceDistance) {
    primaryRay = decodeNormalU(words.z);
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

#endif // MAXENT_SPATIAL_GEOMETRY_GLSL
