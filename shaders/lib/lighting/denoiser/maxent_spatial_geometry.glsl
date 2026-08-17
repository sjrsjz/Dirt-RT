#ifndef MAXENT_SPATIAL_GEOMETRY_GLSL
#define MAXENT_SPATIAL_GEOMETRY_GLSL

// Canonical denoiser-geometry adapter shared by all spatial signals.
//
// colortex3 RGBA32UI ABI, produced by maxent_variance_prepare.glsl:
//   x = floatBitsToUint(primary ray distance), negative means invalid/sky
//   y = oct32 sampling-PDF dominant outgoing direction
//   z = oct32 primary ray direction
//   w = half(perceptual roughness) | half(specular virtual scale)
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
    float distance = uintBitsToFloat(words.x);
    return distance >= 0.0 && !isnan(distance) && !isinf(distance);
}

DenoiserSpatialGeometry denoiserSpatialDecodeGeometry(
        uvec4 words, ivec2 pixel) {
    DenoiserSpatialGeometry geometry;
    float distance = uintBitsToFloat(words.x);
    geometry.pdfDirection = decodeNormalU(words.y);
    geometry.primaryRay = decodeNormalU(words.z);
    vec2 roughnessVirtual = unpackHalf2x16(words.w);
    // Validity and the [0, 1] ranges are owned by the variance-preparation
    // producer. Spatial consumers reject invalid geometry before use.
    geometry.surfaceDistance = distance;
    geometry.roughness = roughnessVirtual.x;
    geometry.virtualScale = roughnessVirtual.y;
    geometry.ggxAlpha = geometry.roughness * geometry.roughness;
    geometry.valid = denoiserSpatialGeometryWordsValid(words);
    return geometry;
}

bool denoiserSpatialTryVirtualWorldPositionFromWords(
        uvec4 geometryWords, uvec4 signalWords, out vec3 position) {
    float distance = uintBitsToFloat(geometryWords.x);
    if (!denoiserSpatialGeometryWordsValid(geometryWords)
            || !denoiserSpatialSignalWordsValid(signalWords)) {
        position = vec3(0.0);
        return false;
    }
    float virtualScale = unpackHalf2x16(geometryWords.w).y;
    float hitDistance = unpackHalf2x16(signalWords.w).y;
    position = decodeNormalU(geometryWords.z)
        * (distance + virtualScale * hitDistance);
    return true;
}

vec3 denoiserSpatialVirtualWorldPositionFromWords(
        uvec4 geometryWords, uvec4 signalWords, vec3 fallback) {
    vec3 position;
    return denoiserSpatialTryVirtualWorldPositionFromWords(
        geometryWords, signalWords, position) ? position : fallback;
}

#endif // MAXENT_SPATIAL_GEOMETRY_GLSL
