#ifndef MAXENT_SPATIAL_GEOMETRY_GLSL
#define MAXENT_SPATIAL_GEOMETRY_GLSL

// Canonical denoiser-geometry adapter shared by all spatial signals.
//
// colortex3 RGBA32UI ABI, produced by maxent_variance_prepare.glsl:
//   x = floatBitsToUint(primary ray distance), negative means invalid/sky
//   y = oct32 sampling-PDF dominant outgoing direction
//   z = half(specular virtual scale) | half(GGX alpha)
//   w = half(perceptual roughness) | materialID16
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

DenoiserSpatialGeometry denoiserSpatialDecodeGeometry(
    uvec4 words, ivec2 pixel) {
    DenoiserSpatialGeometry geometry;
    float distance = uintBitsToFloat(words.x);
    geometry.pdfDirection = decodeNormalU(words.y);
    vec2 virtualAlpha = unpackHalf2x16(words.z);
    geometry.surfaceDistance = max(distance, 0.0);
    geometry.virtualScale = clamp(virtualAlpha.x, 0.0, 1.0);
    geometry.ggxAlpha = clamp(virtualAlpha.y, 0.0, 1.0);
    geometry.roughness = clamp(unpackHalf2x16(words.w).x, 0.0, 1.0);
    geometry.valid = distance >= 0.0
            && !isnan(distance) && !isinf(distance);
    return geometry;
}

#endif // MAXENT_SPATIAL_GEOMETRY_GLSL
