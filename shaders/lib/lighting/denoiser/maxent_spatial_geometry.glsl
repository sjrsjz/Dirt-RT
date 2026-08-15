#ifndef MAXENT_SPATIAL_GEOMETRY_GLSL
#define MAXENT_SPATIAL_GEOMETRY_GLSL

// Canonical denoiser-geometry adapter shared by all spatial signals.
//
// colortex3 RGBA32UI ABI, produced by maxent_variance_prepare.glsl:
//   x = floatBitsToUint(primary ray distance), negative means invalid/sky
//   y = oct32 geometry normal
//   z = oct32 texture/macro normal
//   w = half(perceptual roughness) | materialID16
//
// The variance-preparation policy owns the signal roughness: diffuse writes
// exactly 1.0, while specular writes sqrt(primary GGX alpha).
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_common.glsl"

uniform usampler2D colortex3;

#ifndef DENOISER_SPATIAL_REQUIRE_MATERIAL_MATCH
#define DENOISER_SPATIAL_REQUIRE_MATERIAL_MATCH 0
#endif

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
    geometry.position = distance >= 0.0
        ? reconstructPrimaryRay(uvec2(pixel)) * distance : vec3(0.0);
    geometry.normal = decodeNormalU(words.y);
    geometry.textureNormal = decodeNormalU(words.z);
    geometry.roughness = clamp(unpackHalf2x16(words.w).x, 0.0, 1.0);
    geometry.materialID = words.w >> 16u;
    geometry.valid = distance >= 0.0
        && !isnan(distance) && !isinf(distance);
    return geometry;
}

bool denoiserSpatialGeometryCompatible(DenoiserSpatialGeometry center,
        DenoiserSpatialGeometry neighbor) {
#if DENOISER_SPATIAL_REQUIRE_MATERIAL_MATCH
    return center.materialID == neighbor.materialID;
#else
    return true;
#endif
}

#endif // MAXENT_SPATIAL_GEOMETRY_GLSL
