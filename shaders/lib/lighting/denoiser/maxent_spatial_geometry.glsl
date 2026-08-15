#ifndef MAXENT_SPATIAL_GEOMETRY_GLSL
#define MAXENT_SPATIAL_GEOMETRY_GLSL

// Canonical PrimaryGeometry adapter shared by all spatial signals. It depends
// only on the primary G-buffer, not on diffuse or reflection buffer policies.
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_common.glsl"

#ifndef DENOISER_SPATIAL_REQUIRE_MATERIAL_MATCH
#define DENOISER_SPATIAL_REQUIRE_MATERIAL_MATCH 0
#endif

ivec2 denoiserSpatialImageSize() {
    return ivec2(resolution_global);
}

uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel) {
    return readPrimaryGeometryWords(uvec2(pixel));
}

DenoiserSpatialGeometry denoiserSpatialDecodeGeometry(
        uvec4 words, ivec2 pixel) {
    DenoiserSpatialGeometry geometry;
    float distance = uintBitsToFloat(words.w);
    geometry.position = distance >= 0.0
        ? reconstructPrimaryRay(uvec2(pixel)) * distance : vec3(0.0);
    geometry.normal = decodeNormalU(words.x);
    geometry.textureNormal = decodeNormalU(words.z);
    geometry.roughness = unpackHalf2x16(words.y).x;
    geometry.materialID = words.y >> 16u;
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
