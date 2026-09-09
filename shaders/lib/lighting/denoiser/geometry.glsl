#ifndef MAXENT_DENOISER_GEOMETRY_GLSL
#define MAXENT_DENOISER_GEOMETRY_GLSL

// Canonical denoiser-geometry adapter shared by all spatial signals.
//
// RGBA32UI ABI produced by variance_prepare.glsl:
//   x = floatBitsToUint(first-surface distance), negative means invalid/sky
//   y = oct32 first-surface geometry normal
//   z = reserved, zero
//   w = half(perceptual roughness) | half(center Kish N_eff)
// The primary ray is reconstructed from the pixel coordinate instead of being
// stored.
//
// The variance-preparation policy owns the signal roughness: diffuse writes
// exactly 1.0, while specular writes sqrt(primary GGX alpha).
#include "/lib/buffers/gbuffer.glsl"
#include "/lib/lighting/denoiser/signal.glsl"

// Small kernels supply a cooperatively reconstructed FP32 ray tile. Large
// Poisson gathers reconstruct directly because their footprint is sparse.
#ifdef MAXENT_ATROUS_SMALL_KERNEL
vec3 denoiserSpatialPrimaryRay(ivec2 pixel);
#else
vec3 denoiserSpatialPrimaryRay(ivec2 pixel) {
    return reconstructPrimaryRay(uvec2(pixel));
}
#endif

bool denoiserSpatialGeometryWordsValid(uvec4 words) {
    return uintBitsToFloat(words.x) >= 0.0;
}

struct DenoiserSpatialCenterGeometry {
    vec3 geometryNormal;
    vec3 primaryRay;
    float surfaceDistance;
    float surfacePlaneOffset;
    float ggxAlpha;
    float effectiveSamples;
};

DenoiserSpatialCenterGeometry denoiserSpatialDecodeCenterGeometry(
        uvec4 words, ivec2 pixel) {
    DenoiserSpatialCenterGeometry geometry;
    geometry.surfaceDistance = uintBitsToFloat(words.x);
    geometry.geometryNormal = decodeNormalU(words.y);
    geometry.primaryRay = denoiserSpatialPrimaryRay(pixel);
    geometry.surfacePlaneOffset = geometry.surfaceDistance
        * dot(geometry.geometryNormal, geometry.primaryRay);
    vec2 roughnessEffectiveSamples = unpackHalf2x16(words.w);
    float roughness = roughnessEffectiveSamples.x;
    geometry.ggxAlpha = roughness * roughness;
    geometry.effectiveSamples = roughnessEffectiveSamples.y;
    return geometry;
}

void denoiserSpatialDecodeSampleGeometry(uvec4 words, ivec2 pixel,
        out vec3 primaryRay, out float surfaceDistance,
        out float effectiveSamples) {
    primaryRay = denoiserSpatialPrimaryRay(pixel);
    surfaceDistance = uintBitsToFloat(words.x);
    effectiveSamples = unpackHalf2x16(words.w).y;
}

bool denoiserSpatialTryVirtualWorldPositionFromWords(
        ivec2 pixel, uvec4 signalWords, out vec3 position) {
    if (!denoiserSpatialPreparedSignalWordsValid(signalWords)) {
        position = vec3(0.0);
        return false;
    }
    float virtualDistance = unpackHalf2x16(signalWords.w).y;
    position = denoiserSpatialPrimaryRay(pixel) * virtualDistance;
    return true;
}

vec3 denoiserSpatialVirtualWorldPositionFromWords(
        ivec2 pixel, uvec4 signalWords, vec3 fallback) {
    vec3 position;
    return denoiserSpatialTryVirtualWorldPositionFromWords(
        pixel, signalWords, position) ? position : fallback;
}

#endif // MAXENT_DENOISER_GEOMETRY_GLSL
