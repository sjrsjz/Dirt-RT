// Purpose: prepare diffuse Monte Carlo observation variance and the independent-current branch for A-Trous.
// Dispatch: 16x16.
// Reads: diffuse temporal proposal moments, Raw RT independent-current signal, opaque diffuse geometry.
// Writes: colortex3 geometry/center N_eff, colorimg4 prepared proposal, shared scratch plane A.
// Persistent side effects: prepared diffuse Monte Carlo-variance diagnostic only.
// Invalid representation: negative standardDeviation in signal metadata and negative geometry distance.

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32ui) uniform writeonly uimage2D colorimg3;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

#define DIFFUSE_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/scratch_io.glsl"
#include "/lib/lighting/denoiser/variance_prepare.glsl"

uvec4 denoiserVarianceLoadPrimaryGeometryWords(ivec2 pixel) { return readDiffuseGeometryWords(uvec2(pixel)); }

DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel) {
    uvec4 words = denoiserVarianceLoadPrimaryGeometryWords(pixel);
    DenoiserVarianceGeometry geometry;
    geometry.primaryRay = reconstructPrimaryRay(uvec2(pixel));
    geometry.geometryNormal = decodeNormalU(words.x);
    geometry.surfaceDistance = uintBitsToFloat(words.w);
    geometry.virtualScale = 0.0;
    geometry.signalRoughness = 1.0;
    geometry.valid = geometry.surfaceDistance >= 0.0 && !isnan(geometry.surfaceDistance) && !isinf(geometry.surfaceDistance);
    return geometry;
}

DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel) {
    uvec4 words = readDiffuseSwapRaw(uvec2(pixel));
    DenoiserVarianceSource source;
    source.maxEntY = vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
    source.CoCg = unpackHalf2x16(words.z);
    vec2 effectiveSamplesRootMeanY2 = unpackHalf2x16(words.w);
    source.historyEffectiveSamples = effectiveSamplesRootMeanY2.x;
    source.rootMeanY2 = effectiveSamplesRootMeanY2.y;
    source.hitDistance = 0.0;
    return source;
}

DenoiserVarianceSource denoiserVarianceLoadIndependentCurrentSource(ivec2 pixel) {
    DenoiserVarianceSource source;
    MaxEntEncoding current;
    readDiffuseLightRT(uvec2(pixel), current, source.rootMeanY2);
    source.maxEntY = current.maxEntY;
    source.CoCg = current.CoCg;
    source.historyEffectiveSamples = 1.0;
    source.hitDistance = 0.0;
    return source;
}

void denoiserVarianceStorePrepared(ivec2 pixel, DenoiserMaxEntSignal signal, DenoiserMaxEntSignal independentCurrent,
        uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry, float effectiveSamples) {
    imageStore(colorimg3, pixel, denoiserVariancePackSpatialGeometry(primaryGeometryWords, geometry, effectiveSamples));
    imageStore(colorimg4, pixel, denoiserPackMaxEntSignal(signal));
    denoiserScratchStoreA(pixel, denoiserPackMaxEntSignal(independentCurrent));
}

void denoiserVarianceStoreInvalid(ivec2 pixel) {
    uvec4 invalidSignal = denoiserInvalidMaxEntSignalWords();
    imageStore(colorimg3, pixel, uvec4(floatBitsToUint(-1.0), 0u, 0u, 0u));
    imageStore(colorimg4, pixel, invalidSignal);
    denoiserScratchStoreA(pixel, invalidSignal);
    denoiserVarianceDebugWrite(pixel, -1.0);
}

void denoiserVarianceDebugWrite(ivec2 pixel, float standardDeviation) {
#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_PREPARED_MONTE_CARLO_VARIANCE
    debugWriteDiffusePreparedMonteCarloStandardDeviation(uvec2(pixel), standardDeviation);
#endif
}

void main() { denoiserVariancePrepare(); }
