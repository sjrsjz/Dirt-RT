// Purpose: prepare reflection Monte Carlo observation variance and the independent-current branch for A-Trous.
// Dispatch: 16x16.
// Reads: colortex4 temporal proposal, colortex6 Raw RT observation, compact primary geometry.
// Writes: colortex3 geometry/center N_eff, colorimg5 prepared proposal, shared scratch plane A.
// Persistent side effects: prepared specular Monte Carlo-variance diagnostic only.
// Invalid representation: negative standardDeviation in signal metadata and negative geometry distance.

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32ui) uniform writeonly uimage2D colorimg3;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;
uniform usampler2D colortex4;
uniform usampler2D colortex6;

#define REFLECT_BUFFER
#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/scratch_io.glsl"
#include "/lib/lighting/denoiser/variance_prepare.glsl"

uvec4 denoiserVarianceLoadPrimaryGeometryWords(ivec2 pixel) { return readPrimaryGeometryWords(uvec2(pixel)); }

DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel) {
    uvec4 words = denoiserVarianceLoadPrimaryGeometryWords(pixel);
    DenoiserVarianceGeometry geometry;
    float ggxAlpha = clamp(unpackHalf2x16(words.y).x, 0.0, 1.0);
    geometry.primaryRay = reconstructPrimaryRay(uvec2(pixel));
    vec3 geometryNormal = decodeNormalU(words.x);
    geometry.geometryNormal = geometryNormal;
    geometry.surfaceDistance = uintBitsToFloat(words.w);
    geometry.signalRoughness = sqrt(ggxAlpha);
    geometry.virtualScale = denoiserSpatialSpecularVirtualScale(geometry.primaryRay, geometryNormal, geometry.signalRoughness);
    geometry.pdfDirection = denoiserSpatialSpecularPdfDirectionFromFactor(geometry.primaryRay, geometryNormal, geometry.virtualScale);
    geometry.valid = geometry.surfaceDistance >= 0.0 && !isnan(geometry.surfaceDistance) && !isinf(geometry.surfaceDistance);
    return geometry;
}

DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel) {
    uvec4 words = texelFetch(colortex4, pixel, 0);
    SpecularMaxEnt signal = unpackSpecularMaxEnt(words.xyz);
    DenoiserVarianceSource source;
    source.maxEntY = signal.maxEntY;
    source.CoCg = signal.CoCg;
    vec2 rootMeanY2EffectiveSamples = unpackHalf2x16(words.w);
    source.rootMeanY2 = rootMeanY2EffectiveSamples.x;
    source.historyEffectiveSamples = rootMeanY2EffectiveSamples.y;
    source.hitDistance = max(unpackHalf2x16(texelFetch(colortex6, pixel, 0).w).x, 0.0);
    return source;
}

DenoiserVarianceSource denoiserVarianceLoadIndependentCurrentSource(ivec2 pixel) {
    uvec4 words = texelFetch(colortex6, pixel, 0);
    SpecularMaxEnt current = unpackSpecularMaxEnt(words.xyz);
    DenoiserVarianceSource source;
    source.maxEntY = current.maxEntY;
    source.CoCg = current.CoCg;
    source.rootMeanY2 = abs(current.maxEntY.w);
    source.historyEffectiveSamples = 1.0;
    source.hitDistance = max(unpackHalf2x16(words.w).x, 0.0);
    return source;
}

void denoiserVarianceStorePrepared(ivec2 pixel, DenoiserMaxEntSignal signal, DenoiserMaxEntSignal independentCurrent,
        uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry, float effectiveSamples) {
    imageStore(colorimg3, pixel, denoiserVariancePackSpatialGeometry(primaryGeometryWords, geometry, effectiveSamples));
    imageStore(colorimg5, pixel, denoiserPackMaxEntSignal(signal));
    denoiserScratchStoreA(pixel, denoiserPackMaxEntSignal(independentCurrent));
}

void denoiserVarianceStoreInvalid(ivec2 pixel) {
    uvec4 invalidSignal = denoiserInvalidMaxEntSignalWords();
    imageStore(colorimg3, pixel, uvec4(floatBitsToUint(-1.0), 0u, 0u, 0u));
    imageStore(colorimg5, pixel, invalidSignal);
    denoiserScratchStoreA(pixel, invalidSignal);
    denoiserVarianceDebugWrite(pixel, -1.0);
}

void denoiserVarianceDebugWrite(ivec2 pixel, float standardDeviation) {
#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_PREPARED_MONTE_CARLO_VARIANCE
    debugWriteSpecularPreparedMonteCarloStandardDeviation(uvec2(pixel), standardDeviation);
#endif
}

void main() { denoiserVariancePrepare(); }
