#ifndef MAXENT_VARIANCE_PREPARE_GLSL
#define MAXENT_VARIANCE_PREPARE_GLSL

#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_pdf_direction.glsl"

// Canonical variance-preparation contract shared by diffuse and specular.
//
// Source signal:
//   maxEntY      = (E[Y * direction], E[Y])
//   CoCg         = chroma paired with E[Y]
//   meanY2       = raw temporal second moment E[Y^2]
//   historyLength = Kish effective temporal sample count N_eff
//
// Outputs:
//   colortex3 = canonical RGBA32UI denoiser geometry documented in
//               maxent_spatial_geometry.glsl
//   colortex4 = canonical RGBA32UI MaxEnt signal documented in
//               maxent_spatial_signal.glsl
// The variance carried by colortex4 is the estimator variance of the
// temporally accumulated MaxEnt mean, never the population variance. It is
// computed in F32 and stored as an FP16 square root. Diffuse uses the geometry
// normal as its PDF direction. Specular uses the view-conditioned GGX VNDF
// dominant outgoing direction.
// Iris owns the physical colortex4 ping-pong backing between spatial passes.

#if !defined(MAXENT_VARIANCE_DIFFUSE) && !defined(MAXENT_VARIANCE_SPECULAR)
#error "Select one MaxEnt variance source"
#endif
#if defined(MAXENT_VARIANCE_DIFFUSE) && defined(MAXENT_VARIANCE_SPECULAR)
#error "Select only one MaxEnt variance source"
#endif

layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg3;

struct DenoiserVarianceSource {
    vec4 maxEntY;
    vec2 CoCg;
    float meanY2;
    float historyLength;
    float hitDistance;
};

struct DenoiserVarianceGeometry {
    vec3 pdfDirection;
    float virtualScale;
    float ggxAlpha;
    uint materialID;
    bool valid;
};

DenoiserVarianceSource denoiserVarianceEmptySource() {
    DenoiserVarianceSource source;
    source.maxEntY = vec4(0.0);
    source.CoCg = vec2(0.0);
    source.meanY2 = 0.0;
    source.historyLength = 1.0;
    source.hitDistance = 0.0;
    return source;
}

DenoiserVarianceGeometry denoiserVarianceDecodeGeometry(uvec4 words,
    ivec2 pixel) {
    DenoiserVarianceGeometry geometry;
    float distance;
    float ggxAlpha;
    float pathRoughnessUnused;
    int materialID;
    vec3 positionUnused;
    vec3 geometryNormal;
    unpackPrimaryGeometry(words, uvec2(pixel), positionUnused, distance,
        geometryNormal, ggxAlpha, materialID,
        pathRoughnessUnused);
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    geometry.pdfDirection = geometryNormal;
    geometry.virtualScale = 0.0;
    geometry.ggxAlpha = 1.0;
    #else
    vec3 macroNormal = decodeNormalU(words.z);
    vec3 primaryRay = reconstructPrimaryRay(uvec2(pixel));
    float perceptualRoughness = sqrt(clamp(ggxAlpha, 0.0, 1.0));
    geometry.pdfDirection = denoiserSpatialGgxVndfDominantDirection(
            primaryRay, macroNormal, ggxAlpha);
    geometry.virtualScale = denoiserSpatialSpecularVirtualScale(
            primaryRay, geometryNormal, perceptualRoughness);
    geometry.ggxAlpha = clamp(ggxAlpha, 0.0, 1.0);
    #endif
    geometry.materialID = uint(max(materialID, 0));
    geometry.valid = distance >= 0.0 && !isnan(distance)
            && !isinf(distance);
    return geometry;
}

DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel) {
    return denoiserVarianceDecodeGeometry(
        readPrimaryGeometryWords(uvec2(pixel)), pixel);
}

uvec4 denoiserVariancePackSpatialGeometry(uvec4 primaryWords,
    DenoiserVarianceGeometry geometry) {
    float ggxAlpha = clamp(unpackHalf2x16(primaryWords.y).x, 0.0, 1.0);
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    float signalRoughness = 1.0;
    #else
    float signalRoughness = sqrt(ggxAlpha);
    #endif
    uint roughnessMaterial = (primaryWords.y & 0xffff0000u)
            | (packHalf2x16(vec2(signalRoughness, 0.0)) & 0xffffu);
    uint virtualAlpha = packHalf2x16(clamp(
                vec2(geometry.virtualScale, geometry.ggxAlpha),
                vec2(0.0), vec2(1.0)));
    return uvec4(primaryWords.w, encodeNormalU(geometry.pdfDirection),
        virtualAlpha, roughnessMaterial);
}

DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel) {
    DenoiserVarianceSource source;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    uvec4 words = readDiffuseSwapRaw(uvec2(pixel));
    source.maxEntY = vec4(unpackHalf2x16(words.x),
            unpackHalf2x16(words.y));
    source.CoCg = unpackHalf2x16(words.z);
    vec2 historyRootM2 = unpackHalf2x16(words.w);
    source.historyLength = historyRootM2.x;
    source.meanY2 = historyRootM2.y * historyRootM2.y;
    source.hitDistance = 0.0;
    #else
    uvec4 words = reflectBuffer.data[
        addr(SPEC_N_HISTLIGHT, uvec2(pixel))];
    SpecularMaxEnt signal = unpackSpecularMaxEnt(words.xyz);
    source.maxEntY = signal.maxEntY;
    source.CoCg = signal.CoCg;
    vec2 rootM2History = unpackHalf2x16(words.w);
    source.meanY2 = rootM2History.x * rootM2History.x;
    source.historyLength = rootM2History.y;
    source.hitDistance = unpackHalf2x16(reflectBuffer.data[
            addr(SPEC_N_HISTMETA, uvec2(pixel))].w).x;
    #endif
    return source;
}

void denoiserVarianceStore(ivec2 pixel, DenoiserMaxEntSignal signal,
    uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry) {
    imageStore(colorimg3, pixel,
        denoiserVariancePackSpatialGeometry(primaryGeometryWords, geometry));
    imageStore(colorimg4, pixel, denoiserPackMaxEntSignal(signal));
    #if defined(MAXENT_VARIANCE_SPECULAR)
    #if DEBUG_VIEW == 25
    SpecularMaxEnt debugSignal;
    debugSignal.maxEntY = signal.maxEntY;
    debugSignal.CoCg = signal.CoCg;
    uvec4 historyWords = reflectBuffer.data[
        addr(SPEC_N_HISTLIGHT, uvec2(pixel))];
    float historyLength = unpackHalf2x16(historyWords.w).y;
    writeReflLight(uvec2(pixel), specularMaxEntTotalRgb(debugSignal),
        signal.hitDistance, max(historyLength, 0.0));
    #endif
    #endif
}

void denoiserVarianceStoreInvalid(ivec2 pixel) {
    imageStore(colorimg3, pixel,
        uvec4(floatBitsToUint(-1.0), 0u, 0u, 0u));
    imageStore(colorimg4, pixel, denoiserInvalidMaxEntSignalWords());
}

float denoiserVarianceSanitizeNonnegative(float value) {
    return isnan(value) || isinf(value) ? 0.0 : max(value, 0.0);
}

vec4 denoiserVarianceCanonicalMean(vec4 meanState) {
    if (any(isnan(meanState)) || any(isinf(meanState)))
        return vec4(0.0);
    vec3 directionalMean = meanState.xyz;
    float meanY = max(meanState.w, 0.0);
    float directionalLength2 = dot(directionalMean, directionalMean);
    if (directionalLength2 > meanY * meanY
            && directionalLength2 > 0.0)
        directionalMean *= meanY * inversesqrt(directionalLength2);
    return vec4(directionalMean, meanY);
}

float denoiserVarianceCanonicalMeanY2(vec4 meanState, float meanY2) {
    meanY2 = denoiserVarianceSanitizeNonnegative(meanY2);
    return max(meanY2, max(meanState.w * meanState.w,
            dot(meanState.xyz, meanState.xyz)));
}

float denoiserVariancePopulation(vec4 meanState, float meanY2) {
    // z = (Y * direction, Y), hence E[|z|^2] = 2 E[Y^2].
    return max(2.0 * meanY2 - dot(meanState, meanState), 0.0);
}

float denoiserVarianceOfTemporalMean(vec4 meanState, float meanY2,
    float historyLength) {
    // The stored population estimate is biased by (N - 1) / N. Dividing it
    // by N - 1 therefore gives the variance of the temporal sample mean.
    if (historyLength <= 1.5) return 0.0;
    return denoiserVariancePopulation(meanState, meanY2)
        / (historyLength - 1.0);
}

DenoiserVarianceSource denoiserVarianceSanitizeSource(
    DenoiserVarianceSource source) {
    source.maxEntY = denoiserVarianceCanonicalMean(source.maxEntY);
    if (any(isnan(source.CoCg)) || any(isinf(source.CoCg)))
        source.CoCg = vec2(0.0);
    if (source.maxEntY.w <= 0.0) source.CoCg = vec2(0.0);
    source.meanY2 = denoiserVarianceCanonicalMeanY2(
            source.maxEntY, source.meanY2);
    source.historyLength = clamp(
            denoiserVarianceSanitizeNonnegative(source.historyLength),
            1.0, 65504.0);
    source.hitDistance = clamp(
            denoiserVarianceSanitizeNonnegative(source.hitDistance),
            0.0, DENOISER_SPATIAL_FP16_MAX);
    return source;
}

const uint MAXENT_VARIANCE_TILE_SIZE = 16u;
const uint MAXENT_VARIANCE_HALO = 3u;
const uint MAXENT_VARIANCE_SHARED_WIDTH = 22u;
const uint MAXENT_VARIANCE_SHARED_AREA = 22u * 22u;

// Domain metadata uses one uvec4: x stores the oct32 PDF direction and w packs
// validity in the high bit plus material ID in the remaining bits. MaxEnt and
// moment state remain in their source FP16 representation in shared memory.
shared uvec4 denoiserVarianceGeometryTile[MAXENT_VARIANCE_SHARED_AREA];
shared uvec2 denoiserVarianceMeanTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMomentTile[MAXENT_VARIANCE_SHARED_AREA];
shared float denoiserVarianceEstimateTile[
MAXENT_VARIANCE_TILE_SIZE * MAXENT_VARIANCE_TILE_SIZE];
shared uint denoiserVariancePoolRequired;

const float MAXENT_VARIANCE_KERNEL_DENOM = max(
        MAXENT_VARIANCE_KERNEL_SIGMA * MAXENT_VARIANCE_KERNEL_SIGMA, 1e-6);
const float MAXENT_VARIANCE_KERNEL_1D[4] = {
    1.0,
    exp(-0.5 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-2.0 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-4.5 / MAXENT_VARIANCE_KERNEL_DENOM)
    };
const float MAXENT_VARIANCE_BLUR_1D[2] = { 1.0, 0.6065306597 };

void denoiserVarianceWriteTile(uint index,
    DenoiserVarianceGeometry geometry,
    DenoiserVarianceSource source, bool valid) {
    source = denoiserVarianceSanitizeSource(source);
    denoiserVarianceGeometryTile[index] = uvec4(
            encodeNormalU(geometry.pdfDirection), 0u, 0u,
            valid ? (0x80000000u | (geometry.materialID & 0x7fffffffu)) : 0u);
    denoiserVarianceMeanTile[index] = uvec2(
            packHalf2x16(clamp(source.maxEntY.xy, vec2(-65504.0),
                    vec2(65504.0))),
            packHalf2x16(clamp(source.maxEntY.zw, vec2(-65504.0),
                    vec2(65504.0))));
    denoiserVarianceMomentTile[index] = packHalf2x16(vec2(
                source.historyLength,
                min(sqrt(max(source.meanY2, 0.0)), 65504.0)));
}

void denoiserVarianceLoadTile(uint index, ivec2 pixel, ivec2 imageMax) {
    ivec2 clampedPixel = clamp(pixel, ivec2(0), imageMax);
    bool inBounds = all(equal(pixel, clampedPixel));
    DenoiserVarianceGeometry geometry =
        denoiserVarianceLoadGeometry(clampedPixel);
    bool valid = inBounds && geometry.valid;
    DenoiserVarianceSource source = valid
        ? denoiserVarianceLoadSource(clampedPixel) : denoiserVarianceEmptySource();
    denoiserVarianceWriteTile(index, geometry, source, valid);
}

bool denoiserVarianceTileValid(uint index) {
    return (denoiserVarianceGeometryTile[index].w & 0x80000000u) != 0u;
}

uint denoiserVarianceTileMaterial(uint index) {
    return denoiserVarianceGeometryTile[index].w & 0x7fffffffu;
}

vec3 denoiserVarianceTilePdfDirection(uint index) {
    return decodeNormalU(denoiserVarianceGeometryTile[index].x);
}

vec4 denoiserVarianceTileMean(uint index) {
    return vec4(unpackHalf2x16(denoiserVarianceMeanTile[index].x),
        unpackHalf2x16(denoiserVarianceMeanTile[index].y));
}

vec2 denoiserVarianceTileMomentHistory(uint index) {
    vec2 historyRootM2 = unpackHalf2x16(
            denoiserVarianceMomentTile[index]);
    return vec2(historyRootM2.y * historyRootM2.y,
        historyRootM2.x);
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    uvec2 localPixel = gl_LocalInvocationID.xy;
    uint lane = gl_LocalInvocationIndex;
    ivec2 imageSize = ivec2(resolution_global);
    ivec2 imageMax = imageSize - 1;
    float varianceHistoryBegin = max(MAXENT_VARIANCE_HISTORY_BEGIN, 0.0);
    float varianceHistoryEnd = max(MAXENT_VARIANCE_HISTORY_END,
            varianceHistoryBegin + 1e-3);

    if (lane == 0u) denoiserVariancePoolRequired = 0u;
    barrier();

    uint centerX = localPixel.x + MAXENT_VARIANCE_HALO;
    uint centerY = localPixel.y + MAXENT_VARIANCE_HALO;
    uint centerIndex = centerY * MAXENT_VARIANCE_SHARED_WIDTH + centerX;
    ivec2 centerPixel = ivec2(pixel);
    ivec2 clampedCenter = clamp(centerPixel, ivec2(0), imageMax);
    bool centerInBounds = all(equal(centerPixel, clampedCenter));
    uvec4 centerGeometryWords = readPrimaryGeometryWords(
            uvec2(clampedCenter));
    DenoiserVarianceGeometry centerGeometry =
        denoiserVarianceDecodeGeometry(centerGeometryWords, clampedCenter);
    bool centerValid = centerInBounds && centerGeometry.valid;
    DenoiserVarianceSource centerSource = centerValid
        ? denoiserVarianceLoadSource(clampedCenter) : denoiserVarianceEmptySource();
    centerSource = denoiserVarianceSanitizeSource(centerSource);
    denoiserVarianceWriteTile(centerIndex, centerGeometry, centerSource,
        centerValid);

    if (centerValid && centerSource.historyLength < varianceHistoryEnd)
        atomicOr(denoiserVariancePoolRequired, 1u);
    barrier();

    // Stable workgroups need only their center samples. A group containing
    // any short-history pixel cooperatively loads the 3-pixel halo once.
    if (denoiserVariancePoolRequired != 0u) {
        ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy
                    * MAXENT_VARIANCE_TILE_SIZE) - ivec2(MAXENT_VARIANCE_HALO);
        for (uint index = lane; index < MAXENT_VARIANCE_SHARED_AREA;
            index += MAXENT_VARIANCE_TILE_SIZE
                    * MAXENT_VARIANCE_TILE_SIZE) {
            uint tileX = index % MAXENT_VARIANCE_SHARED_WIDTH;
            uint tileY = index / MAXENT_VARIANCE_SHARED_WIDTH;
            bool interior = tileX >= MAXENT_VARIANCE_HALO
                    && tileX < MAXENT_VARIANCE_HALO
                            + MAXENT_VARIANCE_TILE_SIZE
                    && tileY >= MAXENT_VARIANCE_HALO
                    && tileY < MAXENT_VARIANCE_HALO
                            + MAXENT_VARIANCE_TILE_SIZE;
            if (!interior)
                denoiserVarianceLoadTile(index,
                    tileOrigin + ivec2(tileX, tileY), imageMax);
        }
        barrier();
    }

    float estimatorVariance = 0.0;
    if (centerValid) {
        vec4 centerMean = denoiserVarianceTileMean(centerIndex);
        vec2 centerMomentHistory =
            denoiserVarianceTileMomentHistory(centerIndex);
        float centerMeanY2 = centerMomentHistory.x;
        float centerHistory = centerMomentHistory.y;
        float temporalVariance = denoiserVarianceOfTemporalMean(
                centerMean, centerMeanY2, centerHistory);
        float spatialVariance = temporalVariance;

        if (centerHistory < varianceHistoryEnd
                && denoiserVariancePoolRequired != 0u) {
            float sumMass = 0.0;
            vec4 sumMean = vec4(0.0);
            float sumMeanY2 = 0.0;

            for (int offsetY = -3; offsetY <= 3; ++offsetY) {
                for (int offsetX = -3; offsetX <= 3; ++offsetX) {
                    uint sampleIndex = uint(int(centerY) + offsetY)
                            * MAXENT_VARIANCE_SHARED_WIDTH
                            + uint(int(centerX) + offsetX);
                    if (!denoiserVarianceTileValid(sampleIndex)
                            || denoiserVarianceTileMaterial(sampleIndex)
                                != centerGeometry.materialID)
                        continue;

                    vec3 samplePdfDirection =
                        denoiserVarianceTilePdfDirection(sampleIndex);
                    float spatialWeight =
                        MAXENT_VARIANCE_KERNEL_1D[abs(offsetX)]
                            * MAXENT_VARIANCE_KERNEL_1D[abs(offsetY)]
                            * exp(-denoiserSpatialPdfDirectionExponent(
                                    centerGeometry.pdfDirection,
                                    samplePdfDirection));
                    vec2 sampleMomentHistory =
                        denoiserVarianceTileMomentHistory(sampleIndex);
                    float mass = spatialWeight * sampleMomentHistory.y;

                    sumMass += mass;
                    sumMean += mass
                            * denoiserVarianceTileMean(sampleIndex);
                    sumMeanY2 += mass * sampleMomentHistory.x;
                }
            }

            if (sumMass > 1e-8) {
                float inverseMass = 1.0 / sumMass;
                vec4 pooledMean = denoiserVarianceCanonicalMean(
                        sumMean * inverseMass);
                float pooledMeanY2 = denoiserVarianceCanonicalMeanY2(
                        pooledMean, sumMeanY2 * inverseMass);
                float pooledPopulationVariance =
                    denoiserVariancePopulation(pooledMean, pooledMeanY2);
                spatialVariance = pooledPopulationVariance
                        / max(centerHistory, 1.0);
            }
        }

        float temporalTrust = smoothstep(varianceHistoryBegin,
                varianceHistoryEnd, centerHistory);
        if (centerHistory <= 1.5) temporalTrust = 0.0;
        float estimatorSigma = mix(sqrt(max(spatialVariance, 0.0)),
                sqrt(max(temporalVariance, 0.0)), temporalTrust);
        estimatorVariance = denoiserVarianceSanitizeNonnegative(
                estimatorSigma * estimatorSigma);
    }
    denoiserVarianceEstimateTile[localPixel.y
        * MAXENT_VARIANCE_TILE_SIZE + localPixel.x] = estimatorVariance;
    barrier();

    if (!centerInBounds) return;
    if (!centerValid) {
        denoiserVarianceStoreInvalid(centerPixel);
        return;
    }

    // A compact 3x3 variance-only blur suppresses isolated estimator spikes.
    // It intentionally never modifies the center MaxEnt signal itself.
    float blurredVariance = 0.0;
    float blurWeight = 0.0;
    int minOffsetX = max(-1, -int(localPixel.x));
    int maxOffsetX = min(1,
            int(MAXENT_VARIANCE_TILE_SIZE - 1u - localPixel.x));
    int minOffsetY = max(-1, -int(localPixel.y));
    int maxOffsetY = min(1,
            int(MAXENT_VARIANCE_TILE_SIZE - 1u - localPixel.y));
    for (int offsetY = minOffsetY; offsetY <= maxOffsetY; ++offsetY) {
        for (int offsetX = minOffsetX; offsetX <= maxOffsetX; ++offsetX) {
            uint sampleX = uint(int(centerX) + offsetX);
            uint sampleY = uint(int(centerY) + offsetY);
            uint sampleIndex = sampleY * MAXENT_VARIANCE_SHARED_WIDTH
                    + sampleX;
            if (!denoiserVarianceTileValid(sampleIndex)
                    || denoiserVarianceTileMaterial(sampleIndex)
                        != centerGeometry.materialID)
                continue;
            float weight = MAXENT_VARIANCE_BLUR_1D[abs(offsetX)]
                    * MAXENT_VARIANCE_BLUR_1D[abs(offsetY)]
                    * exp(-denoiserSpatialPdfDirectionExponent(
                            centerGeometry.pdfDirection,
                            denoiserVarianceTilePdfDirection(sampleIndex)));
            uint varianceIndex = uint(int(localPixel.y) + offsetY)
                    * MAXENT_VARIANCE_TILE_SIZE
                    + uint(int(localPixel.x) + offsetX);
            blurredVariance += weight
                    * denoiserVarianceEstimateTile[varianceIndex];
            blurWeight += weight;
        }
    }

    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = centerSource.maxEntY;
    outputSignal.CoCg = centerSource.CoCg;
    outputSignal.variance = denoiserVarianceSanitizeNonnegative(
            blurredVariance / max(blurWeight, 1e-8));
    outputSignal.hitDistance = centerSource.hitDistance;
    denoiserVarianceStore(centerPixel, outputSignal, centerGeometryWords,
        centerGeometry);
}

#endif // MAXENT_VARIANCE_PREPARE_GLSL
