#ifndef MAXENT_VARIANCE_PREPARE_GLSL
#define MAXENT_VARIANCE_PREPARE_GLSL

#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_virtual_projection.glsl"
#include "/lib/math/statistics.glsl"

// Canonical variance-preparation contract shared by diffuse and specular.
//
// Source signal:
//   maxEntY      = (E[Y * direction], E[Y])
//   CoCg         = chroma paired with E[Y]
//   rootMeanY2   = temporal root second moment sqrt(E[Y^2])
//   historyLength = Kish effective temporal sample count N_eff
//
// Outputs:
//   colortex3 = canonical RGBA32UI denoiser geometry documented in
//               maxent_spatial_geometry.glsl
//   colortex4 = canonical RGBA32UI MaxEnt signal documented in
//               maxent_spatial_signal.glsl
//   DIF_N_CURRENT_A = independent one-frame Raw estimator with its separately
//               prepared variance; A-Trous propagates it in A/B SSBO planes
//               using the exact weights selected from colortex4/5.
// The variance carried by colortex4 is the trace uncertainty of the encoded
// linear moment vector (E[Y u], E[Y]).  Since |(Y u, Y)|^2 = 2Y^2, it is
// obtained directly from the stored moments without a decoder distribution or
// covariance-shape assumption. It is stored and propagated as an FP16
// standard deviation. First-surface plane
// consistency constrains both variance pooling stages. The signal's upper
// metadata half contains the already constructed radial virtual distance, not
// reflection hit distance.
// Iris owns the physical colortex4 ping-pong backing between spatial passes.

#if !defined(MAXENT_VARIANCE_DIFFUSE) && !defined(MAXENT_VARIANCE_SPECULAR)
#error "Select one MaxEnt variance source"
#endif
#if defined(MAXENT_VARIANCE_DIFFUSE) && defined(MAXENT_VARIANCE_SPECULAR)
#error "Select only one MaxEnt variance source"
#endif

layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg3;
#if defined(MAXENT_VARIANCE_SPECULAR)
uniform usampler2D colortex4;
uniform usampler2D colortex6;
#endif

struct DenoiserVarianceSource {
    vec4 maxEntY;
    vec2 CoCg;
    float rootMeanY2;
    float historyLength;
    float hitDistance;
};

struct DenoiserVarianceGeometry {
    vec3 surfaceNormal;
    vec3 primaryRay;
    float surfaceDistance;
    float virtualScale;
    float ggxAlpha;
    uint materialID;
    bool valid;
};

DenoiserVarianceSource denoiserVarianceEmptySource() {
    DenoiserVarianceSource source;
    source.maxEntY = vec4(0.0);
    source.CoCg = vec2(0.0);
    source.rootMeanY2 = 0.0;
    source.historyLength = 1.0;
    source.hitDistance = 0.0;
    return source;
}

DenoiserVarianceGeometry denoiserVarianceDecodeGeometry(uvec4 words, ivec2 pixel) {
    DenoiserVarianceGeometry geometry;
    float distance = uintBitsToFloat(words.w);
    float ggxAlpha = unpackHalf2x16(words.y).x;
    geometry.primaryRay = reconstructPrimaryRay(uvec2(pixel));
    geometry.surfaceNormal = decodeNormalU(words.x);
    geometry.surfaceDistance = distance;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    geometry.virtualScale = 0.0;
    geometry.ggxAlpha = 1.0;
    #else
    ggxAlpha = clamp(ggxAlpha, 0.0, 1.0);
    float perceptualRoughness = sqrt(ggxAlpha);
    geometry.virtualScale = denoiserSpatialSpecularVirtualScale(
            geometry.primaryRay, geometry.surfaceNormal, perceptualRoughness);
    geometry.ggxAlpha = ggxAlpha;
    #endif
    geometry.materialID = words.y >> 16u;
    geometry.valid = distance >= 0.0 && !isnan(distance)
            && !isinf(distance);
    return geometry;
}

DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel) {
    return denoiserVarianceDecodeGeometry(readPrimaryGeometryWords(uvec2(pixel)), pixel);
}

uvec4 denoiserVariancePackSpatialGeometry(uvec4 primaryWords, DenoiserVarianceGeometry geometry) {
    float ggxAlpha = geometry.ggxAlpha;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    float signalRoughness = 1.0;
    #else
    float signalRoughness = sqrt(ggxAlpha);
    #endif
    uint packedRoughness = packHalf2x16(vec2(signalRoughness, 0.0));
    return uvec4(primaryWords.w, encodeNormalU(geometry.surfaceNormal),
        encodeNormalU(geometry.primaryRay), packedRoughness);
}

DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel) {
    DenoiserVarianceSource source;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    uvec4 words = readDiffuseSwapRaw(uvec2(pixel));
    source.maxEntY = vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
    source.CoCg = unpackHalf2x16(words.z);
    vec2 historyRootM2 = unpackHalf2x16(words.w);
    source.historyLength = historyRootM2.x;
    source.rootMeanY2 = historyRootM2.y;
    source.hitDistance = 0.0;
    #else
    // Composite63 preserves the raw reprojected history in N2. This image is
    // the provisional temporal proposal; the independent Raw estimator is
    // loaded separately by denoiserVarianceLoadIndependentCurrentSource().
    uvec4 words = texelFetch(colortex4, pixel, 0);
    SpecularMaxEnt signal = unpackSpecularMaxEnt(words.xyz);
    source.maxEntY = signal.maxEntY;
    source.CoCg = signal.CoCg;
    vec2 rootM2History = unpackHalf2x16(words.w);
    source.rootMeanY2 = rootM2History.x;
    source.historyLength = rootM2History.y;
    source.hitDistance = max(unpackHalf2x16(
        texelFetch(colortex6, pixel, 0).w).x, 0.0);
    #endif
    return source;
}

// Independent current-frame observation. This branch is never temporally
// premixed; variance preparation and every later spatial pass apply the same
// geometry and signal weights as the proposal branch.
DenoiserVarianceSource denoiserVarianceLoadIndependentCurrentSource(
        ivec2 pixel) {
    DenoiserVarianceSource source;
    #if defined(MAXENT_VARIANCE_DIFFUSE)
    MaxEntEncoding current;
    readDiffuseLightRT(uvec2(pixel), current, source.rootMeanY2);
    source.maxEntY = current.maxEntY;
    source.CoCg = current.CoCg;
    source.hitDistance = 0.0;
    #else
    uvec4 currentWords = texelFetch(colortex6, pixel, 0);
    SpecularMaxEnt current = unpackSpecularMaxEnt(currentWords.xyz);
    source.maxEntY = current.maxEntY;
    source.CoCg = current.CoCg;
    source.rootMeanY2 = max(current.maxEntY.w, 0.0);
    source.hitDistance = max(unpackHalf2x16(currentWords.w).x, 0.0);
    #endif
    source.historyLength = 1.0;
    return source;
}

void denoiserVarianceStore(ivec2 pixel, DenoiserMaxEntSignal signal,
    DenoiserMaxEntSignal independentCurrent,
    uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry) {
    imageStore(colorimg3, pixel, denoiserVariancePackSpatialGeometry(primaryGeometryWords, geometry));
    imageStore(colorimg4, pixel, denoiserPackMaxEntSignal(signal));
    uvec4 currentWords = denoiserPackMaxEntSignal(independentCurrent);
    writeDiffuseIndependentCurrentA(uvec2(pixel), currentWords);
}

void denoiserVarianceStoreInvalid(ivec2 pixel) {
    imageStore(colorimg3, pixel, uvec4(floatBitsToUint(-1.0), 0u, 0u, 0u));
    imageStore(colorimg4, pixel, denoiserInvalidMaxEntSignalWords());
    writeDiffuseIndependentCurrentA(uvec2(pixel),
        denoiserInvalidMaxEntSignalWords());

    #if defined(MAXENT_VARIANCE_DIFFUSE) && DEBUG_VIEW == 26
    debugWriteDiffuseVariancePreparationStandardDeviation(
        uvec2(pixel), -1.0);
    #elif defined(MAXENT_VARIANCE_SPECULAR) && DEBUG_VIEW == 27
    debugWriteSpecularVariancePreparationStandardDeviation(
        uvec2(pixel), -1.0);
    #endif
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

float denoiserVarianceCanonicalRootMeanY2(vec4 meanState,
    float rootMeanY2) {
    rootMeanY2 = denoiserVarianceSanitizeNonnegative(rootMeanY2);
    // denoiserVarianceCanonicalMean() already guarantees |E[Y u]| <= E[Y],
    // so enforcing sqrt(E[Y^2]) >= E[Y] also enforces the directional bound.
    return max(rootMeanY2, meanState.w);
}

float denoiserVarianceBiasedCentralMoment(vec4 meanState,
        float rootMeanY2) {
    meanState = denoiserVarianceCanonicalMean(meanState);
    rootMeanY2 = denoiserVarianceCanonicalRootMeanY2(
            meanState, rootMeanY2);
    // z = (Y u, Y), |u| = 1, hence E[|z|^2] = 2 E[Y^2]. For
    // finite weighted empirical moments this is the biased central moment;
    // N_eff is required before it can represent Monte Carlo variance.
    return statisticsBiasedCentralSecondMoment(
        2.0 * rootMeanY2 * rootMeanY2, meanState);
}

float denoiserVarianceOfTemporalMean(vec4 meanState, float rootMeanY2,
    float historyLength) {
    // For normalized temporal weights, the stored central second moment has
    // expectation (1 - 1/N_eff) tr Cov[z].  Dividing by N_eff - 1 therefore
    // estimates tr Cov[weighted mean].
    return statisticsEstimatorVarianceFromBiasedCentralMoment(
        denoiserVarianceBiasedCentralMoment(meanState, rootMeanY2),
        historyLength);
}

DenoiserVarianceSource denoiserVarianceSanitizeSource(DenoiserVarianceSource source) {
    source.maxEntY = denoiserVarianceCanonicalMean(source.maxEntY);
    if (any(isnan(source.CoCg)) || any(isinf(source.CoCg)))
        source.CoCg = vec2(0.0);
    if (source.maxEntY.w <= 0.0) source.CoCg = vec2(0.0);
    source.rootMeanY2 = denoiserVarianceCanonicalRootMeanY2(
            source.maxEntY, source.rootMeanY2);
    source.hitDistance = clamp(denoiserVarianceSanitizeNonnegative(source.hitDistance),
            0.0, DENOISER_SPATIAL_FP16_MAX);
    return source;
}

const uint MAXENT_VARIANCE_TILE_SIZE = 16u;
const uint MAXENT_VARIANCE_HALO = 3u;
const uint MAXENT_VARIANCE_SHARED_WIDTH = 22u;
const uint MAXENT_VARIANCE_SHARED_AREA = 22u * 22u;

// First-surface distance and primary-ray direction reconstruct the sample
// position used by the plane test. Validity/material metadata is separate so
// the geometry domain occupies 12 bytes per tile entry instead of a uvec4.
shared uvec2 denoiserVarianceSurfaceTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMetadataTile[MAXENT_VARIANCE_SHARED_AREA];
shared uvec2 denoiserVarianceMeanTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMomentTile[MAXENT_VARIANCE_SHARED_AREA];
shared uvec2 denoiserVarianceCurrentMeanTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceCurrentMomentTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVariancePoolRequired;

// The exposed sigma range starts at 0.5, so the denominator is >= 0.25.
const float MAXENT_VARIANCE_KERNEL_DENOM = MAXENT_VARIANCE_KERNEL_SIGMA * MAXENT_VARIANCE_KERNEL_SIGMA;
const float MAXENT_VARIANCE_KERNEL_1D[4] = {
    1.0,
    exp(-0.5 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-2.0 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-4.5 / MAXENT_VARIANCE_KERNEL_DENOM)
    };

void denoiserVarianceWriteTile(uint index, DenoiserVarianceGeometry geometry,
    DenoiserVarianceSource source,
    DenoiserVarianceSource independentCurrent, bool valid) {
    source = denoiserVarianceSanitizeSource(source);
    independentCurrent = denoiserVarianceSanitizeSource(independentCurrent);
    denoiserVarianceSurfaceTile[index] = uvec2(floatBitsToUint(geometry.surfaceDistance),
            encodeNormalU(geometry.primaryRay));
    denoiserVarianceMetadataTile[index] = valid
        ? (0x80000000u | (geometry.materialID & 0x7fffffffu)) : 0u;
    denoiserVarianceMeanTile[index] = uvec2(
            packHalf2x16(clamp(source.maxEntY.xy, vec2(-65504.0),
                    vec2(65504.0))),
            packHalf2x16(clamp(source.maxEntY.zw, vec2(-65504.0),
                    vec2(65504.0))));
    denoiserVarianceMomentTile[index] = packHalf2x16(vec2(
                source.historyLength, min(source.rootMeanY2, 65504.0)));
    denoiserVarianceCurrentMeanTile[index] = uvec2(
            packHalf2x16(clamp(independentCurrent.maxEntY.xy,
                    vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(independentCurrent.maxEntY.zw,
                    vec2(-65504.0), vec2(65504.0))));
    denoiserVarianceCurrentMomentTile[index] = packHalf2x16(vec2(
            independentCurrent.historyLength,
            min(independentCurrent.rootMeanY2, 65504.0)));
}

void denoiserVarianceLoadTile(uint index, ivec2 pixel, ivec2 imageMax) {
    ivec2 clampedPixel = clamp(pixel, ivec2(0), imageMax);
    bool inBounds = all(equal(pixel, clampedPixel));
    DenoiserVarianceGeometry geometry = denoiserVarianceLoadGeometry(clampedPixel);
    bool valid = inBounds && geometry.valid;
    DenoiserVarianceSource source = valid
        ? denoiserVarianceLoadSource(clampedPixel) : denoiserVarianceEmptySource();
    DenoiserVarianceSource independentCurrent = valid
        ? denoiserVarianceLoadIndependentCurrentSource(clampedPixel)
        : denoiserVarianceEmptySource();
    denoiserVarianceWriteTile(index, geometry, source,
        independentCurrent, valid);
}

bool denoiserVarianceTileValid(uint index) {
    return (denoiserVarianceMetadataTile[index] & 0x80000000u) != 0u;
}

uint denoiserVarianceTileMaterial(uint index) {
    return denoiserVarianceMetadataTile[index] & 0x7fffffffu;
}

float denoiserVarianceTileSurfacePlaneDepthExponent(uint index, float centerPlaneOffset,
    vec3 centerSurfaceNormal, float surfaceRejectionScale) {
    uvec2 words = denoiserVarianceSurfaceTile[index];
    return denoiserSpatialSurfacePlaneDepthExponent(centerPlaneOffset, centerSurfaceNormal,
        decodeNormalU(words.y), uintBitsToFloat(words.x), surfaceRejectionScale);
}

vec4 denoiserVarianceTileMean(uint index, bool independentCurrent) {
    uvec2 words = independentCurrent
        ? denoiserVarianceCurrentMeanTile[index]
        : denoiserVarianceMeanTile[index];
    return vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
}

vec2 denoiserVarianceTileMomentHistory(uint index,
        bool independentCurrent) {
    uint words = independentCurrent
        ? denoiserVarianceCurrentMomentTile[index]
        : denoiserVarianceMomentTile[index];
    vec2 historyRootM2 = unpackHalf2x16(words);
    return historyRootM2.yx;
}

float denoiserVariancePreparedStandardDeviation(uint centerIndex,
        uint centerX, uint centerY,
        DenoiserVarianceGeometry centerGeometry,
        float centerPlaneOffset, float surfaceRejectionScale,
        float varianceHistoryBegin, float varianceHistoryEnd,
        bool independentCurrent) {
    vec4 centerMean = denoiserVarianceTileMean(
        centerIndex, independentCurrent);
    vec2 centerMomentHistory = denoiserVarianceTileMomentHistory(
        centerIndex, independentCurrent);
    float centerRootMeanY2 = centerMomentHistory.x;
    float centerHistory = centerMomentHistory.y;
    float temporalVariance = denoiserVarianceOfTemporalMean(
        centerMean, centerRootMeanY2, centerHistory);
    float spatialVariance = temporalVariance;

    if (centerHistory < varianceHistoryEnd
            && denoiserVariancePoolRequired != 0u) {
        float sumMass = 0.0;
        float sumSquaredObservationWeight = 0.0;
        vec4 sumMean = vec4(0.0);
        float sumMeanY2 = 0.0;

        // Pool the raw observations represented by each pixel. For pixel i,
        // mass_i=k_i*N_i and its squared normalized observation-weight
        // contribution is k_i^2*N_i. The independent-current branch has
        // N_i=1 by construction and therefore estimates one-frame variance.
        for (int offsetY = -3; offsetY <= 3; ++offsetY) {
            for (int offsetX = -3; offsetX <= 3; ++offsetX) {
                uint sampleIndex = uint(int(centerY) + offsetY)
                    * MAXENT_VARIANCE_SHARED_WIDTH
                    + uint(int(centerX) + offsetX);
                if (!denoiserVarianceTileValid(sampleIndex)
                        || denoiserVarianceTileMaterial(sampleIndex)
                            != centerGeometry.materialID)
                    continue;

                float spatialWeight =
                    MAXENT_VARIANCE_KERNEL_1D[abs(offsetX)]
                    * MAXENT_VARIANCE_KERNEL_1D[abs(offsetY)]
                    * exp(-denoiserVarianceTileSurfacePlaneDepthExponent(
                        sampleIndex, centerPlaneOffset,
                        centerGeometry.surfaceNormal,
                        surfaceRejectionScale));
                vec2 sampleMomentHistory =
                    denoiserVarianceTileMomentHistory(
                        sampleIndex, independentCurrent);
                float sampleRootMeanY2 = sampleMomentHistory.x;
                float sampleHistory = sampleMomentHistory.y;
                float mass = spatialWeight * sampleHistory;

                sumMass += mass;
                sumSquaredObservationWeight += spatialWeight
                    * spatialWeight * sampleHistory;
                sumMean += mass * denoiserVarianceTileMean(
                    sampleIndex, independentCurrent);
                sumMeanY2 += mass * sampleRootMeanY2
                    * sampleRootMeanY2;
            }
        }

        if (sumMass > 1e-8) {
            float inverseMass = 1.0 / sumMass;
            vec4 pooledMean = denoiserVarianceCanonicalMean(
                sumMean * inverseMass);
            float pooledRootMeanY2 =
                denoiserVarianceCanonicalRootMeanY2(
                    pooledMean, sqrt(max(sumMeanY2 * inverseMass, 0.0)));
            float pooledBiasedObservationVariance =
                denoiserVarianceBiasedCentralMoment(
                    pooledMean, pooledRootMeanY2);
            float squaredWeightMass = sumSquaredObservationWeight
                * inverseMass * inverseMass;
            float besselDenominator = 1.0 - squaredWeightMass;
            if (besselDenominator > 1e-6) {
                float pooledEffectiveSamples = 1.0 / squaredWeightMass;
                float observationVariance =
                    statisticsObservationVarianceFromBiasedCentralMoment(
                        pooledBiasedObservationVariance,
                        pooledEffectiveSamples);
                spatialVariance =
                    statisticsEstimatorVarianceFromObservationVariance(
                        observationVariance, centerHistory);
            }
        }
    }

    float spatialTrust = 1.0 - smoothstep(varianceHistoryBegin,
        varianceHistoryEnd, centerHistory);
    float preparedVariance = mix(temporalVariance,
        spatialVariance, spatialTrust);
    return sqrt(max(preparedVariance, temporalVariance));
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    uvec2 localPixel = gl_LocalInvocationID.xy;
    uint lane = gl_LocalInvocationIndex;
    ivec2 imageSize = ivec2(resolution_global);
    ivec2 imageMax = imageSize - 1;
    // Begin is nonnegative by its setting range. End is an independent
    // setting, so preserve an ordered smoothstep interval.
    float varianceHistoryBegin = MAXENT_VARIANCE_HISTORY_BEGIN;
    float varianceHistoryEnd = max(MAXENT_VARIANCE_HISTORY_END, varianceHistoryBegin + 1e-3);

    if (lane == 0u) denoiserVariancePoolRequired = 0u;
    barrier();

    uint centerX = localPixel.x + MAXENT_VARIANCE_HALO;
    uint centerY = localPixel.y + MAXENT_VARIANCE_HALO;
    uint centerIndex = centerY * MAXENT_VARIANCE_SHARED_WIDTH + centerX;
    ivec2 centerPixel = ivec2(pixel);
    ivec2 clampedCenter = clamp(centerPixel, ivec2(0), imageMax);
    bool centerInBounds = all(equal(centerPixel, clampedCenter));
    uvec4 centerGeometryWords = readPrimaryGeometryWords(uvec2(clampedCenter));
    DenoiserVarianceGeometry centerGeometry = denoiserVarianceDecodeGeometry(centerGeometryWords, clampedCenter);
    bool centerValid = centerInBounds && centerGeometry.valid;
    float surfaceRejectionScale = denoiserSpatialSurfaceRejectionScale(
            centerGeometry.surfaceDistance, float(imageSize.y));
    float centerPlaneOffset = centerGeometry.surfaceDistance
            * dot(centerGeometry.surfaceNormal, centerGeometry.primaryRay);
    DenoiserVarianceSource centerSource = centerValid
        ? denoiserVarianceLoadSource(clampedCenter) : denoiserVarianceEmptySource();
    DenoiserVarianceSource centerIndependentCurrent = centerValid
        ? denoiserVarianceLoadIndependentCurrentSource(clampedCenter)
        : denoiserVarianceEmptySource();
    centerSource = denoiserVarianceSanitizeSource(centerSource);
    centerIndependentCurrent = denoiserVarianceSanitizeSource(
        centerIndependentCurrent);
    denoiserVarianceWriteTile(centerIndex, centerGeometry, centerSource,
        centerIndependentCurrent, centerValid);

    if (centerValid && (centerSource.historyLength < varianceHistoryEnd
            || centerIndependentCurrent.historyLength < varianceHistoryEnd))
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
                denoiserVarianceLoadTile(index, tileOrigin + ivec2(tileX, tileY), imageMax);
        }
        barrier();
    }

    float estimatorStandardDeviation = 0.0;
    float independentCurrentStandardDeviation = 0.0;
    if (centerValid) {
        estimatorStandardDeviation =
            denoiserVariancePreparedStandardDeviation(
                centerIndex, centerX, centerY, centerGeometry,
                centerPlaneOffset, surfaceRejectionScale,
                varianceHistoryBegin, varianceHistoryEnd, false);
        independentCurrentStandardDeviation =
            denoiserVariancePreparedStandardDeviation(
                centerIndex, centerX, centerY, centerGeometry,
                centerPlaneOffset, surfaceRejectionScale,
                varianceHistoryBegin, varianceHistoryEnd, true);
    }

    if (!centerInBounds) return;
    if (!centerValid) {
        denoiserVarianceStoreInvalid(centerPixel);
        return;
    }

    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = centerSource.maxEntY;
    outputSignal.CoCg = centerSource.CoCg;
    outputSignal.standardDeviation = estimatorStandardDeviation;
    outputSignal.virtualDistance = centerGeometry.surfaceDistance + centerGeometry.virtualScale * centerSource.hitDistance;

    DenoiserMaxEntSignal independentCurrentSignal;
    independentCurrentSignal.maxEntY = centerIndependentCurrent.maxEntY;
    independentCurrentSignal.CoCg = centerIndependentCurrent.CoCg;
    independentCurrentSignal.standardDeviation =
        independentCurrentStandardDeviation;
    independentCurrentSignal.virtualDistance = centerGeometry.surfaceDistance
        + centerGeometry.virtualScale * centerIndependentCurrent.hitDistance;

    #if defined(MAXENT_VARIANCE_DIFFUSE) && DEBUG_VIEW == 26
    debugWriteDiffuseVariancePreparationStandardDeviation(
        pixel, estimatorStandardDeviation);
    #elif defined(MAXENT_VARIANCE_SPECULAR) && DEBUG_VIEW == 27
    debugWriteSpecularVariancePreparationStandardDeviation(
        pixel, estimatorStandardDeviation);
    #endif

    denoiserVarianceStore(centerPixel, outputSignal,
        independentCurrentSignal, centerGeometryWords, centerGeometry);
}

#endif // MAXENT_VARIANCE_PREPARE_GLSL
