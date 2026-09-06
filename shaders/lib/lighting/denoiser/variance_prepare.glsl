#ifndef MAXENT_DENOISER_VARIANCE_PREPARE_GLSL
#define MAXENT_DENOISER_VARIANCE_PREPARE_GLSL

#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/math/statistics.glsl"

// Domain-neutral variance-preparation algorithm shared by diffuse and reflection.
//
// Source signal:
//   maxEntY      = (E[Y * direction], E[Y])
//   CoCg         = chroma paired with E[Y]
//   rootMeanY2   = temporal root second moment sqrt(E[Y^2])
//   historyEffectiveSamples = Kish effective temporal sample count N_eff
//
// Outputs are returned through the including domain adapter. standardDeviation
// stores sqrt(local Bures estimator variance): observation variance divided by
// the center temporal N_eff for the proposal, and N=1 for independent current.
// The stored
// E[Y^2] supplies the radial term; the alpha=1 g^-3 joint closure supplies the
// otherwise unidentified R^2-weighted angular moments. The resulting scalar is
// matched to maxentLightSampleDistanceSq and remains finite at kappa=0 and 1.
//
// The short-history fallback first reconstructs the linear temporal fields
// E[Y u], E[Y], and E[Y^2] in space and only then evaluates the closure. This
// order includes between-pixel directional spread in the spatial variance.
// Kish N_eff is reconstructed separately from the same weights and applies the
// finite-sample correction. The signal's upper metadata half stores constructed
// radial virtual distance, not reflection hit distance.

struct DenoiserVarianceSource {
    vec4 maxEntY;
    vec2 CoCg;
    float rootMeanY2;
    float historyEffectiveSamples;
    float hitDistance;
};

struct DenoiserVarianceGeometry {
    vec3 geometryNormal;
    vec3 pdfDirection;
    vec3 primaryRay;
    float surfaceDistance;
    float virtualScale;
    float signalRoughness;
    bool valid;
};

DenoiserVarianceSource denoiserVarianceEmptySource() {
    DenoiserVarianceSource source;
    source.maxEntY = vec4(0.0);
    source.CoCg = vec2(0.0);
    source.rootMeanY2 = 0.0;
    source.historyEffectiveSamples = 1.0;
    source.hitDistance = 0.0;
    return source;
}

uvec4 denoiserVariancePackSpatialGeometry(uvec4 primaryWords, DenoiserVarianceGeometry geometry, float effectiveSamples) {
    uint packedRoughnessSamples = packHalf2x16(vec2(geometry.signalRoughness, effectiveSamples));
    return uvec4(primaryWords.w, encodeNormalU(geometry.geometryNormal), encodeNormalU(geometry.pdfDirection), packedRoughnessSamples);
}

// Required adapter callbacks. Raw storage and image bindings remain outside this file.
uvec4 denoiserVarianceLoadPrimaryGeometryWords(ivec2 pixel);
DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel);
DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel);
DenoiserVarianceSource denoiserVarianceLoadIndependentCurrentSource(ivec2 pixel);
void denoiserVarianceStorePrepared(ivec2 pixel, DenoiserMaxEntSignal signal, DenoiserMaxEntSignal independentCurrent,
        uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry, float effectiveSamples);
void denoiserVarianceStoreInvalid(ivec2 pixel);
void denoiserVarianceDebugWrite(ivec2 pixel, float standardDeviation);

float denoiserVarianceSanitizeNonnegative(float value) {
    return isnan(value) || isinf(value) ? 0.0 : max(value, 0.0);
}

vec4 denoiserVarianceFiniteMean(vec4 meanState) {
    return any(isnan(meanState)) || any(isinf(meanState)) ? vec4(0.0) : meanState;
}

bool denoiserVarianceMomentsValid(vec4 m, float rms) {
    if (any(isnan(m)) || any(isinf(m)) || !(m.w >= 0.0)
            || !(rms >= 0.0) || isnan(rms) || isinf(rms)
            || rms > DENOISER_SPATIAL_FP16_MAX
            || any(greaterThan(abs(m), vec4(DENOISER_SPATIAL_FP16_MAX)))) return false;
    if (m.w == 0.0) return rms == 0.0 && all(equal(m.xyz, vec3(0.0)));
    // Permit FP16 rounding at the cone/Jensen boundary, not arbitrary loss of
    // the paired first or second moment. Zero mean + residual RMS stays unknown.
    float tolerance = max(0.002 * m.w, 1.1920928955078125e-7);
    return rms + tolerance >= m.w && length(m.xyz) <= m.w + tolerance;
}

float denoiserVarianceBiasedBuresG3Moment(vec4 meanState, float rootMeanY2) {
    if (!denoiserVarianceMomentsValid(meanState, rootMeanY2))
        return DENOISER_UNKNOWN_UNCERTAINTY;
    float meanY = max(meanState.w, 0.0);
    if (meanY == 0.0) return 0.0;

    rootMeanY2 = max(rootMeanY2, meanY);
    float meanY2 = rootMeanY2 * rootMeanY2;
    float kappaSquared = clamp(dot(meanState.xyz, meanState.xyz)
        / (meanY * meanY), 0.0, 1.0);

    // The R-weighted angular marginal is g^-3; its R^2-weighted moments use
    // g^-4. Contracting their elementary first and second angular moments with
    // the local Bures metric cancels the apparent 1/(1-kappa^2) singularity.
    // This is the cancellation-free form of
    // [2 E[Y^2](3-k^2)/(3+k^2)-E[Y]^2] / [4 E[Y]].
    float radialVariance = (rootMeanY2 - meanY)
        * (rootMeanY2 + meanY);
    float angularScale = 3.0 * (1.0 - kappaSquared)
        / (3.0 + kappaSquared);
    return (radialVariance + angularScale * meanY2)
        / (4.0 * meanY);
}

float denoiserMonteCarloVarianceFromTemporalMoments(vec4 meanState, float rootMeanY2, float historyEffectiveSamples) {
    // Finite-history plug-in correction for the g^-3 angular closure.
    // Consumers divide the returned per-observation variance by estimator
    // N_eff exactly once.
    float biased = denoiserVarianceBiasedBuresG3Moment(meanState, rootMeanY2);
    if (!denoiserVarianceKnown(biased)
            || !statisticsValidEffectiveSampleCount(historyEffectiveSamples))
        return DENOISER_UNKNOWN_UNCERTAINTY;
    if (historyEffectiveSamples <= 1.0)
        return meanState.w == 0.0 && rootMeanY2 == 0.0 ? 0.0 : DENOISER_UNKNOWN_UNCERTAINTY;
    float variance = statisticsObservationVarianceFromBiasedCentralMoment(biased, historyEffectiveSamples);
    return denoiserVarianceKnown(variance) ? variance : DENOISER_UNKNOWN_UNCERTAINTY;
}

DenoiserVarianceSource denoiserVarianceSanitizeSource(DenoiserVarianceSource source) {
    bool momentsValid = denoiserVarianceMomentsValid(source.maxEntY, source.rootMeanY2)
        && statisticsValidEffectiveSampleCount(source.historyEffectiveSamples);
    source.maxEntY = denoiserVarianceFiniteMean(source.maxEntY);
    if (any(isnan(source.CoCg)) || any(isinf(source.CoCg))) source.CoCg = vec2(0.0);
    if (!momentsValid) source.rootMeanY2 = DENOISER_UNKNOWN_UNCERTAINTY;
    if (!statisticsValidEffectiveSampleCount(source.historyEffectiveSamples))
        source.historyEffectiveSamples = 1.0;
    source.hitDistance = clamp(denoiserVarianceSanitizeNonnegative(source.hitDistance),
            0.0, DENOISER_SPATIAL_FP16_MAX);
    return source;
}

const uint MAXENT_VARIANCE_TILE_SIZE = 16u;
const uint MAXENT_VARIANCE_HALO = 3u;
const uint MAXENT_VARIANCE_SHARED_WIDTH = 22u;
const uint MAXENT_VARIANCE_SHARED_AREA = 22u * 22u;

// First-surface distance and primary-ray direction reconstruct the sample
// position used by the geometry-normal plane test. The PDF direction supplies
// an independent soft cosine weight. Variance pooling does not distinguish
// materials.
shared uvec2 denoiserVarianceSurfaceTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVariancePdfDirectionTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMetadataTile[MAXENT_VARIANCE_SHARED_AREA];
shared uvec2 denoiserVarianceMeanTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVarianceMomentTile[MAXENT_VARIANCE_SHARED_AREA];

// The exposed sigma range starts at 0.5, so the denominator is >= 0.25.
const float MAXENT_VARIANCE_KERNEL_DENOM = MAXENT_VARIANCE_KERNEL_SIGMA * MAXENT_VARIANCE_KERNEL_SIGMA;
const float MAXENT_VARIANCE_KERNEL_1D[4] = {
    1.0,
    exp(-0.5 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-2.0 / MAXENT_VARIANCE_KERNEL_DENOM),
    exp(-4.5 / MAXENT_VARIANCE_KERNEL_DENOM)
    };

void denoiserVarianceWriteTile(uint index, DenoiserVarianceGeometry geometry, DenoiserVarianceSource source, bool valid) {
    valid = valid && statisticsValidEffectiveSampleCount(source.historyEffectiveSamples)
        && denoiserVarianceMomentsValid(source.maxEntY, source.rootMeanY2);
    source = denoiserVarianceSanitizeSource(source);
    float effectiveSamples = valid ? min(source.historyEffectiveSamples, 65504.0) : 1.0;
    denoiserVarianceSurfaceTile[index] = uvec2(floatBitsToUint(geometry.surfaceDistance),
            encodeNormalU(geometry.primaryRay));
    denoiserVariancePdfDirectionTile[index] = encodeNormalU(geometry.pdfDirection);
    denoiserVarianceMetadataTile[index] = valid ? 0x80000000u : 0u;
    denoiserVarianceMeanTile[index] = uvec2(
            packHalf2x16(clamp(source.maxEntY.xy,
                    vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(source.maxEntY.zw,
                    vec2(-65504.0), vec2(65504.0))));
    denoiserVarianceMomentTile[index] = packHalf2x16(vec2(
            effectiveSamples, min(source.rootMeanY2, 65504.0)));
}

void denoiserVarianceLoadTile(uint index, ivec2 pixel, ivec2 imageMax) {
    ivec2 clampedPixel = clamp(pixel, ivec2(0), imageMax);
    bool inBounds = all(equal(pixel, clampedPixel));
    DenoiserVarianceGeometry geometry = denoiserVarianceLoadGeometry(clampedPixel);
    bool valid = inBounds && geometry.valid;
    DenoiserVarianceSource source = valid ? denoiserVarianceLoadSource(clampedPixel) : denoiserVarianceEmptySource();
    denoiserVarianceWriteTile(index, geometry, source, valid);
}

bool denoiserVarianceTileValid(uint index) {
    return (denoiserVarianceMetadataTile[index] & 0x80000000u) != 0u;
}

float denoiserVarianceTileSurfacePlaneExponent(uint index, float centerPlaneOffset,
    vec3 centerGeometryNormal, float surfaceRejectionScale) {
    uvec2 words = denoiserVarianceSurfaceTile[index];
    return denoiserSpatialAxialDistanceExponent(centerPlaneOffset, centerGeometryNormal,
        decodeNormalU(words.y), uintBitsToFloat(words.x), surfaceRejectionScale);
}

float denoiserVarianceTilePdfDirectionExponent(uint index, vec3 centerPdfDirection) {
    return denoiserSpatialPdfDirectionExponent(
        centerPdfDirection, decodeNormalU(denoiserVariancePdfDirectionTile[index]));
}

vec4 denoiserVarianceTileMean(uint index) {
    uvec2 words = denoiserVarianceMeanTile[index];
    return vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
}

vec2 denoiserVarianceTileRootMeanY2EffectiveSamples(uint index) {
    return unpackHalf2x16(denoiserVarianceMomentTile[index]).yx;
}

float denoiserVariancePreparedSpatialMonteCarloVariance(uint centerX, uint centerY, DenoiserVarianceGeometry centerGeometry,
        float centerPlaneOffset, float surfaceRejectionScale) {
    float sumWeight = 0.0;
    float sumSquaredWeightOverEffectiveSamples = 0.0;
    vec4 sumMean = vec4(0.0);
    float sumMeanY2 = 0.0;

    // Linear moments use normalized spatial weights. Kish N_eff alone uses
    // the nonlinear weighted-estimator reconstruction W^2/sum(w_i^2/N_i).
    for (int offsetY = -3; offsetY <= 3; ++offsetY) {
        for (int offsetX = -3; offsetX <= 3; ++offsetX) {
            uint sampleIndex = uint(int(centerY) + offsetY) * MAXENT_VARIANCE_SHARED_WIDTH + uint(int(centerX) + offsetX);
            if (!denoiserVarianceTileValid(sampleIndex)) continue;

            float spatialWeight = MAXENT_VARIANCE_KERNEL_1D[abs(offsetX)] * MAXENT_VARIANCE_KERNEL_1D[abs(offsetY)]
                * exp(-(denoiserVarianceTileSurfacePlaneExponent(sampleIndex, centerPlaneOffset, centerGeometry.geometryNormal, surfaceRejectionScale)
                    + denoiserVarianceTilePdfDirectionExponent(sampleIndex, centerGeometry.pdfDirection)));
            if (!(spatialWeight > 0.0) || isnan(spatialWeight) || isinf(spatialWeight)) continue;
            vec2 rootMeanY2EffectiveSamples = denoiserVarianceTileRootMeanY2EffectiveSamples(sampleIndex);
            float sampleEffectiveSamples = rootMeanY2EffectiveSamples.y;
            sumWeight += spatialWeight;
            sumSquaredWeightOverEffectiveSamples += spatialWeight * spatialWeight / sampleEffectiveSamples;
            sumMean += spatialWeight * denoiserVarianceTileMean(sampleIndex);
            sumMeanY2 += spatialWeight * rootMeanY2EffectiveSamples.x * rootMeanY2EffectiveSamples.x;
        }
    }

    if (!(sumWeight > 1e-8)) return DENOISER_UNKNOWN_UNCERTAINTY;
    float inverseWeight = 1.0 / sumWeight;
    vec4 pooledMean = sumMean * inverseWeight;
    float pooledRootMeanY2 = sqrt(sumMeanY2 * inverseWeight);
    float pooledEffectiveSamples = statisticsKishEffectiveSampleCount(sumWeight, sumSquaredWeightOverEffectiveSamples);
    return denoiserMonteCarloVarianceFromTemporalMoments(pooledMean, pooledRootMeanY2, pooledEffectiveSamples);
}

void denoiserVariancePrepare() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    uvec2 localPixel = gl_LocalInvocationID.xy;
    uint lane = gl_LocalInvocationIndex;
    ivec2 imageSize = ivec2(resolution_global);
    ivec2 imageMax = imageSize - 1;
    float spatialOnlySamples = MAXENT_VARIANCE_SPATIAL_ONLY_SAMPLES;
    float varianceTransitionEnd = spatialOnlySamples + max(MAXENT_VARIANCE_TRANSITION_SAMPLES, 1e-3);

    uint centerX = localPixel.x + MAXENT_VARIANCE_HALO;
    uint centerY = localPixel.y + MAXENT_VARIANCE_HALO;
    uint centerIndex = centerY * MAXENT_VARIANCE_SHARED_WIDTH + centerX;
    ivec2 centerPixel = ivec2(pixel);
    ivec2 clampedCenter = clamp(centerPixel, ivec2(0), imageMax);
    bool centerInBounds = all(equal(centerPixel, clampedCenter));
    uvec4 centerGeometryWords = denoiserVarianceLoadPrimaryGeometryWords(clampedCenter);
    DenoiserVarianceGeometry centerGeometry = denoiserVarianceLoadGeometry(clampedCenter);
    bool centerValid = centerInBounds && centerGeometry.valid;
    float surfaceRejectionScale = denoiserSpatialDistanceRejectionScale(
            centerGeometry.surfaceDistance, float(imageSize.y));
    float centerPlaneOffset = centerGeometry.surfaceDistance
            * dot(centerGeometry.geometryNormal, centerGeometry.primaryRay);
    DenoiserVarianceSource centerSource = centerValid
        ? denoiserVarianceLoadSource(clampedCenter) : denoiserVarianceEmptySource();
    DenoiserVarianceSource centerIndependentCurrent = centerValid
        ? denoiserVarianceLoadIndependentCurrentSource(clampedCenter)
        : denoiserVarianceEmptySource();
    centerSource = denoiserVarianceSanitizeSource(centerSource);
    centerIndependentCurrent = denoiserVarianceSanitizeSource(
        centerIndependentCurrent);
    denoiserVarianceWriteTile(centerIndex, centerGeometry, centerSource, centerValid);

    // Interior invocations publish their temporal moment states; cooperative
    // loads only have to fill the three-pixel halo.
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * MAXENT_VARIANCE_TILE_SIZE) - ivec2(MAXENT_VARIANCE_HALO);
    for (uint index = lane; index < MAXENT_VARIANCE_SHARED_AREA; index += MAXENT_VARIANCE_TILE_SIZE * MAXENT_VARIANCE_TILE_SIZE) {
        uint tileX = index % MAXENT_VARIANCE_SHARED_WIDTH;
        uint tileY = index / MAXENT_VARIANCE_SHARED_WIDTH;
        bool interior = tileX >= MAXENT_VARIANCE_HALO && tileX < MAXENT_VARIANCE_HALO + MAXENT_VARIANCE_TILE_SIZE
            && tileY >= MAXENT_VARIANCE_HALO && tileY < MAXENT_VARIANCE_HALO + MAXENT_VARIANCE_TILE_SIZE;
        if (!interior) denoiserVarianceLoadTile(index, tileOrigin + ivec2(tileX, tileY), imageMax);
    }
    barrier();

    float standardDeviation = DENOISER_UNKNOWN_UNCERTAINTY;
    float independentCurrentStandardDeviation = DENOISER_UNKNOWN_UNCERTAINTY;
    if (centerValid) {
        float spatialVariance = denoiserVariancePreparedSpatialMonteCarloVariance(
            centerX, centerY, centerGeometry, centerPlaneOffset, surfaceRejectionScale);
        float temporalVariance = denoiserMonteCarloVarianceFromTemporalMoments(
            centerSource.maxEntY, centerSource.rootMeanY2, centerSource.historyEffectiveSamples);
        float temporalTrust = smoothstep(spatialOnlySamples, varianceTransitionEnd, centerSource.historyEffectiveSamples);
        bool spatialKnown = denoiserVarianceKnown(spatialVariance);
        bool temporalKnown = denoiserVarianceKnown(temporalVariance);
        float preparedVariance = spatialKnown && temporalKnown
            ? mix(spatialVariance, temporalVariance, temporalTrust)
            : (spatialKnown ? spatialVariance : temporalVariance);
        // Convert the chosen observation variance exactly once. The pooled
        // Kish count estimates the observation distribution; the proposal mean
        // still has the center's temporal count.
        standardDeviation = denoiserVarianceKnown(preparedVariance)
            ? denoiserVarianceToSigma(preparedVariance / max(centerSource.historyEffectiveSamples, 1.0))
            : DENOISER_UNKNOWN_UNCERTAINTY;
        independentCurrentStandardDeviation = denoiserVarianceToSigma(
            spatialKnown ? spatialVariance : temporalVariance);
    }

    if (!centerInBounds) return;
    if (!centerValid) {
        denoiserVarianceStoreInvalid(centerPixel);
        return;
    }

    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = centerSource.maxEntY;
    outputSignal.CoCg = centerSource.CoCg;
    outputSignal.standardDeviation = standardDeviation;
    outputSignal.virtualDistance = centerGeometry.surfaceDistance + centerGeometry.virtualScale * centerSource.hitDistance;

    DenoiserMaxEntSignal independentCurrentSignal;
    independentCurrentSignal.maxEntY = centerIndependentCurrent.maxEntY;
    independentCurrentSignal.CoCg = centerIndependentCurrent.CoCg;
    independentCurrentSignal.standardDeviation = independentCurrentStandardDeviation;
    independentCurrentSignal.virtualDistance = centerGeometry.surfaceDistance
        + centerGeometry.virtualScale * centerIndependentCurrent.hitDistance;

    denoiserVarianceDebugWrite(centerPixel, standardDeviation);
    denoiserVarianceStorePrepared(centerPixel, outputSignal, independentCurrentSignal, centerGeometryWords,
        centerGeometry, centerSource.historyEffectiveSamples);
}

#endif // MAXENT_DENOISER_VARIANCE_PREPARE_GLSL
