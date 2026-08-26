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
// Outputs are returned through the including domain adapter. estimatorStdDev is
// the trace uncertainty of the encoded linear moment vector (E[Y u], E[Y]). Since |(Y u, Y)|^2 = 2Y^2, it is
// obtained directly from the stored moments without a decoder distribution or
// covariance-shape assumption. It is stored and propagated as an FP16 standard deviation. The estimated
// sampling-PDF direction constrains both variance pooling stages. The signal's upper
// metadata half contains the already constructed radial virtual distance, not
// reflection hit distance. No decoder distribution is involved.

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
    uint materialID;
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

uvec4 denoiserVariancePackSpatialGeometry(uvec4 primaryWords, DenoiserVarianceGeometry geometry) {
    uint packedRoughness = packHalf2x16(vec2(geometry.signalRoughness, 0.0));
    return uvec4(primaryWords.w, encodeNormalU(geometry.geometryNormal), encodeNormalU(geometry.pdfDirection), packedRoughness);
}

// Required adapter callbacks. Raw storage and image bindings remain outside this file.
uvec4 denoiserVarianceLoadPrimaryGeometryWords(ivec2 pixel);
DenoiserVarianceGeometry denoiserVarianceLoadGeometry(ivec2 pixel);
DenoiserVarianceSource denoiserVarianceLoadSource(ivec2 pixel);
DenoiserVarianceSource denoiserVarianceLoadIndependentCurrentSource(ivec2 pixel);
void denoiserVarianceStorePrepared(ivec2 pixel, DenoiserMaxEntSignal signal, DenoiserMaxEntSignal independentCurrent,
        uvec4 primaryGeometryWords, DenoiserVarianceGeometry geometry);
void denoiserVarianceStoreInvalid(ivec2 pixel);
void denoiserVarianceDebugWrite(ivec2 pixel, float estimatorStdDev);

float denoiserVarianceSanitizeNonnegative(float value) {
    return isnan(value) || isinf(value) ? 0.0 : max(value, 0.0);
}

vec4 denoiserVarianceFiniteMean(vec4 meanState) {
    return any(isnan(meanState)) || any(isinf(meanState)) ? vec4(0.0) : meanState;
}

float denoiserVarianceBiasedCentralMoment(vec4 meanState, float rootMeanY2) {
    meanState = denoiserVarianceFiniteMean(meanState);
    rootMeanY2 = denoiserVarianceSanitizeNonnegative(rootMeanY2);
    // z = (Y u, Y), |u| = 1, hence E[|z|^2] = 2 E[Y^2]. For
    // finite weighted empirical moments this is the biased central moment;
    // N_eff is required before it can represent Monte Carlo variance.
    return statisticsBiasedCentralSecondMoment(2.0 * rootMeanY2 * rootMeanY2, meanState);
}

float denoiserVarianceOfTemporalMean(vec4 meanState, float rootMeanY2, float historyEffectiveSamples) {
    // For normalized temporal weights, the stored central second moment has
    // expectation (1 - 1/N_eff) tr Cov[z].  Dividing by N_eff - 1 therefore
    // estimates tr Cov[weighted mean].
    return statisticsEstimatorVarianceFromBiasedCentralMoment(
        denoiserVarianceBiasedCentralMoment(meanState, rootMeanY2), historyEffectiveSamples);
}

DenoiserVarianceSource denoiserVarianceSanitizeSource(DenoiserVarianceSource source) {
    source.maxEntY = denoiserVarianceFiniteMean(source.maxEntY);
    if (any(isnan(source.CoCg)) || any(isinf(source.CoCg))) source.CoCg = vec2(0.0);
    source.rootMeanY2 = denoiserVarianceSanitizeNonnegative(source.rootMeanY2);
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
// an independent soft cosine weight. Geometry and material metadata occupy
// 16 bytes per tile entry.
shared uvec2 denoiserVarianceSurfaceTile[MAXENT_VARIANCE_SHARED_AREA];
shared uint denoiserVariancePdfDirectionTile[MAXENT_VARIANCE_SHARED_AREA];
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
    denoiserVariancePdfDirectionTile[index] = encodeNormalU(geometry.pdfDirection);
    denoiserVarianceMetadataTile[index] = valid
        ? (0x80000000u | (geometry.materialID & 0x7fffffffu)) : 0u;
    denoiserVarianceMeanTile[index] = uvec2(
            packHalf2x16(clamp(source.maxEntY.xy, vec2(-65504.0),
                    vec2(65504.0))),
            packHalf2x16(clamp(source.maxEntY.zw, vec2(-65504.0),
                    vec2(65504.0))));
    denoiserVarianceMomentTile[index] = packHalf2x16(vec2(
                source.historyEffectiveSamples, min(source.rootMeanY2, 65504.0)));
    denoiserVarianceCurrentMeanTile[index] = uvec2(
            packHalf2x16(clamp(independentCurrent.maxEntY.xy,
                    vec2(-65504.0), vec2(65504.0))),
            packHalf2x16(clamp(independentCurrent.maxEntY.zw,
                    vec2(-65504.0), vec2(65504.0))));
    denoiserVarianceCurrentMomentTile[index] = packHalf2x16(vec2(
            independentCurrent.historyEffectiveSamples,
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

vec4 denoiserVarianceTileMean(uint index, bool independentCurrent) {
    uvec2 words = independentCurrent
        ? denoiserVarianceCurrentMeanTile[index]
        : denoiserVarianceMeanTile[index];
    return vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
}

vec2 denoiserVarianceTileRootMeanY2EffectiveSamples(uint index, bool independentCurrent) {
    uint words = independentCurrent
        ? denoiserVarianceCurrentMomentTile[index]
        : denoiserVarianceMomentTile[index];
    vec2 historyRootM2 = unpackHalf2x16(words);
    return historyRootM2.yx;
}

float denoiserVariancePreparedEstimatorStdDev(uint centerIndex, uint centerX, uint centerY, DenoiserVarianceGeometry centerGeometry,
        float centerPlaneOffset, float surfaceRejectionScale,
        float varianceHistoryBegin, float varianceHistoryEnd, bool independentCurrent) {
    vec4 centerMean = denoiserVarianceTileMean(centerIndex, independentCurrent);
    vec2 centerRootMeanY2EffectiveSamples = denoiserVarianceTileRootMeanY2EffectiveSamples(centerIndex, independentCurrent);
    float centerRootMeanY2 = centerRootMeanY2EffectiveSamples.x;
    float centerEffectiveSamples = centerRootMeanY2EffectiveSamples.y;
    float temporalVariance = denoiserVarianceOfTemporalMean(centerMean, centerRootMeanY2, centerEffectiveSamples);
    float spatialVariance = temporalVariance;

    if (centerEffectiveSamples < varianceHistoryEnd && denoiserVariancePoolRequired != 0u) {
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
                    * exp(-(denoiserVarianceTileSurfacePlaneExponent(
                            sampleIndex, centerPlaneOffset,
                            centerGeometry.geometryNormal,
                            surfaceRejectionScale)
                        + denoiserVarianceTilePdfDirectionExponent(
                            sampleIndex, centerGeometry.pdfDirection)));
                vec2 sampleRootMeanY2EffectiveSamples =
                    denoiserVarianceTileRootMeanY2EffectiveSamples(sampleIndex, independentCurrent);
                float sampleRootMeanY2 = sampleRootMeanY2EffectiveSamples.x;
                float sampleEffectiveSamples = sampleRootMeanY2EffectiveSamples.y;
                float mass = spatialWeight * sampleEffectiveSamples;

                sumMass += mass;
                sumSquaredObservationWeight += spatialWeight
                    * spatialWeight * sampleEffectiveSamples;
                sumMean += mass * denoiserVarianceTileMean(
                    sampleIndex, independentCurrent);
                sumMeanY2 += mass * sampleRootMeanY2
                    * sampleRootMeanY2;
            }
        }

        if (sumMass > 1e-8) {
            float inverseMass = 1.0 / sumMass;
            vec4 pooledMean = sumMean * inverseMass;
            float pooledRootMeanY2 = sqrt(sumMeanY2 * inverseMass);
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
                        observationVariance, centerEffectiveSamples);
            }
        }
    }

    float spatialTrust = 1.0 - smoothstep(varianceHistoryBegin,
        varianceHistoryEnd, centerEffectiveSamples);
    float preparedVariance = mix(temporalVariance,
        spatialVariance, spatialTrust);
    return sqrt(max(preparedVariance, temporalVariance));
}

void denoiserVariancePrepare() {
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
    denoiserVarianceWriteTile(centerIndex, centerGeometry, centerSource,
        centerIndependentCurrent, centerValid);

    if (centerValid && (centerSource.historyEffectiveSamples < varianceHistoryEnd
            || centerIndependentCurrent.historyEffectiveSamples < varianceHistoryEnd))
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

    float estimatorStdDev = 0.0;
    float independentCurrentEstimatorStdDev = 0.0;
    if (centerValid) {
        estimatorStdDev =
            denoiserVariancePreparedEstimatorStdDev(
                centerIndex, centerX, centerY, centerGeometry,
                centerPlaneOffset, surfaceRejectionScale,
                varianceHistoryBegin, varianceHistoryEnd, false);
        independentCurrentEstimatorStdDev =
            denoiserVariancePreparedEstimatorStdDev(
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
    outputSignal.estimatorStdDev = estimatorStdDev;
    outputSignal.virtualDistance = centerGeometry.surfaceDistance + centerGeometry.virtualScale * centerSource.hitDistance;

    DenoiserMaxEntSignal independentCurrentSignal;
    independentCurrentSignal.maxEntY = centerIndependentCurrent.maxEntY;
    independentCurrentSignal.CoCg = centerIndependentCurrent.CoCg;
    independentCurrentSignal.estimatorStdDev = independentCurrentEstimatorStdDev;
    independentCurrentSignal.virtualDistance = centerGeometry.surfaceDistance
        + centerGeometry.virtualScale * centerIndependentCurrent.hitDistance;

    denoiserVarianceDebugWrite(centerPixel, estimatorStdDev);
    denoiserVarianceStorePrepared(centerPixel, outputSignal, independentCurrentSignal, centerGeometryWords, centerGeometry);
}

#endif // MAXENT_DENOISER_VARIANCE_PREPARE_GLSL
