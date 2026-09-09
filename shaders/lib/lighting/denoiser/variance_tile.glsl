#ifndef MAXENT_DENOISER_VARIANCE_TILE_GLSL
#define MAXENT_DENOISER_VARIANCE_TILE_GLSL

// Internal tile staging and spatial moment pooling. Included after the shared
// variance source/geometry types, adapter callbacks and moment closure.

const uint MAXENT_VARIANCE_TILE_SIZE = 16u;
const uint MAXENT_VARIANCE_HALO = 3u;
const uint MAXENT_VARIANCE_SHARED_WIDTH = 22u;
const uint MAXENT_VARIANCE_SHARED_AREA = 22u * 22u;

// Decode each ray once when publishing the tile, rather than in each of the
// 49 neighborhood taps. xyz retains the existing octahedral quantization;
// w is radial distance with its sign bit carrying validity (including -0).
// The extra two shared words per entry trade LDS capacity for fewer repeated
// normalizations. Both signal domains use this same representation.
shared vec4 denoiserVarianceSurfaceTile[MAXENT_VARIANCE_SHARED_AREA];
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
    // A valid radial distance is nonnegative. Its sign bit carries tile
    // validity, saving a separate 484-word shared array. Invalid entries are
    // never decoded, so their remaining fields need no initialization.
    denoiserVarianceSurfaceTile[index].w = uintBitsToFloat(
            floatBitsToUint(abs(geometry.surfaceDistance)) | (valid ? 0u : 0x80000000u));
    if (!valid) return;
    denoiserVarianceSurfaceTile[index].xyz = decodeNormalU(encodeNormalU(geometry.primaryRay));
    // The validation above already establishes finite, FP16-bounded moments;
    // chroma and hit distance are not part of this tile's statistical input.
    float effectiveSamples = min(source.historyEffectiveSamples, 65504.0);
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
    return (floatBitsToUint(denoiserVarianceSurfaceTile[index].w) & 0x80000000u) == 0u;
}

float denoiserVarianceTileSurfacePlaneExponent(uint index, float centerPlaneOffset,
    vec3 centerGeometryNormal, float surfaceRejectionScale) {
    vec4 surface = denoiserVarianceSurfaceTile[index];
    return denoiserSpatialAxialDistanceExponent(centerPlaneOffset, centerGeometryNormal,
        surface.xyz, surface.w, surfaceRejectionScale);
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
                * exp(-denoiserVarianceTileSurfacePlaneExponent(sampleIndex,
                    centerPlaneOffset, centerGeometry.geometryNormal,
                    surfaceRejectionScale));
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

#endif // MAXENT_DENOISER_VARIANCE_TILE_GLSL
