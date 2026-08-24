#ifndef MAXENT_TEMPORAL_ROBUST_TILE_GLSL
#define MAXENT_TEMPORAL_ROBUST_TILE_GLSL

// Both history-resolve programs use 16x16 workgroups. The including pass
// supplies maxentTemporalRobustLoadSignalWords(), which reads the independently
// current-frame-filtered branch rather than the temporally premixed proposal.
// A two-pixel halo gives each invocation a complete 5x5 neighborhood while
// every source signal and geometry record is fetched only once per workgroup.
const int MAXENT_TEMPORAL_ROBUST_GROUP_SIZE = 16;
const int MAXENT_TEMPORAL_ROBUST_RADIUS = 2;
const int MAXENT_TEMPORAL_ROBUST_DIAMETER = 5;
const int MAXENT_TEMPORAL_ROBUST_SAMPLE_COUNT = 25;
const int MAXENT_TEMPORAL_ROBUST_TILE_SIZE = 20;
const int MAXENT_TEMPORAL_ROBUST_TILE_AREA = 400;
const float MAXENT_TEMPORAL_ROBUST_MAX_PLANE_EXPONENT = 1.0;

shared uvec4 maxentTemporalRobustSignalTile[
    MAXENT_TEMPORAL_ROBUST_TILE_AREA];
shared uvec4 maxentTemporalRobustGeometryTile[
    MAXENT_TEMPORAL_ROBUST_TILE_AREA];

struct MaxEntTemporalRobustEstimate {
    vec4 moment;
    float standardDeviation;
};

int maxentTemporalRobustTileIndex(ivec2 tilePixel) {
    return tilePixel.y * MAXENT_TEMPORAL_ROBUST_TILE_SIZE + tilePixel.x;
}

ivec2 maxentTemporalRobustTilePixel(ivec2 offset) {
    return ivec2(gl_LocalInvocationID.xy)
        + ivec2(MAXENT_TEMPORAL_ROBUST_RADIUS) + offset;
}

void maxentTemporalRobustLoadSharedTile() {
    int localIndex = int(gl_LocalInvocationIndex);
    ivec2 imageSize = textureSize(colortex4, 0);
    ivec2 groupOrigin = ivec2(gl_WorkGroupID.xy)
        * MAXENT_TEMPORAL_ROBUST_GROUP_SIZE;
    for (int tileIndex = localIndex;
            tileIndex < MAXENT_TEMPORAL_ROBUST_TILE_AREA;
            tileIndex += MAXENT_TEMPORAL_ROBUST_GROUP_SIZE
                * MAXENT_TEMPORAL_ROBUST_GROUP_SIZE) {
        ivec2 tilePixel = ivec2(
            tileIndex % MAXENT_TEMPORAL_ROBUST_TILE_SIZE,
            tileIndex / MAXENT_TEMPORAL_ROBUST_TILE_SIZE);
        ivec2 pixel = groupOrigin + tilePixel
            - ivec2(MAXENT_TEMPORAL_ROBUST_RADIUS);
        bool validPixel = all(greaterThanEqual(pixel, ivec2(0)))
            && all(lessThan(pixel, imageSize));
        maxentTemporalRobustSignalTile[tileIndex] = validPixel
            ? maxentTemporalRobustLoadSignalWords(pixel)
            : denoiserInvalidMaxEntSignalWords();
        maxentTemporalRobustGeometryTile[tileIndex] = validPixel
            ? readPrimaryGeometryWords(uvec2(pixel))
            : uvec4(0u, 0u, 0u, floatBitsToUint(-1.0));
    }
    memoryBarrierShared();
    barrier();
}

uvec4 maxentTemporalRobustTileSignalWords(ivec2 offset) {
    return maxentTemporalRobustSignalTile[maxentTemporalRobustTileIndex(
        maxentTemporalRobustTilePixel(offset))];
}

uvec4 maxentTemporalRobustTileGeometryWords(ivec2 offset) {
    return maxentTemporalRobustGeometryTile[maxentTemporalRobustTileIndex(
        maxentTemporalRobustTilePixel(offset))];
}

vec4 maxentTemporalRobustTileMoment(ivec2 offset) {
    uvec4 words = maxentTemporalRobustTileSignalWords(offset);
    return vec4(unpackHalf2x16(words.x), unpackHalf2x16(words.y));
}

float maxentTemporalRobustTileStandardDeviation(ivec2 offset) {
    return max(unpackHalf2x16(
        maxentTemporalRobustTileSignalWords(offset).w).x, 0.0);
}

int maxentTemporalRobustNeighborhoodIndex(ivec2 offset) {
    ivec2 neighborhoodPixel = offset + ivec2(MAXENT_TEMPORAL_ROBUST_RADIUS);
    return neighborhoodPixel.y * MAXENT_TEMPORAL_ROBUST_DIAMETER + neighborhoodPixel.x;
}

uint maxentTemporalRobustNeighborhoodBit(ivec2 offset) {
    return 1u << uint(maxentTemporalRobustNeighborhoodIndex(offset));
}

bool maxentTemporalRobustMaskContains(uint acceptedMask, int sampleIndex) {
    return (acceptedMask & (1u << uint(sampleIndex))) != 0u;
}

// The caller constructs acceptedMask while performing geometry rejection once.
// The two reconstruction sweeps then read only shared denoiser output: first
// the scalar population variance, then one isotropic Gaussian reweighted mean.
// The same frozen weights propagate the supplied estimator trace variance.
MaxEntTemporalRobustEstimate maxentTemporalGaussianReweightedTileEstimate(
        uint acceptedMask, int sampleCount, vec4 momentSum,
        vec4 fallbackMoment, float fallbackStandardDeviation) {
    MaxEntTemporalRobustEstimate estimate;
    estimate.moment = fallbackMoment;
    estimate.standardDeviation = max(fallbackStandardDeviation, 0.0);
    if (sampleCount <= 0) return estimate;
    vec4 initialMean = momentSum / float(sampleCount);

    float scalarVariance = 0.0;
    float initialSquaredWeightVarianceSum = 0.0;
    float initialWeightedStddevSum = 0.0;
    for (int sampleIndex = 0;
            sampleIndex < MAXENT_TEMPORAL_ROBUST_SAMPLE_COUNT; ++sampleIndex) {
        if (!maxentTemporalRobustMaskContains(acceptedMask, sampleIndex))
            continue;
        ivec2 offset = ivec2(
            sampleIndex % MAXENT_TEMPORAL_ROBUST_DIAMETER,
            sampleIndex / MAXENT_TEMPORAL_ROBUST_DIAMETER)
            - ivec2(MAXENT_TEMPORAL_ROBUST_RADIUS);
        scalarVariance += maxentMomentDistanceSq(
            maxentTemporalRobustTileMoment(offset), initialMean);
        float standardDeviation =
            maxentTemporalRobustTileStandardDeviation(offset);
        initialSquaredWeightVarianceSum +=
            standardDeviation * standardDeviation;
        initialWeightedStddevSum += standardDeviation;
    }
    scalarVariance /= float(sampleCount);
    float inverseSampleCount = 1.0 / float(sampleCount);
    estimate.moment = initialMean;
    estimate.standardDeviation = statisticsWeightedMeanStandardDeviation(
        initialSquaredWeightVarianceSum, initialWeightedStddevSum,
        inverseSampleCount,
        MAXENT_TEMPORAL_ROBUST_PROPAGATION_CORRELATION);
    if (!(scalarVariance > 0.0) || isnan(scalarVariance)
            || isinf(scalarVariance)) return estimate;

    float inverseTwoVariance = 0.5 / scalarVariance;
    vec4 weightedMomentSum = vec4(0.0);
    float weightSum = 0.0;
    float squaredWeightVarianceSum = 0.0;
    float weightedStddevSum = 0.0;
    for (int sampleIndex = 0;
            sampleIndex < MAXENT_TEMPORAL_ROBUST_SAMPLE_COUNT; ++sampleIndex) {
        if (!maxentTemporalRobustMaskContains(acceptedMask, sampleIndex))
            continue;
        ivec2 offset = ivec2(
            sampleIndex % MAXENT_TEMPORAL_ROBUST_DIAMETER,
            sampleIndex / MAXENT_TEMPORAL_ROBUST_DIAMETER)
            - ivec2(MAXENT_TEMPORAL_ROBUST_RADIUS);
        vec4 moment = maxentTemporalRobustTileMoment(offset);
        float residualSq = maxentMomentDistanceSq(moment, initialMean);
        float weight = exp(-residualSq * inverseTwoVariance);
        weightedMomentSum += weight * moment;
        weightSum += weight;
        float standardDeviation =
            maxentTemporalRobustTileStandardDeviation(offset);
        squaredWeightVarianceSum += weight * weight
            * standardDeviation * standardDeviation;
        weightedStddevSum += weight * standardDeviation;
    }
    if (!(weightSum > 0.0) || isnan(weightSum) || isinf(weightSum))
        return estimate;

    vec4 reweightedMean = weightedMomentSum / weightSum;
    if (any(isnan(reweightedMean)) || any(isinf(reweightedMean)))
        return estimate;
    estimate.moment = reweightedMean;
    estimate.standardDeviation = statisticsWeightedMeanStandardDeviation(
        squaredWeightVarianceSum, weightedStddevSum, 1.0 / weightSum,
        MAXENT_TEMPORAL_ROBUST_PROPAGATION_CORRELATION);
    return estimate;
}

#endif // MAXENT_TEMPORAL_ROBUST_TILE_GLSL
