#ifndef LIB_MATH_STATISTICS_GLSL
#define LIB_MATH_STATISTICS_GLSL

// Pure statistical operations shared by the denoiser. This file knows
// nothing about MaxEnt, geometry, pass identity, or storage formats.

bool statisticsValidEffectiveSampleCount(float effectiveSamples) {
    return effectiveSamples >= 1.0 && !isnan(effectiveSamples)
        && !isinf(effectiveSamples);
}

// Kish ESS of a normalized weighted mixture of estimators:
// N_eff = (sum_i w_i)^2 / sum_i(w_i^2 / N_i).
float statisticsKishEffectiveSampleCount(float weightSum,
        float squaredWeightOverEffectiveSamplesSum) {
    if (!(weightSum > 1e-8)
            || !(squaredWeightOverEffectiveSamplesSum > 1e-12)
            || isnan(weightSum) || isinf(weightSum)
            || isnan(squaredWeightOverEffectiveSamplesSum)
            || isinf(squaredWeightOverEffectiveSamplesSum))
        return 0.0;
    return weightSum * weightSum
        / squaredWeightOverEffectiveSamplesSum;
}

// Effective sample count of correlated estimators under the constant-
// correlation covariance closure:
//
// N_eff = W^2 / ((1-p) sum_i(w_i^2 / N_i)
//                  + p (sum_i(w_i / sqrt(N_i)))^2).
//
// This is Kish ESS at p=0. At p=1 it describes resampling one estimator
// field: interpolating equal-N taps leaves N unchanged at every subpixel
// phase instead of manufacturing up to four independent histories.
float statisticsCorrelatedEffectiveSampleCount(float weightSum,
        float squaredWeightOverEffectiveSamplesSum,
        float weightOverRootEffectiveSamplesSum, float correlation) {
    if (!(weightSum > 1e-8)
            || !(weightOverRootEffectiveSamplesSum > 1e-8)
            || isnan(weightSum) || isinf(weightSum)
            || isnan(squaredWeightOverEffectiveSamplesSum)
            || isinf(squaredWeightOverEffectiveSamplesSum)
            || isnan(weightOverRootEffectiveSamplesSum)
            || isinf(weightOverRootEffectiveSamplesSum))
        return 0.0;
    float p = clamp(correlation, 0.0, 1.0);
    float denominator = (1.0 - p)
            * max(squaredWeightOverEffectiveSamplesSum, 0.0)
        + p * weightOverRootEffectiveSamplesSum
            * weightOverRootEffectiveSamplesSum;
    if (!(denominator > 1e-12) || isnan(denominator)
            || isinf(denominator))
        return 0.0;
    return weightSum * weightSum / denominator;
}

// Bilinear/history-field reconstruction is a single resampling operation,
// not four independent observations. This p=1 specialization avoids carrying
// the unused independent-estimator term through every reprojection tap.
float statisticsReconstructedEffectiveSampleCount(float weightSum,
        float weightOverRootEffectiveSamplesSum) {
    if (!(weightSum > 1e-8)
            || !(weightOverRootEffectiveSamplesSum > 1e-8)
            || isnan(weightSum) || isinf(weightSum)
            || isnan(weightOverRootEffectiveSamplesSum)
            || isinf(weightOverRootEffectiveSamplesSum))
        return 0.0;
    float ratio = weightSum / weightOverRootEffectiveSamplesSum;
    return ratio * ratio;
}

float statisticsKishBlendEffectiveSampleCounts(float effectiveSamplesA,
        float effectiveSamplesB, float alpha) {
    alpha = clamp(alpha, 0.0, 1.0);
    if (alpha <= 0.0) return statisticsValidEffectiveSampleCount(
        effectiveSamplesA) ? effectiveSamplesA : 0.0;
    if (alpha >= 1.0) return statisticsValidEffectiveSampleCount(
        effectiveSamplesB) ? effectiveSamplesB : 0.0;
    if (!statisticsValidEffectiveSampleCount(effectiveSamplesA)
            || !statisticsValidEffectiveSampleCount(effectiveSamplesB))
        return 0.0;
    float weightA = 1.0 - alpha;
    return statisticsKishEffectiveSampleCount(1.0,
        weightA * weightA / effectiveSamplesA
            + alpha * alpha / effectiveSamplesB);
}

// X' = (1-alpha) X + alpha x, where x is one independent observation.
// Alpha is the mixing coefficient; N_eff is only its statistical consequence.
float statisticsKishUpdateEffectiveSampleCount(float historyEffectiveSamples,
        float alpha) {
    if (!statisticsValidEffectiveSampleCount(historyEffectiveSamples))
        return 1.0;
    alpha = clamp(alpha, 0.0, 1.0);
    float retainedWeight = 1.0 - alpha;
    float squaredWeightSum = retainedWeight * retainedWeight
            / historyEffectiveSamples + alpha * alpha;
    return 1.0 / max(squaredWeightSum, 1e-12);
}

// Minimum current weight for a finite temporal window. During growth,
// alpha=1/(N+1) is ordinary equal-weight accumulation and advances Kish N by
// exactly one. Once the requested window is reached, alpha=2/(N+1) makes the
// Kish update stationary (N'=N), yielding a finite-window EMA without ever
// clamping or otherwise editing N_eff directly.
float statisticsKishFiniteWindowCurrentWeightFloor(
        float historyEffectiveSamples, float maximumEffectiveSamples) {
    if (!statisticsValidEffectiveSampleCount(historyEffectiveSamples))
        return 1.0;
    maximumEffectiveSamples = max(maximumEffectiveSamples, 1.0);
    float numerator = historyEffectiveSamples >= maximumEffectiveSamples
        ? 2.0 : 1.0;
    return clamp(numerator / (historyEffectiveSamples + 1.0), 0.0, 1.0);
}

// Trace-variance closure when only two scalar standard deviations and one
// correlation coefficient are retained.
float statisticsDifferenceVarianceFromStandardDeviations(
        float standardDeviationA, float standardDeviationB,
        float correlation) {
    standardDeviationA = max(standardDeviationA, 0.0);
    standardDeviationB = max(standardDeviationB, 0.0);
    float p = clamp(correlation, -1.0, 1.0);
    return max(standardDeviationA * standardDeviationA
        + standardDeviationB * standardDeviationB
        - 2.0 * p * standardDeviationA * standardDeviationB, 0.0);
}

// Positive-part estimate of a deterministic squared bias from one observed
// estimator difference. For independent estimators,
// E[|C-H|^2] = |b|^2 + V_C + V_H.
float statisticsPositivePartSquaredBias(float squaredDifference,
        float differenceNoiseVariance) {
    if (!(squaredDifference >= 0.0) || isnan(squaredDifference)
            || isinf(squaredDifference)
            || !(differenceNoiseVariance >= 0.0)
            || isnan(differenceNoiseVariance)
            || isinf(differenceNoiseVariance))
        return 0.0;
    return max(squaredDifference - differenceNoiseVariance, 0.0);
}

// Minimum-MSE current weight for
//   H = mu + b + e_H, C = mu + e_C,
// with independent zero-mean estimator errors. The covariance term is
// deliberately zero by contract.
float statisticsMinimumMseIndependentCurrentWeight(float squaredBias,
        float historyVariance, float currentVariance) {
    squaredBias = max(squaredBias, 0.0);
    historyVariance = max(historyVariance, 0.0);
    currentVariance = max(currentVariance, 0.0);
    float historyMse = squaredBias + historyVariance;
    float denominator = historyMse + currentVariance;
    if (!(denominator > 0.0) || isnan(denominator)
            || isinf(denominator))
        return 0.0;
    return clamp(historyMse / denominator, 0.0, 1.0);
}

float statisticsIndependentBlendVariance(float historyVariance,
        float currentVariance, float currentWeight) {
    currentWeight = clamp(currentWeight, 0.0, 1.0);
    float historyWeight = 1.0 - currentWeight;
    return max(historyWeight * historyWeight
            * max(historyVariance, 0.0)
        + currentWeight * currentWeight
            * max(currentVariance, 0.0), 0.0);
}

// Constant-correlation expansion for a normalized weighted mean. Callers
// accumulate Q=sum(w_i^2 V_i), S=sum(w_i sqrt(V_i)), and W=sum(w_i).
float statisticsWeightedMeanVariance(float squaredWeightVarianceSum,
        float weightedStandardDeviationSum, float inverseWeightSum,
        float correlation) {
    float p = clamp(correlation, -1.0, 1.0);
    float numerator = (1.0 - p) * max(squaredWeightVarianceSum, 0.0)
        + p * weightedStandardDeviationSum
            * weightedStandardDeviationSum;
    return max(numerator * inverseWeightSum * inverseWeightSum, 0.0);
}

float statisticsWeightedMeanStandardDeviation(
        float squaredWeightVarianceSum,
        float weightedStandardDeviationSum, float inverseWeightSum,
        float correlation) {
    return sqrt(statisticsWeightedMeanVariance(
        squaredWeightVarianceSum, weightedStandardDeviationSum,
        inverseWeightSum, correlation));
}

float statisticsBiasedCentralSecondMoment(float expectedSquaredNorm,
        vec4 expectedValue) {
    return max(expectedSquaredNorm - dot(expectedValue, expectedValue), 0.0);
}

// For a weighted empirical estimator, the biased central moment and the
// estimator variance satisfy B = (N_eff - 1) Var[mean]. These direct
// conversions keep moment reconstruction and variance preparation exact
// inverses without passing through per-observation variance unnecessarily.
float statisticsEstimatorVarianceFromBiasedCentralMoment(
        float biasedCentralMoment, float effectiveSamples) {
    if (!(effectiveSamples > 1.0) || isnan(effectiveSamples)
            || isinf(effectiveSamples))
        return 0.0;
    return max(biasedCentralMoment, 0.0)
        / (effectiveSamples - 1.0);
}

float statisticsEstimatorVarianceFromMoments(float expectedSquaredNorm,
        vec4 expectedValue, float effectiveSamples) {
    return statisticsEstimatorVarianceFromBiasedCentralMoment(
        statisticsBiasedCentralSecondMoment(
            expectedSquaredNorm, expectedValue),
        effectiveSamples);
}

float statisticsBiasedCentralMomentFromEstimatorVariance(
        float estimatorVariance, float effectiveSamples) {
    if (!statisticsValidEffectiveSampleCount(effectiveSamples)) return 0.0;
    return max(effectiveSamples - 1.0, 0.0)
        * max(estimatorVariance, 0.0);
}

float statisticsExpectedSquaredNormFromEstimatorVariance(
        vec4 expectedValue, float estimatorVariance,
        float effectiveSamples) {
    return dot(expectedValue, expectedValue)
        + statisticsBiasedCentralMomentFromEstimatorVariance(
            estimatorVariance, effectiveSamples);
}

// E[S_biased] = (1 - 1/N_eff) Var[Z]. First recover Monte Carlo
// per-observation variance, then separately scale it for an estimator.
float statisticsObservationVarianceFromBiasedCentralMoment(
        float biasedCentralMoment, float effectiveSamples) {
    if (!(effectiveSamples > 1.0)) return 0.0;
    return max(biasedCentralMoment, 0.0)
        / (1.0 - 1.0 / effectiveSamples);
}

float statisticsEstimatorVarianceFromObservationVariance(
        float observationVariance, float effectiveSamples) {
    if (!statisticsValidEffectiveSampleCount(effectiveSamples)) return 0.0;
    return max(observationVariance, 0.0) / effectiveSamples;
}

#endif // LIB_MATH_STATISTICS_GLSL
