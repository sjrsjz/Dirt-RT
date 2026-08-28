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

// Minimum-variance current weight V_H/(V_H+V_C) for two independent unbiased
// estimators. This is exposed only as a diagnostic; temporal response uses its
// separately tuned moment-distance function.
float statisticsMinimumVarianceIndependentCurrentWeight(float historyVariance, float currentVariance) {
    historyVariance = max(historyVariance, 0.0);
    currentVariance = max(currentVariance, 0.0);
    float denominator = historyVariance + currentVariance;
    if (!(denominator > 0.0) || isnan(denominator)
            || isinf(denominator))
        return 0.0;
    return clamp(historyVariance / denominator, 0.0, 1.0);
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
        vec3 expectedValue) {
    return max(expectedSquaredNorm - dot(expectedValue, expectedValue), 0.0);
}

// E[S_biased] = (1 - 1/N_eff) Var[Z]. Recover the Monte Carlo
// per-observation variance without converting it to temporal mean variance.
float statisticsObservationVarianceFromBiasedCentralMoment(
        float biasedCentralMoment, float effectiveSamples) {
    if (!(effectiveSamples > 1.0)) return 0.0;
    return max(biasedCentralMoment, 0.0)
        / (1.0 - 1.0 / effectiveSamples);
}

#endif // LIB_MATH_STATISTICS_GLSL
