#ifndef MAXENT_MOMENT_STATISTICS_GLSL
#define MAXENT_MOMENT_STATISTICS_GLSL

// The denoiser operates on the encoder's linear moment vector
//
//     m = (E[R u], E[R]).
//
// No decoder distribution is involved.  For a unit direction u, a raw sample
// z = (R u, R) satisfies |z|^2 = 2 R^2, so E[R^2] is sufficient to recover
// the trace of the sampling covariance of z.  The full covariance matrix, and
// covariance between two already filtered estimators, are not stored.

float maxentMomentDifferenceCorrelationForSpatialStep(int stepRadius) {
    // Kernel-weighted center/tap trace correlation inherited from all
    // preceding fixed-kernel passes.
    if (stepRadius <= 1) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_16;
    return MAXENT_SPATIAL_DIFFERENCE_CORRELATION_STEP_32;
}

float maxentMomentPropagationCorrelationForSpatialStep(int stepRadius) {
    // Kernel-pair-weighted trace correlation among all nine pass inputs.
    if (stepRadius <= 1) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_1;
    if (stepRadius <= 2) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_2;
    if (stepRadius <= 4) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_4;
    if (stepRadius <= 8) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_8;
    if (stepRadius <= 16) return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_16;
    return MAXENT_SPATIAL_PROPAGATION_CORRELATION_STEP_32;
}

float maxentMomentDistanceSq(vec4 a, vec4 b) {
    vec4 delta = a - b;
    return max(dot(delta, delta), 0.0);
}

// Scalar covariance closure for estimator-error vectors:
//
//   E[epsilon_a . epsilon_b] ~= p_context sigma_a sigma_b,
//   sigma_i^2 = V_i = E[|epsilon_i|^2].
//
// This is a closure for the trace cross-covariance, not a claim that the
// unavailable component-wise covariance matrix has been reconstructed.
float maxentMomentDifferenceVarianceFromStandardDeviations(
    float standardDeviationA, float standardDeviationB,
    float correlation) {
    standardDeviationA = max(standardDeviationA, 0.0);
    standardDeviationB = max(standardDeviationB, 0.0);
    float p = clamp(correlation, -0.125, 1.0);
    float varianceA = standardDeviationA * standardDeviationA;
    float varianceB = standardDeviationB * standardDeviationB;
    float covarianceTrace = p * standardDeviationA * standardDeviationB;
    return max(varianceA + varianceB - 2.0 * covarianceTrace, 0.0);
}

// Complete covariance expansion for a normalized weighted mean.  Callers
// accumulate
//
//   Q = sum_i w_i^2 V_i,       S = sum_i w_i sqrt(V_i),
//
// then the constant-correlation closure gives
//
//   Var(sum_i w_i X_i / W)
//     = ((1-p) Q + p S^2) / W^2.
float maxentMomentWeightedMeanVariance(float squaredWeightVarianceSum,
    float weightedStddevSum, float inverseWeightSum,
    float correlation) {
    // -1/8 is conservative for every caller: the largest mixture is the
    // nine-estimator A-Trous pass.  Reprojection callers use positive p.
    float p = clamp(correlation, -0.125, 1.0);
    float numerator = (1.0 - p) * max(squaredWeightVarianceSum, 0.0)
            + p * weightedStddevSum * weightedStddevSum;
    return max(numerator * inverseWeightSum * inverseWeightSum, 0.0);
}

float maxentMomentWeightedMeanStandardDeviation(
    float squaredWeightVarianceSum, float weightedStddevSum,
    float inverseWeightSum, float correlation) {
    return sqrt(maxentMomentWeightedMeanVariance(
            squaredWeightVarianceSum, weightedStddevSum, inverseWeightSum,
            correlation));
}

float maxentClampHistoryWeightByMomentDifference(float historyWeight,
    vec4 currentMoment, float currentStddev,
    vec4 historyMoment, float historyStddev,
    float historySamples, float validWeight, float temporalCurrentWeight,
    float maximumHistory, float tolerance, out float normalizedDistance) {
    normalizedDistance = -1.0;
    historyWeight = isnan(historyWeight) || isinf(historyWeight)
        ? 0.0 : max(historyWeight, 0.0);
    // if (historySamples <= MAXENT_TEMPORAL_DIFFERENCE_COLD_START_HISTORY)
    //     return historyWeight;
    if (any(isnan(currentMoment)) || any(isinf(currentMoment))
            || any(isnan(historyMoment)) || any(isinf(historyMoment))
            || !(currentStddev >= 0.0) || isnan(currentStddev)
            || isinf(currentStddev)
            || !(historyStddev >= 0.0) || isnan(historyStddev)
            || isinf(historyStddev)
            || !(historySamples >= 1.0) || isnan(historySamples)
            || isinf(historySamples)
            || !(validWeight > 0.0) || isnan(validWeight)
            || isinf(validWeight)
            || !(temporalCurrentWeight > 0.0)
            || isnan(temporalCurrentWeight)
            || isinf(temporalCurrentWeight)) return historyWeight;

    float distanceSq = maxentMomentDistanceSq(currentMoment, historyMoment);
    // The compared values are the robust current spatial estimator C and the
    // reprojected previous denoised estimator H. Their standard deviations are
    // both available, so do not reconstruct sigma_C from N_eff * Var(H).
    //
    // C retains (1-alpha) of H. The fixed-kernel overlap constant accounts for
    // the additional 5x5 current reconstruction versus the history response.
    // Expressing the known shared-history covariance in correlation form keeps
    // the same scalar trace-covariance closure used by every spatial pass:
    //
    //   Cov(C,H) ~= p_overlap (1-alpha) Var(H)
    //            = p_CH sigma_C sigma_H.
    float sharedHistoryWeight = clamp(1.0 - temporalCurrentWeight, 0.0, 1.0);
    float currentHistoryCorrelation = 0.0;
    if (currentStddev > 0.0 && historyStddev > 0.0) {
        currentHistoryCorrelation = clamp(
                MAXENT_TEMPORAL_ROBUST_HISTORY_OVERLAP_CORRELATION
                    * sharedHistoryWeight * historyStddev / currentStddev,
                -0.125, 1.0);
    }
    float observedDifferenceVariance =
        maxentMomentDifferenceVarianceFromStandardDeviations(
            currentStddev, historyStddev, currentHistoryCorrelation)
            + MAXENT_TEMPORAL_DENOISER_INTRINSIC_VARIANCE;
    float normalizedDistanceSq = distanceSq / observedDifferenceVariance;
    normalizedDistance = sqrt(normalizedDistanceSq);
    if (isnan(normalizedDistance) || isinf(normalizedDistance)) {
        normalizedDistance = -1.0;
        return historyWeight;
    }

    // // Only trace variance is identifiable from E[R^2], so this remains a
    // // trace-standardized evidence measure rather than a chi-square/Wald test.
    float historyCap = min(
            tolerance * exp(-min(0.25*(normalizedDistanceSq), 80.0)), maximumHistory);
    return min(historyWeight, historyCap);
}

#endif // MAXENT_MOMENT_STATISTICS_GLSL
