#ifndef MAXENT_TEMPORAL_STATISTICS_GLSL
#define MAXENT_TEMPORAL_STATISTICS_GLSL

// Kish effective sample count for normalized estimator weights:
//
//     N_eff = 1 / sum_i(w_i^2)
//
// Tracking N_eff rather than elapsed frames keeps the variance of an
// adaptively weighted temporal mean consistent with V_population/(N_eff-1).

float maxentTemporalFiniteEffectiveSamples(float sampleCount) {
    if (isnan(sampleCount) || isinf(sampleCount)) return 0.0;
    return max(sampleCount, 0.0);
}

// Bilinear reprojection taps are correlated spatial reconstructions of one
// temporal estimator, not additional independent observations.  Use their
// weighted harmonic temporal count so a uniform N field remains exactly N at
// every sub-pixel phase.  Applying Kish across the spatial tap weights would
// incorrectly turn N into 2N/4N at texel edges/corners and imprint a grid on
// the following frame's history blend.
float maxentTemporalReprojectedEffectiveSamples(float sumWeight,
        float sumWeightOverSamples, float maximumSamples) {
    if (!(sumWeight > 1e-8) || !(sumWeightOverSamples > 1e-12)
            || isnan(sumWeightOverSamples)
            || isinf(sumWeightOverSamples))
        return 0.0;
    float effectiveSamples = sumWeight / sumWeightOverSamples;
    return clamp(effectiveSamples, 1.0, max(maximumSamples, 1.0));
}

float maxentTemporalUpdatedEffectiveSamples(float historySamples,
        float currentWeight, float maximumSamples) {
    historySamples = max(
        maxentTemporalFiniteEffectiveSamples(historySamples), 1.0);
    currentWeight = clamp(currentWeight, 0.0, 1.0);
    float historyWeight = 1.0 - currentWeight;
    float sumSquaredWeights = historyWeight * historyWeight / historySamples
        + currentWeight * currentWeight;
    return clamp(1.0 / max(sumSquaredWeights, 1e-12), 1.0,
        max(maximumSamples, 1.0));
}

// Two reprojected estimators may both be blended with the same current sample
// before being combined (surface-motion and virtual-motion histories). Count
// that shared current sample once rather than treating the two branches as
// independent observations.
float maxentTemporalSharedCurrentEffectiveSamples(
        float historySamplesA, float currentWeightA, bool historyValidA,
        float historySamplesB, float currentWeightB, bool historyValidB,
        float branchBWeight, float maximumSamples) {
    float branchWeightB = clamp(branchBWeight, 0.0, 1.0);
    float branchWeightA = 1.0 - branchWeightB;
    currentWeightA = clamp(currentWeightA, 0.0, 1.0);
    currentWeightB = clamp(currentWeightB, 0.0, 1.0);

    float sharedCurrentWeight = 0.0;
    float sumSquaredWeights = 0.0;
    if (historyValidA) {
        float historyWeight = branchWeightA * (1.0 - currentWeightA);
        sumSquaredWeights += historyWeight * historyWeight / max(
            maxentTemporalFiniteEffectiveSamples(historySamplesA), 1.0);
        sharedCurrentWeight += branchWeightA * currentWeightA;
    } else {
        sharedCurrentWeight += branchWeightA;
    }
    if (historyValidB) {
        float historyWeight = branchWeightB * (1.0 - currentWeightB);
        sumSquaredWeights += historyWeight * historyWeight / max(
            maxentTemporalFiniteEffectiveSamples(historySamplesB), 1.0);
        sharedCurrentWeight += branchWeightB * currentWeightB;
    } else {
        sharedCurrentWeight += branchWeightB;
    }
    sumSquaredWeights += sharedCurrentWeight * sharedCurrentWeight;
    return clamp(1.0 / max(sumSquaredWeights, 1e-12), 1.0,
        max(maximumSamples, 1.0));
}

#endif // MAXENT_TEMPORAL_STATISTICS_GLSL
