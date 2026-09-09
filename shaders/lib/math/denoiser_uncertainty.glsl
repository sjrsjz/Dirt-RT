#ifndef DENOISER_UNCERTAINTY_GLSL
#define DENOISER_UNCERTAINTY_GLSL

// Existing FP16 sigma lane: -1 = invalid light, -2 = valid light / unknown
// uncertainty. Never square either sentinel. No additional stored fields.
const float DENOISER_UNKNOWN_UNCERTAINTY = -2.0;
bool denoiserTemporalMomentsFinite(vec4 moment, vec2 chroma, float rms) {
    return !any(isnan(moment)) && !any(isinf(moment))
        && !any(isnan(chroma)) && !any(isinf(chroma))
        && moment.w >= 0.0 && rms >= 0.0 && !isnan(rms) && !isinf(rms);
}
bool denoiserSigmaKnown(float sigma) {
    // Ordered bounds also reject both infinities and NaN.
    return sigma >= 0.0 && sigma <= 65504.0;
}
bool denoiserSigmaUsable(float sigma) {
    return denoiserSigmaKnown(sigma) || sigma == DENOISER_UNKNOWN_UNCERTAINTY;
}
bool denoiserVarianceKnown(float variance) {
    return variance >= 0.0 && variance <= 65504.0 * 65504.0;
}
float denoiserSigmaOrUnknown(float sigma) {
    return denoiserSigmaKnown(sigma) ? sigma : DENOISER_UNKNOWN_UNCERTAINTY;
}
float denoiserVarianceToSigma(float variance) {
    return denoiserVarianceKnown(variance) ? sqrt(variance) : DENOISER_UNKNOWN_UNCERTAINTY;
}
// Variance of a weighted estimator under a constant-correlation closure.
// Accumulators are invocation-local registers, not additional stored channels.
// independentVariance/weightedSigma are needed even with all sigmas known.
// knownWeight/unknownSquaredWeight support the missing-statistics donor;
// totalWeight is also the normalization of the filtered signal.
struct DenoiserEstimatorVarianceAccumulator {
    float independentVariance;
    float weightedSigma;
    float knownWeight;
    float totalWeight;
    float unknownSquaredWeight;
};
DenoiserEstimatorVarianceAccumulator denoiserBeginEstimatorVariance() {
    return DenoiserEstimatorVarianceAccumulator(0.0, 0.0, 0.0, 0.0, 0.0);
}
void denoiserAccumulateEstimatorVariance(inout DenoiserEstimatorVarianceAccumulator a,
        float sigma, float weight) {
    if (!(weight > 0.0) || isnan(weight) || isinf(weight)) return;
    a.totalWeight += weight;
    if (denoiserSigmaKnown(sigma)) {
        float weightedSigma = weight * sigma;
        a.independentVariance += weightedSigma * weightedSigma;
        a.weightedSigma += weightedSigma;
        a.knownWeight += weight;
    } else {
        a.unknownSquaredWeight += weight * weight;
    }
}
float denoiserResolveEstimatorSigma(DenoiserEstimatorVarianceAccumulator a, float correlation) {
    if (!(a.knownWeight > 0.0) || !(a.totalWeight > 0.0))
        return DENOISER_UNKNOWN_UNCERTAINTY;
    // Missing statistics borrow the local known-sigma average. Include their
    // actual signal weights: dropping them from the normalization would describe
    // a different estimator. This is a local-stationarity fallback, not a bound.
    float donorSigma = a.weightedSigma / a.knownWeight;
    float independentVariance = a.independentVariance
        + a.unknownSquaredWeight * donorSigma * donorSigma;
    float weightedSigma = a.weightedSigma
        + max(a.totalWeight - a.knownWeight, 0.0) * donorSigma;
    float inverseWeight = 1.0 / a.totalWeight;
    float variance = mix(independentVariance, weightedSigma * weightedSigma,
        clamp(correlation, 0.0, 1.0)) * inverseWeight * inverseWeight;
    // Known endpoint sigma and the unknown-sigma donor are bounded by
    // 65504. Their normalized constant-correlation estimate has the same
    // bound. Correct a small finite FP32 overshoot here only: the generic
    // variance ingress remains strict, and NaN/Inf/larger violations fail.
    const float maximumVariance = 65504.0 * 65504.0;
    return variance >= 0.0
            && variance <= maximumVariance * (1.0 + 0.00001)
        ? sqrt(min(variance, maximumVariance))
        : DENOISER_UNKNOWN_UNCERTAINTY;
}
float denoiserMixEstimatorSigma(float a, float b, float t, float correlation) {
    DenoiserEstimatorVarianceAccumulator accum = denoiserBeginEstimatorVariance();
    t = clamp(t, 0.0, 1.0);
    denoiserAccumulateEstimatorVariance(accum, a, 1.0-t);
    denoiserAccumulateEstimatorVariance(accum, b, t);
    return denoiserResolveEstimatorSigma(accum, correlation);
}
#endif
