#ifndef MAXENT_BURES_GLSL
#define MAXENT_BURES_GLSL

struct DenoiserSpatialBuresData {
    vec2 stddev;
    float trace;
};

DenoiserSpatialBuresData denoiserSpatialMakeBuresData(vec4 maxEntY) {
    DenoiserSpatialBuresData data;
    float parallelExcess = 0.5 * dot(maxEntY.xyz, maxEntY.xyz);
    float energy = maxEntY.w;
    float energy2 = energy * energy;
    float traceRoot = sqrt(max(4.0 * energy2 - 1.5 * parallelExcess, 0.0));
    float analyticTrace = (2.0 * energy2 + energy * traceRoot) * (1.0 / 3.0) - parallelExcess;
    float perpendicularVariance = max((analyticTrace - parallelExcess) * (1.0 / 3.0), 0.0);
    data.trace = 3.0 * perpendicularVariance + parallelExcess;
    data.stddev = sqrt(vec2(perpendicularVariance, perpendicularVariance + parallelExcess));
    return data;
}

DenoiserSpatialBuresData denoiserSpatialMakeBuresDataFromStddev(vec2 stddev) {
    DenoiserSpatialBuresData data;
    data.stddev = stddev;
    vec2 axisVariance = stddev * stddev;
    data.trace = 2.0 * axisVariance.x + axisVariance.y;
    return data;
}

float denoiserSpatialBuresDistanceSq(vec4 centerMaxEntY, DenoiserSpatialBuresData centerData,
    vec4 sampleMaxEntY, DenoiserSpatialBuresData sampleData) {
    float meanDot = dot(centerMaxEntY.xyz, sampleMaxEntY.xyz);
    float crossAxes = centerData.stddev.x * sampleData.stddev.y + centerData.stddev.y * sampleData.stddev.x;
    float cross2d = sqrt(crossAxes * crossAxes + 0.25 * meanDot * meanDot);
    float crossTrace = centerData.stddev.x * sampleData.stddev.x + cross2d;
    vec3 meanDelta = centerMaxEntY.xyz - sampleMaxEntY.xyz;
    return max(dot(meanDelta, meanDelta) + centerData.trace + sampleData.trace - 2.0 * crossTrace, 0.0);
}

float maxentClampHistoryWeightByDenoisedDifference(float historyWeight,
    vec4 currentMaxEntY, vec4 historyMaxEntY, float historyStddev,
    float historySamples, float validWeight, float temporalCurrentWeight,
    float maximumHistory, float tolerance, out float normalizedDistance) {
    normalizedDistance = -1.0;
    historyWeight = isnan(historyWeight) || isinf(historyWeight) ? 1.0 : max(historyWeight, 1.0);
    if (historyWeight <= MAXENT_TEMPORAL_DIFFERENCE_COLD_START_HISTORY) return historyWeight;
    if (any(isnan(currentMaxEntY)) || any(isinf(currentMaxEntY))
            || any(isnan(historyMaxEntY)) || any(isinf(historyMaxEntY))
            || !(historyStddev >= 0.0) || isnan(historyStddev) || isinf(historyStddev)
            || !(historySamples >= 1.0) || isnan(historySamples) || isinf(historySamples)
            || !(validWeight > 0.0) || isnan(validWeight) || isinf(validWeight)
            || !(temporalCurrentWeight > 0.0) || isnan(temporalCurrentWeight)
            || isinf(temporalCurrentWeight)) return historyWeight;

    DenoiserSpatialBuresData currentBures = denoiserSpatialMakeBuresData(currentMaxEntY);
    DenoiserSpatialBuresData historyBures = denoiserSpatialMakeBuresData(historyMaxEntY);
    float distanceSq = denoiserSpatialBuresDistanceSq(currentMaxEntY, currentBures, historyMaxEntY, historyBures);
    // T = (1-alpha)H + alpha*X, so D(T,H)/alpha estimates X-H. The stored
    // history variance is Var(H)=Var(X)/N_eff, hence
    // Var(X-H)=(N_eff+1)*Var(H). T and H are correlated and must not be added
    // as independent estimators.
    float innovationStddev = historyStddev * sqrt(historySamples + 1.0);
    normalizedDistance = sqrt(clamp(validWeight, 0.0, 1.0) * distanceSq)
            / max(temporalCurrentWeight * innovationStddev, 1e-6);
    if (isnan(normalizedDistance) || isinf(normalizedDistance)) {
        normalizedDistance = -1.0;
        return historyWeight;
    }

    float k = min(normalizedDistance, 80.0) / max(tolerance, 1e-6);
    float agreement = exp(-k);
    float historyCap = 1.0 + (max(maximumHistory, 1.0) - 1.0) * agreement;
    return min(historyWeight, historyCap);
}

#endif // MAXENT_BURES_GLSL
