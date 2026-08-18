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

#endif // MAXENT_BURES_GLSL
