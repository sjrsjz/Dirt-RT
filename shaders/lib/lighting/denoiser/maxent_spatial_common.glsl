#ifndef MAXENT_SPATIAL_COMMON_GLSL
#define MAXENT_SPATIAL_COMMON_GLSL

#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"

// Geometry-side contract. Signal-specific wrappers own decoding and decide
// whether two samples are compatible (for example, specular rejects material
// ID discontinuities while diffuse deliberately does not).
struct DenoiserSpatialGeometry {
    vec3 position;
    vec3 normal;
    vec3 textureNormal;
    float roughness;
    uint materialID;
    bool valid;
};

struct DenoiserSpatialBuresData {
    vec2 stddev;
    float trace;
    float anisotropy;
    float invMeanLength2;
};

struct DenoiserSpatialAccumulator {
    vec4 maxEntY;
    vec2 CoCg;
    vec2 varianceEnergy;
    float weight;
};

DenoiserSpatialBuresData denoiserSpatialMakeBuresData(vec4 maxEntY) {
    DenoiserSpatialBuresData data;
    float meanLength2 = dot(maxEntY.xyz, maxEntY.xyz);
    float rho = sqrt(meanLength2) / max(maxEntY.w, 1e-8);
    rho = clamp(rho, 0.0, 1.0 - 1e-6);
    float kappa = 3.0 * rho /
        (2.0 + sqrt(max(4.0 - 3.0 * rho * rho, 1e-12)));
    data.stddev = maxent_eigen_std(max(maxEntY.w, 0.0), kappa);
    vec2 axisVariance = data.stddev * data.stddev;
    data.trace = 2.0 * axisVariance.x + axisVariance.y;
    data.anisotropy = axisVariance.y - axisVariance.x;
    data.invMeanLength2 = meanLength2 > 1e-16
        ? 1.0 / meanLength2 : 0.0;
    return data;
}

DenoiserSpatialBuresData denoiserSpatialMakeBuresDataFromStddev(
        vec4 maxEntY, vec2 stddev) {
    DenoiserSpatialBuresData data;
    data.stddev = stddev;
    vec2 axisVariance = stddev * stddev;
    data.trace = 2.0 * axisVariance.x + axisVariance.y;
    data.anisotropy = axisVariance.y - axisVariance.x;
    float meanLength2 = dot(maxEntY.xyz, maxEntY.xyz);
    data.invMeanLength2 = meanLength2 > 1e-16
        ? 1.0 / meanLength2 : 0.0;
    return data;
}

float denoiserSpatialBuresDistanceSq(vec4 centerMaxEntY,
        DenoiserSpatialBuresData centerData, vec4 sampleMaxEntY,
        DenoiserSpatialBuresData sampleData) {
    float directionCosine2 = 0.0;
    if (centerData.invMeanLength2 > 0.0
            && sampleData.invMeanLength2 > 0.0) {
        float meanDot = dot(centerMaxEntY.xyz, sampleMaxEntY.xyz);
        directionCosine2 = min(1.0, meanDot * meanDot
            * centerData.invMeanLength2 * sampleData.invMeanLength2);
    }

    float crossAxes = centerData.stddev.x * sampleData.stddev.y
        + centerData.stddev.y * sampleData.stddev.x;
    float cross2d = sqrt(max(crossAxes * crossAxes
        + directionCosine2 * centerData.anisotropy
            * sampleData.anisotropy, 0.0));
    float crossTrace = centerData.stddev.x * sampleData.stddev.x
        + cross2d;
    vec3 meanDelta = centerMaxEntY.xyz - sampleMaxEntY.xyz;
    return max(dot(meanDelta, meanDelta) + centerData.trace
        + sampleData.trace - 2.0 * crossTrace, 0.0);
}

float denoiserSpatialPlaneExponent(DenoiserSpatialGeometry center,
        DenoiserSpatialGeometry neighbor, float resolutionY) {
    resolutionY = max(resolutionY, 1.0);
    float centerDistance = max(length(center.position), 0.001);
    float footprintDistance = max(centerDistance, resolutionY * 1e-5);
    float invPixelFootprint = resolutionY / max(
        MAXENT_SPATIAL_PLANE_DISTANCE_TOLERANCE * footprintDistance, resolutionY * 1e-6);
    return abs(dot(neighbor.position, center.normal)
        - dot(center.position, center.normal)) * invPixelFootprint;
}

float denoiserSpatialTextureNormalExponent(
        DenoiserSpatialGeometry center,
        DenoiserSpatialGeometry neighbor) {
    float normalCosine = clamp(dot(center.textureNormal,
        neighbor.textureNormal), -1.0, 1.0);
    // This exponent is folded into the geometry/light exponent below, so a
    // tap evaluates only one exp(). Unit roughness disables this term.
    return max(MAXENT_SPATIAL_NORMAL_SENSITIVITY, 0.0)
        * (1.0 - clamp(center.roughness, 0.0, 1.0))
        * (1.0 - normalCosine);
}

float denoiserSpatialLightSourceToleranceScale(float roughness) {
    return sqrt(clamp(roughness, 0.0, 1.0));
}

float denoiserSpatialWeight(DenoiserMaxEntSignal centerSignal,
        DenoiserSpatialBuresData centerBures,
        DenoiserSpatialGeometry centerGeometry,
        DenoiserMaxEntSignal sampleSignal,
        DenoiserSpatialBuresData sampleBures,
        DenoiserSpatialGeometry sampleGeometry, float kernelWeight,
        float lightSourceToleranceScale, float phiLuminance,
        float resolutionY) {
    float exponent = denoiserSpatialPlaneExponent(centerGeometry,
        sampleGeometry, resolutionY);
    exponent += denoiserSpatialTextureNormalExponent(centerGeometry,
        sampleGeometry);
    float distanceSq = denoiserSpatialBuresDistanceSq(
        centerSignal.maxEntY, centerBures,
        sampleSignal.maxEntY, sampleBures);
    float variance = centerSignal.variance + sampleSignal.variance;
    exponent += phiLuminance * distanceSq
        / max(lightSourceToleranceScale * variance, 1e-12);
    return kernelWeight * exp(-exponent);
}

float denoiserSpatialVariancePower(int stepRadius) {
    float coefficient;
    if (stepRadius <= 1) coefficient = 0.3339015144;
    else if (stepRadius <= 2) coefficient = 0.4375201036;
    else if (stepRadius <= 4) coefficient = 0.4592464660;
    else if (stepRadius <= 8) coefficient = 0.4644479501;
    else if (stepRadius <= 16) coefficient = 0.4657344365;
    else coefficient = 0.4660551979;
    return max(1.0, 2.0 - MAXENT_SPATIAL_VARIANCE_ADAPTATION * coefficient);
}

DenoiserSpatialAccumulator denoiserSpatialBeginAccumulation(
        DenoiserMaxEntSignal center) {
    DenoiserSpatialAccumulator accum;
    accum.maxEntY = center.maxEntY;
    accum.CoCg = center.CoCg;
    float centerVariance = max(center.variance, 1e-16);
    accum.varianceEnergy = vec2(centerVariance);
    accum.weight = 1.0;
    return accum;
}

void denoiserSpatialAccumulate(inout DenoiserSpatialAccumulator accum,
        DenoiserMaxEntSignal neighbor, float weight) {
    accum.maxEntY += neighbor.maxEntY * weight;
    accum.CoCg += neighbor.CoCg * weight;
    float weightedVariance = weight * neighbor.variance;
    accum.varianceEnergy += vec2(weightedVariance,
        weight * weightedVariance);
    accum.weight += weight;
}

DenoiserMaxEntSignal denoiserSpatialResolve(
        DenoiserSpatialAccumulator accum, int stepRadius) {
    float invWeight = 1.0 / max(accum.weight, 1e-6);
    DenoiserMaxEntSignal outputSignal;
    outputSignal.maxEntY = accum.maxEntY * invWeight;
    outputSignal.CoCg = accum.CoCg * invWeight;
    float power = denoiserSpatialVariancePower(stepRadius);
    float varianceMix = 2.0 - exp2(2.0 - power);
    outputSignal.variance = mix(accum.varianceEnergy.x,
        accum.varianceEnergy.y, varianceMix) * pow(invWeight, power);
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_SPATIAL_COMMON_GLSL
