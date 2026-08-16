#ifndef MAXENT_SPATIAL_COMMON_GLSL
#define MAXENT_SPATIAL_COMMON_GLSL

#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_signal.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_pdf_direction.glsl"

// Geometry-side contract. PDF direction is the sole spatial-domain geometry
// descriptor; material IDs do not participate in A-trous filtering.
struct DenoiserSpatialGeometry {
    vec3 pdfDirection;
    float roughness;
    float surfaceDistance;
    float virtualScale;
    float ggxAlpha;
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
    float hitDistance;
    float hitWeight;
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

float denoiserSpatialLightSourceToleranceScale(float roughness) {
    return 1.0; //0.5 + 0.5 * sqrt(clamp(roughness, 0.0, 1.0));
}

bool denoiserSpatialSkyHit(float hitDistance) {
    float storedSkyDistance = min(max(VPROJDIST_SKY, 0.0),
            DENOISER_SPATIAL_FP16_MAX);
    return hitDistance >= 0.99 * storedSkyDistance;
}

float denoiserSpatialVirtualHitExponent(
    DenoiserSpatialGeometry centerGeometry, float centerHitDistance,
    DenoiserSpatialGeometry sampleGeometry, float sampleHitDistance) {
    // This term belongs to the light-field signal only. Rough GGX proposals
    // make virtual position increasingly irrelevant.
    float constraint = exp2(-200.0 * max(centerGeometry.ggxAlpha,
                    sampleGeometry.ggxAlpha));
    if (constraint <= 0.0) return 0.0;

    bool centerSky = denoiserSpatialSkyHit(centerHitDistance);
    bool sampleSky = denoiserSpatialSkyHit(sampleHitDistance);
    if (centerSky && sampleSky) return 0.0;
    if (centerSky != sampleSky)
        return max(MAXENT_SPATIAL_SPECULAR_VIRTUAL_POSITION_SENSITIVITY, 0.0)
            * constraint;

    float centerExtension = centerGeometry.virtualScale
            * centerHitDistance;
    float sampleExtension = sampleGeometry.virtualScale
            * sampleHitDistance;
    // Primary distance only sets the scale of the comparison. Its difference
    // is deliberately absent: PDF direction remains the surface-domain test.
    float normalization = max(max(centerGeometry.surfaceDistance,
                sampleGeometry.surfaceDistance) + max(centerHitDistance,
                    sampleHitDistance), 1e-5);
    float relativeVirtualError = abs(centerExtension - sampleExtension)
            / normalization;
    return max(MAXENT_SPATIAL_SPECULAR_VIRTUAL_POSITION_SENSITIVITY, 0.0)
        * constraint * relativeVirtualError;
}

// exp(-E_alpha) = alpha. This is the only non-geometric control applied to
// hit-distance filtering: alpha=0 preserves the center, alpha=1 applies the
// full spatial kernel. The clamp turns an exact delta into a negligible tap.
float denoiserSpatialHitDistanceAlphaExponent(float ggxAlpha) {
    return -log(max(clamp(ggxAlpha, 0.0, 1.0), 1e-8));
}

float denoiserSpatialWeight(DenoiserMaxEntSignal centerSignal,
    DenoiserSpatialBuresData centerBures,
    DenoiserSpatialGeometry centerGeometry,
    DenoiserMaxEntSignal sampleSignal,
    DenoiserSpatialBuresData sampleBures,
    DenoiserSpatialGeometry sampleGeometry, float kernelWeight,
    float lightSourceToleranceScale, float phiLuminance,
    float hitDistanceAlphaExponent,
    out float hitDistanceWeight) {
    float geometryExponent = denoiserSpatialPdfDirectionExponent(
            centerGeometry.pdfDirection, sampleGeometry.pdfDirection);
    float hitDistanceExponent = geometryExponent
            + hitDistanceAlphaExponent;
    hitDistanceWeight = kernelWeight * exp(-hitDistanceExponent);
    float signalExponent = geometryExponent
            + denoiserSpatialVirtualHitExponent(centerGeometry,
                centerSignal.hitDistance, sampleGeometry,
                sampleSignal.hitDistance);
    float distanceSq = denoiserSpatialBuresDistanceSq(
            centerSignal.maxEntY, centerBures,
            sampleSignal.maxEntY, sampleBures);
    float variance = centerSignal.variance + sampleSignal.variance;
    signalExponent += phiLuminance * distanceSq
            / max(lightSourceToleranceScale * variance, 1e-12);
    return kernelWeight * exp(-signalExponent);
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
    accum.hitDistance = center.hitDistance;
    accum.hitWeight = 1.0;
    return accum;
}

void denoiserSpatialAccumulate(inout DenoiserSpatialAccumulator accum,
    DenoiserMaxEntSignal neighbor, float weight,
    float hitDistanceWeight) {
    if (weight > 1e-6) {
        accum.maxEntY += neighbor.maxEntY * weight;
        accum.CoCg += neighbor.CoCg * weight;
        float weightedVariance = weight * neighbor.variance;
        accum.varianceEnergy += vec2(weightedVariance,
                weight * weightedVariance);
        accum.weight += weight;
    }
    if (hitDistanceWeight > 1e-6) {
        accum.hitDistance += neighbor.hitDistance * hitDistanceWeight;
        accum.hitWeight += hitDistanceWeight;
    }
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
    outputSignal.hitDistance = accum.hitDistance
            / max(accum.hitWeight, 1e-6);
    return denoiserSanitizeMaxEntSignal(outputSignal);
}

#endif // MAXENT_SPATIAL_COMMON_GLSL
