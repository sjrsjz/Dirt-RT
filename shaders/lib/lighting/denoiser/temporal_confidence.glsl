#ifndef MAXENT_TEMPORAL_CONFIDENCE_GLSL
#define MAXENT_TEMPORAL_CONFIDENCE_GLSL

#include "/lib/lighting/denoiser/virtual_projection.glsl"
#include "/lib/math/statistics.glsl"

// Experimental split-sample resolve. All quantities here use Euclidean
// linear-moment trace variance, in radiance^2. Spatial A-Trous keeps Bures
// observation variance. Only the persistent filtered-history slot changes:
// sigma = sqrt(estimator trace variance), N = 1. No buffer is added.
// The domain adapter reads immutable CURRENT RAW light, never resolved history.
void maxentConfidenceLoadRaw(ivec2 pixel, out vec4 moment, out vec2 chroma);

struct MaxentConfidenceGroup {
    vec4 moment;
    vec2 chroma;
    float observationVariance;
    float inverseSamples;
    bool valid;
};

MaxentConfidenceGroup maxentConfidenceFinishGroup(vec4 momentSum, vec2 chromaSum,
        float normSum, float weight, float weightSquared) {
    MaxentConfidenceGroup g;
    g.valid = weight > 1e-8;
    float invWeight = g.valid ? 1.0 / weight : 0.0;
    g.moment = momentSum * invWeight;
    g.chroma = chromaSum * invWeight;
    g.inverseSamples = clamp(weightSquared * invWeight * invWeight, 0.0, 1.0);
    float central = max(normSum * invWeight - dot(g.moment, g.moment), 0.0);
    g.observationVariance = g.inverseSamples < 0.999
        ? central / (1.0 - g.inverseSamples) : 0.0;
    return g;
}

uvec4 maxentConfidenceGeometryWords(uvec2 pixel) {
#ifdef REFLECT_BUFFER
    return readPrimaryGeometryWords(pixel);
#else
    return readDiffuseGeometryWords(pixel);
#endif
}

void maxentConfidenceGather(ivec2 pixel, out MaxentConfidenceGroup pilot,
        out MaxentConfidenceGroup checkA, out MaxentConfidenceGroup checkB,
        out MaxentConfidenceGroup current) {
    uvec4 centerWords = maxentConfidenceGeometryWords(uvec2(pixel));
    vec3 normal = decodeNormalU(centerWords.x);
    float distance = uintBitsToFloat(centerWords.w);
    float planeOffset = distance * dot(normal, reconstructPrimaryRay(uvec2(pixel)));
    float rejectionScale = denoiserSpatialDistanceRejectionScale(distance, float(resolution_global.y));
    vec4 momentSum[3] = vec4[3](vec4(0.0), vec4(0.0), vec4(0.0));
    vec2 chromaSum[3] = vec2[3](vec2(0.0), vec2(0.0), vec2(0.0));
    float normSum[3] = float[3](0.0, 0.0, 0.0);
    float weight[3] = float[3](0.0, 0.0, 0.0);
    float weightSquared[3] = float[3](0.0, 0.0, 0.0);
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 p = pixel + ivec2(x, y);
            if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, ivec2(resolution_global)))) continue;
            uvec4 words = maxentConfidenceGeometryWords(uvec2(p));
            float tapDistance = uintBitsToFloat(words.w);
            if (!(tapDistance >= 0.0) || isinf(tapDistance)) continue;
            vec3 tapNormal = decodeNormalU(words.x);
            if (dot(normal, tapNormal) < 0.95) continue;
#ifdef REFLECT_BUFFER
            // Same encoded material and nearly equal GGX alpha. Lighting
            // differences never enter these weights or the sample partition.
            if ((words.y >> 16u) != (centerWords.y >> 16u)) continue;
            if (abs(unpackHalf2x16(words.y).x - unpackHalf2x16(centerWords.y).x) > 0.02) continue;
#endif
            float exponent = 0.5 * float(x * x + y * y)
                + denoiserSpatialAxialDistanceExponent(planeOffset, normal,
                    reconstructPrimaryRay(uvec2(p)), tapDistance, rejectionScale);
            float a = exp(-exponent);
            if (!(a > 1e-8)) continue;
            vec4 m;
            vec2 chroma;
            maxentConfidenceLoadRaw(p, m, chroma);
            if (any(isnan(m)) || any(isinf(m)) || any(isnan(chroma))
                    || any(isinf(chroma)) || m.w < 0.0) continue;
            // Groups 0/1 partition the pilot; group 2 contains the center.
            // No raw sample contributes to both groups for this output pixel.
            bool pilotTap = ((abs(x) + abs(y)) & 1) == 1;
            int group = pilotTap ? ((x < 0 || (x == 0 && y < 0)) ? 0 : 1) : 2;
            momentSum[group] += a * m;
            chromaSum[group] += a * chroma;
            normSum[group] += a * dot(m, m);
            weight[group] += a;
            weightSquared[group] += a * a;
        }
    }
    checkA = maxentConfidenceFinishGroup(momentSum[0], chromaSum[0], normSum[0], weight[0], weightSquared[0]);
    checkB = maxentConfidenceFinishGroup(momentSum[1], chromaSum[1], normSum[1], weight[1], weightSquared[1]);
    pilot = maxentConfidenceFinishGroup(momentSum[0]+momentSum[1], chromaSum[0]+chromaSum[1],
        normSum[0]+normSum[1], weight[0]+weight[1], weightSquared[0]+weightSquared[1]);
    current = maxentConfidenceFinishGroup(momentSum[2], chromaSum[2], normSum[2], weight[2], weightSquared[2]);
    pilot.valid = pilot.valid && pilot.inverseSamples < 0.999;
}

float maxentConfidencePriorObservationVariance(vec4 moment, float rms, float samples) {
    // Exact for X=(R*u,R), |u|=1; an upper second-norm closure for an
    // already combined realizable directional packet. Fixed raw EMA weights
    // keep the pilot/response selection out of these population moments.
    float w = max(moment.w, 0.0);
    rms = max(rms, w);
    float radial = (rms - w) * (rms + w);
    float angular = max(rms * rms - dot(moment.xyz, moment.xyz), 0.0);
    return statisticsObservationVarianceFromBiasedCentralMoment(radial + angular, samples);
}

void maxentConfidenceResolve(MaxentConfidenceGroup pilot, MaxentConfidenceGroup checkA,
        MaxentConfidenceGroup checkB, MaxentConfidenceGroup current,
        vec4 history, vec2 historyChroma, float historyVariance, bool hasHistory,
        float priorObservationVariance, float temporalHistorySamples, float reprojectionFloor,
        out vec4 result, out vec2 resultChroma, out float resultVariance, out float gain) {
    // Pilot-only plug-in noise scale: gain never uses current-group brightness
    // or its sample variance. Missing tails are supplied by the fixed raw EMA.
    float observationVariance = max(priorObservationVariance,
        pilot.valid ? pilot.observationVariance : 0.0);
    float scale = hasHistory ? dot(history, history) : 0.0;
    if (pilot.valid) scale = max(scale, dot(pilot.moment, pilot.moment));
    float varianceFloor = max(1e-12, 1e-8 * scale);
    float currentVariance = max(observationVariance * current.inverseSamples, varianceFloor);
    if (!hasHistory) {
        result = current.moment;
        resultChroma = current.chroma;
        resultVariance = currentVariance;
        gain = 1.0;
        return;
    }
    historyVariance = max(historyVariance, 0.0);
    float beta = 0.0;
    float pilotVariance = 0.0;
    float distanceSquared = 0.0;
    if (pilot.valid && MAXENT_TEMPORAL_CONFIDENCE_FAILURE_PROBABILITY > 0.0) {
        pilotVariance = max(observationVariance * pilot.inverseSamples, varianceFloor);
        vec4 difference = pilot.moment - history;
        distanceSquared = dot(difference, difference);
        float radiusSquared = (historyVariance + pilotVariance)
            / max(float(MAXENT_TEMPORAL_CONFIDENCE_FAILURE_PROBABILITY), 1e-6);
        // Require both disjoint pilot halves to support the same change.
        // The pooled test alone can select rare bright pilot samples, even
        // with oracle variance. This guard is also independent of current.
        vec4 da = checkA.moment - history;
        vec4 db = checkB.moment - history;
        float ra2 = (historyVariance + max(observationVariance * checkA.inverseSamples, varianceFloor))
            / max(float(MAXENT_TEMPORAL_CONFIDENCE_FAILURE_PROBABILITY), 1e-6);
        float rb2 = (historyVariance + max(observationVariance * checkB.inverseSamples, varianceFloor))
            / max(float(MAXENT_TEMPORAL_CONFIDENCE_FAILURE_PROBABILITY), 1e-6);
        bool corroborated = checkA.valid && checkB.valid && dot(da, db) > 0.0
            && dot(da, da) > ra2 && dot(db, db) > rb2;
        if (corroborated && distanceSquared > radiusSquared)
            beta = 1.0 - sqrt(radiusSquared / distanceSquared);
    }
    vec4 clippedHistory = mix(history, pilot.valid ? pilot.moment : history, beta);
    vec2 clippedChroma = mix(historyChroma, pilot.valid ? pilot.chroma : historyChroma, beta);
    // Frozen-coefficient correlation envelope plus a between-hypothesis term.
    // This is an uncertainty proxy after adaptive projection, not exact Kish
    // variance or a proof of unconditional confidence coverage.
    float clippedSigma = (1.0 - beta) * sqrt(historyVariance) + beta * sqrt(pilotVariance);
    float clippedVariance = clippedSigma * clippedSigma
        + beta * (1.0 - beta) * distanceSquared;
    // Constrained minimum-variance scalar gain. The leverage cap prevents an
    // underestimated current variance from assigning an unseen tail sample a
    // large weight. Only independent pilot evidence relaxes this cap.
    // This is an explicit bounded-leverage policy, not a coverage theorem.
    float minimumGain = float(MAXENT_TEMPORAL_FIXED_ALPHA);
    float stationaryCap = max(minimumGain, 1.0 / (1.0 + max(temporalHistorySamples, 1.0)));
    float maximumGain = mix(stationaryCap, 1.0, beta);
    gain = max(clamp(reprojectionFloor, 0.0, 1.0),
        clamp(statisticsMinimumVarianceIndependentCurrentWeight(clippedVariance, currentVariance),
            minimumGain, maximumGain));
    result = mix(clippedHistory, current.moment, gain);
    resultChroma = mix(clippedChroma, current.chroma, gain);
    resultVariance = (1.0 - gain) * (1.0 - gain) * clippedVariance
        + gain * gain * currentVariance;
}

#endif
