#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex5;
uniform usampler2D colortex6;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    MaxEntGeometry geometry = maxentLoadGeometry(pixel);
    if (!geometry.valid) {
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTMETA, pixel)] = uvec4(0u);
        return;
    }

    MaxEntSlowSignal slow = maxentUnpackSlow(
        texelFetch(colortex4, ivec2(pixel), 0));
    MaxEntFastSignal fast = maxentUnpackFast(
        texelFetch(colortex5, ivec2(pixel), 0));
    // These are the Kish counts of the linear temporal estimators produced by
    // the preceding pass.  Everything below is a deterministic, correlated
    // estimator-space correction; it does not add an independent observation.
    // Preserve the temporal counts instead of feeding clamp/anti-lag amounts
    // into the independent-current-sample ESS recurrence.
    float slowTemporalSamples = slow.historyLength;
    float fastTemporalSamples = fast.historyLength;
    MaxEntSpecularInput noisyCenter = maxentUnpackSpecularInput(
        texelFetch(colortex6, ivec2(pixel), 0));

    vec3 fastM1 = vec3(0.0);
    vec3 fastM2 = vec3(0.0);
    vec3 noisyM1 = vec3(0.0);
    vec4 noisyMaxEntY = vec4(0.0);
    vec2 noisyCoCg = vec2(0.0);
    float noisyY2 = 0.0;
    float sampleCount = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 q = ivec2(pixel) + ivec2(x, y);
            if (!maxentInBounds(q, size)) continue;
            MaxEntFastSignal qFast = maxentUnpackFast(
                texelFetch(colortex5, q, 0));
            uvec4 qGeometry = readPrimaryGeometryWords(uvec2(q));
            if (uintBitsToFloat(qGeometry.w) < 0.0
                    || (qGeometry.y >> 16u) != geometry.materialID)
                continue;
            MaxEntSpecularInput qNoisy = maxentUnpackSpecularInput(
                texelFetch(colortex6, q, 0));
            vec3 qNoisyYCoCg = maxentMaxEntYCoCg(qNoisy.signal);
            vec3 qFastYCoCg = maxentMaxEntYCoCg(qFast.signal);
            fastM1 += qFastYCoCg;
            fastM2 += qFastYCoCg * qFastYCoCg;
            noisyM1 += qNoisyYCoCg;
            noisyMaxEntY += qNoisy.signal.maxEntY;
            noisyCoCg += qNoisy.signal.CoCg;
            noisyY2 += qNoisyYCoCg.x * qNoisyYCoCg.x;
            sampleCount += 1.0;
        }
    }

    vec3 slowYCoCg = maxentMaxEntYCoCg(slow.signal);
    float oldSlowY = slowYCoCg.x;
    if (sampleCount > 0.0) {
        float invCount = 1.0 / sampleCount;
        fastM1 *= invCount;
        fastM2 *= invCount;
        noisyM1 *= invCount;
        noisyMaxEntY *= invCount;
        noisyCoCg *= invCount;
        noisyY2 *= invCount;
        vec3 sigma = sqrt(max(fastM2 - fastM1 * fastM1, vec3(0.0)));
        vec3 boxMin = fastM1 - MAXENT_SPECULAR_TEMPORAL_CLAMP_SIGMA * sigma;
        vec3 boxMax = fastM1 + MAXENT_SPECULAR_TEMPORAL_CLAMP_SIGMA * sigma;
        vec3 fastYCoCg = maxentMaxEntYCoCg(fast.signal);
        boxMin = min(boxMin, fastYCoCg);
        boxMax = max(boxMax, fastYCoCg);

        vec3 clampedYCoCg = slowYCoCg;
        if (MAXENT_SPECULAR_TEMPORAL_MAX_FAST_HISTORY < MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY)
            clampedYCoCg = clamp(slowYCoCg, boxMin, boxMax);
        if (slow.historyLength <= MAXENT_SPECULAR_TEMPORAL_HISTORY_FIX_THRESHOLD)
            clampedYCoCg = fastYCoCg;

        float clampDenominator = fastYCoCg.x - slowYCoCg.x;
        float clampingFactor = abs(clampedYCoCg.x - slowYCoCg.x) > 1e-8
                && abs(clampDenominator) > 1e-8
            ? clamp((clampedYCoCg.x - slowYCoCg.x) /
                clampDenominator, 0.0, 1.0) : 0.0;
        if (slow.historyLength <= MAXENT_SPECULAR_TEMPORAL_HISTORY_FIX_THRESHOLD)
            clampingFactor = 1.0;

        slow.signal = maxentMixMaxEnt(slow.signal, fast.signal,
            clampingFactor);
        slow.signal = maxentSetMaxEntYCoCg(slow.signal, clampedYCoCg);

        SpecularMaxEnt noisyMean;
        noisyMean.maxEntY = noisyMaxEntY;
        noisyMean.CoCg = noisyCoCg;
        noisyMean = sanitizeSpecularMaxEnt(noisyMean);
        float historyDifference = abs(fastYCoCg.x - slowYCoCg.x);
        float distanceToNoisy = abs(noisyM1.x - fastYCoCg.x);
        float accelerationDistance = 0.33 * 10.0
            * MAXENT_SPECULAR_TEMPORAL_ANTI_LAG * historyDifference
            * clampingFactor;
        float acceleration = distanceToNoisy > 1e-8
            ? min(accelerationDistance / distanceToNoisy, 1.0) : 0.0;
        if (slow.historyLength <= MAXENT_SPECULAR_TEMPORAL_HISTORY_FIX_THRESHOLD)
            acceleration = 0.0;
        // NRD adds the same responsive-to-noisy correction to both histories.
        // Apply that correction to every MaxEnt component, including direction.
        vec4 accelerationMaxEntY =
            (noisyMean.maxEntY - fast.signal.maxEntY) * acceleration;
        vec2 accelerationCoCg =
            (noisyMean.CoCg - fast.signal.CoCg) * acceleration;
        slow.signal.maxEntY += accelerationMaxEntY;
        slow.signal.CoCg += accelerationCoCg;
        fast.signal.maxEntY += accelerationMaxEntY;
        fast.signal.CoCg += accelerationCoCg;
        slow.signal = sanitizeSpecularMaxEnt(slow.signal);
        fast.signal = sanitizeSpecularMaxEnt(fast.signal);

        float temporalSigma = MAXENT_SPECULAR_TEMPORAL_RESET_TEMPORAL_SIGMA *
            sqrt(max(noisyY2 - noisyM1.x * noisyM1.x, 0.0));
        float spatialSigma = MAXENT_SPECULAR_TEMPORAL_RESET_SPATIAL_SIGMA * sigma.x;
        float reset = 0.5 * MAXENT_SPECULAR_TEMPORAL_RESET_AMOUNT * max(0.0,
            abs(oldSlowY - noisyM1.x) - spatialSigma - temporalSigma) /
            max(max(oldSlowY, noisyM1.x) + spatialSigma + temporalSigma,
                1e-6);
        reset = clamp(reset, 0.0, 1.0);
        slow.signal = maxentMixMaxEnt(slow.signal, noisyCenter.signal, reset);
        fast.signal = maxentMixMaxEnt(fast.signal, noisyCenter.signal, reset);
        slowYCoCg = maxentMaxEntYCoCg(slow.signal);
    }

    slow.signal = sanitizeSpecularMaxEnt(slow.signal);
    slow.secondMoment = max(slow.secondMoment +
        slowYCoCg.x * slowYCoCg.x - oldSlowY * oldSlowY, 0.0);
    slow.historyLength = slowTemporalSamples;
    fast.historyLength = fastTemporalSamples;

    MaxEntSpecularHistory history;
    history.surfacePosition = geometry.position;
    history.geometryNormal = geometry.normal;
    history.slowSignal = slow.signal;
    history.secondMoment = slow.secondMoment;
    history.responsiveSignal = fast.signal;
    history.hitDistance = fast.hitDistance;
    history.roughness = geometry.roughness;
    history.historyLength = slow.historyLength;
    history.responsiveHistoryLength = fast.historyLength;
    history.materialID = geometry.materialID;
    writeMaxEntSpecularHistory(pixel, history);
}
