#ifndef MAXENT_SPATIAL_SIGNAL_GLSL
#define MAXENT_SPATIAL_SIGNAL_GLSL

// Canonical input/output ABI for the MaxEnt spatial denoiser.
//
// Image format: RGBA32UI
//   x = packHalf2x16(maxEntY.xy)
//   y = packHalf2x16(maxEntY.zw)
//   z = packHalf2x16(CoCg.xy)
//   w = packHalf2x16(sqrt(variance), hitDistance)
//
// maxEntY.xyz is the directional first moment, maxEntY.w is total luminance,
// and CoCg carries chroma. Variance is evaluated in F32 but stored as an FP16
// standard deviation and squared after load. The reflection hit distance is
// propagated independently in the upper FP16 lane. Input and output use the
// same layout, so every spatial pass may ping-pong the same RGBA32UI resources.
//
// A negative FP16 standard deviation is the invalid/no-surface sentinel.
// Geometry is supplied separately by the signal policy.

const float DENOISER_SPATIAL_FP16_MAX = 65504.0;
const float DENOISER_SPATIAL_VARIANCE_MAX =
    DENOISER_SPATIAL_FP16_MAX * DENOISER_SPATIAL_FP16_MAX;

struct DenoiserMaxEntSignal {
    vec4 maxEntY;
    vec2 CoCg;
    float variance;
    float hitDistance;
};

DenoiserMaxEntSignal denoiserEmptyMaxEntSignal() {
    DenoiserMaxEntSignal signal;
    signal.maxEntY = vec4(0.0);
    signal.CoCg = vec2(0.0);
    signal.variance = 0.0;
    signal.hitDistance = 0.0;
    return signal;
}

bool denoiserSpatialSignalWordsValid(uvec4 words) {
    float standardDeviation = unpackHalf2x16(words.w).x;
    return standardDeviation >= 0.0
        && !isnan(standardDeviation) && !isinf(standardDeviation);
}

uvec4 denoiserInvalidMaxEntSignalWords() {
    return uvec4(0u, 0u, 0u, packHalf2x16(vec2(-1.0, 0.0)));
}

DenoiserMaxEntSignal denoiserSanitizeMaxEntSignal(
        DenoiserMaxEntSignal signal) {
    if (any(isnan(signal.maxEntY)) || any(isinf(signal.maxEntY)))
        signal.maxEntY = vec4(0.0);
    if (any(isnan(signal.CoCg)) || any(isinf(signal.CoCg)))
        signal.CoCg = vec2(0.0);
    if (isnan(signal.variance) || isinf(signal.variance))
        signal.variance = 0.0;
    if (isnan(signal.hitDistance) || isinf(signal.hitDistance))
        signal.hitDistance = 0.0;

    signal.maxEntY.w = max(signal.maxEntY.w, 0.0);
    float meanLength2 = dot(signal.maxEntY.xyz, signal.maxEntY.xyz);
    float totalEnergy2 = signal.maxEntY.w * signal.maxEntY.w;
    if (meanLength2 > totalEnergy2 && meanLength2 > 0.0)
        signal.maxEntY.xyz *= signal.maxEntY.w * inversesqrt(meanLength2);

    if (signal.maxEntY.w <= 0.0) {
        signal.maxEntY = vec4(0.0);
        signal.CoCg = vec2(0.0);
    }
    signal.variance = clamp(signal.variance, 0.0,
        DENOISER_SPATIAL_VARIANCE_MAX);
    signal.hitDistance = clamp(signal.hitDistance, 0.0,
        DENOISER_SPATIAL_FP16_MAX);
    return signal;
}

DenoiserMaxEntSignal denoiserUnpackMaxEntSignal(uvec4 words) {
    DenoiserMaxEntSignal signal;
    signal.maxEntY = vec4(unpackHalf2x16(words.x),
        unpackHalf2x16(words.y));
    signal.CoCg = unpackHalf2x16(words.z);
    vec2 standardDeviationHitDistance = unpackHalf2x16(words.w);
    float standardDeviation = max(standardDeviationHitDistance.x, 0.0);
    signal.variance = standardDeviation * standardDeviation;
    signal.hitDistance = max(standardDeviationHitDistance.y, 0.0);
    return denoiserSanitizeMaxEntSignal(signal);
}

uvec4 denoiserPackMaxEntSignal(DenoiserMaxEntSignal signal) {
    signal = denoiserSanitizeMaxEntSignal(signal);
    return uvec4(
        packHalf2x16(clamp(signal.maxEntY.xy,
            vec2(-DENOISER_SPATIAL_FP16_MAX),
            vec2(DENOISER_SPATIAL_FP16_MAX))),
        packHalf2x16(clamp(signal.maxEntY.zw,
            vec2(-DENOISER_SPATIAL_FP16_MAX),
            vec2(DENOISER_SPATIAL_FP16_MAX))),
        packHalf2x16(clamp(signal.CoCg,
            vec2(-DENOISER_SPATIAL_FP16_MAX),
            vec2(DENOISER_SPATIAL_FP16_MAX))),
        packHalf2x16(vec2(min(sqrt(signal.variance),
            DENOISER_SPATIAL_FP16_MAX), signal.hitDistance)));
}

#endif // MAXENT_SPATIAL_SIGNAL_GLSL
