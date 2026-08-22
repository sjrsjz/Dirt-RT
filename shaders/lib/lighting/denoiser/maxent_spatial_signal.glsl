#ifndef MAXENT_SPATIAL_SIGNAL_GLSL
#define MAXENT_SPATIAL_SIGNAL_GLSL

// Canonical input/output ABI for the MaxEnt spatial denoiser.
//
// Image format: RGBA32UI
//   x = packHalf2x16(maxEntY.xy)
//   y = packHalf2x16(maxEntY.zw)
//   z = packHalf2x16(CoCg.xy)
//   w = packHalf2x16(standardDeviation, virtualDistance)
//
// maxEntY.xyz is the directional first moment, maxEntY.w is total luminance,
// and CoCg carries chroma. Estimator uncertainty remains an explicit standard
// deviation on both sides of this ABI; pack/unpack never performs a hidden
// sqrt or square. Camera-relative radial virtual distance is propagated
// independently in the upper FP16 lane. Input and
// output use the same layout, so every spatial pass may ping-pong the same
// RGBA32UI resources.
//
// A negative FP16 standard deviation is the invalid/no-surface sentinel.
// Geometry is supplied separately by the signal policy.

const float DENOISER_SPATIAL_FP16_MAX = 65504.0;
struct DenoiserMaxEntSignal {
    vec4 maxEntY;
    vec2 CoCg;
    float standardDeviation;
    float virtualDistance;
};

DenoiserMaxEntSignal denoiserEmptyMaxEntSignal() {
    DenoiserMaxEntSignal signal;
    signal.maxEntY = vec4(0.0);
    signal.CoCg = vec2(0.0);
    signal.standardDeviation = 0.0;
    signal.virtualDistance = 0.0;
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
    if (isnan(signal.standardDeviation) || isinf(signal.standardDeviation))
        signal.standardDeviation = 0.0;
    if (isnan(signal.virtualDistance) || isinf(signal.virtualDistance))
        signal.virtualDistance = 0.0;

    signal.maxEntY.w = max(signal.maxEntY.w, 0.0);
    float meanLength2 = dot(signal.maxEntY.xyz, signal.maxEntY.xyz);
    float totalEnergy2 = signal.maxEntY.w * signal.maxEntY.w;
    if (meanLength2 > totalEnergy2 && meanLength2 > 0.0)
        signal.maxEntY.xyz *= signal.maxEntY.w * inversesqrt(meanLength2);

    if (signal.maxEntY.w <= 0.0) {
        signal.maxEntY = vec4(0.0);
        signal.CoCg = vec2(0.0);
    }
    signal.standardDeviation = clamp(signal.standardDeviation, 0.0,
        DENOISER_SPATIAL_FP16_MAX);
    signal.virtualDistance = clamp(signal.virtualDistance, 0.0,
        DENOISER_SPATIAL_FP16_MAX);
    return signal;
}

DenoiserMaxEntSignal denoiserUnpackMaxEntSignalTrusted(uvec4 words) {
    DenoiserMaxEntSignal signal;
    signal.maxEntY = vec4(unpackHalf2x16(words.x),
        unpackHalf2x16(words.y));
    signal.CoCg = unpackHalf2x16(words.z);
    vec2 standardDeviationVirtualDistance = unpackHalf2x16(words.w);
    signal.standardDeviation = standardDeviationVirtualDistance.x;
    signal.virtualDistance = standardDeviationVirtualDistance.y;
    return signal;
}

// Spatial ping-pong inputs have already crossed a sanitizing pack boundary.
// Keep the checked entry points for producers and diagnostics, while the hot
// A-trous path avoids repeating the full finite/energy validation per tap.
DenoiserMaxEntSignal denoiserUnpackMaxEntSignal(uvec4 words) {
    DenoiserMaxEntSignal signal = denoiserUnpackMaxEntSignalTrusted(words);
    vec2 standardDeviationVirtualDistance = unpackHalf2x16(words.w);
    signal.standardDeviation = max(standardDeviationVirtualDistance.x, 0.0);
    signal.virtualDistance = max(standardDeviationVirtualDistance.y, 0.0);
    return denoiserSanitizeMaxEntSignal(signal);
}

uvec4 denoiserPackMaxEntSignalTrusted(DenoiserMaxEntSignal signal) {
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
        packHalf2x16(vec2(signal.standardDeviation,
            signal.virtualDistance)));
}

uvec4 denoiserPackMaxEntSignal(DenoiserMaxEntSignal signal) {
    return denoiserPackMaxEntSignalTrusted(
        denoiserSanitizeMaxEntSignal(signal));
}

#endif // MAXENT_SPATIAL_SIGNAL_GLSL
