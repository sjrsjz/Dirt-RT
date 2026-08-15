#ifndef MAXENT_SPATIAL_SIGNAL_GLSL
#define MAXENT_SPATIAL_SIGNAL_GLSL

// Canonical input/output ABI for the MaxEnt spatial denoiser.
//
// Image format: RGBA32UI
//   x = packHalf2x16(maxEntY.xy)
//   y = packHalf2x16(maxEntY.zw)
//   z = packHalf2x16(CoCg.xy)
//   w = floatBitsToUint(variance)
//
// maxEntY.xyz is the directional first moment, maxEntY.w is total luminance,
// and CoCg carries chroma. Variance is a native, non-negative F32 estimator
// variance; it is never packed as FP16. Input and output use the same layout,
// so every spatial pass may ping-pong the same RGBA32UI resources.
//
// A negative F32 variance is reserved as the invalid/no-surface sentinel.
// Geometry is supplied separately by the signal policy and must provide a
// camera-relative position, geometry normal, texture normal, GGX roughness,
// material ID, and validity bit. Sky may therefore be rejected either by its
// geometry distance or by this signal sentinel without a separate mask.

const float DENOISER_SPATIAL_FP16_MAX = 65504.0;
const float DENOISER_SPATIAL_VARIANCE_MAX = 1e30;

struct DenoiserMaxEntSignal {
    vec4 maxEntY;
    vec2 CoCg;
    float variance;
};

DenoiserMaxEntSignal denoiserEmptyMaxEntSignal() {
    DenoiserMaxEntSignal signal;
    signal.maxEntY = vec4(0.0);
    signal.CoCg = vec2(0.0);
    signal.variance = 0.0;
    return signal;
}

bool denoiserSpatialSignalWordsValid(uvec4 words) {
    float variance = uintBitsToFloat(words.w);
    return variance >= 0.0 && !isnan(variance) && !isinf(variance);
}

uvec4 denoiserInvalidMaxEntSignalWords() {
    return uvec4(0u, 0u, 0u, floatBitsToUint(-1.0));
}

DenoiserMaxEntSignal denoiserSanitizeMaxEntSignal(
        DenoiserMaxEntSignal signal) {
    if (any(isnan(signal.maxEntY)) || any(isinf(signal.maxEntY)))
        signal.maxEntY = vec4(0.0);
    if (any(isnan(signal.CoCg)) || any(isinf(signal.CoCg)))
        signal.CoCg = vec2(0.0);
    if (isnan(signal.variance) || isinf(signal.variance))
        signal.variance = 0.0;

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
    return signal;
}

DenoiserMaxEntSignal denoiserUnpackMaxEntSignal(uvec4 words) {
    DenoiserMaxEntSignal signal;
    signal.maxEntY = vec4(unpackHalf2x16(words.x),
        unpackHalf2x16(words.y));
    signal.CoCg = unpackHalf2x16(words.z);
    signal.variance = uintBitsToFloat(words.w);
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
        floatBitsToUint(signal.variance));
}

#endif // MAXENT_SPATIAL_SIGNAL_GLSL
