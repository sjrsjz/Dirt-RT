#ifndef MAXENT_DENOISER_SIGNAL_GLSL
#define MAXENT_DENOISER_SIGNAL_GLSL

#include "/lib/math/denoiser_uncertainty.glsl"

// Canonical input/output ABI for the MaxEnt spatial denoiser.
//
// Image format: RGBA32UI
//   x = packHalf2x16(maxEntY.xy)
//   y = packHalf2x16(maxEntY.zw)
//   z = packHalf2x16(CoCg.xy)
//   w = packHalf2x16(standardDeviation, virtualDistance)
//
// maxEntY.xyz is the directional first moment, maxEntY.w is total luminance,
// and CoCg carries chroma. standardDeviation stores sqrt(estimator variance)
// in the local Bures metric: preparation divides observation variance by the
// center's raw temporal N_eff once (independent-current starts at N=1).
// Each A-Trous pass propagates uncertainty with its actual squared weights and
// a per-step overlap-correlation closure. No spatial N_eff is propagated.
// Pack/unpack never performs a hidden sqrt or square.
// Camera-relative radial virtual distance is independent. Input and
// output use the same layout, so every spatial pass may ping-pong the same
// RGBA32UI resources.
//
// sigma=-1 marks invalid light; sigma=-2 preserves light with unknown variance.
// Geometry is supplied separately by the signal policy.
// Persistent filtered history reuses the same packing; that history ABI is
// consumed by reprojection/resolve only, never directly by A-Trous.

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
    return denoiserSigmaUsable(standardDeviation);
}

uvec4 denoiserInvalidMaxEntSignalWords() {
    return uvec4(0u, 0u, 0u, packHalf2x16(vec2(-1.0, 0.0)));
}

DenoiserMaxEntSignal denoiserSanitizeMaxEntSignal(
        DenoiserMaxEntSignal signal) {
    if (any(isnan(signal.maxEntY)) || any(isinf(signal.maxEntY))) {
        signal.maxEntY = vec4(0.0);
        signal.standardDeviation = DENOISER_UNKNOWN_UNCERTAINTY;
    }
    if (any(isnan(signal.CoCg)) || any(isinf(signal.CoCg)))
        signal.CoCg = vec2(0.0);
    // Invalid uncertainty must not become a valid zero-noise estimate.
    if (signal.standardDeviation != -1.0)
        signal.standardDeviation = denoiserSigmaOrUnknown(signal.standardDeviation);
    if (isnan(signal.virtualDistance) || isinf(signal.virtualDistance))
        signal.virtualDistance = 0.0;

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
    signal.standardDeviation = standardDeviationVirtualDistance.x;
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
        packHalf2x16(vec2(signal.standardDeviation, signal.virtualDistance)));
}

uvec4 denoiserPackMaxEntSignal(DenoiserMaxEntSignal signal) {
    return denoiserPackMaxEntSignalTrusted(
        denoiserSanitizeMaxEntSignal(signal));
}

#endif // MAXENT_DENOISER_SIGNAL_GLSL
