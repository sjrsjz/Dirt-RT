#version 430 core

// Unified MaxEnt A-Trous entry for diffuse and specular signals.
// Required macros:
//   MAXENT_ATROUS_STEP
//   DENOISER_SPATIAL_PHI_LUMINANCE
// Optional macros:
//   MAXENT_ATROUS_SMALL_KERNEL
//   MAXENT_ATROUS_FINAL_RESOLVE (reflection output hook)

#ifndef DENOISER_SPATIAL_PHI_LUMINANCE
#error "DENOISER_SPATIAL_PHI_LUMINANCE must be configured by the pass"
#endif
#if defined(MAXENT_ATROUS_SMALL_KERNEL)
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 8, local_size_y = 8) in;
#endif

#include "/lib/common.glsl"
#include "/lib/lighting/denoiser/maxent_spatial_geometry.glsl"

#if defined(MAXENT_ATROUS_FINAL_RESOLVE)
#include "/lib/buffers/specular_buffer.glsl"
#include "/lib/lighting/denoiser/maxent_bures.glsl"
#endif

uniform usampler2D colortex4;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel) {
    return texelFetch(colortex4, pixel, 0);
}

#if defined(MAXENT_ATROUS_FINAL_RESOLVE)
SpecularMaxEnt maxentAtrousSpecularSignal(DenoiserMaxEntSignal signal) {
    SpecularMaxEnt result;
    result.maxEntY = signal.maxEntY;
    result.CoCg = signal.CoCg;
    // The spatial resolve already sanitized this signal. The eventual
    // SpecularMaxEnt pack/RGB conversion remains the owning safety boundary.
    return result;
}

void maxentClampSpecularHistoryByDenoisedDifference(uvec2 pixel,
    SpecularMaxEnt currentSignal, float currentStddev) {
    uvec4 historyWords = reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)];
    vec2 momentHistory = unpackHalf2x16(historyWords.w);
    if (!(momentHistory.y >= 1.0) || any(isnan(momentHistory)) || any(isinf(momentHistory))
            || momentHistory.y <= MAXENT_TEMPORAL_DIFFERENCE_COLD_START_HISTORY) return;

    SpecularMaxEnt historySignal;
    float historyStddev, temporalCurrentWeight;
    if (!readMaxEntSpecularDenoisedReprojection(pixel, historySignal,
            historyStddev, temporalCurrentWeight)) return;

    DenoiserSpatialBuresData currentBures = denoiserSpatialMakeBuresData(currentSignal.maxEntY);
    DenoiserSpatialBuresData historyBures = denoiserSpatialMakeBuresData(historySignal.maxEntY);
    float distanceSq = denoiserSpatialBuresDistanceSq(currentSignal.maxEntY,
            currentBures, historySignal.maxEntY, historyBures);
    float combinedStddev = sqrt(currentStddev * currentStddev + historyStddev * historyStddev);
    float normalizedDistance = sqrt(distanceSq) / max(temporalCurrentWeight * combinedStddev, 1e-6);
    if (isnan(normalizedDistance) || isinf(normalizedDistance)) return;

    float k = min(normalizedDistance, 80.0) * (1.0 / MAXENT_SPECULAR_TEMPORAL_DIFFERENCE_TOLERANCE);
    float agreement = (1 + k) * exp(-k);
    float historyCap = 1.0 + (max(float(MAXENT_SPECULAR_TEMPORAL_MAX_HISTORY), 1.0) - 1.0) * agreement;
    historyWords.w = pack2HalfClampedU(momentHistory.x, min(momentHistory.y, historyCap));
    reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = historyWords;
}
#endif

void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal) {
    // denoiserSpatialResolve() has already sanitized the signal once. Avoid a
    // second identical validation on the hot ping-pong store path.
    uvec4 packedSignal = denoiserPackMaxEntSignalTrusted(signal);
    imageStore(colorimg4, pixel, packedSignal);

    #if defined(MAXENT_ATROUS_FINAL_RESOLVE)
    SpecularMaxEnt specular = maxentAtrousSpecularSignal(signal);
    float standardDeviation = sqrt(max(signal.variance, 0.0));
    #if DEBUG_VIEW != 8
    maxentClampSpecularHistoryByDenoisedDifference(uvec2(pixel), specular, standardDeviation);
    #endif
    writeMaxEntSpecularDenoisedHistory(uvec2(pixel), specular, standardDeviation);
    #if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14
    // Preserve the diagnostic value written by its owning pass.
    #else
    writeReflMaxEnt(uvec2(pixel), specular, signal.virtualDistance, 1.0);
    #endif
    #endif
}

void denoiserSpatialStoreInvalid(ivec2 pixel) {
    uvec4 invalidSignal = denoiserInvalidMaxEntSignalWords();
    imageStore(colorimg4, pixel, invalidSignal);

    #if defined(MAXENT_ATROUS_FINAL_RESOLVE)
    writeMaxEntSpecularDenoisedHistoryInvalid(uvec2(pixel));
    #if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14
    #else
    writeReflMaxEnt(uvec2(pixel), emptySpecularMaxEnt(), 0.0, 0.0);
    #endif
    #endif
}

#define DENOISER_SPATIAL_STEP MAXENT_ATROUS_STEP

#if defined(MAXENT_ATROUS_SMALL_KERNEL)
#include "/lib/lighting/denoiser/maxent_spatial_atrous_small.glsl"
#else
#include "/lib/lighting/denoiser/maxent_spatial_atrous_large.glsl"

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    DenoiserMaxEntSignal signal;
    if (!denoiserSpatialFilterLarge(pixel, signal)) {
        if (any(greaterThanEqual(gl_GlobalInvocationID.xy,
                    resolution_global))) return;
        denoiserSpatialStoreInvalid(pixel);
        return;
    }
    denoiserSpatialStore(pixel, signal);
}
#endif
