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
#endif

#if defined(MAXENT_ATROUS_WRITE_ALTERNATE)
uniform usampler2D colortex4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;
#else
uniform usampler2D colortex5;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
#endif

uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel) {
    #if defined(MAXENT_ATROUS_WRITE_ALTERNATE)
    return texelFetch(colortex4, pixel, 0);
    #else
    return texelFetch(colortex5, pixel, 0);
    #endif
}

void maxentAtrousStoreSignalWords(ivec2 pixel, uvec4 words) {
    #if defined(MAXENT_ATROUS_WRITE_ALTERNATE)
    imageStore(colorimg5, pixel, words);
    #else
    imageStore(colorimg4, pixel, words);
    #endif
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

#endif

void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal) {
    // denoiserSpatialResolve() has already sanitized the signal once. Avoid a
    // second identical validation on the hot ping-pong store path.
    uvec4 packedSignal = denoiserPackMaxEntSignalTrusted(signal);
    maxentAtrousStoreSignalWords(pixel, packedSignal);

    #if defined(MAXENT_ATROUS_FINAL_RESOLVE)
    SpecularMaxEnt specular = maxentAtrousSpecularSignal(signal);
    writeMaxEntSpecularDenoisedHistory(uvec2(pixel), specular,
        signal.standardDeviation);
    // SPEC_N_LIGHT still contains the reprojected denoised-history scratch.
    // The following full-screen resolve reads the completed 5x5 final-output
    // neighborhood, clamps history confidence, then publishes reflection N0.
    #endif
}

void denoiserSpatialStoreInvalid(ivec2 pixel) {
    uvec4 invalidSignal = denoiserInvalidMaxEntSignalWords();
    maxentAtrousStoreSignalWords(pixel, invalidSignal);

    #if defined(MAXENT_ATROUS_FINAL_RESOLVE)
    writeMaxEntSpecularDenoisedHistoryInvalid(uvec2(pixel));
    // Final reflection invalidation is deferred with the valid resolve above.
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
