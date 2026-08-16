#version 430 core

// Unified MaxEnt A-Trous entry for diffuse and specular signals.
// Required macros:
//   MAXENT_ATROUS_STEP
//   DENOISER_SPATIAL_PHI_LUMINANCE
// Optional macros:
//   MAXENT_ATROUS_SMALL_KERNEL
//   MAXENT_ATROUS_DEBUG_VIEW (reflection output hook)
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

#if defined(MAXENT_ATROUS_DEBUG_VIEW) || \
        defined(MAXENT_ATROUS_FINAL_RESOLVE)
#define MAXENT_ATROUS_REFLECTION_OUTPUT
#include "/lib/buffers/specular_buffer.glsl"
#endif

uniform usampler2D colortex4;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;

uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel) {
    return texelFetch(colortex4, pixel, 0);
}

#if defined(MAXENT_ATROUS_REFLECTION_OUTPUT)
SpecularMaxEnt maxentAtrousSpecularSignal(DenoiserMaxEntSignal signal) {
    SpecularMaxEnt result;
    result.maxEntY = signal.maxEntY;
    result.CoCg = signal.CoCg;
    return sanitizeSpecularMaxEnt(result);
}
#endif

void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal) {
    uvec4 packedSignal = denoiserPackMaxEntSignal(signal);
    imageStore(colorimg4, pixel, packedSignal);

#if defined(MAXENT_ATROUS_REFLECTION_OUTPUT)
    SpecularMaxEnt specular = maxentAtrousSpecularSignal(signal);
    #if defined(MAXENT_ATROUS_DEBUG_VIEW) && \
            DEBUG_VIEW == MAXENT_ATROUS_DEBUG_VIEW
    writeReflLight(uvec2(pixel), specularMaxEntTotalRgb(specular),
        signal.hitDistance, 1.0);
    #endif
    #if defined(MAXENT_ATROUS_FINAL_RESOLVE)
        #if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14 || \
                (DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30)
        // Preserve the diagnostic value written by its owning pass.
        #else
        writeReflMaxEnt(uvec2(pixel), specular, signal.hitDistance, 1.0);
        #endif
    #endif
#endif
}

void denoiserSpatialStoreInvalid(ivec2 pixel) {
    uvec4 invalidSignal = denoiserInvalidMaxEntSignalWords();
    imageStore(colorimg4, pixel, invalidSignal);

#if defined(MAXENT_ATROUS_REFLECTION_OUTPUT)
    #if defined(MAXENT_ATROUS_DEBUG_VIEW) && \
            DEBUG_VIEW == MAXENT_ATROUS_DEBUG_VIEW
    writeReflLight(uvec2(pixel), vec3(0.0), 0.0, 0.0);
    #endif
    #if defined(MAXENT_ATROUS_FINAL_RESOLVE)
        #if DEBUG_VIEW == 9 || DEBUG_VIEW == 12 || DEBUG_VIEW == 14 || \
                (DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30)
        #else
        writeReflMaxEnt(uvec2(pixel), emptySpecularMaxEnt(), 0.0, 0.0);
        #endif
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
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy,
            resolution_global))) return;

    DenoiserMaxEntSignal signal;
    if (!denoiserSpatialFilterLarge(pixel, signal)) {
        denoiserSpatialStoreInvalid(pixel);
        return;
    }
    denoiserSpatialStore(pixel, signal);
}
#endif
