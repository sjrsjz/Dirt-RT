// Purpose: one diffuse A-Trous iteration over the proposal and independent-current signals.
// Dispatch: 16x16 for steps 1/2/4, otherwise 8x8.
// Reads: colortex3 geometry; colortex4/5 proposal; scratch A/B in the bloom images.
// Writes: opposite scratch image; opposite colortex4/5 proposal for steps 1-16.
// Step 32 keeps proposal inputs for weights, but emits no proposal image;
// proposal accumulation is retained only for filtered-variance debug views.
// Persistent side effects: none.
// Metadata: sigma=-1 invalid; sigma=-2 is usable light with unknown variance.

#ifndef DENOISER_SPATIAL_PHI_LUMINANCE
#error "DENOISER_SPATIAL_PHI_LUMINANCE must be configured by the pass"
#endif

#ifdef MAXENT_ATROUS_SMALL_KERNEL
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 8, local_size_y = 8) in;
#endif

#include "/lib/common.glsl"
#include "/lib/buffers/debug_buffer.glsl"
#include "/lib/lighting/denoiser/atrous_filter.glsl"
#include "/lib/lighting/denoiser/scratch_io.glsl"

uniform usampler2D colortex3;
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
uniform usampler2D colortex4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;
#else
uniform usampler2D colortex5;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
#endif

ivec2 denoiserSpatialImageSize() { return textureSize(colortex3, 0); }
uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel) { return texelFetch(colortex3, pixel, 0); }

uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel) {
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
    return texelFetch(colortex4, pixel, 0);
#else
    return texelFetch(colortex5, pixel, 0);
#endif
}

uvec4 denoiserSpatialLoadIndependentCurrentWords(ivec2 pixel) {
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
    return denoiserScratchLoadA(pixel);
#else
    return denoiserScratchLoadB(pixel);
#endif
}

void denoiserSpatialStoreSignalWords(ivec2 pixel, uvec4 words) {
#if DENOISER_SPATIAL_WRITE_PROPOSAL
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
    imageStore(colorimg5, pixel, words);
#else
    imageStore(colorimg4, pixel, words);
#endif
#endif
}

void denoiserSpatialStoreIndependentCurrentWords(ivec2 pixel, uvec4 words) {
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
    denoiserScratchStoreB(pixel, words);
#else
    denoiserScratchStoreA(pixel, words);
#endif
}

void denoiserSpatialStore(ivec2 pixel, DenoiserMaxEntSignal signal, DenoiserMaxEntSignal independentCurrent) {
    denoiserSpatialStoreSignalWords(pixel, denoiserPackMaxEntSignalTrusted(signal));
    denoiserSpatialStoreIndependentCurrentWords(pixel, denoiserPackMaxEntSignalTrusted(independentCurrent));
#if MAXENT_ATROUS_STEP == 32 && DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_FILTERED_MONTE_CARLO_VARIANCE
    debugWriteDiffuseFilteredMonteCarloStandardDeviation(uvec2(pixel), signal.standardDeviation);
#endif
}

void denoiserSpatialStoreInvalid(ivec2 pixel) {
    uvec4 invalidWords = denoiserInvalidMaxEntSignalWords();
    denoiserSpatialStoreSignalWords(pixel, invalidWords);
    denoiserSpatialStoreIndependentCurrentWords(pixel, invalidWords);
#if MAXENT_ATROUS_STEP == 32 && DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_FILTERED_MONTE_CARLO_VARIANCE
    debugWriteDiffuseFilteredMonteCarloStandardDeviation(uvec2(pixel), -1.0);
#endif
}

#define DENOISER_SPATIAL_STEP MAXENT_ATROUS_STEP
#ifdef MAXENT_ATROUS_SMALL_KERNEL
#include "/lib/lighting/denoiser/atrous_small.glsl"
#else
#include "/lib/lighting/denoiser/atrous_large.glsl"

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    DenoiserMaxEntSignal signal;
    DenoiserMaxEntSignal independentCurrent;
    if (!denoiserSpatialFilterLarge(pixel, signal, independentCurrent)) {
        if (any(greaterThanEqual(gl_GlobalInvocationID.xy, resolution_global))) return;
        denoiserSpatialStoreInvalid(pixel);
        return;
    }
    denoiserSpatialStore(pixel, signal, independentCurrent);
}
#endif
