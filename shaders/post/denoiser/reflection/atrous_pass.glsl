// Purpose: one reflection A-Trous iteration over the proposal and independent-current signals.
// Dispatch: 16x16 for steps 1/2/4, otherwise 8x8.
// Reads: colortex3 geometry; colortex4/5 signal input; shared scratch plane A/B.
// Writes: the opposite colortex4/5 image and the opposite scratch plane.
// Persistent side effects: none; reflection history is published only by resolve.glsl.
// Invalid representation: negative estimatorStdDev in signal metadata.

#ifndef DENOISER_SPATIAL_PHI_LUMINANCE
#error "DENOISER_SPATIAL_PHI_LUMINANCE must be configured by the pass"
#endif

#ifdef MAXENT_ATROUS_SMALL_KERNEL
layout(local_size_x = 16, local_size_y = 16) in;
#else
layout(local_size_x = 8, local_size_y = 8) in;
#endif

#include "/lib/common.glsl"
#include "/lib/lighting/denoiser/atrous_filter.glsl"
#include "/lib/lighting/denoiser/scratch_io.glsl"

uniform usampler2D colortex3;
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
uniform usampler2D colortex5;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
#else
uniform usampler2D colortex4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;
#endif

ivec2 denoiserSpatialImageSize() { return textureSize(colortex3, 0); }
uvec4 denoiserSpatialLoadGeometryWords(ivec2 pixel) { return texelFetch(colortex3, pixel, 0); }

uvec4 denoiserSpatialLoadSignalWords(ivec2 pixel) {
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
    return texelFetch(colortex5, pixel, 0);
#else
    return texelFetch(colortex4, pixel, 0);
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
#ifdef MAXENT_ATROUS_WRITE_ALTERNATE
    imageStore(colorimg4, pixel, words);
#else
    imageStore(colorimg5, pixel, words);
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
}

void denoiserSpatialStoreInvalid(ivec2 pixel) {
    uvec4 invalidWords = denoiserInvalidMaxEntSignalWords();
    denoiserSpatialStoreSignalWords(pixel, invalidWords);
    denoiserSpatialStoreIndependentCurrentWords(pixel, invalidWords);
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
