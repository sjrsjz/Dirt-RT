#ifndef MAXENT_DENOISER_SCRATCH_IO_GLSL
#define MAXENT_DENOISER_SCRATCH_IO_GLSL

#include "/lib/buffers/diffuse_buffer.glsl"

// The diffuse and reflection pipelines execute serially and share these two
// transient RGBA32UI SSBO planes. The words use the signal.glsl ABI.
uvec4 denoiserScratchLoadA(ivec2 pixel) { return readDiffuseIndependentCurrentA(uvec2(pixel)); }
uvec4 denoiserScratchLoadB(ivec2 pixel) { return readDiffuseIndependentCurrentB(uvec2(pixel)); }
void denoiserScratchStoreA(ivec2 pixel, uvec4 words) { writeDiffuseIndependentCurrentA(uvec2(pixel), words); }
void denoiserScratchStoreB(ivec2 pixel, uvec4 words) { writeDiffuseIndependentCurrentB(uvec2(pixel), words); }

#endif // MAXENT_DENOISER_SCRATCH_IO_GLSL
