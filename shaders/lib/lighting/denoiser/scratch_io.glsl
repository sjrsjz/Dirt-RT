#ifndef MAXENT_DENOISER_SCRATCH_IO_GLSL
#define MAXENT_DENOISER_SCRATCH_IO_GLSL

// These images are idle until the bloom passes. Use their texture cache for
// spatial gathers, then let bloom overwrite them after both denoiser domains.
uniform sampler2D bloomAtlas_Sampler;
uniform sampler2D bloomBlur_Sampler;
layout(rgba32f) uniform writeonly image2D bloomAtlas;
layout(rgba32f) uniform writeonly image2D bloomBlur;

// A word contains two finite FP16 values, so its FP32 exponent is at most 247.
// Bias that exponent by one before storing in RGBA32F: all transport values
// become finite normal FP32 bit patterns, including originally subnormal words.
// Integer subtraction recovers every bit; no float arithmetic touches payloads.
vec4 denoiserScratchToImage(uvec4 words) {
    return uintBitsToFloat(words + uvec4(0x00800000u));
}
uvec4 denoiserScratchFromImage(vec4 values) {
    return floatBitsToUint(values) - uvec4(0x00800000u);
}
uvec4 denoiserScratchLoadA(ivec2 pixel) {
    return denoiserScratchFromImage(texelFetch(bloomAtlas_Sampler, pixel, 0));
}
uvec4 denoiserScratchLoadB(ivec2 pixel) {
    return denoiserScratchFromImage(texelFetch(bloomBlur_Sampler, pixel, 0));
}
void denoiserScratchStoreA(ivec2 pixel, uvec4 words) {
    imageStore(bloomAtlas, pixel, denoiserScratchToImage(words));
}
void denoiserScratchStoreB(ivec2 pixel, uvec4 words) {
    imageStore(bloomBlur, pixel, denoiserScratchToImage(words));
}

#endif // MAXENT_DENOISER_SCRATCH_IO_GLSL
