#version 430 core
// Axis adapter for the shared separable bloom kernel.
layout(local_size_x = 256, local_size_y = 1) in;
layout(rgba32f) uniform readonly image2D bloomAtlas;
layout(rgba32f) uniform writeonly image2D bloomBlur;
#define BLOOM_BLUR_INPUT bloomAtlas
#define BLOOM_BLUR_OUTPUT bloomBlur
#define BLOOM_BLUR_AXIS ivec2(1, 0)
#include "/lib/post_processing/gaussian_blur.glsl"
