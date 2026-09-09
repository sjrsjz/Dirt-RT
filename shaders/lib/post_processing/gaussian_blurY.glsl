#version 430 core
// Axis adapter for the shared separable bloom kernel.
layout(local_size_x = 1, local_size_y = 256) in;
layout(rgba32f) uniform readonly image2D bloomBlur;
layout(rgba32f) uniform writeonly image2D bloomAtlas;
#define BLOOM_BLUR_INPUT bloomBlur
#define BLOOM_BLUR_OUTPUT bloomAtlas
#define BLOOM_BLUR_AXIS ivec2(0, 1)
#include "/lib/post_processing/gaussian_blur.glsl"
