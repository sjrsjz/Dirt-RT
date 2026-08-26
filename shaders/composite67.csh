#version 430 core

#define DENOISER_SPATIAL_PHI_LUMINANCE MAXENT_SPATIAL_SPECULAR_LIGHT_FIELD_SENSITIVITY
#define MAXENT_ATROUS_SMALL_KERNEL
#define MAXENT_ATROUS_STEP 2
#include "/post/denoiser/reflection/atrous_pass.glsl"
