#version 430 core

#define DENOISER_SPATIAL_PHI_LUMINANCE MAXENT_SPATIAL_DIFFUSE_LIGHT_FIELD_SENSITIVITY
#define MAXENT_ATROUS_STEP 32
#include "/post/denoiser/diffuse/atrous_pass.glsl"
