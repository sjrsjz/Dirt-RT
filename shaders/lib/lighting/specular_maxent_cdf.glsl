#ifndef DIRT_RT_LIB_LIGHTING_SPECULAR_MAXENT_CDF_GLSL
#define DIRT_RT_LIB_LIGHTING_SPECULAR_MAXENT_CDF_GLSL

// Iris texture binding adapter for the specular response decoder.

#define QLI_CDF_ATAN2 atan
#define QLI_CDF_SAMPLE(p) textureLod(ggxQLiCdfLut,p,0.0)
uniform sampler3D ggxQLiCdfLut;
#include "/lib/lighting/specular_cdf/response.glsl"
#undef QLI_CDF_SAMPLE
#undef QLI_CDF_ATAN2

#endif // DIRT_RT_LIB_LIGHTING_SPECULAR_MAXENT_CDF_GLSL
