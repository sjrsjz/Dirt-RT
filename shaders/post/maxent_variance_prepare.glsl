layout(local_size_x = 16, local_size_y = 16) in;

#if defined(MAXENT_VARIANCE_DIFFUSE)
#define DIFFUSE_BUFFER
#elif defined(MAXENT_VARIANCE_SPECULAR)
#define REFLECT_BUFFER
#endif

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/maxent_variance_prepare.glsl"
