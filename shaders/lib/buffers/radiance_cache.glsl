#ifndef RADIANCE_CACHE_GLSL
#define RADIANCE_CACHE_GLSL

#include "/lib/settings.glsl"
#include "/lib/buffers/addr.glsl"
#include "/lib/lighting/maxent_encode.glsl"

// Storage/request operations are shared by every consumer. The allocator is
// called only by ray4; sampling observes immutable mapping metadata between
// execution-group barriers. Keep this facade for existing include sites.
#include "/lib/buffers/radiance_cache/storage.glsl"
#include "/lib/buffers/radiance_cache/allocator.glsl"
#include "/lib/buffers/radiance_cache/sampling.glsl"

#endif // RADIANCE_CACHE_GLSL
