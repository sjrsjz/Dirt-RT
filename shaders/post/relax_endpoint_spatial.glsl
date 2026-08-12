#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

layout(rgba32ui) uniform writeonly uimage2D colorimg6;

// Kept in composite59 so the public pass schedule does not change. The old
// 7x7 endpoint-moment fit is gone; this is now only the raw MaxEnt+hit upload.
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec3 position;
    float primaryDistance;
    readGeo0(GEO_N_GEO, pixel, position, primaryDistance);
    if (primaryDistance < -0.5) {
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    SpecularMaxEnt signal;
    float hitDistance, unusedWeight;
    readReflMaxEnt(pixel, signal, hitDistance, unusedWeight);
    RelaxPrepassSignal outSignal;
    outSignal.signal = signal;
    outSignal.hitDistance = hitDistance;
    imageStore(colorimg6, ivec2(pixel), relaxPackPrepass(outSignal));
}
