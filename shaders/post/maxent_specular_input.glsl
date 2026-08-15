#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"

layout(rgba32ui) uniform writeonly uimage2D colorimg6;

// Raw MaxEnt + hit-distance upload for the specular denoiser.
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    if (readPrimaryDistance(pixel) < 0.0) {
        imageStore(colorimg6, ivec2(pixel), uvec4(0u));
        return;
    }

    SpecularMaxEnt signal;
    float hitDistance, unusedWeight;
    readReflMaxEnt(pixel, signal, hitDistance, unusedWeight);
    MaxEntSpecularInput outSignal;
    outSignal.signal = signal;
    outSignal.hitDistance = hitDistance;
    imageStore(colorimg6, ivec2(pixel),
        maxentPackSpecularInput(outSignal));
}
