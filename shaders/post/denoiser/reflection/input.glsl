#version 430 core

// Purpose: adapt Raw RT reflection output to the temporal denoiser input ABI.
// Dispatch: 8x8.
// Reads: reflection MaxEnt output, reflection hit distance, compact primary geometry.
// Writes: colorimg6 Raw reflection signal.
// Persistent side effects: none.
// Invalid representation: sky pixels write zero words.

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/post/denoiser/reflection/common.glsl"

layout(rgba32ui) uniform writeonly uimage2D colorimg6;

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
    imageStore(colorimg6, ivec2(pixel), maxentPackSpecularInput(outSignal));
}
