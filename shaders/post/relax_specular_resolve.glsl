#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex5;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;
    RelaxSpatialSignal signal = relaxUnpackSpatial(
        texelFetch(colortex5, ivec2(pixel), 0));
    writeReflLight(pixel, relaxFiniteColor(signal.radiance),
        signal.hitDistance, signal.historyLength);
}
