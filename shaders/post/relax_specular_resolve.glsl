#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex5;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

#if DEBUG_VIEW == 12 || DEBUG_VIEW == 14 || (DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30) || (DEBUG_VIEW >= 38 && DEBUG_VIEW <= 40)
    // A selected diagnostic stage stored its output in N=1.  Do not overwrite
    // it with the final spatially filtered result from colortex5.
    return;
#else
    RelaxSpatialSignal signal = relaxUnpackSpatial(
        texelFetch(colortex5, ivec2(pixel), 0));
    writeReflLight(pixel, relaxFiniteColor(signal.radiance),
        signal.hitDistance, signal.historyLength);
#endif
}
