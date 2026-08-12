#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex5;
layout(rgba32f) uniform writeonly image2D colorimg3;

// MaxEnt6 cannot be sparsely repaired through a four-float target without
// desynchronising direction/luminance from CoCg. Temporal clamp and variance
// perform the coherent repair later, so this scheduled pass only exposes Y4.
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    RelaxSlowSignal slow = relaxUnpackSlow(
        texelFetch(colortex4, ivec2(pixel), 0));
    imageStore(colorimg3, ivec2(pixel), slow.signal.aliceY);
}
