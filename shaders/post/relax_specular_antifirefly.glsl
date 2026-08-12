#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex9;
uniform usampler2D colortex4;
layout(rgba32f) uniform writeonly image2D colorimg3;

// Scalar outlier replacement would rotate AliceY without updating CoCg in
// this four-float target. The coherent YCoCg clamp already ran in composite63,
// so preserve the complete MaxEnt direction here.
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;
    vec4 aliceY = texelFetch(colortex9, ivec2(pixel), 0);
    imageStore(colorimg3, ivec2(pixel), aliceY);

#if DEBUG_VIEW == 24
    RelaxPostSignal meta = relaxUnpackPost(
        texelFetch(colortex4, ivec2(pixel), 0));
    SpecularMaxEnt signal;
    signal.aliceY = aliceY;
    signal.CoCg = meta.CoCg;
    writeReflLight(pixel, specularMaxEntTotalRgb(signal),
        meta.hitDistance, meta.historyLength);
#endif
}
