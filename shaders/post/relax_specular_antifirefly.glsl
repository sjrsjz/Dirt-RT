#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_common.glsl"

uniform sampler2D colortex9;
uniform usampler2D colortex4;
layout(rgba32f) uniform writeonly image2D colorimg3;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    vec4 center = texelFetch(colortex9, ivec2(pixel), 0);
    RelaxFastSignal centerFast = relaxUnpackFast(texelFetch(colortex4, ivec2(pixel), 0));

    if (centerFast.confidence <= 0.0 || RELAX_ANTIFIREFLY_ENABLE == 0) {
        imageStore(colorimg3, ivec2(pixel), center);
#if DEBUG_VIEW == 24
        writeReflLight(pixel, relaxFiniteColor(center.rgb),
            centerFast.hitDistance, centerFast.historyLength);
#endif
        return;
    }

    float centerLuminance = relaxLuma(center.rgb);
    float minLuminance = 1e30;
    float maxLuminance = -1.0;
    vec3 minColor = center.rgb;
    vec3 maxColor = center.rgb;
    int validCount = 0;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        if (x == 0 && y == 0) continue;
        ivec2 q = ivec2(pixel) + ivec2(x, y);
        if (!relaxInBounds(q, size)) continue;
        RelaxFastSignal qFast = relaxUnpackFast(texelFetch(colortex4, q, 0));
        if (qFast.confidence <= 0.0 || qFast.materialID != centerFast.materialID) continue;
        vec4 sampleSignal = texelFetch(colortex9, q, 0);
        float luminance = relaxLuma(sampleSignal.rgb);
        if (luminance < minLuminance) {
            minLuminance = luminance;
            minColor = sampleSignal.rgb;
        }
        if (luminance > maxLuminance) {
            maxLuminance = luminance;
            maxColor = sampleSignal.rgb;
        }
        ++validCount;
    }

    vec3 outputColor = center.rgb;
    if (validCount > 0) {
        if (centerLuminance > maxLuminance) outputColor = maxColor;
        if (centerLuminance < minLuminance) outputColor = minColor;
    }
    imageStore(colorimg3, ivec2(pixel), vec4(outputColor, center.a));
#if DEBUG_VIEW == 24
    writeReflLight(pixel, relaxFiniteColor(outputColor),
        centerFast.hitDistance, centerFast.historyLength);
#endif
}
