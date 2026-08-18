#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"

uniform usampler2D colortex4;
uniform usampler2D colortex5;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    MaxEntGeometry geometry = maxentLoadGeometry(pixel);
    if (!geometry.valid) {
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        return;
    }

    MaxEntTemporalSignal temporal = maxentUnpackTemporal(texelFetch(colortex4, ivec2(pixel), 0));
    MaxEntSpecularHistory history;
    history.surfacePosition = geometry.position;
    history.geometryNormal = geometry.normal;
    history.signal = temporal.signal;
    history.secondMoment = temporal.secondMoment;
    history.hitDistance = maxentUnpackTemporalHitDistance(texelFetch(colortex5, ivec2(pixel), 0));
    history.roughness = geometry.roughness;
    history.historyLength = temporal.historyLength;
    history.materialID = geometry.materialID;
    writeMaxEntSpecularTemporalHistory(pixel, history);
}
