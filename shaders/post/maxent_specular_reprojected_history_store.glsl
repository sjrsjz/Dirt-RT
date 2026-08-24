layout(local_size_x = 16, local_size_y = 16) in;

// Race-free transfer of the raw reprojected estimator into N1/N2 scratch.
// The final resolve overwrites these planes with the actually accepted update.

#define REFLECT_BUFFER
#include "/lib/lighting/denoiser/maxent_specular_temporal_common.glsl"

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

    MaxEntTemporalSignal temporal = maxentUnpackTemporal(
        texelFetch(colortex5, ivec2(pixel), 0));
    SpecularMaxEnt denoisedSignal;
    float denoisedStddev, reprojectionAlphaFloor;
    float historyHitDistance;
    if (!statisticsValidEffectiveSampleCount(temporal.historyLength)
            || !readMaxEntSpecularDenoisedReprojection(pixel,
                denoisedSignal, denoisedStddev, reprojectionAlphaFloor,
                historyHitDistance)) {
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        return;
    }
    MaxEntSpecularHistory history;
    history.surfacePosition = geometry.position;
    history.geometryNormal = geometry.normal;
    history.signal = temporal.signal;
    history.rootMeanY2 = temporal.rootMeanY2;
    history.hitDistance = historyHitDistance;
    history.roughness = geometry.roughness;
    history.historyLength = temporal.historyLength;
    history.materialID = geometry.materialID;
    writeMaxEntSpecularTemporalHistory(pixel, history);
}
