// Purpose: preserve the raw reprojected reflection estimator before colortex5 becomes A-Trous input.
// Dispatch: 16x16.
// Reads: colortex5 raw reprojection, current geometry, filtered-history reprojection scratch.
// Writes: reflection N1/N2 transient history staging planes.
// Persistent side effects: none; resolve.glsl overwrites these planes with the accepted update.
// Invalid representation: zero staging records.

layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/post/denoiser/reflection/common.glsl"

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
    float denoisedMonteCarloStandardDeviation, denoisedEffectiveSamples, reprojectionAlphaFloor;
    float currentTrackingHitDistance;
    if (!statisticsValidEffectiveSampleCount(temporal.historyEffectiveSamples)
            || !readMaxEntSpecularDenoisedReprojection(pixel,
                denoisedSignal, denoisedMonteCarloStandardDeviation, denoisedEffectiveSamples, reprojectionAlphaFloor,
                currentTrackingHitDistance)) {
        reflectBuffer.data[addr(SPEC_N_HISTGEO, pixel)] = uvec4(0u);
        reflectBuffer.data[addr(SPEC_N_HISTLIGHT, pixel)] = uvec4(0u);
        return;
    }
    MaxEntSpecularHistory history;
    history.surfacePosition = geometry.position;
    history.geometryNormal = geometry.normal;
    history.signal = temporal.signal;
    history.rootMeanY2 = temporal.rootMeanY2;
    history.hitDistance = currentTrackingHitDistance;
    history.roughness = geometry.roughness;
    history.historyEffectiveSamples = temporal.historyEffectiveSamples;
    history.materialID = maxentReflectionHistoryMaterialID(
        geometry.materialID);
    writeMaxEntSpecularTemporalHistory(pixel, history);
}
