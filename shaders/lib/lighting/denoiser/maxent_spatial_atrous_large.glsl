#ifndef MAXENT_SPATIAL_ATROUS_LARGE_GLSL
#define MAXENT_SPATIAL_ATROUS_LARGE_GLSL

// Same policy interface as maxent_spatial_atrous_small.glsl, except stores are
// owned by the including pass. This lets the shared Poisson implementation be
// used by both fragment and compute entry points.

const vec4 DENOISER_SPATIAL_POISSON_8[8] = vec4[](
    vec4(-0.4706069, -0.4427112, 0.6461146, 0.81170),
    vec4(-0.9057375,  0.3003471, 0.9542373, 0.63422),
    vec4(-0.3487388,  0.4037880, 0.5335386, 0.86734),
    vec4( 0.1023042,  0.6439373, 0.6520134, 0.80847),
    vec4( 0.5699277,  0.3513750, 0.6695386, 0.79925),
    vec4( 0.2939128, -0.1131226, 0.3149309, 0.95161),
    vec4( 0.7836658, -0.4208784, 0.8895339, 0.67328),
    vec4( 0.1564120, -0.8198990, 0.8346850, 0.70589));

bool denoiserSpatialLargeInBounds(ivec2 pixel, ivec2 size) {
    return all(greaterThanEqual(pixel, ivec2(0)))
        && all(lessThan(pixel, size));
}

bool denoiserSpatialFilterLarge(ivec2 pixel,
        out DenoiserMaxEntSignal outputSignal) {
    ivec2 size = denoiserSpatialImageSize();
    outputSignal = denoiserEmptyMaxEntSignal();
    if (!denoiserSpatialLargeInBounds(pixel, size)) return false;

    uvec4 centerGeometryWords =
        denoiserSpatialLoadGeometryWords(pixel);
    uvec4 centerSignalWords = denoiserSpatialLoadSignalWords(pixel);
    DenoiserSpatialGeometry centerGeometry =
        denoiserSpatialDecodeGeometry(centerGeometryWords, pixel);
    if (!centerGeometry.valid
            || !denoiserSpatialSignalWordsValid(centerSignalWords))
        return false;

    DenoiserMaxEntSignal centerSignal =
        denoiserUnpackMaxEntSignal(centerSignalWords);
    DenoiserSpatialBuresData centerBures =
        denoiserSpatialMakeBuresData(centerSignal.maxEntY);
    float lightSourceToleranceScale =
        denoiserSpatialLightSourceToleranceScale(centerGeometry.roughness);
    DenoiserSpatialAccumulator accum =
        denoiserSpatialBeginAccumulation(centerSignal);

    float rotationAngle = 2.0 * PI * fract(rand(vec2(pixel))
        + float(DENOISER_SPATIAL_STEP) * 0.6180339887498949);
    float cs = cos(rotationAngle);
    float sn = sin(rotationAngle);
    mat2 rotation = mat2(cs, -sn, sn, cs)
        * (float(DENOISER_SPATIAL_STEP) * 1.75);

    for (int i = 0; i < 8; ++i) {
        ivec2 samplePixel = pixel + ivec2(round(rotation
            * DENOISER_SPATIAL_POISSON_8[i].xy));
        if (!denoiserSpatialLargeInBounds(samplePixel, size)) continue;

        uvec4 sampleGeometryWords =
            denoiserSpatialLoadGeometryWords(samplePixel);
        uvec4 sampleSignalWords =
            denoiserSpatialLoadSignalWords(samplePixel);
        DenoiserSpatialGeometry sampleGeometry =
            denoiserSpatialDecodeGeometry(sampleGeometryWords, samplePixel);
        if (!sampleGeometry.valid
                || !denoiserSpatialSignalWordsValid(sampleSignalWords)
                || !denoiserSpatialGeometryCompatible(
                    centerGeometry, sampleGeometry))
            continue;

        DenoiserMaxEntSignal sampleSignal =
            denoiserUnpackMaxEntSignal(sampleSignalWords);
        DenoiserSpatialBuresData sampleBures =
            denoiserSpatialMakeBuresData(sampleSignal.maxEntY);
        float weight = denoiserSpatialWeight(centerSignal, centerBures,
            centerGeometry, sampleSignal, sampleBures, sampleGeometry,
            DENOISER_SPATIAL_POISSON_8[i].w,
            lightSourceToleranceScale,
            DENOISER_SPATIAL_PHI_LUMINANCE, float(size.y));
        if (weight <= 1e-6) continue;
        denoiserSpatialAccumulate(accum, sampleSignal, weight);
    }

    outputSignal = denoiserSpatialResolve(accum,
        DENOISER_SPATIAL_STEP);
    return true;
}

#endif // MAXENT_SPATIAL_ATROUS_LARGE_GLSL
