#ifndef DIRT_RT_LIB_RT_RAYTRACE_GUIDING_GLSL
#define DIRT_RT_LIB_RT_RAYTRACE_GUIDING_GLSL

// Reprojected incident-light proposals and their complete mixture PDFs.

// ===========================================================================
// MaxEnt Path Guiding
// ===========================================================================

GuideInfo computeMaxEntGuide(uvec2 pixel, uint currentFrameId,
        float strengthMultiplier) {
    GuideInfo g = emptyGuideInfo();

    vec4 guideY;
    vec2 guideChroma;
    float guideSigma, validCoverage;
    if (!readDiffuseDenoisedReprojectionForFrame(pixel, currentFrameId,
            guideY, guideChroma, guideSigma, validCoverage)) return g;
    vec3 x = guideY.xyz;
    float omega = guideY.w;
    float length_x = max(length(x), 1e-20);
    omega = max(omega, length_x);
    g.axis = x / length_x;
    float rho = clamp(length_x / omega, 0.0, 1.0);
    g.kappa = maxent_kappa(length_x, omega);
    g.valid = length_x > 1e-8;
    g.prob = float(g.valid) * strengthMultiplier * rho
        * validCoverage;
    return g;
}

// ray0 owns the current primary surface, the Vulkanite camera, and a barrier
// before continuation tracing. It therefore performs the surface-history
// reprojection once and publishes the result in the existing N4 scratch.
void prepareSpecularDenoisedSurfaceReprojection(uvec2 currentPixel,
        vec3 currentSurfacePosition, vec3 currentGeometryNormal,
        uint currentMaterialID, vec3 cameraDelta, float motionValid,
        mat4 historyViewProjection, mat4 currentViewProjection) {
    if (motionValid < 0.5) {
        writeMaxEntSpecularPreparedSurfaceDenoisedInvalid(currentPixel);
        return;
    }

    vec4 previousClip = historyViewProjection
        * vec4(currentSurfacePosition + cameraDelta, 1.0);
    if (previousClip.w <= 1e-8 || any(isnan(previousClip))
            || any(isinf(previousClip))) {
        writeMaxEntSpecularPreparedSurfaceDenoisedInvalid(currentPixel);
        return;
    }
    vec3 projected = previousClip.xyz / previousClip.w * 0.5 + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0
            || any(lessThan(projected.xy, vec2(0.0)))
            || any(greaterThan(projected.xy, vec2(1.0)))) {
        writeMaxEntSpecularPreparedSurfaceDenoisedInvalid(currentPixel);
        return;
    }

    ivec2 size = ivec2(resolution_global);
    vec2 pixelPosition = projected.xy * vec2(size);
    ivec2 origin = ivec2(floor(pixelPosition));
    vec2 f = fract(pixelPosition);
    vec4 bilinear = vec4((1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    SpecularMaxEnt signalSum = emptySpecularMaxEnt();
    DenoiserEstimatorVarianceAccumulator uncertainty =
        denoiserBeginEstimatorVariance();
    float sumWeight = 0.0;
    float depthThreshold = MAXENT_SPECULAR_TEMPORAL_DISOCCLUSION_THRESHOLD
        * max(length(currentSurfacePosition), 1.0);
    uint expectedMaterial = specularHistoryMaterialSignature(
        currentMaterialID);

    for (int i = 0; i < 4; ++i) {
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size)))
            continue;

        uvec2 historyPixel = uvec2(p);
        uvec4 geometryWords = reflectBuffer.data[
            addr(SPEC_N_HISTGEO, historyPixel)];
        uvec4 rawWords = reflectBuffer.data[
            addr(SPEC_N_HISTLIGHT, historyPixel)];
        float historyDistance = uintBitsToFloat(geometryWords.x);
        vec2 rawMetadata = unpackHalf2x16(rawWords.w);
        if (!(historyDistance >= 0.0) || isnan(historyDistance)
                || isinf(historyDistance) || !(rawMetadata.x >= 0.0)
                || !(rawMetadata.y >= 1.0) || any(isnan(rawMetadata))
                || any(isinf(rawMetadata)))
            continue;
        vec4 rawMoment = vec4(unpackHalf2x16(rawWords.x),
            unpackHalf2x16(rawWords.y));
        vec2 rawChroma = unpackHalf2x16(rawWords.z);
        if (!denoiserTemporalMomentsFinite(
                rawMoment, rawChroma, rawMetadata.x))
            continue;

        vec3 historyNormal;
        uint historyMaterial;
        unpackMaxEntHistoryNormalMaterial(geometryWords.z,
            historyNormal, historyMaterial);
        if (historyMaterial != expectedMaterial
                || dot(currentGeometryNormal, historyNormal) <= 0.0)
            continue;

        vec3 historyWorld = prevRaytracingCamPos
            + decodeNormalU(geometryWords.y) * historyDistance;
        vec3 historyPosition = historyWorld - prevRaytracingCamPos;
        vec3 historyPositionCurrent = historyPosition - cameraDelta;
        if (abs(dot(historyPositionCurrent - currentSurfacePosition,
                currentGeometryNormal)) > depthThreshold)
            continue;

        vec4 currentClip = currentViewProjection
            * vec4(historyPositionCurrent, 1.0);
        if (currentClip.w <= 1e-8 || any(isnan(currentClip))
                || any(isinf(currentClip)))
            continue;
        vec2 currentHistoryPixel = (currentClip.xy / currentClip.w
            * 0.5 + 0.5) * vec2(size);
        if (any(greaterThan(abs(currentHistoryPixel - vec2(currentPixel)),
                vec2(MAXENT_SPECULAR_TEMPORAL_REPROJECTION_RADIUS + 1e-5))))
            continue;

        SpecularMaxEnt tap;
        float tapSigma, tapEffectiveSamples;
        if (!readMaxEntSpecularDenoisedHistory(historyPixel, tap,
                tapSigma, tapEffectiveSamples))
            continue;

        float w = bilinear[i];
        signalSum.maxEntY += tap.maxEntY * w;
        signalSum.CoCg += tap.CoCg * w;
        denoiserAccumulateEstimatorVariance(uncertainty, tapSigma, w);
        sumWeight += w;
    }

    if (sumWeight <= 1e-6) {
        writeMaxEntSpecularPreparedSurfaceDenoisedInvalid(currentPixel);
        return;
    }
    float inverseWeight = 1.0 / sumWeight;
    signalSum.maxEntY *= inverseWeight;
    signalSum.CoCg *= inverseWeight;
    float sigma = denoiserResolveEstimatorSigma(uncertainty, 1.0);
    writeMaxEntSpecularPreparedSurfaceDenoised(currentPixel,
        signalSum, sigma, clamp(sumWeight, 0.0, 1.0));
}

GuideInfo computeSpecularMaxEntGuide(uvec2 pixel,
        float strengthMultiplier) {
    GuideInfo g = emptyGuideInfo();
    SpecularMaxEnt signal;
    float sigma, validCoverage;
    if (strengthMultiplier <= 0.0
            || !readMaxEntSpecularPreparedSurfaceDenoised(
                pixel, signal, sigma, validCoverage))
        return g;
    vec4 moment = signal.maxEntY;
    float directionalLength = length(moment.xyz);
    float totalEnergy = max(moment.w, directionalLength);
    if (!(totalEnergy > 1e-8) || !(directionalLength > 1e-8)
            || any(isnan(moment)) || any(isinf(moment)))
        return g;

    float rho = clamp(directionalLength / totalEnergy, 0.0, 1.0);
    g.axis = moment.xyz / directionalLength;
    g.kappa = min(maxent_kappa(directionalLength, totalEnergy),
        1.0 - 1e-6);
    g.prob = min(0.75, strengthMultiplier * rho
        * validCoverage);
    g.valid = g.prob > 0.0;
    return g;
}

float specularGuideMixturePdf(GuideInfo guide, float vndfPdf,
        vec3 direction) {
    float guidePdf = guide.prob > 0.0
        ? maxent_guiding_pdf(direction, guide.axis, guide.kappa) : 0.0;
    return (1.0 - guide.prob) * vndfPdf + guide.prob * guidePdf;
}

// ===========================================================================
// Diffuse Direction Sampling with MaxEnt MIS
// ===========================================================================

vec3 sampleDiffuseWithGuide(vec3 geometryNormal, vec3 shadingNormal,
    vec3 ro_o, GuideInfo guide, vec2 xi,
    out vec3 next_rd, out float guideWeight, out float sampledPdf) {
    sampledPdf = 0.0;
    bool useGuide = getRandom() < guide.prob;
    if (useGuide) {
        next_rd = sample_maxent_guiding(guide.axis, guide.kappa, xi);
    } else {
        next_rd = SampleUniformHemisphere(shadingNormal, xi);
    }

    float NoL = max(0.0, dot(shadingNormal, next_rd));
    float geometryNoL = dot(geometryNormal, next_rd);
    // 如果采样到了几何半球下方，直接裁切
    if (NoL <= 0.0 || geometryNoL <= 0.0) {
        guideWeight = 0.0;
        return vec3(0.0);
    }

    float pdfUniform = 1.0 / (2.0 * PI);
    float pdfMaxEnt = guide.prob > 0.0 ? maxent_guiding_pdf(next_rd, guide.axis, guide.kappa) : 0.0;
    float pdfMix = (1.0 - guide.prob) * pdfUniform + guide.prob * pdfMaxEnt;
    sampledPdf = pdfMix;
    guideWeight = (pdfMix > 1e-20) ? (pdfUniform / pdfMix) : 0.0;
    return vec3(guideWeight);
}

#endif // DIRT_RT_LIB_RT_RAYTRACE_GUIDING_GLSL
