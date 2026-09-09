#ifndef DENOISER_DIFFUSE_REPROJECTION_GLSL
#define DENOISER_DIFFUSE_REPROJECTION_GLSL

#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/math/statistics.glsl"

#ifndef MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE
#define MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE 1.0
#endif

#ifndef MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS
#define MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS 1.5
#endif

#ifndef MAXENT_TEMPORAL_GEOMETRY_EPSILON
#define MAXENT_TEMPORAL_GEOMETRY_EPSILON 1e-5
#endif

struct DiffuseTemporalFootprint {
    vec3 origin;
    vec3 geometryNormal;
    vec3 tangent;
    vec3 bitangent;
    vec2 plane0;
    vec2 plane1;
    vec2 plane2;
    vec2 plane3;
    float depthHalfExtent;
    float planeEdgeEpsilon;
};

float diffuseTemporalCross2(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

vec3 diffuseTemporalPrimaryRay(vec2 pixel, mat3 currentModelView,
        vec4 currentProjectionParams) {
    vec2 safeResolution = max(vec2(resolution_global), vec2(1.0));
    vec2 inverseScale = vec2(
        abs(currentProjectionParams.x) > 1e-8
            ? 1.0 / currentProjectionParams.x : 0.0,
        abs(currentProjectionParams.y) > 1e-8
            ? 1.0 / currentProjectionParams.y : 0.0);
    vec2 viewSlope = (pixel / safeResolution * 2.0 - 1.0
        + currentProjectionParams.zw) * inverseScale;
    return transpose(currentModelView)
        * normalize(vec3(viewSlope, -1.0));
}

bool diffuseTemporalPlanePoint(vec2 pixel, vec3 currentPosition,
        vec3 surfaceNormal, vec3 cameraDelta,
        mat3 currentModelView, vec4 currentProjectionParams,
        out vec3 historySpacePoint) {
    vec3 rayDirection = diffuseTemporalPrimaryRay(pixel,
        currentModelView, currentProjectionParams);
    float denominator = dot(surfaceNormal, rayDirection);
    if (abs(denominator) < 1e-6) return false;

    float rayDistance = dot(surfaceNormal, currentPosition) / denominator;
    if (!(rayDistance > 0.0) || isnan(rayDistance) || isinf(rayDistance))
        return false;

    historySpacePoint = rayDirection * rayDistance + cameraDelta;
    return !any(isnan(historySpacePoint))
        && !any(isinf(historySpacePoint));
}

bool buildDiffuseTemporalFootprint(uvec2 currentPixel,
        vec3 currentPosition, vec3 surfaceNormal, vec3 cameraDelta,
        mat3 currentModelView, vec4 currentProjectionParams,
        out DiffuseTemporalFootprint footprint) {
    float normalLengthSquared = dot(surfaceNormal, surfaceNormal);
    if (normalLengthSquared < 1e-8) return false;

    footprint.geometryNormal = surfaceNormal
        * inversesqrt(normalLengthSquared);
    if (footprint.geometryNormal.z < -0.999999) {
        footprint.tangent = vec3(0.0, -1.0, 0.0);
        footprint.bitangent = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + footprint.geometryNormal.z);
        float c = -footprint.geometryNormal.x
            * footprint.geometryNormal.y * a;
        footprint.tangent = vec3(
            1.0 - footprint.geometryNormal.x
                * footprint.geometryNormal.x * a,
            c, -footprint.geometryNormal.x);
        footprint.bitangent = vec3(c,
            1.0 - footprint.geometryNormal.y
                * footprint.geometryNormal.y * a,
            -footprint.geometryNormal.y);
    }

    float radius = max(
        float(MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS), 0.5);
    vec2 pixelMin = vec2(currentPixel) - radius;
    vec2 pixelMax = vec2(currentPixel) + radius;
    vec3 corner0, corner1, corner2, corner3;
    if (!diffuseTemporalPlanePoint(vec2(pixelMin.x, pixelMin.y),
            currentPosition, footprint.geometryNormal, cameraDelta,
            currentModelView, currentProjectionParams,
            corner0)) return false;
    if (!diffuseTemporalPlanePoint(vec2(pixelMax.x, pixelMin.y),
            currentPosition, footprint.geometryNormal, cameraDelta,
            currentModelView, currentProjectionParams,
            corner1)) return false;
    if (!diffuseTemporalPlanePoint(vec2(pixelMax.x, pixelMax.y),
            currentPosition, footprint.geometryNormal, cameraDelta,
            currentModelView, currentProjectionParams,
            corner2)) return false;
    if (!diffuseTemporalPlanePoint(vec2(pixelMin.x, pixelMax.y),
            currentPosition, footprint.geometryNormal, cameraDelta,
            currentModelView, currentProjectionParams,
            corner3)) return false;

    footprint.origin = currentPosition + cameraDelta;
    footprint.plane0 = vec2(
        dot(corner0 - footprint.origin, footprint.tangent),
        dot(corner0 - footprint.origin, footprint.bitangent));
    footprint.plane1 = vec2(
        dot(corner1 - footprint.origin, footprint.tangent),
        dot(corner1 - footprint.origin, footprint.bitangent));
    footprint.plane2 = vec2(
        dot(corner2 - footprint.origin, footprint.tangent),
        dot(corner2 - footprint.origin, footprint.bitangent));
    footprint.plane3 = vec2(
        dot(corner3 - footprint.origin, footprint.tangent),
        dot(corner3 - footprint.origin, footprint.bitangent));

    float footprintDiameter = max(
        length(footprint.plane2 - footprint.plane0),
        length(footprint.plane3 - footprint.plane1));
    footprint.depthHalfExtent = footprintDiameter
        * MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE;
    footprint.planeEdgeEpsilon = MAXENT_TEMPORAL_GEOMETRY_EPSILON
        * max(footprintDiameter, 1.0);
    return true;
}

bool diffuseTemporalFootprintContains(vec3 historyPosition,
        DiffuseTemporalFootprint footprint) {
    vec3 historyDelta = historyPosition - footprint.origin;
    if (abs(dot(historyDelta, footprint.geometryNormal))
            > footprint.depthHalfExtent) return false;

    vec2 planePoint = vec2(
        dot(historyDelta, footprint.tangent),
        dot(historyDelta, footprint.bitangent));
    float edge0 = diffuseTemporalCross2(
        footprint.plane1 - footprint.plane0,
        planePoint - footprint.plane0);
    float edge1 = diffuseTemporalCross2(
        footprint.plane2 - footprint.plane1,
        planePoint - footprint.plane1);
    float edge2 = diffuseTemporalCross2(
        footprint.plane3 - footprint.plane2,
        planePoint - footprint.plane2);
    float edge3 = diffuseTemporalCross2(
        footprint.plane0 - footprint.plane3,
        planePoint - footprint.plane3);
    float epsilon = footprint.planeEdgeEpsilon;
    bool positive = edge0 >= -epsilon && edge1 >= -epsilon
        && edge2 >= -epsilon && edge3 >= -epsilon;
    bool negative = edge0 <= epsilon && edge1 <= epsilon
        && edge2 <= epsilon && edge3 <= epsilon;
    return positive || negative;
}

float diffuseTemporalSurfaceSampleScale(vec3 historyPosition,
        DiffuseTemporalFootprint footprint, vec3 currentPosition,
        vec3 cameraDelta) {
    float previousDistanceSquared = dot(historyPosition, historyPosition);
    vec3 historyCurrentSpace = historyPosition - cameraDelta;
    float currentDistanceSquared = dot(
        historyCurrentSpace, historyCurrentSpace);
    float centerDistanceSquared = dot(currentPosition, currentPosition);
    if (!(previousDistanceSquared > 1e-8)
            || !(currentDistanceSquared > 1e-8)
            || !(centerDistanceSquared > 1e-8)) return 0.0;

    float centerNoV = dot(currentPosition
        * inversesqrt(centerDistanceSquared), footprint.geometryNormal);
    float historyNoV = dot(historyCurrentSpace
        * inversesqrt(currentDistanceSquared), footprint.geometryNormal);
    return clamp(currentDistanceSquared * abs(historyNoV)
        / max(previousDistanceSquared * abs(centerNoV), 1e-3),
        0.0, 1.0);
}

void invalidatePreparedDiffuseHistory(uvec2 pixel, uint currentFrameId) {
    writeDiffuseDenoisedReprojectionInvalidForFrame(pixel, currentFrameId);
    diffuseBuffer.data[addr(DIF_N_SWAP, pixel)] = uvec4(0u);
}

// ray1 owns the final diffuse surface after transmissive-background
// substitution. One accepted tap set reconstructs both raw and final-denoised
// histories. The current denoised parity and DIF_N_SWAP are then shared by path
// guiding and temporal; resolve later overwrites both with current history.
void prepareDiffuseDenoisedSurfaceReprojection(uvec2 currentPixel,
        vec3 currentPosition, vec3 currentGeometryNormal,
        vec3 cameraDelta, float motionValid, uint currentFrameId,
        mat4 historyViewProjection, mat3 currentModelView,
        vec4 currentProjectionParams) {
    if (motionValid < 0.5) {
        invalidatePreparedDiffuseHistory(currentPixel, currentFrameId);
        return;
    }

    vec4 previousClip = historyViewProjection
        * vec4(currentPosition + cameraDelta, 1.0);
    if (previousClip.w <= 1e-8 || any(isnan(previousClip))
            || any(isinf(previousClip))) {
        invalidatePreparedDiffuseHistory(currentPixel, currentFrameId);
        return;
    }
    vec3 previousScreen = previousClip.xyz / previousClip.w
        * 0.5 + 0.5;
    if (previousScreen.z < 0.0 || previousScreen.z > 1.0
            || any(lessThan(previousScreen.xy, vec2(0.0)))
            || any(greaterThan(previousScreen.xy, vec2(1.0)))) {
        invalidatePreparedDiffuseHistory(currentPixel, currentFrameId);
        return;
    }

    DiffuseTemporalFootprint footprint;
    if (!buildDiffuseTemporalFootprint(currentPixel, currentPosition,
            currentGeometryNormal, cameraDelta, currentModelView,
            currentProjectionParams, footprint)) {
        invalidatePreparedDiffuseHistory(currentPixel, currentFrameId);
        return;
    }

    vec2 previousCoordinate = previousScreen.xy
        * vec2(resolution_global);
    ivec2 previousBase = ivec2(floor(previousCoordinate));
    vec2 previousFraction = fract(previousCoordinate);
    float totalKernelWeight = 0.0;
    float validKernelWeight = 0.0;
    float sumWeightOverRootSamples = 0.0;
    float weightedMeanY2 = 0.0;
    MaxEntEncoding weightedRaw = init_maxent();
    vec4 weightedDenoisedMoment = vec4(0.0);
    vec2 weightedDenoisedChroma = vec2(0.0);
    DenoiserEstimatorVarianceAccumulator uncertainty =
        denoiserBeginEstimatorVariance();

    for (int tapIndex = 0; tapIndex < 4; ++tapIndex) {
        ivec2 sampleTexel = previousBase
            + ivec2(tapIndex & 1, tapIndex >> 1);
        float weightX = (tapIndex & 1) == 0
            ? 1.0 - previousFraction.x : previousFraction.x;
        float weightY = (tapIndex & 2) == 0
            ? 1.0 - previousFraction.y : previousFraction.y;
        float reconstructionWeight = weightX * weightY;
        totalKernelWeight += reconstructionWeight;
        if (reconstructionWeight <= 0.0) continue;
        if (any(lessThan(sampleTexel, ivec2(0)))
                || any(greaterThanEqual(sampleTexel,
                    ivec2(resolution_global)))) continue;

        uvec2 historyTexel = uvec2(sampleTexel);
        uvec4 historyWords = diffuseBuffer.data[
            addr(DIF_N_HIST, historyTexel)];
        vec2 historyMetadata = unpackHalf2x16(historyWords.w);
        if (!statisticsValidEffectiveSampleCount(historyMetadata.x)
                || historyMetadata.y < 0.0
                || isnan(historyMetadata.y)
                || isinf(historyMetadata.y)) continue;
        vec4 historyMoment = vec4(unpackHalf2x16(historyWords.x),
            unpackHalf2x16(historyWords.y));
        vec2 historyChroma = unpackHalf2x16(historyWords.z);
        if (any(isnan(historyMoment)) || any(isinf(historyMoment))
                || any(isnan(historyChroma))
                || any(isinf(historyChroma))) continue;

        uvec4 geometryWords = readDiffuseHistGeoRawForFrame(
            historyTexel, currentFrameId);
        float geometryHistoryWeight;
        if (!unpackDiffusePreviousHistoryWeightForFrame(geometryWords.w,
                currentFrameId,
                geometryHistoryWeight)) continue;
        float historyDistance = uintBitsToFloat(geometryWords.x);
        vec3 historyPosition = decodeDiffuseHistoryNormalU(geometryWords.y)
            * historyDistance;
        if (!diffuseTemporalFootprintContains(historyPosition,
                footprint)) continue;

        uvec4 denoisedWords = readDiffuseDenoisedPreviousRawForFrame(
            historyTexel, currentFrameId);
        if (!denoiserSpatialSignalWordsValid(denoisedWords)) continue;
        vec2 denoisedMetadata = unpackHalf2x16(denoisedWords.w);
        if (!statisticsValidEffectiveSampleCount(-denoisedMetadata.y))
            continue;

        vec4 tapMoment = vec4(unpackHalf2x16(denoisedWords.x),
            unpackHalf2x16(denoisedWords.y));
        vec2 tapChroma = unpackHalf2x16(denoisedWords.z);
        if (any(isnan(tapMoment)) || any(isinf(tapMoment))
                || any(isnan(tapChroma)) || any(isinf(tapChroma)))
            continue;

        float tapSigma = denoisedMetadata.x;
        float surfaceSampleScale = diffuseTemporalSurfaceSampleScale(
            historyPosition, footprint, currentPosition, cameraDelta);
        float correctedTapSamples = max(
            historyMetadata.x * surfaceSampleScale, 1.0);
        if (denoiserSigmaKnown(tapSigma))
            tapSigma /= sqrt(clamp(surfaceSampleScale, 1e-4, 1.0));
        MaxEntEncoding tapRaw;
        tapRaw.maxEntY = historyMoment;
        tapRaw.CoCg = historyChroma;
        accumulate_maxent(weightedRaw, tapRaw, reconstructionWeight);
        weightedMeanY2 += reconstructionWeight
            * historyMetadata.y * historyMetadata.y;
        sumWeightOverRootSamples += reconstructionWeight
            * inversesqrt(correctedTapSamples);
        weightedDenoisedMoment += reconstructionWeight * tapMoment;
        weightedDenoisedChroma += reconstructionWeight * tapChroma;
        denoiserAccumulateEstimatorVariance(
            uncertainty, tapSigma, reconstructionWeight);
        validKernelWeight += reconstructionWeight;
    }

    if (validKernelWeight < 1e-5) {
        invalidatePreparedDiffuseHistory(currentPixel, currentFrameId);
        return;
    }
    float inverseWeight = 1.0 / validKernelWeight;
    float historyEffectiveSamples =
        statisticsReconstructedEffectiveSampleCount(
            validKernelWeight, sumWeightOverRootSamples);
    if (!statisticsValidEffectiveSampleCount(historyEffectiveSamples)) {
        invalidatePreparedDiffuseHistory(currentPixel, currentFrameId);
        return;
    }
    float validCoverage = clamp(validKernelWeight
        / max(totalKernelWeight, 1e-8), 0.0, 1.0);
    MaxEntEncoding reprojectedRaw = scale_maxent(
        weightedRaw, inverseWeight);
    writeDiffuseSwap(currentPixel, reprojectedRaw,
        historyEffectiveSamples,
        sqrt(weightedMeanY2 * inverseWeight));
    writeDiffuseDenoisedReprojectionForFrame(currentPixel, currentFrameId,
        weightedDenoisedMoment * inverseWeight,
        weightedDenoisedChroma * inverseWeight,
        denoiserResolveEstimatorSigma(uncertainty, 1.0),
        validCoverage);
}

#endif
