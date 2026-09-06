#version 430 core

// Purpose: reproject diffuse raw/filtered histories and build the provisional temporal proposal.
// Dispatch: 16x8.
// Reads: Raw RT diffuse signal, previous raw/geometry history, previous filtered history, motion.
// Writes: diffuse swap proposal, filtered-history reprojection scratch, colorimg6 raw reprojection.
// Persistent side effects: none; resolve.glsl owns the accepted history update.
// Invalid representation: zero raw state plus invalid filtered reprojection metadata.

// A 128-thread group admits another resident block when registers are the
// limiting launch resource, while preserving 16-wide horizontal coherence.
layout(local_size_x = 16, local_size_y = 8) in;
layout(rgba32ui) uniform writeonly uimage2D colorimg6;

#define DIFFUSE_BUFFER_MIN
#define PREV_DIFFUSE_BUFFER

#include "/lib/constants.glsl"
#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/denoiser/internal_constants.glsl"
#include "/lib/lighting/denoiser/signal.glsl"
#include "/lib/math/statistics.glsl"

uniform vec2 resolution;

#ifndef MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE
#define MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE 1.0
#endif

#ifndef MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS
#define MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS 1.5
#endif

#ifndef MAXENT_TEMPORAL_GEOMETRY_EPSILON
#define MAXENT_TEMPORAL_GEOMETRY_EPSILON 1e-5
#endif

// Reprojection footprints.

struct TemporalFootprint {
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

vec3 prevScreenPos;
vec3 cameraDelta;
vec3 geometryNormal;
vec3 currentPosition;

MaxEntEncoding outputMaxEnt;
float outputMeanY2;
float outputEffectiveSamples = 0.0;

uvec4 currentLightPacked() {
    return readDiffuseLightRTRaw(uvec2(gl_GlobalInvocationID.xy));
}

void unpackCurrentLight(out MaxEntEncoding maxEnt, out float meanY2) {
    uvec4 packedLight = currentLightPacked();
    maxEnt.maxEntY = vec4(unpackHalf2x16(packedLight.x), unpackHalf2x16(packedLight.y));
    maxEnt.CoCg = unpackHalf2x16(packedLight.z);
    float rootMeanY2 = unpackHalf2x16(packedLight.w).y;
    meanY2 = rootMeanY2 * rootMeanY2;
}

// Footprint geometry.

float diffuseTemporalCross2(vec2 a, vec2 b) { return a.x * b.y - a.y * b.x; }

vec3 diffuseTemporalPrimaryRay(vec2 pixel) {
    vec2 safeResolution = max(vec2(resolution_global), vec2(1.0));
    vec2 inverseScale = vec2(abs(rtProjectionParams.x) > 1e-8 ? 1.0 / rtProjectionParams.x : 0.0,
        abs(rtProjectionParams.y) > 1e-8 ? 1.0 / rtProjectionParams.y : 0.0);
    vec2 viewSlope = (pixel / safeResolution * 2.0 - 1.0 + rtProjectionParams.zw) * inverseScale;
    return transpose(mat3(rtModelView)) * normalize(vec3(viewSlope, -1.0));
}

bool diffuseTemporalPlanePoint(vec2 pixel, vec3 currentPos, vec3 surfaceNormal, vec3 camDelta, out vec3 historySpacePoint) {
    vec3 rayDirection = diffuseTemporalPrimaryRay(pixel);
    float denominator = dot(surfaceNormal, rayDirection);
    if (abs(denominator) < 1e-6) return false;

    float rayDistance = dot(surfaceNormal, currentPos) / denominator;
    if (!(rayDistance > 0.0) || isnan(rayDistance) || isinf(rayDistance)) return false;

    historySpacePoint = rayDirection * rayDistance + camDelta;
    return !any(isnan(historySpacePoint)) && !any(isinf(historySpacePoint));
}

bool buildDiffuseTemporalFootprint(vec3 currentPos, vec3 surfaceNormal, vec3 camDelta, out TemporalFootprint fp) {
    float normalLengthSquared = dot(surfaceNormal, surfaceNormal);
    if (normalLengthSquared < 1e-8) return false;

    fp.geometryNormal = surfaceNormal * inversesqrt(normalLengthSquared);
    if (fp.geometryNormal.z < -0.999999) {
        fp.tangent = vec3(0.0, -1.0, 0.0);
        fp.bitangent = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + fp.geometryNormal.z);
        float c = -fp.geometryNormal.x * fp.geometryNormal.y * a;
        fp.tangent = vec3(1.0 - fp.geometryNormal.x * fp.geometryNormal.x * a, c, -fp.geometryNormal.x);
        fp.bitangent = vec3(c, 1.0 - fp.geometryNormal.y * fp.geometryNormal.y * a, -fp.geometryNormal.y);
    }

    float radius = max(float(MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS), 0.5);
    vec2 currentPixel = vec2(gl_GlobalInvocationID.xy);
    vec2 pixelMin = currentPixel - radius;
    vec2 pixelMax = currentPixel + radius;
    vec3 corner0, corner1, corner2, corner3;
    if (!diffuseTemporalPlanePoint(vec2(pixelMin.x, pixelMin.y), currentPos, fp.geometryNormal, camDelta, corner0)) return false;
    if (!diffuseTemporalPlanePoint(vec2(pixelMax.x, pixelMin.y), currentPos, fp.geometryNormal, camDelta, corner1)) return false;
    if (!diffuseTemporalPlanePoint(vec2(pixelMax.x, pixelMax.y), currentPos, fp.geometryNormal, camDelta, corner2)) return false;
    if (!diffuseTemporalPlanePoint(vec2(pixelMin.x, pixelMax.y), currentPos, fp.geometryNormal, camDelta, corner3)) return false;

    fp.origin = currentPos + camDelta;
    fp.plane0 = vec2(dot(corner0 - fp.origin, fp.tangent), dot(corner0 - fp.origin, fp.bitangent));
    fp.plane1 = vec2(dot(corner1 - fp.origin, fp.tangent), dot(corner1 - fp.origin, fp.bitangent));
    fp.plane2 = vec2(dot(corner2 - fp.origin, fp.tangent), dot(corner2 - fp.origin, fp.bitangent));
    fp.plane3 = vec2(dot(corner3 - fp.origin, fp.tangent), dot(corner3 - fp.origin, fp.bitangent));

    float footprintDiameter = max(length(fp.plane2 - fp.plane0), length(fp.plane3 - fp.plane1));
    fp.depthHalfExtent = footprintDiameter * MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE;
    fp.planeEdgeEpsilon = MAXENT_TEMPORAL_GEOMETRY_EPSILON * max(footprintDiameter, 1.0);
    return true;
}

bool diffuseTemporalFootprintContains(vec3 historyPosition, TemporalFootprint fp) {
    vec3 historyDelta = historyPosition - fp.origin;
    if (abs(dot(historyDelta, fp.geometryNormal)) > fp.depthHalfExtent) return false;

    vec2 planePoint = vec2(dot(historyDelta, fp.tangent), dot(historyDelta, fp.bitangent));
    float edge0 = diffuseTemporalCross2(fp.plane1 - fp.plane0, planePoint - fp.plane0);
    float edge1 = diffuseTemporalCross2(fp.plane2 - fp.plane1, planePoint - fp.plane1);
    float edge2 = diffuseTemporalCross2(fp.plane3 - fp.plane2, planePoint - fp.plane2);
    float edge3 = diffuseTemporalCross2(fp.plane0 - fp.plane3, planePoint - fp.plane3);
    float epsilon = fp.planeEdgeEpsilon;
    bool positive = edge0 >= -epsilon && edge1 >= -epsilon && edge2 >= -epsilon && edge3 >= -epsilon;
    bool negative = edge0 <= epsilon && edge1 <= epsilon && edge2 <= epsilon && edge3 <= epsilon;
    return positive || negative;
}

float diffuseTemporalSurfaceSampleScale(vec3 historyPosition, TemporalFootprint fp) {
    float previousDistanceSquared = dot(historyPosition, historyPosition);
    vec3 historyCurrentSpace = historyPosition - cameraDelta;
    float currentDistanceSquared = dot(historyCurrentSpace, historyCurrentSpace);
    float centerDistanceSquared = dot(currentPosition, currentPosition);
    if (!(previousDistanceSquared > 1e-8) || !(currentDistanceSquared > 1e-8) || !(centerDistanceSquared > 1e-8)) return 0.0;

    float centerNoV = dot(currentPosition * inversesqrt(centerDistanceSquared), fp.geometryNormal);
    float historyNoV = dot(historyCurrentSpace * inversesqrt(currentDistanceSquared), fp.geometryNormal);
    return clamp(currentDistanceSquared * abs(historyNoV) / max(previousDistanceSquared * abs(centerNoV), 1e-3), 0.0, 1.0);
}

// Temporal proposal construction.

void resetToCurrentSample() {
    outputEffectiveSamples = 1.0;
    unpackCurrentLight(outputMaxEnt, outputMeanY2);
}

void publishDenoisedReprojection(vec4 weightedMaxEntY, vec2 weightedCoCg,
        DenoiserEstimatorVarianceAccumulator uncertainty, float acceptedWeight, float validCoverage) {
    float inverseWeight = 1.0 / acceptedWeight;
    float standardDeviation = denoiserResolveEstimatorSigma(uncertainty, 1.0);
    writeDiffuseDenoisedReprojection(
        gl_GlobalInvocationID.xy,
        weightedMaxEntY * inverseWeight,
        weightedCoCg * inverseWeight,
        standardDeviation,
        1.0,
        validCoverage
    );
}

void buildDiffuseTemporalProposal() {
    if (any(lessThan(prevScreenPos, vec3(0.0))) ||
            any(greaterThan(prevScreenPos, vec3(1.0)))) {
        writeDiffuseDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    TemporalFootprint fp;
    if (!buildDiffuseTemporalFootprint(currentPosition, geometryNormal, cameraDelta, fp)) {
        writeDiffuseDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    vec2 prevCoord = prevScreenPos.xy * vec2(resolution_global);
    ivec2 prevBase = ivec2(floor(prevCoord));
    vec2 prevFraction = fract(prevCoord);

    MaxEntEncoding accumMaxEnt = init_maxent();
    float totalKernelWeight = 0.0;
    float validKernelWeight = 0.0;
    float sumWeightOverRootSamples = 0.0;
    float weightedMeanY2 = 0.0;
    vec4 denoisedMaxEntY = vec4(0.0);
    vec2 denoisedCoCg = vec2(0.0);
    DenoiserEstimatorVarianceAccumulator denoisedUncertainty = denoiserBeginEstimatorVariance();

    for (int tapIndex = 0; tapIndex < 4; tapIndex++) {
        ivec2 sampleTexel = prevBase + ivec2(tapIndex & 1, tapIndex >> 1);
        float weightX = (tapIndex & 1) == 0 ? 1.0 - prevFraction.x : prevFraction.x;
        float weightY = (tapIndex & 2) == 0 ? 1.0 - prevFraction.y : prevFraction.y;
        float reconstructionWeight = weightX * weightY;
        totalKernelWeight += reconstructionWeight;

        if (any(lessThan(sampleTexel, ivec2(0))) || any(greaterThanEqual(sampleTexel, ivec2(resolution_global)))) continue;

        uvec2 historyTexel = uvec2(sampleTexel);
        uvec4 packedHistory = diffuseBuffer.data[addr(DIF_N_HIST, historyTexel)];
        vec2 historyMeta = unpackHalf2x16(packedHistory.w);

        if (!statisticsValidEffectiveSampleCount(historyMeta.x) || historyMeta.y < 0.0
                || isnan(historyMeta.y) || isinf(historyMeta.y)) continue;

        uvec4 packedGeometry = readDiffuseHistGeoRaw(historyTexel);

        float geometryHistoryWeight;
        if (!unpackDiffusePreviousHistoryWeight(packedGeometry.w, geometryHistoryWeight)) continue;

        float historyDistance = uintBitsToFloat(packedGeometry.x);
        vec3 historyPosition = decodeDiffuseHistoryNormalU(packedGeometry.y) * historyDistance;
        if (!diffuseTemporalFootprintContains(historyPosition, fp)) continue;

        uvec4 denoisedWords = readDiffuseDenoisedPreviousRaw(historyTexel);
        if (!denoiserSpatialSignalWordsValid(denoisedWords)) continue;

        vec2 denoisedMetadata = unpackHalf2x16(denoisedWords.w);
        float tapDenoisedEffectiveSamples = -denoisedMetadata.y;
        if (!statisticsValidEffectiveSampleCount(tapDenoisedEffectiveSamples)) continue;

        float tapSamples = historyMeta.x;
        // A surviving estimator still owns one Kish sample; the Jacobian discards only its accumulated excess.
        float surfaceSampleScale = diffuseTemporalSurfaceSampleScale(historyPosition, fp);
        float correctedTapSamples = max(tapSamples * surfaceSampleScale, 1.0);
        float tapWeight = reconstructionWeight;

        MaxEntEncoding tapMaxEnt;
        tapMaxEnt.maxEntY = vec4(unpackHalf2x16(packedHistory.x), unpackHalf2x16(packedHistory.y));
        tapMaxEnt.CoCg = unpackHalf2x16(packedHistory.z);
        if (any(isnan(tapMaxEnt.maxEntY)) || any(isinf(tapMaxEnt.maxEntY))
                || any(isnan(tapMaxEnt.CoCg)) || any(isinf(tapMaxEnt.CoCg))) continue;

        vec4 tapDenoisedMaxEntY = vec4(unpackHalf2x16(denoisedWords.x), unpackHalf2x16(denoisedWords.y));
        vec2 tapDenoisedChroma = unpackHalf2x16(denoisedWords.z);
        if (any(isnan(tapDenoisedMaxEntY)) || any(isinf(tapDenoisedMaxEntY))
                || any(isnan(tapDenoisedChroma)) || any(isinf(tapDenoisedChroma))) continue;
        float tapDenoisedStdDev = denoisedMetadata.x;
        // Stored sigma is estimator uncertainty. Inflate for footprint loss.
        if (denoiserSigmaKnown(tapDenoisedStdDev))
            tapDenoisedStdDev /= sqrt(clamp(surfaceSampleScale, 1e-4, 1.0));
        denoisedMaxEntY += tapWeight * tapDenoisedMaxEntY;
        denoisedCoCg += tapWeight * tapDenoisedChroma;
        denoiserAccumulateEstimatorVariance(denoisedUncertainty, tapDenoisedStdDev, tapWeight);

        accumulate_maxent(accumMaxEnt, tapMaxEnt, tapWeight);
        float tapMeanY2 = historyMeta.y * historyMeta.y;
        weightedMeanY2 += tapWeight * tapMeanY2;
        validKernelWeight += tapWeight;
        sumWeightOverRootSamples += tapWeight * inversesqrt(correctedTapSamples);
    }

    if (validKernelWeight < 1e-5) {
        writeDiffuseDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    float inverseKernelWeight = 1.0 / validKernelWeight;
    MaxEntEncoding histMaxEnt = scale_maxent(
        accumMaxEnt, inverseKernelWeight);

    float historyEffectiveSamples = statisticsReconstructedEffectiveSampleCount(
        validKernelWeight, sumWeightOverRootSamples);

    if (!statisticsValidEffectiveSampleCount(historyEffectiveSamples)) {
        writeDiffuseDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    // Raw moments use normalized bilinear weights; the Jacobian reduces raw
    // N_eff and inflates filtered estimator variance. Overlap uses correlation=1.
    float histMeanY2 = weightedMeanY2 * inverseKernelWeight;

    MaxEntEncoding currentMaxEnt;
    float currentMeanY2;
    unpackCurrentLight(currentMaxEnt, currentMeanY2);
    float validFootprintCoverage = clamp(validKernelWeight / max(totalKernelWeight, 1e-8), 0.0, 1.0);
    publishDenoisedReprojection(
        denoisedMaxEntY,
        denoisedCoCg,
        denoisedUncertainty,
        validKernelWeight,
        validFootprintCoverage
    );

    // Build the provisional estimator requested by the late-resolve pipeline:
    // one independent current observation is provisionally inserted into the
    // reprojected history.  Variance preparation must see this growing Kish
    // estimator; feeding it Neff=1 every frame permanently locks the spatial
    // denoiser in its blurry cold-start regime.
    // The provisional branch uses the configured fixed weight so variance preparation and A-Trous see a stable
    // estimator. Resolve evaluates the denoised response and corrects forward with the independent-current branch.
    float proposalAlpha = clamp(
        float(MAXENT_TEMPORAL_FIXED_ALPHA), 0.0, 1.0);
    outputEffectiveSamples = statisticsKishUpdateEffectiveSampleCount(historyEffectiveSamples, proposalAlpha);
    outputMeanY2 = mix(histMeanY2, currentMeanY2, proposalAlpha);
    outputMaxEnt = mix_maxent(histMaxEnt, currentMaxEnt, proposalAlpha);

    // Keep the raw reprojected temporal estimator alive until final resolve.
    // DIF_N_SWAP/colortex4 contains the provisional estimator above; resolve
    // later commits the estimator selected by the actual response alpha.
    imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy),
        packDiffuseTemporalState(histMaxEnt, historyEffectiveSamples,
            sqrt(histMeanY2)));
}

// Pass entry.

void main() {
    uvec2 pix = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pix, uvec2(resolution)))) return;

    float infoDistance;
    readDiffusePrimaryGeometry(pix, currentPosition, infoDistance);

    if (infoDistance < -0.5) {
        diffuseBuffer.data[addr(DIF_N_SWAP, pix)] = uvec4(0u);
        writeDiffuseDenoisedReprojectionInvalid(pix);
        imageStore(colorimg6, ivec2(pix), uvec4(0u));
        return;
    }

    geometryNormal = readDiffuseGeometryNormal(pix);
    cameraDelta = camPos - prevRaytracingCamPos;

    vec3 surfaceMotion;
    float motionValid;
    readDiffuseMotion(pix, surfaceMotion, motionValid);

    if (motionValid < 0.5) {
        resetToCurrentSample();
        writeDiffuseSwap(pix, outputMaxEnt, outputEffectiveSamples,
            sqrt(outputMeanY2));
        writeDiffuseDenoisedReprojectionInvalid(pix);
        imageStore(colorimg6, ivec2(pix), uvec4(0u));
        return;
    }

    cameraDelta -= surfaceMotion;

    vec4 clipPos = rtPrevViewProjection * vec4(currentPosition + cameraDelta, 1.0);
    prevScreenPos = abs(clipPos.w) > 1e-6
        ? (clipPos.xyz / clipPos.w) * 0.5 + 0.5 : vec3(-1.0);

    buildDiffuseTemporalProposal();
    writeDiffuseSwap(pix, outputMaxEnt, outputEffectiveSamples,
        sqrt(outputMeanY2));
}
