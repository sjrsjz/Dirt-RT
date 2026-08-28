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

struct TemporalFootprintFast {
    vec3 origin;
    vec3 geometryNormal;
    float depthHalfExtent;
};

struct TemporalJacobianKernel {
    mat2 currentFromPrevious;
    mat2 geometryCurrentFromPrevious;
    vec2 previousExtent;
    float currentRadius;
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

bool buildTemporalFootprintFast(vec3 currentPos, vec3 surfaceNormal,
    vec3 camDelta, out TemporalFootprintFast fp) {
    float normalLengthSquared = dot(surfaceNormal, surfaceNormal);
    if (normalLengthSquared < 1e-8) return false;

    fp.geometryNormal = surfaceNormal * inversesqrt(normalLengthSquared);
    fp.origin = currentPos + camDelta;

    float positionLengthSquared = dot(currentPos, currentPos);
    float positionLength = sqrt(max(positionLengthSquared, 1e-8));
    float noV = abs(dot(currentPos, fp.geometryNormal)) / positionLength;
    float pixelWorldSize = max(positionLength / max(float(resolution_global.y), 1.0), 1e-4);

    fp.depthHalfExtent = max(
            4.0 * MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS *
                pixelWorldSize * MAXENT_DIFFUSE_TEMPORAL_DEPTH_SCALE / max(noV, 0.05),
            1e-5
        );
    return true;
}

vec3 diffuseTemporalPrimaryRay(vec2 pixel) {
    vec2 safeResolution = max(vec2(resolution_global), vec2(1.0));
    vec2 inverseScale = vec2(abs(rtProjectionParams.x) > 1e-8 ? 1.0 / rtProjectionParams.x : 0.0,
        abs(rtProjectionParams.y) > 1e-8 ? 1.0 / rtProjectionParams.y : 0.0);
    vec2 viewSlope = (pixel / safeResolution * 2.0 - 1.0 + rtProjectionParams.zw) * inverseScale;
    return transpose(mat3(rtModelView)) * normalize(vec3(viewSlope, -1.0));
}

bool projectDiffuseTemporalPlaneSample(vec2 pixel, vec3 currentPos, vec3 surfaceNormal,
        vec3 camDelta, out vec2 previousPixel) {
    vec3 rayDirection = diffuseTemporalPrimaryRay(pixel);
    float denominator = dot(surfaceNormal, rayDirection);
    if (abs(denominator) < 1e-6) return false;

    float rayDistance = dot(surfaceNormal, currentPos) / denominator;
    if (!(rayDistance > 0.0) || isnan(rayDistance) || isinf(rayDistance)) return false;

    vec4 clip = rtPrevViewProjection * vec4(rayDirection * rayDistance + camDelta, 1.0);
    if (!(clip.w > 1e-7) || isinf(clip.w)) return false;

    previousPixel = (clip.xy / clip.w * 0.5 + 0.5) * vec2(resolution_global);
    return !any(isnan(previousPixel)) && !any(isinf(previousPixel));
}

bool regularizeDiffuseTemporalJacobian(mat2 inputJacobian, out mat2 outputJacobian) {
    vec2 columnX = inputJacobian[0];
    vec2 columnY = inputJacobian[1];
    float gramXX = dot(columnX, columnX);
    float gramXY = dot(columnX, columnY);
    float gramYY = dot(columnY, columnY);
    float discriminant = sqrt(max((gramXX - gramYY) * (gramXX - gramYY) + 4.0 * gramXY * gramXY, 0.0));
    float eigenvalueMax = 0.5 * (gramXX + gramYY + discriminant);
    float eigenvalueMin = 0.5 * (gramXX + gramYY - discriminant);
    if (!(eigenvalueMax > 1e-8) || eigenvalueMin < -1e-5 || isnan(eigenvalueMax) || isinf(eigenvalueMax)) return false;

    vec2 rightMax = abs(gramXY) > 1e-6 ? normalize(vec2(gramXY, eigenvalueMax - gramXX))
        : (gramXX >= gramYY ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 rightMin = vec2(-rightMax.y, rightMax.x);
    float singularMax = sqrt(max(eigenvalueMax, 1e-8));
    float singularMin = sqrt(max(eigenvalueMin, 1e-8));
    vec2 leftMax = inputJacobian * rightMax / singularMax;
    vec2 leftMin = inputJacobian * rightMin / singularMin;
    if (dot(leftMax, leftMax) < 1e-8) return false;

    leftMax = normalize(leftMax);
    leftMin -= leftMax * dot(leftMax, leftMin);
    if (dot(leftMin, leftMin) < 1e-8) {
        float orientation = determinant(inputJacobian) < 0.0 ? -1.0 : 1.0;
        leftMin = orientation * vec2(-leftMax.y, leftMax.x);
    } else {
        leftMin = normalize(leftMin);
    }

    singularMax = clamp(singularMax, 1.0, MAXENT_DIFFUSE_TEMPORAL_MAX_JACOBIAN_STRETCH);
    singularMin = clamp(singularMin, 1.0, MAXENT_DIFFUSE_TEMPORAL_MAX_JACOBIAN_STRETCH);
    outputJacobian = mat2(leftMax * singularMax * rightMax.x + leftMin * singularMin * rightMin.x,
        leftMax * singularMax * rightMax.y + leftMin * singularMin * rightMin.y);
    return !any(isnan(outputJacobian[0])) && !any(isnan(outputJacobian[1]))
        && !any(isinf(outputJacobian[0])) && !any(isinf(outputJacobian[1]));
}

bool buildDiffuseTemporalJacobianKernel(vec3 currentPos, vec3 surfaceNormal, vec3 camDelta,
        out TemporalJacobianKernel kernel) {
    vec2 currentPixel = vec2(gl_GlobalInvocationID.xy);
    vec2 previousLeft, previousRight, previousDown, previousUp;
    if (!projectDiffuseTemporalPlaneSample(currentPixel - vec2(1.0, 0.0), currentPos, surfaceNormal, camDelta, previousLeft)) return false;
    if (!projectDiffuseTemporalPlaneSample(currentPixel + vec2(1.0, 0.0), currentPos, surfaceNormal, camDelta, previousRight)) return false;
    if (!projectDiffuseTemporalPlaneSample(currentPixel - vec2(0.0, 1.0), currentPos, surfaceNormal, camDelta, previousDown)) return false;
    if (!projectDiffuseTemporalPlaneSample(currentPixel + vec2(0.0, 1.0), currentPos, surfaceNormal, camDelta, previousUp)) return false;

    mat2 exactPreviousFromCurrent = mat2(0.5 * (previousRight - previousLeft), 0.5 * (previousUp - previousDown));
    float exactDeterminant = determinant(exactPreviousFromCurrent);
    if (abs(exactDeterminant) < 1e-6 || isnan(exactDeterminant) || isinf(exactDeterminant)) return false;

    mat2 reconstructionPreviousFromCurrent;
    if (!regularizeDiffuseTemporalJacobian(exactPreviousFromCurrent, reconstructionPreviousFromCurrent)) return false;

    kernel.currentFromPrevious = inverse(reconstructionPreviousFromCurrent);
    kernel.geometryCurrentFromPrevious = inverse(exactPreviousFromCurrent);
    kernel.currentRadius = max(float(MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS), 1.0);
    kernel.previousExtent = kernel.currentRadius * (abs(reconstructionPreviousFromCurrent[0]) + abs(reconstructionPreviousFromCurrent[1]));
    return !any(isnan(kernel.previousExtent)) && !any(isinf(kernel.previousExtent));
}

bool strictHistoryGeometryTestFast(vec3 historyPosition, TemporalFootprintFast fp,
        vec2 expectedCurrentPixel, vec3 camDelta) {
    vec3 historyDelta = historyPosition - fp.origin;
    if (abs(dot(historyDelta, fp.geometryNormal)) > fp.depthHalfExtent) return false;

    vec3 historyCurrentSpace = historyPosition - camDelta;
    vec4 clip = rtViewProjection * vec4(historyCurrentSpace, 1.0);
    if (!(clip.w > 1e-7) || isinf(clip.w)) return false;

    vec2 projectedPixel = (clip.xy / clip.w * 0.5 + 0.5) * vec2(resolution_global);
    vec2 extent = vec2(MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS + MAXENT_TEMPORAL_GEOMETRY_EPSILON);

    return all(lessThanEqual(abs(projectedPixel - expectedCurrentPixel), extent));
}

// Temporal proposal construction.

void resetToCurrentSample() {
    outputEffectiveSamples = 1.0;
    unpackCurrentLight(outputMaxEnt, outputMeanY2);
}

void publishDenoisedReprojection(vec4 weightedMaxEntY, vec2 weightedCoCg,
    float squaredWeightVarianceSum, float weightedStdDevSum,
    float acceptedWeight, float validCoverage) {
    float inverseWeight = 1.0 / acceptedWeight;
    float standardDeviation = statisticsWeightedMeanStandardDeviation(
        squaredWeightVarianceSum, weightedStdDevSum, inverseWeight,
        MAXENT_TEMPORAL_REPROJECTION_CORRELATION);
    writeDiffuseDenoisedReprojection(
        gl_GlobalInvocationID.xy,
        weightedMaxEntY * inverseWeight,
        weightedCoCg * inverseWeight,
        standardDeviation,
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

    TemporalFootprintFast fp;
    if (!buildTemporalFootprintFast(currentPosition, geometryNormal, cameraDelta, fp)) {
        writeDiffuseDenoisedReprojectionInvalid(gl_GlobalInvocationID.xy);
        resetToCurrentSample();
        imageStore(colorimg6, ivec2(gl_GlobalInvocationID.xy), uvec4(0u));
        return;
    }

    vec2 prevCoord = prevScreenPos.xy * vec2(resolution_global);
    TemporalJacobianKernel kernel;
    if (!buildDiffuseTemporalJacobianKernel(currentPosition, fp.geometryNormal, cameraDelta, kernel)) {
        kernel.currentFromPrevious = mat2(1.0);
        kernel.geometryCurrentFromPrevious = mat2(1.0);
        kernel.currentRadius = max(float(MAXENT_DIFFUSE_TEMPORAL_REPROJECTION_RADIUS), 1.0);
        kernel.previousExtent = vec2(kernel.currentRadius);
    }

    // Pull the separable tent kernel through the local tangent-plane Jacobian. It remains ordinary bilinear
    // reconstruction when the mapping is one-to-one, and grows only when one current pixel covers more history.
    ivec2 kernelMin = ivec2(floor(prevCoord - kernel.previousExtent)) + ivec2(1);
    ivec2 kernelMax = ivec2(ceil(prevCoord + kernel.previousExtent)) - ivec2(1);

    MaxEntEncoding accumMaxEnt = init_maxent();
    float totalKernelWeight = 0.0;
    float validKernelWeight = 0.0;
    float sumWeightOverRootSamples = 0.0;
    float weightedMeanY2 = 0.0;
    vec4 denoisedMaxEntY = vec4(0.0);
    vec2 denoisedCoCg = vec2(0.0);
    float denoisedSquaredWeightVariance = 0.0;
    float denoisedWeightedStdDev = 0.0;

    for (int sampleY = kernelMin.y; sampleY <= kernelMax.y; sampleY++) {
        for (int sampleX = kernelMin.x; sampleX <= kernelMax.x; sampleX++) {
            ivec2 sampleTexel = ivec2(sampleX, sampleY);
            vec2 previousOffset = vec2(sampleTexel) - prevCoord;
            vec2 currentOffset = kernel.currentFromPrevious * previousOffset;
            vec2 axisWeight = max(vec2(1.0) - abs(currentOffset) / kernel.currentRadius, vec2(0.0));
            float reconstructionWeight = axisWeight.x * axisWeight.y;
            if (reconstructionWeight <= 0.0) continue;
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

            vec2 geometryCurrentOffset = kernel.geometryCurrentFromPrevious * previousOffset;
            vec2 expectedCurrentPixel = vec2(gl_GlobalInvocationID.xy) + geometryCurrentOffset;
            if (!strictHistoryGeometryTestFast(historyPosition, fp, expectedCurrentPixel, cameraDelta)) continue;

            vec3 historyNormal = decodeDiffuseHistoryNormalU(packedGeometry.z);
            float normalWeight = max(dot(fp.geometryNormal, historyNormal), 0.0);
            if (normalWeight <= 0.0) continue;

            uvec4 denoisedWords = readDiffuseDenoisedPreviousRaw(historyTexel);
            if (!denoiserSpatialSignalWordsValid(denoisedWords)) continue;

            float tapSamples = historyMeta.x;
            float tapWeight = reconstructionWeight * normalWeight;

            MaxEntEncoding tapMaxEnt;
            tapMaxEnt.maxEntY = vec4(unpackHalf2x16(packedHistory.x), unpackHalf2x16(packedHistory.y));
            tapMaxEnt.CoCg = unpackHalf2x16(packedHistory.z);
            if (any(isnan(tapMaxEnt.maxEntY)) || any(isinf(tapMaxEnt.maxEntY))
                    || any(isnan(tapMaxEnt.CoCg)) || any(isinf(tapMaxEnt.CoCg))) continue;

            vec4 tapDenoisedMaxEntY = vec4(unpackHalf2x16(denoisedWords.x), unpackHalf2x16(denoisedWords.y));
            float tapDenoisedStdDev = unpackHalf2x16(denoisedWords.w).x;
            denoisedMaxEntY += tapWeight * tapDenoisedMaxEntY;
            denoisedCoCg += tapWeight * unpackHalf2x16(denoisedWords.z);
            denoisedSquaredWeightVariance += tapWeight * tapWeight * tapDenoisedStdDev * tapDenoisedStdDev;
            denoisedWeightedStdDev += tapWeight * tapDenoisedStdDev;

            accumulate_maxent(accumMaxEnt, tapMaxEnt, tapWeight);
            float tapMeanY2 = historyMeta.y * historyMeta.y;
            weightedMeanY2 += tapWeight * tapMeanY2;
            validKernelWeight += tapWeight;
            sumWeightOverRootSamples += tapWeight * inversesqrt(tapSamples);
        }
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

    // E[R u], E[R], CoCg and E[R^2] are all linear moments. Reprojection applies the same normalized Jacobian
    // weights to each of them; only Kish N_eff has a separate nonlinear reconstruction rule.
    float histMeanY2 = weightedMeanY2 * inverseKernelWeight;

    MaxEntEncoding currentMaxEnt;
    float currentMeanY2;
    unpackCurrentLight(currentMaxEnt, currentMeanY2);
    float validFootprintCoverage = clamp(validKernelWeight / max(totalKernelWeight, 1e-8), 0.0, 1.0);
    publishDenoisedReprojection(
        denoisedMaxEntY,
        denoisedCoCg,
        denoisedSquaredWeightVariance,
        denoisedWeightedStdDev,
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

    geometryNormal = readPrimaryGeometryNormal(pix);
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
