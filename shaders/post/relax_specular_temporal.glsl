#version 430 core

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

#ifndef TEMPORAL_GEOMETRY_EPSILON
#define TEMPORAL_GEOMETRY_EPSILON 1e-5
#endif

// Curvature is only a translation-parallax correction.  It is faded out for
// broad lobes, where a one-pixel normal derivative is not a stable coherent
// mirror model, and bounded away from the thin-lens focal singularity.
const float RELAX_CURVATURE_MIN_PARALLAX_PIXELS = 1.0 / 256.0;
const float RELAX_CURVATURE_ROUGHNESS_FADE_BEGIN = 0.2;
const float RELAX_CURVATURE_ROUGHNESS_FADE_END = 0.65;
const float RELAX_CURVATURE_MIN_FOCUS_DENOMINATOR = 0.25;
const float RELAX_CURVATURE_MAX_MOTION_ACCELERATION = 4.0;

struct RelaxReprojectedHistory {
    vec3 slowRadiance;
    float secondMoment;
    vec3 fastRadiance;
    RelaxEndpointMoments endpoint;
    vec3 normal;
    float roughness;
    float historyLength;
    float confidence;
    float footprintQuality;
    bool found;
};

bool relaxSurfaceFootprintContains(
    uvec2 currentPixel,
    vec3 historyPositionCurrentSpace
) {
    // For a point already accepted by the plane-distance test, projecting it
    // into the current frame is equivalent to testing it against the four
    // ray/plane footprint corners. This replaces eight inverse-VP transforms,
    // four ray-plane intersections and a large live footprint structure with
    // at most one forward projection per candidate history tap.
    vec4 clip = rtViewProjection * vec4(historyPositionCurrentSpace, 1.0);
    if (clip.w <= 1e-7 || any(isnan(clip)) || any(isinf(clip))) return false;
    vec2 projectedPixel = (clip.xy / clip.w * 0.5 + 0.5) *
        vec2(resolution_global);
    vec2 extent = vec2(TEMPORAL_CLIP_PIXEL_RADIUS +
        TEMPORAL_GEOMETRY_EPSILON);
    return all(lessThanEqual(abs(projectedPixel - vec2(currentPixel)), extent));
}

RelaxReprojectedHistory relaxEmptyHistory() {
    RelaxReprojectedHistory h;
    h.slowRadiance = vec3(0.0);
    h.secondMoment = 0.0;
    h.fastRadiance = vec3(0.0);
    h.endpoint = emptyRelaxEndpointMoments();
    h.normal = vec3(0.0, 1.0, 0.0);
    h.roughness = 1.0;
    h.historyLength = 0.0;
    h.confidence = 0.0;
    h.footprintQuality = 0.0;
    h.found = false;
    return h;
}

RelaxReprojectedHistory relaxLoadHistory(
    vec2 uv,
    uvec2 currentPixel,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    vec3 cameraDelta,
    bool requireFullFootprint,
    bool requireSurfaceFootprint,
    bool loadEndpoint
) {
    RelaxReprojectedHistory outHistory = relaxEmptyHistory();
    ivec2 size = ivec2(resolution_global);
    vec2 pixelPosition = uv * vec2(size);
    ivec2 origin = ivec2(floor(pixelPosition));
    vec2 f = fract(pixelPosition);
    vec4 bilinear = vec4(
        (1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y, f.x * f.y);

    float sumWeight = 0.0;
    float validBilinearWeight = 0.0;
    int validTapCount = 0;
    vec3 normalSum = vec3(0.0);
    vec3 endpointMeanSum = vec3(0.0);
    float endpointSecondMomentSum = 0.0;
    float endpointWeight = 0.0;
    float depthThreshold = RELAX_DISOCCLUSION_THRESHOLD *
        max(length(currentSurfacePosition), 1.0);
    for (int i = 0; i < 4; ++i) {
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (!relaxInBounds(p, size)) continue;
        RelaxSpecularHistory h = readRelaxSpecularHistory(uvec2(p),
            loadEndpoint);
        if (h.historyLength < 0.5 || h.materialID != currentMaterial) continue;

        vec3 previousSurfaceCurrentSpace = h.surfacePosition - cameraDelta;
        float planeDistance = abs(dot(
            previousSurfaceCurrentSpace - currentSurfacePosition,
            currentNormal));
        if (planeDistance > depthThreshold) continue;
        if (dot(currentNormal, h.geometryNormal) <= 0.0) continue;
        if (requireSurfaceFootprint && !relaxSurfaceFootprintContains(
            currentPixel, previousSurfaceCurrentSpace)) continue;

        float w = bilinear[i];
        validBilinearWeight += w;
        ++validTapCount;
        if (w <= 0.0) continue;

        outHistory.slowRadiance += h.slowRadiance * w;
        outHistory.secondMoment += h.secondMoment * w;
        outHistory.fastRadiance += h.responsiveRadiance * w;
        if (loadEndpoint && relaxEndpointMomentsValid(h.endpoint)) {
            // History moments are relative to the history pixel's primary
            // surface. Rebase them to the current primary surface expressed
            // in previous-frame camera-relative coordinates before mixing.
            vec3 targetPreviousSurface = currentSurfacePosition + cameraDelta;
            vec3 originDelta = (h.surfacePosition - targetPreviousSurface) /
                relaxEndpointDistanceScale();
            RelaxEndpointMoments rebasedEndpoint;
            rebasedEndpoint.mean = h.endpoint.mean + originDelta;
            rebasedEndpoint.secondMoment = h.endpoint.secondMoment +
                dot(originDelta, h.endpoint.mean + rebasedEndpoint.mean);
            rebasedEndpoint = sanitizeRelaxEndpointMoments(rebasedEndpoint);
            if (relaxEndpointMomentsValid(rebasedEndpoint)) {
                endpointMeanSum += rebasedEndpoint.mean * w;
                endpointSecondMomentSum += rebasedEndpoint.secondMoment * w;
                endpointWeight += w;
            }
        }
        outHistory.roughness += (h.roughness - 1.0) * w;
        outHistory.historyLength += h.historyLength * w;
        outHistory.confidence += h.reprojectionConfidence * w;
        normalSum += h.geometryNormal * w;
        sumWeight += w;
    }

    outHistory.footprintQuality = clamp(validBilinearWeight, 0.0, 1.0);
    bool footprintAccepted = requireFullFootprint
        ? (validTapCount == 4 && validBilinearWeight > 0.999)
        : (sumWeight > 1e-5);
    if (!footprintAccepted || sumWeight <= 1e-5) return relaxEmptyHistory();

    float invWeight = 1.0 / sumWeight;
    outHistory.slowRadiance *= invWeight;
    outHistory.secondMoment *= invWeight;
    outHistory.fastRadiance *= invWeight;
    if (endpointWeight > 1e-5) {
        float inverseEndpointWeight = 1.0 / endpointWeight;
        outHistory.endpoint.mean = endpointMeanSum * inverseEndpointWeight;
        outHistory.endpoint.secondMoment = endpointSecondMomentSum *
            inverseEndpointWeight;
        outHistory.endpoint = sanitizeRelaxEndpointMoments(outHistory.endpoint);
    }
    outHistory.roughness = clamp(1.0 + (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyLength *= invWeight;
    outHistory.confidence *= invWeight;
    outHistory.normal = relaxSafeNormalize(normalSum * invWeight, currentNormal);
    outHistory.found = true;
    return outHistory;
}

struct RelaxEndpointProjection {
    vec2 uv;
    float confidence;
    bool valid;
};

bool relaxProjectRelative(mat4 viewProjection, vec3 position, out vec2 uv) {
    vec4 clip = viewProjection * vec4(position, 1.0);
    if (clip.w <= 1e-7 || any(isnan(clip)) || any(isinf(clip))) {
        uv = vec2(-2.0);
        return false;
    }
    uv = clip.xy / clip.w * 0.5 + 0.5;
    return !any(isnan(uv)) && !any(isinf(uv));
}

bool relaxLoadCurvatureEdge(
    ivec2 samplePixel,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    out vec3 tangentEdge,
    out vec3 normalDelta
) {
    tangentEdge = vec3(0.0);
    normalDelta = vec3(0.0);
    ivec2 size = ivec2(resolution_global);
    if (!relaxInBounds(samplePixel, size)) return false;

    vec3 samplePosition;
    float samplePrimaryDistance;
    readGeo0(GEO_N_GEO, uvec2(samplePixel), samplePosition,
        samplePrimaryDistance);
    if (samplePrimaryDistance < -0.5 || any(isnan(samplePosition)) ||
            any(isinf(samplePosition)))
        return false;

    vec3 sampleNormal;
    float sampleAlpha, samplePathRoughness;
    int sampleMaterial;
    readGeo1(GEO_N_NORMALS, uvec2(samplePixel), sampleNormal, sampleAlpha,
        sampleMaterial, samplePathRoughness);
    if (uint(max(sampleMaterial, 0)) != currentMaterial ||
            dot(currentNormal, sampleNormal) <= 0.5)
        return false;

    // Do not estimate a derivative through silhouettes or disconnected
    // surfaces. The tangent-plane intersection below supplies the edge used
    // by the second fundamental form; the actual neighbor is only a guard.
    float planeThreshold = RELAX_DISOCCLUSION_THRESHOLD *
        max(length(currentSurfacePosition), 1.0);
    if (abs(dot(samplePosition - currentSurfacePosition, currentNormal)) >
            planeThreshold)
        return false;

    vec3 sampleRay = relaxSafeNormalize(samplePosition, vec3(0.0));
    float rayPlaneDenominator = dot(currentNormal, sampleRay);
    if (abs(rayPlaneDenominator) <= 1e-4) return false;
    float tangentDistance = dot(currentSurfacePosition, currentNormal) /
        rayPlaneDenominator;
    if (tangentDistance <= 1e-5 || isnan(tangentDistance) ||
            isinf(tangentDistance))
        return false;

    tangentEdge = sampleRay * tangentDistance - currentSurfacePosition;
    normalDelta = sampleNormal - currentNormal;
    return dot(tangentEdge, tangentEdge) > 1e-10;
}

struct RelaxDirectionalCurvature {
    float value;
    float parallaxPixels;
};

RelaxDirectionalCurvature relaxEstimateDirectionalCurvature(
    uvec2 currentPixel,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    float perceptualRoughness,
    vec3 cameraDelta,
    vec2 surfacePreviousUv
) {
    RelaxDirectionalCurvature result;
    result.value = 0.0;
    result.parallaxPixels = 0.0;

    // Subtracting these two previous-frame projections removes camera
    // rotation and leaves only translation/object-motion parallax. Therefore
    // pure optical-center rotation can never select or apply curvature.
    vec2 zeroParallaxUv;
    if (!relaxProjectRelative(rtPrevViewProjection,
            currentSurfacePosition, zeroParallaxUv) ||
            any(isnan(surfacePreviousUv)) || any(isinf(surfacePreviousUv)))
        return result;
    vec2 parallaxPixels = (zeroParallaxUv - surfacePreviousUv) *
        vec2(resolution_global);
    result.parallaxPixels = length(parallaxPixels);
    if (result.parallaxPixels < RELAX_CURVATURE_MIN_PARALLAX_PIXELS)
        return result;

    vec2 direction = parallaxPixels / result.parallaxPixels;
    ivec2 pixel = ivec2(currentPixel);
    ivec2 xOffset = ivec2(direction.x < 0.0 ? -1 : 1, 0);
    ivec2 yOffset = ivec2(0, direction.y < 0.0 ? -1 : 1);
    vec3 edge = vec3(0.0);
    vec3 normalDelta = vec3(0.0);
    float usedWeight = 0.0;

    vec3 axisEdge, axisNormalDelta;
    float axisWeight = abs(direction.x);
    if (axisWeight > 1e-4 && relaxLoadCurvatureEdge(pixel + xOffset,
            currentSurfacePosition, currentNormal, currentMaterial,
            axisEdge, axisNormalDelta)) {
        edge += axisEdge * axisWeight;
        normalDelta += axisNormalDelta * axisWeight;
        usedWeight += axisWeight;
    }
    axisWeight = abs(direction.y);
    if (axisWeight > 1e-4 && relaxLoadCurvatureEdge(pixel + yOffset,
            currentSurfacePosition, currentNormal, currentMaterial,
            axisEdge, axisNormalDelta)) {
        edge += axisEdge * axisWeight;
        normalDelta += axisNormalDelta * axisWeight;
        usedWeight += axisWeight;
    }
    if (usedWeight <= 1e-4) return result;

    edge /= usedWeight;
    normalDelta /= usedWeight;
    float edgeLengthSquared = dot(edge, edge);
    if (edgeLengthSquared <= 1e-10) return result;
    float curvature = dot(normalDelta, edge) / edgeLengthSquared;
    if (isnan(curvature) || isinf(curvature)) return result;

    float coherentMirrorAmount = 1.0 - smoothstep(
        RELAX_CURVATURE_ROUGHNESS_FADE_BEGIN,
        RELAX_CURVATURE_ROUGHNESS_FADE_END,
        perceptualRoughness);
    result.value = curvature * coherentMirrorAmount;
    return result;
}

bool relaxApplyThinMirror(float offset, float curvature, out float focused) {
    float denominator = 1.0 + 2.0 * curvature * offset;
    if (denominator < RELAX_CURVATURE_MIN_FOCUS_DENOMINATOR ||
            isnan(denominator) || isinf(denominator)) {
        focused = offset;
        return false;
    }
    focused = offset / denominator;
    return !isnan(focused) && !isinf(focused) &&
        (offset == 0.0 || focused * offset > 0.0);
}

RelaxEndpointProjection relaxBuildEndpointProjection(
    vec3 currentSurfacePosition,
    vec3 cameraDelta,
    RelaxEndpointMoments endpoint,
    RelaxDirectionalCurvature directionalCurvature
) {
    RelaxEndpointProjection projection;
    projection.uv = vec2(-2.0);
    projection.confidence = 0.0;
    projection.valid = false;
    endpoint = sanitizeRelaxEndpointMoments(endpoint);
    if (!relaxEndpointMomentsValid(endpoint)) return projection;

    float distanceScale = relaxEndpointDistanceScale();
    vec3 meanWorld = endpoint.mean * distanceScale;
    float centralEnergyNormalized = max(endpoint.secondMoment
        - dot(endpoint.mean, endpoint.mean), 0.0);
    float axialVariance = centralEnergyNormalized
        * distanceScale * distanceScale / 3.0;

    // A temporal sample belongs to the primary pixel, not to the screen-space
    // projection of its rough VNDF endpoint. Collapse the measured endpoint
    // moments onto the current primary ray so every support point projects to
    // the current pixel. Pure camera rotation is then exactly depth-invariant.
    float surfaceDistance = length(currentSurfacePosition);
    if (surfaceDistance <= 1e-7) return projection;
    vec3 primaryRay = currentSurfacePosition / surfaceDistance;
    float axialMeanOffset = dot(meanWorld, primaryRay);
    float axialSigma = sqrt(max(axialVariance, 0.0));
    float nearAxialOffset = axialMeanOffset - axialSigma;
    float farAxialOffset = axialMeanOffset + axialSigma;

    // Apply the local thin-mirror equation to the virtual offset measured
    // from the primary surface, never to the camera-to-surface distance. The
    // two transformed support points remain scalar multiples of primaryRay,
    // preserving exact pure-rotation reprojection.
    float focusedNearOffset, focusedFarOffset;
    bool curvatureApplied = abs(directionalCurvature.value) > 1e-12 &&
        relaxApplyThinMirror(nearAxialOffset, directionalCurvature.value,
            focusedNearOffset) &&
        relaxApplyThinMirror(farAxialOffset, directionalCurvature.value,
            focusedFarOffset);
    if (!curvatureApplied) {
        focusedNearOffset = nearAxialOffset;
        focusedFarOffset = farAxialOffset;
    }

    float nearVirtualDistance = surfaceDistance + focusedNearOffset;
    float farVirtualDistance = surfaceDistance + focusedFarOffset;
    if (nearVirtualDistance <= 1e-5 || isnan(farVirtualDistance)
            || isinf(farVirtualDistance))
        return projection;

    vec3 nearPrevious = primaryRay * nearVirtualDistance + cameraDelta;
    vec3 farPrevious = primaryRay * farVirtualDistance + cameraDelta;
    vec2 nearPreviousUv, farPreviousUv;
    if (!relaxProjectRelative(rtPrevViewProjection,
            nearPrevious, nearPreviousUv) ||
        !relaxProjectRelative(rtPrevViewProjection,
            farPrevious, farPreviousUv))
        return projection;

    projection.uv = 0.5 * (nearPreviousUv + farPreviousUv);

    // Reject a noisy/focal curvature estimate if it accelerates the virtual
    // address by more than the underlying translation parallax can explain.
    // Fall back to the uncurved axial model instead of discarding history.
    if (curvatureApplied) {
        vec2 uncurvedMeanUv;
        vec3 uncurvedMeanPrevious = primaryRay *
            (surfaceDistance + axialMeanOffset) + cameraDelta;
        bool uncurvedMeanValid = relaxProjectRelative(rtPrevViewProjection,
            uncurvedMeanPrevious, uncurvedMeanUv);
        float correctionPixels = uncurvedMeanValid
            ? length((projection.uv - uncurvedMeanUv) *
                vec2(resolution_global))
            : 1e20;
        float allowedCorrectionPixels = max(1.0,
            RELAX_CURVATURE_MAX_MOTION_ACCELERATION *
                directionalCurvature.parallaxPixels);
        if (correctionPixels > allowedCorrectionPixels) {
            float uncurvedNearDistance = surfaceDistance + nearAxialOffset;
            float uncurvedFarDistance = surfaceDistance + farAxialOffset;
            if (uncurvedNearDistance <= 1e-5 ||
                    isnan(uncurvedFarDistance) || isinf(uncurvedFarDistance))
                return projection;
            nearPrevious = primaryRay *
                uncurvedNearDistance + cameraDelta;
            farPrevious = primaryRay *
                uncurvedFarDistance + cameraDelta;
            if (!relaxProjectRelative(rtPrevViewProjection,
                    nearPrevious, nearPreviousUv) ||
                !relaxProjectRelative(rtPrevViewProjection,
                    farPrevious, farPreviousUv))
                return projection;
            projection.uv = 0.5 * (nearPreviousUv + farPreviousUv);
        }
    }

    vec2 sigmaMotionPixels = 0.5 * (farPreviousUv - nearPreviousUv)
        * vec2(resolution_global);
    float motionVariancePixels = dot(sigmaMotionPixels, sigmaMotionPixels);
    projection.confidence = 1.0 / (1.0 + motionVariancePixels);
    projection.valid = !any(isnan(projection.uv)) &&
        !any(isinf(projection.uv)) &&
        all(greaterThanEqual(projection.uv, vec2(0.0))) &&
        all(lessThanEqual(projection.uv, vec2(1.0))) &&
        projection.confidence > 1e-5;
    return projection;
}

void relaxAccumulatePath(
    RelaxReprojectedHistory h,
    vec3 noisyRadiance,
    float noisyM2,
    float slowConfidence,
    float responsiveConfidence,
    out RelaxSlowSignal slow,
    out RelaxFastSignal fast,
    out float historyContribution
) {
    float historyLength = h.found ? max(h.historyLength, 0.0) : 0.0;
    float slowFrames = min(historyLength, float(RELAX_SPEC_MAX_HISTORY));
    float fastFrames = min(historyLength, float(RELAX_SPEC_MAX_FAST_HISTORY));
    float slowAlpha = max(1.0 - slowConfidence,
        1.0 / (1.0 + slowFrames));
    float fastAlpha = max(1.0 - responsiveConfidence,
        1.0 / (1.0 + fastFrames));
    historyContribution = h.found ? (1.0 - slowAlpha) : 0.0;

    slow.radiance = mix(h.slowRadiance, noisyRadiance, slowAlpha);
    slow.secondMoment = mix(h.secondMoment, noisyM2, slowAlpha);
    slow.historyLength = min(historyLength + 1.0, float(RELAX_SPEC_MAX_HISTORY));
    slow.confidence = slowConfidence;

    fast.radiance = mix(h.fastRadiance, noisyRadiance, fastAlpha);
    fast.endpointDistance = 0.0;
    fast.historyLength = slow.historyLength;
    fast.confidence = responsiveConfidence;
    fast.materialID = 0u;
}

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel, resolution_global))) return;

    RelaxPrepassSignal noisy =
        relaxUnpackPrepass(texelFetch(colortex6, ivec2(pixel), 0));
    float noisyLuminance = relaxLuma(noisy.radiance);
    float noisyM2 = noisyLuminance * noisyLuminance;

    vec3 currentPos;
    float primaryDistance;
    readGeo0(GEO_N_GEO, pixel, currentPos, primaryDistance);
    if (primaryDistance < -0.5) {
        RelaxSlowSignal slow;
        slow.radiance = noisy.radiance;
        slow.secondMoment = noisyM2;
        slow.historyLength = 0.0;
        slow.confidence = 0.0;
        RelaxFastSignal fast;
        fast.radiance = noisy.radiance;
        fast.endpointDistance = 0.0;
        fast.historyLength = 0.0;
        fast.confidence = 0.0;
        fast.materialID = 0u;
        imageStore(colorimg4, ivec2(pixel), relaxPackSlow(slow));
        imageStore(colorimg5, ivec2(pixel), relaxPackFast(fast));
#if DEBUG_VIEW == 12 || (DEBUG_VIEW >= 38 && DEBUG_VIEW <= 40)
        writeReflLight(pixel, slow.radiance, fast.endpointDistance,
            slow.historyLength);
#elif DEBUG_VIEW == 14
        writeReflLight(pixel, vec3(0.0), fast.endpointDistance, 0.0);
#endif
        return;
    }

    vec3 currentNormal;
    float currentAlpha, currentPathRoughness;
    int currentMaterialInt;
    readGeo1(GEO_N_NORMALS, pixel, currentNormal, currentAlpha,
        currentMaterialInt, currentPathRoughness);
    uint currentMaterial = uint(max(currentMaterialInt, 0));
    float currentRoughness = relaxPerceptualRoughness(currentAlpha);
    vec3 cameraDelta = camPos - prevRaytracingCamPos;
    vec3 surfaceMotion;
    float motionValid;
    readSurfaceMotion(pixel, surfaceMotion, motionValid);
    cameraDelta -= surfaceMotion;
    vec3 viewDirection = relaxSafeNormalize(currentPos, vec3(0.0, 0.0, 1.0));
    vec3 V = -viewDirection;
    float NoV = abs(dot(currentNormal, V));
    vec2 surfaceUv = relaxProjectPrevious(currentPos, cameraDelta);
    RelaxReprojectedHistory surface = relaxLoadHistory(
        surfaceUv, pixel, currentPos, currentNormal, currentMaterial,
        cameraDelta, false, true, true);
    surface.found = surface.found && motionValid >= 0.5;
    vec3 previousV = -relaxSafeNormalize(currentPos + cameraDelta, -V);
    // Compare unit-vector chord lengths in dot space. For theta in the RELAX
    // lobe range, chord(theta) tracks theta closely and avoids atan + acos.
    float lobeTangent = relaxSpecLobeTanHalfAngle(currentRoughness, 0.75);
    float lobeChord = sqrt(max(2.0 - 2.0 * inversesqrt(
        1.0 + lobeTangent * lobeTangent), 0.0));
    lobeChord = max(lobeChord, 1.5 / 255.0);
    float surfaceViewWeight = 0.0;
    if (surface.found) {
        float viewChord = sqrt(max(2.0 - 2.0 *
            clamp(dot(V, previousV), -1.0, 1.0), 0.0));
        float acceptedChord = max(lobeChord * max(NoV, 0.05), 1e-4);
        surfaceViewWeight = clamp(1.0 - viewChord / acceptedChord,
            0.0, 1.0);
        surfaceViewWeight *= surface.footprintQuality;
    }

    // The current-frame 7x7 estimate drives virtual reprojection directly.
    // It is intentionally not mixed with a different view-conditioned
    // endpoint distribution from history.
    RelaxEndpointMoments currentFrameEndpoint =
        sanitizeRelaxEndpointMoments(noisy.endpoint);

    // Endpoint moments belong to the primary surface, so their temporal
    // correspondence is the surface reprojection (not the reflected image
    // reprojection). The exact moment rebase above makes both E[X] and
    // E[|X|^2] refer to the current surface before this EMA is evaluated.
    RelaxEndpointMoments temporalEndpoint = currentFrameEndpoint;
    if (relaxEndpointMomentsValid(currentFrameEndpoint) && surface.found &&
        relaxEndpointMomentsValid(surface.endpoint)) {
        float endpointFrames = min(surface.historyLength,
            float(RELAX_SPEC_MAX_HISTORY));
        float endpointAlpha = max(1.0 - surface.footprintQuality,
            1.0 / (1.0 + endpointFrames));
        temporalEndpoint.mean = mix(surface.endpoint.mean,
            currentFrameEndpoint.mean, endpointAlpha);
        temporalEndpoint.secondMoment = mix(surface.endpoint.secondMoment,
            currentFrameEndpoint.secondMoment, endpointAlpha);
        temporalEndpoint = sanitizeRelaxEndpointMoments(temporalEndpoint);
    }
    float endpointDistance = relaxEndpointMeanDistance(temporalEndpoint);
    RelaxDirectionalCurvature directionalCurvature;
    directionalCurvature.value = 0.0;
    directionalCurvature.parallaxPixels = 0.0;
    if (motionValid >= 0.5 &&
            relaxEndpointMomentsValid(currentFrameEndpoint) &&
            currentRoughness < RELAX_CURVATURE_ROUGHNESS_FADE_END) {
        directionalCurvature = relaxEstimateDirectionalCurvature(pixel,
            currentPos, currentNormal, currentMaterial, currentRoughness,
            cameraDelta, surfaceUv);
    }
    RelaxEndpointProjection endpointProjection =
        relaxBuildEndpointProjection(
            currentPos, cameraDelta,
            currentFrameEndpoint, directionalCurvature);
    // A finite endpoint model cannot represent an infinity/sky component.
    // Retain the previous finite moments in storage, but do not use them to
    // reproject the current sky sample.
    endpointProjection.valid = endpointProjection.valid &&
        relaxEndpointMomentsValid(noisy.endpoint);
    RelaxReprojectedHistory virtualHistory = endpointProjection.valid
        ? relaxLoadHistory(endpointProjection.uv, pixel,
            currentPos, currentNormal, currentMaterial,
            cameraDelta, true, false, DEBUG_VIEW == 40)
        : relaxEmptyHistory();
    virtualHistory.found = virtualHistory.found && motionValid >= 0.5;

    float virtualAmount = virtualHistory.found
        ? endpointProjection.confidence : 0.0;
    float virtualAccumulationConfidence = 0.0;
    float virtualResponsiveConfidence = 0.0;
    if (virtualHistory.found) {
        vec2 normalParams = relaxNormalWeightParams(currentRoughness, 5.0, 1.0);
        float normalWeight = relaxSpecularNormalWeight(normalParams,
            currentNormal, virtualHistory.normal, V, V);
        vec2 roughnessParams = relaxRoughnessWeightParams(
            currentRoughness * currentRoughness, RELAX_ROUGHNESS_FRACTION);
        float roughnessWeight = relaxExponentialWeight(
            virtualHistory.roughness * virtualHistory.roughness,
            roughnessParams);
        virtualAccumulationConfidence =
            endpointProjection.confidence * roughnessWeight;
        virtualResponsiveConfidence =
            virtualAccumulationConfidence * normalWeight *
            virtualHistory.footprintQuality;
        virtualAmount *= normalWeight * roughnessWeight *
            virtualHistory.footprintQuality;
    }
    virtualAmount = clamp(virtualAmount, 0.0, 1.0);

    RelaxSlowSignal surfaceSlow, virtualSlow;
    RelaxFastSignal surfaceFast, virtualFast;
    float surfaceHistoryContribution;
    float virtualHistoryContribution;
    relaxAccumulatePath(surface, noisy.radiance, noisyM2,
        surfaceViewWeight, surfaceViewWeight,
        surfaceSlow, surfaceFast,
        surfaceHistoryContribution);
    relaxAccumulatePath(virtualHistory, noisy.radiance, noisyM2,
        virtualAccumulationConfidence,
        virtualResponsiveConfidence,
        virtualSlow, virtualFast,
        virtualHistoryContribution);

    RelaxSlowSignal outputSlow;
    RelaxFastSignal outputFast;
    outputSlow.radiance = mix(surfaceSlow.radiance, virtualSlow.radiance, virtualAmount);
    outputSlow.secondMoment = mix(surfaceSlow.secondMoment, virtualSlow.secondMoment, virtualAmount);
    outputSlow.historyLength = mix(surfaceSlow.historyLength, virtualSlow.historyLength, virtualAmount);
    outputSlow.historyLength *= sqrt(max(surface.footprintQuality, 1.0 / max(outputSlow.historyLength, 1.0)));
    outputSlow.historyLength = clamp(outputSlow.historyLength, 1.0, float(RELAX_SPEC_MAX_HISTORY));
    outputSlow.confidence = mix(surfaceViewWeight,
        virtualAccumulationConfidence, virtualAmount);
    outputFast.radiance = mix(surfaceFast.radiance, virtualFast.radiance, virtualAmount);
    outputFast.endpointDistance = endpointDistance;
    outputFast.historyLength = outputSlow.historyLength;
    outputFast.confidence = outputSlow.confidence;
    outputFast.materialID = currentMaterial;
    float outputHistoryContribution = mix(surfaceHistoryContribution,
        virtualHistoryContribution, virtualAmount);

    if (outputSlow.secondMoment == 0.0)
        outputSlow.secondMoment = RELAX_SPEC_VARIANCE_BOOST *
            (1.0 - outputSlow.confidence);
    imageStore(colorimg4, ivec2(pixel), relaxPackSlow(outputSlow));
    imageStore(colorimg5, ivec2(pixel), relaxPackFast(outputFast));
#if DEBUG_VIEW == 38
    // Current prepass signal. This bypasses all temporal and spatial reuse.
    writeReflLight(pixel, noisy.radiance, endpointDistance, 0.0);
#elif DEBUG_VIEW == 39
    // The history value actually fetched through surface reprojection.
    writeReflLight(pixel,
        surface.found ? relaxFiniteColor(surface.slowRadiance) : vec3(0.0),
        relaxEndpointMeanDistance(surface.endpoint),
        surface.found ? 1.0 : 0.0);
#elif DEBUG_VIEW == 40
    // History fetched with the offline-calibrated GGX-VNDF visual point.
    writeReflLight(pixel,
        virtualHistory.found
            ? relaxFiniteColor(virtualHistory.slowRadiance) : vec3(0.0),
        relaxEndpointMeanDistance(virtualHistory.endpoint),
        virtualHistory.found ? 1.0 : 0.0);
#elif DEBUG_VIEW == 12
    // Preserve the temporal-only result in ReflectBuffer N=1.  The later
    // RELAX passes may still execute, but resolve leaves this value untouched
    // in temporal diagnostic views, so no spatial stage contributes.
    writeReflLight(pixel, relaxFiniteColor(outputSlow.radiance),
        endpointDistance, outputHistoryContribution);
#elif DEBUG_VIEW == 14
    writeReflLight(pixel, vec3(outputHistoryContribution),
        endpointDistance, 0.0);
#endif
#if DEBUG_VIEW == 9
    // View 9 remains the current-frame 7x7 spatial result by definition.
    writeReflEndpointMoments(pixel, currentFrameEndpoint);
#else
    // Keep moment history intact even when a diagnostic write above replaces
    // the other words of ReflectBuffer N=1.
    writeReflEndpointMoments(pixel, temporalEndpoint);
#endif
}
