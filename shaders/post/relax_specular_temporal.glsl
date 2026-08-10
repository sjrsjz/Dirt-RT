#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

uniform usampler2D colortex6;
layout(rgba32ui) uniform writeonly uimage2D colorimg4;
layout(rgba32ui) uniform writeonly uimage2D colorimg5;

#ifndef TEMPORAL_GEOMETRY_EPSILON
#define TEMPORAL_GEOMETRY_EPSILON 1e-5
#endif

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

float relaxCross2(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

struct RelaxSurfaceFootprint {
    vec3 origin;
    vec3 tangent;
    vec3 bitangent;
    vec2 c0;
    vec2 c1;
    vec2 c2;
    vec2 c3;
    float epsilon;
    bool valid;
};

RelaxSurfaceFootprint relaxBuildSurfaceFootprint(
    uvec2 pixel,
    vec3 currentPosition,
    vec3 currentNormal
) {
    RelaxSurfaceFootprint fp;
    fp.origin = currentPosition;
    fp.valid = false;
    vec3 n = relaxSafeNormalize(currentNormal, vec3(0.0, 1.0, 0.0));
    if (n.z < -0.999999) {
        fp.tangent = vec3(0.0, -1.0, 0.0);
        fp.bitangent = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        fp.tangent = vec3(1.0 - n.x * n.x * a, b, -n.x);
        fp.bitangent = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    mat4 inverseCurrentViewProjection = inverse(rtProjection * rtModelView);
    vec2 size = vec2(resolution_global);
    vec2 uvMin = (vec2(pixel) - TEMPORAL_CLIP_PIXEL_RADIUS) / size * 2.0 - 1.0;
    vec2 uvMax = (vec2(pixel) + TEMPORAL_CLIP_PIXEL_RADIUS) / size * 2.0 - 1.0;
    vec2 corners[4] = vec2[4](
        vec2(uvMin.x, uvMin.y), vec2(uvMax.x, uvMin.y),
        vec2(uvMax.x, uvMax.y), vec2(uvMin.x, uvMax.y));
    vec2 planeCorners[4];

    for (int i = 0; i < 4; ++i) {
        vec4 nearH = inverseCurrentViewProjection * vec4(corners[i], -1.0, 1.0);
        vec4 farH = inverseCurrentViewProjection * vec4(corners[i], 1.0, 1.0);
        if (abs(nearH.w) < 1e-8 || abs(farH.w) < 1e-8) return fp;
        vec3 rayOrigin = nearH.xyz / nearH.w;
        vec3 rayDirection = farH.xyz / farH.w - rayOrigin;
        float denominator = dot(rayDirection, n);
        if (abs(denominator) < 1e-7) return fp;
        vec3 cornerPosition = rayOrigin + rayDirection *
            (dot(currentPosition - rayOrigin, n) / denominator);
        vec3 cornerDelta = cornerPosition - currentPosition;
        planeCorners[i] = vec2(dot(cornerDelta, fp.tangent),
            dot(cornerDelta, fp.bitangent));
    }

    fp.c0 = planeCorners[0];
    fp.c1 = planeCorners[1];
    fp.c2 = planeCorners[2];
    fp.c3 = planeCorners[3];
    float footprintDiameter = max(
        length(fp.c2 - fp.c0), length(fp.c3 - fp.c1));
    fp.epsilon = TEMPORAL_GEOMETRY_EPSILON * max(footprintDiameter, 1.0);
    fp.valid = true;
    return fp;
}

bool relaxSurfaceFootprintContains(
    RelaxSurfaceFootprint fp,
    vec3 historyPositionCurrentSpace
) {
    if (!fp.valid) return false;
    vec3 delta = historyPositionCurrentSpace - fp.origin;
    vec2 p = vec2(dot(delta, fp.tangent), dot(delta, fp.bitangent));
    float e0 = relaxCross2(fp.c1 - fp.c0, p - fp.c0);
    float e1 = relaxCross2(fp.c2 - fp.c1, p - fp.c1);
    float e2 = relaxCross2(fp.c3 - fp.c2, p - fp.c2);
    float e3 = relaxCross2(fp.c0 - fp.c3, p - fp.c3);
    return (e0 >= -fp.epsilon && e1 >= -fp.epsilon &&
            e2 >= -fp.epsilon && e3 >= -fp.epsilon) ||
        (e0 <= fp.epsilon && e1 <= fp.epsilon &&
            e2 <= fp.epsilon && e3 <= fp.epsilon);
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
    bool requireSurfaceFootprint
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
    RelaxSurfaceFootprint surfaceFootprint;
    if (requireSurfaceFootprint)
        surfaceFootprint = relaxBuildSurfaceFootprint(
            currentPixel, currentSurfacePosition, currentNormal);

    for (int i = 0; i < 4; ++i) {
        ivec2 p = origin + ivec2(i & 1, i >> 1);
        if (!relaxInBounds(p, size)) continue;
        RelaxSpecularHistory h = readRelaxSpecularHistory(uvec2(p));
        if (h.historyLength < 0.5 || h.materialID != currentMaterial) continue;

        vec3 previousSurfaceCurrentSpace = h.surfacePosition - cameraDelta;
        float planeDistance = abs(dot(
            previousSurfaceCurrentSpace - currentSurfacePosition,
            currentNormal));
        if (planeDistance > depthThreshold) continue;
        if (dot(currentNormal, h.geometryNormal) <= 0.0) continue;
        if (requireSurfaceFootprint && !relaxSurfaceFootprintContains(
            surfaceFootprint, previousSurfaceCurrentSpace)) continue;

        float w = bilinear[i];
        validBilinearWeight += w;
        ++validTapCount;
        if (w <= 0.0) continue;

        outHistory.slowRadiance += h.slowRadiance * w;
        outHistory.secondMoment += h.secondMoment * w;
        outHistory.fastRadiance += h.responsiveRadiance * w;
        if (relaxEndpointMomentsValid(h.endpoint)) {
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

// Offline-calibrated Heitz GGX-VNDF representative point.  The fit uses
// 4096 samples per (alpha, NoV, hit-distance law) group and minimizes the MSE
// against the Monte-Carlo mean projected motion.  GGX/view dependence is
// already present in the measured endpoint moments.  The fitted form is
//
// zeta = 1 + spread * P2(spread, axial).
//
// The fitted rational denominator coefficient converged to 1.15e-23, so it
// is identically one at FP32 precision and is deliberately omitted.
//
// Multiplication by spread is a mathematical boundary condition: q=|m|^2 is
// a deterministic endpoint, for which pVisual=P+m and zeta must be exactly 1.
float relaxVisualPointZeta(float spread, float axial) {
    spread = clamp(spread, 0.0, 1.0);
    axial = clamp(axial, -1.0, 1.0);
    float numerator =
          0.28423406448
        - 0.98314270477 * spread
        + 0.88688920343 * axial
        + 0.71462985137 * spread * spread
        - 0.28210119537 * spread * axial
        - 0.16915468475 * axial * axial;
    return 1.0 + spread * numerator;
}

RelaxEndpointProjection relaxBuildEndpointProjection(
    vec3 currentSurfacePosition,
    vec3 cameraDelta,
    RelaxEndpointMoments endpoint,
    vec2 currentUv
) {
    RelaxEndpointProjection projection;
    projection.uv = vec2(-2.0);
    projection.confidence = 0.0;
    projection.valid = false;
    endpoint = sanitizeRelaxEndpointMoments(endpoint);
    if (!relaxEndpointMomentsValid(endpoint)) return projection;

    float distanceScale = relaxEndpointDistanceScale();
    vec3 meanWorld = endpoint.mean * distanceScale;
    float meanSquared = dot(endpoint.mean, endpoint.mean);
    // Form the dimensionless central-energy ratio before restoring world
    // scale. This avoids subtracting two O(VPROJDIST_SKY^2) FP32 numbers and
    // preserves the q=|m|^2 deterministic boundary after FP16 decoding.
    float spread = clamp((endpoint.secondMoment - meanSquared) /
        max(endpoint.secondMoment, 1e-12), 0.0, 1.0);
    vec3 surfaceView = mat3(rtModelView) * currentSurfacePosition;
    vec3 meanView = mat3(rtModelView) * meanWorld;
    float rawAxial = meanView.z / max(-surfaceView.z, 1e-6);
    float axial = rawAxial / (1.0 + abs(rawAxial));
    float zeta = relaxVisualPointZeta(spread, axial);

    vec3 visualCurrent = currentSurfacePosition + zeta * meanWorld;
    vec3 visualPrevious = visualCurrent + cameraDelta;
    vec2 visualCurrentUv, visualPreviousUv;
    if (!relaxProjectRelative(rtProjection * rtModelView,
            visualCurrent, visualCurrentUv) ||
        !relaxProjectRelative(rtPrevProjection * rtPrevModelView,
            visualPrevious, visualPreviousUv))
        return projection;

    // Difference the visual-point projections and apply that motion to the
    // actual integer-coordinate primary-ray UV. This preserves Vulkanite's
    // sampling protocol and makes identical current/previous cameras an
    // identity without a camera-stationary branch.
    projection.uv = currentUv + visualPreviousUv - visualCurrentUv;
    projection.confidence = 1.0;
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
        cameraDelta, false, true);
    surface.found = surface.found && motionValid >= 0.5;
    vec3 previousV = -relaxSafeNormalize(currentPos + cameraDelta, -V);
    float lobeAngle = max(atan(relaxSpecLobeTanHalfAngle(currentRoughness, 0.75)),
        1.5 / 255.0);
    float surfaceViewWeight = 0.0;
    if (surface.found) {
        float angle = acos(clamp(dot(V, previousV), -1.0, 1.0));
        surfaceViewWeight = clamp(1.0 - angle / max(lobeAngle * max(NoV, 0.05), 1e-4), 0.0, 1.0);
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
    RelaxEndpointProjection endpointProjection =
        relaxBuildEndpointProjection(
            currentPos, cameraDelta,
            currentFrameEndpoint,
            relaxCurrentUv(pixel));
    // A finite endpoint model cannot represent an infinity/sky component.
    // Retain the previous finite moments in storage, but do not use them to
    // reproject the current sky sample.
    endpointProjection.valid = endpointProjection.valid &&
        relaxEndpointMomentsValid(noisy.endpoint);
    RelaxReprojectedHistory virtualHistory = endpointProjection.valid
        ? relaxLoadHistory(endpointProjection.uv, pixel,
            currentPos, currentNormal, currentMaterial,
            cameraDelta, true, false)
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
