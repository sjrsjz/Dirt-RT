#version 430 compatibility

layout(local_size_x = 8, local_size_y = 8) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"
#include "/lib/denoise/relax_ggx_endpoint_moments.glsl"

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
    float hitDistance;
    vec3 normal;
    float roughness;
    float historyLength;
    float confidence;
    float footprintQuality;
    vec3 endpointMean;
    float endpointSecondMoment;
    float endpointWeight;
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
    h.hitDistance = 0.0;
    h.normal = vec3(0.0, 1.0, 0.0);
    h.roughness = 1.0;
    h.historyLength = 0.0;
    h.confidence = 0.0;
    h.footprintQuality = 0.0;
    h.endpointMean = vec3(0.0);
    h.endpointSecondMoment = 0.0;
    h.endpointWeight = 0.0;
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
        outHistory.hitDistance += h.hitDistance * w;
        outHistory.roughness += (h.roughness - 1.0) * w;
        outHistory.historyLength += h.historyLength * w;
        outHistory.confidence += h.reprojectionConfidence * w;
        normalSum += h.geometryNormal * w;
        if (h.endpointSecondMoment > 0.0) {
            endpointMeanSum += h.endpointMean * w;
            endpointSecondMomentSum += h.endpointSecondMoment * w;
            endpointWeight += w;
        }
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
    outHistory.hitDistance *= invWeight;
    outHistory.roughness = clamp(1.0 + (outHistory.roughness - 1.0) * invWeight, 0.0, 1.0);
    outHistory.historyLength *= invWeight;
    outHistory.confidence *= invWeight;
    outHistory.normal = relaxSafeNormalize(normalSum * invWeight, currentNormal);
    if (endpointWeight > 1e-6) {
        RelaxEndpointMoments endpoint;
        endpoint.mean = endpointMeanSum / endpointWeight;
        endpoint.secondMoment = endpointSecondMomentSum / endpointWeight;
        endpoint = sanitizeRelaxEndpointMoments(endpoint);
        outHistory.endpointMean = endpoint.mean;
        outHistory.endpointSecondMoment = endpoint.secondMoment;
        outHistory.endpointWeight =
            clamp(endpointWeight * invWeight, 0.0, 1.0);
    }
    outHistory.found = true;
    return outHistory;
}

struct RelaxEndpointProjection {
    vec2 uv;
    vec2 majorAxisUv;
    float sigmaMajorPixels;
    float sigmaMinorPixels;
    float confidence;
    bool valid;
};

float relaxEndpointCovarianceForm(
    vec3 lhs, vec3 rhs, float q, mat3 angularSecondMoment, vec3 mean
) {
    return q * dot(lhs, angularSecondMoment * rhs) -
        dot(lhs, mean) * dot(rhs, mean);
}

RelaxEndpointMoments relaxAccumulateEndpointMoments(
    RelaxReprojectedHistory surfaceHistory,
    RelaxEndpointMoments currentEndpoint,
    float surfaceConfidence
) {
    currentEndpoint = sanitizeRelaxEndpointMoments(currentEndpoint);
    RelaxEndpointMoments previousEndpoint;
    previousEndpoint.mean = surfaceHistory.endpointMean;
    previousEndpoint.secondMoment =
        surfaceHistory.endpointSecondMoment;
    previousEndpoint = sanitizeRelaxEndpointMoments(previousEndpoint);
    bool currentValid = relaxEndpointMomentsValid(currentEndpoint);
    bool previousValid = surfaceHistory.found &&
        relaxEndpointMomentsValid(previousEndpoint) &&
        surfaceHistory.endpointWeight > 1e-5;

    if (!currentValid) return previousValid
        ? previousEndpoint : emptyRelaxEndpointMoments();
    if (!previousValid) return currentEndpoint;

    float historyFrames = min(max(surfaceHistory.historyLength, 0.0),
        float(RELAX_SPEC_MAX_HISTORY));
    float alpha = max(1.0 - clamp(surfaceConfidence, 0.0, 1.0),
        1.0 / (1.0 + historyFrames));
    RelaxEndpointMoments result;
    result.mean = mix(previousEndpoint.mean, currentEndpoint.mean, alpha);
    // Decode RMS -> q, linearly filter q, and only then encode sqrt(q) at
    // the FP16 store boundary.
    result.secondMoment = mix(previousEndpoint.secondMoment,
        currentEndpoint.secondMoment, alpha);
    return sanitizeRelaxEndpointMoments(result);
}

RelaxEndpointProjection relaxBuildEndpointProjection(
    vec3 previousSurfacePosition,
    RelaxEndpointMoments endpoint,
    float ggxAlpha,
    vec3 macroNormal,
    vec3 V
) {
    RelaxEndpointProjection projection;
    projection.uv = vec2(-2.0);
    projection.majorAxisUv = vec2(0.0);
    projection.sigmaMajorPixels = 0.0;
    projection.sigmaMinorPixels = 0.0;
    projection.confidence = 0.0;
    projection.valid = false;
    endpoint = sanitizeRelaxEndpointMoments(endpoint);
    if (!relaxEndpointMomentsValid(endpoint)) return projection;

    vec3 angularMean;
    mat3 angularSecondMoment;
    float NoV = abs(dot(macroNormal, V));
    relaxLookupGGXAngularMoments(ggxAlpha, NoV, macroNormal,
        V, angularMean, angularSecondMoment);

    float a2 = max(dot(angularMean, angularMean), 1e-8);
    float meanDistance = dot(angularMean, endpoint.mean) / a2;
    vec3 separabilityResidual =
        endpoint.mean - angularMean * meanDistance;
    float residualRatio = dot(separabilityResidual, separabilityResidual) /
        max(endpoint.secondMoment, 1e-12);
    float separabilityConfidence =
        1.0 - clamp(residualRatio, 0.0, 1.0);
    float distanceScale = relaxEndpointDistanceScale();
    vec3 meanWorld = endpoint.mean * distanceScale;
    float qWorld = endpoint.secondMoment *
        distanceScale * distanceScale;
    vec3 previousMeanEndpoint = previousSurfacePosition + meanWorld;
    vec4 previousCameraH =
        rtPrevModelView * vec4(previousMeanEndpoint, 1.0);
    vec3 meanCamera = previousCameraH.xyz;
    if (any(isnan(meanCamera)) || any(isinf(meanCamera)) ||
            meanCamera.z >= -1e-4) return projection;

    mat3 previousRotation = mat3(rtPrevModelView);
    vec3 meanOffsetCamera = previousRotation * meanWorld;
    mat3 angularSecondMomentCamera = previousRotation *
        angularSecondMoment * transpose(previousRotation);
    vec3 ex = vec3(1.0, 0.0, 0.0);
    vec3 ey = vec3(0.0, 1.0, 0.0);
    vec3 ez = vec3(0.0, 0.0, 1.0);
    float varX = max(relaxEndpointCovarianceForm(ex, ex, qWorld,
        angularSecondMomentCamera, meanOffsetCamera), 0.0);
    float varY = max(relaxEndpointCovarianceForm(ey, ey, qWorld,
        angularSecondMomentCamera, meanOffsetCamera), 0.0);
    float varZ = max(relaxEndpointCovarianceForm(ez, ez, qWorld,
        angularSecondMomentCamera, meanOffsetCamera), 0.0);
    float covXZ = relaxEndpointCovarianceForm(ex, ez, qWorld,
        angularSecondMomentCamera, meanOffsetCamera);
    float covYZ = relaxEndpointCovarianceForm(ey, ez, qWorld,
        angularSecondMomentCamera, meanOffsetCamera);
    covXZ = clamp(covXZ, -sqrt(varX * varZ), sqrt(varX * varZ));
    covYZ = clamp(covYZ, -sqrt(varY * varZ), sqrt(varY * varZ));

    projection.uv = relaxProjectPreviousRelative(previousMeanEndpoint);
    float z = meanCamera.z;
    float invZ = 1.0 / z;
    float invZ2 = invZ * invZ;
    float invZ3 = invZ2 * invZ;
    float fxUv = -0.5 * rtPrevProjection[0][0];
    float fyUv = -0.5 * rtPrevProjection[1][1];
    vec2 perspectiveBias = vec2(
        fxUv * (meanCamera.x * varZ * invZ3 - covXZ * invZ2),
        fyUv * (meanCamera.y * varZ * invZ3 - covYZ * invZ2));

    vec2 resolution = vec2(resolution_global);
    vec3 jx = vec3(fxUv * resolution.x * invZ, 0.0,
        -fxUv * resolution.x * meanCamera.x * invZ2);
    vec3 jy = vec3(0.0, fyUv * resolution.y * invZ,
        -fyUv * resolution.y * meanCamera.y * invZ2);
    float covarianceXX = max(relaxEndpointCovarianceForm(jx, jx, qWorld,
        angularSecondMomentCamera, meanOffsetCamera), 0.0);
    float covarianceYY = max(relaxEndpointCovarianceForm(jy, jy, qWorld,
        angularSecondMomentCamera, meanOffsetCamera), 0.0);
    float covarianceXY = relaxEndpointCovarianceForm(jx, jy, qWorld,
        angularSecondMomentCamera, meanOffsetCamera);
    covarianceXY = clamp(covarianceXY,
        -sqrt(covarianceXX * covarianceYY),
         sqrt(covarianceXX * covarianceYY));

    float trace = covarianceXX + covarianceYY;
    float discriminant = sqrt(max(
        (covarianceXX - covarianceYY) *
        (covarianceXX - covarianceYY) +
        4.0 * covarianceXY * covarianceXY, 0.0));
    float lambdaMajor = max(0.5 * (trace + discriminant), 0.0);
    float lambdaMinor = max(0.5 * (trace - discriminant), 0.0);
    projection.sigmaMajorPixels = sqrt(lambdaMajor);
    projection.sigmaMinorPixels = sqrt(lambdaMinor);

    vec2 majorAxisPixels;
    if (abs(covarianceXY) > 1e-8) {
        majorAxisPixels = normalize(vec2(
            covarianceXY, lambdaMajor - covarianceXX));
    } else {
        majorAxisPixels = covarianceXX >= covarianceYY
            ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }
    projection.majorAxisUv = majorAxisPixels / resolution;

    // The second-order expansion is invalid close to its projective pole.
    // Bound the correction by the reconstructed footprint instead of allowing
    // a finite input moment to generate an Inf/NaN screen coordinate.
    vec2 biasPixels = perspectiveBias * resolution;
    float maxBiasPixels = max(0.5, 2.0 * projection.sigmaMajorPixels);
    float biasLength = length(biasPixels);
    if (biasLength > maxBiasPixels)
        perspectiveBias *= maxBiasPixels / biasLength;
    projection.uv += perspectiveBias;

    float footprintConfidence = 1.0 - smoothstep(
        8.0, 32.0, projection.sigmaMajorPixels);
    projection.confidence =
        separabilityConfidence * footprintConfidence;
    projection.valid = !any(isnan(projection.uv)) &&
        !any(isinf(projection.uv)) &&
        all(greaterThanEqual(projection.uv, vec2(0.0))) &&
        all(lessThanEqual(projection.uv, vec2(1.0))) &&
        projection.confidence > 1e-5;
    return projection;
}

RelaxReprojectedHistory relaxLoadEllipseHistory(
    RelaxEndpointProjection projection,
    uvec2 currentPixel,
    vec3 currentSurfacePosition,
    vec3 currentNormal,
    uint currentMaterial,
    vec3 cameraDelta
) {
    RelaxReprojectedHistory result = relaxEmptyHistory();
    if (!projection.valid) return result;
    int sampleCount = projection.sigmaMajorPixels > 0.75 ? 3 : 1;
    float radiusPixels = clamp(
        0.5 * projection.sigmaMajorPixels, 0.5, 2.0);
    float sumWeight = 0.0;
    vec3 normalSum = vec3(0.0);
    for (int i = 0; i < 3; ++i) {
        if (i >= sampleCount) break;
        float signedOffset = i == 0 ? 0.0 : (i == 1 ? -1.0 : 1.0);
        float tapWeight = i == 0 ? (sampleCount == 1 ? 1.0 : 0.5) : 0.25;
        vec2 uv = projection.uv + projection.majorAxisUv *
            (signedOffset * radiusPixels);
        RelaxReprojectedHistory tap = relaxLoadHistory(
            uv, currentPixel, currentSurfacePosition, currentNormal,
            currentMaterial, cameraDelta, true, false);
        if (!tap.found) continue;
        result.slowRadiance += tap.slowRadiance * tapWeight;
        result.secondMoment += tap.secondMoment * tapWeight;
        result.fastRadiance += tap.fastRadiance * tapWeight;
        result.hitDistance += tap.hitDistance * tapWeight;
        result.roughness += (tap.roughness - 1.0) * tapWeight;
        result.historyLength += tap.historyLength * tapWeight;
        result.confidence += tap.confidence * tapWeight;
        result.footprintQuality += tap.footprintQuality * tapWeight;
        normalSum += tap.normal * tapWeight;
        sumWeight += tapWeight;
    }
    if (sumWeight <= 1e-5) return relaxEmptyHistory();
    float invWeight = 1.0 / sumWeight;
    result.slowRadiance *= invWeight;
    result.secondMoment *= invWeight;
    result.fastRadiance *= invWeight;
    result.hitDistance *= invWeight;
    result.roughness = clamp(
        1.0 + (result.roughness - 1.0) * invWeight, 0.0, 1.0);
    result.historyLength *= invWeight;
    result.confidence *= invWeight;
    result.footprintQuality =
        clamp(result.footprintQuality * invWeight, 0.0, 1.0);
    result.normal = relaxSafeNormalize(normalSum * invWeight, currentNormal);
    result.found = true;
    return result;
}

void relaxAccumulatePath(
    RelaxReprojectedHistory h,
    vec3 noisyRadiance,
    float noisyM2,
    float noisyHitDistance,
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
    slow.hitDistance = mix(h.hitDistance, noisyHitDistance, max(slowAlpha, 0.1));
    slow.historyLength = min(historyLength + 1.0, float(RELAX_SPEC_MAX_HISTORY));
    slow.confidence = slowConfidence;

    fast.radiance = mix(h.fastRadiance, noisyRadiance, fastAlpha);
    fast.hitDistance = mix(h.hitDistance, noisyHitDistance, max(fastAlpha, 0.1));
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
        slow.hitDistance = noisy.hitDistance;
        slow.historyLength = 0.0;
        slow.confidence = 0.0;
        RelaxFastSignal fast;
        fast.radiance = noisy.radiance;
        fast.hitDistance = noisy.hitDistance;
        fast.historyLength = 0.0;
        fast.confidence = 0.0;
        fast.materialID = 0u;
        imageStore(colorimg4, ivec2(pixel), relaxPackSlow(slow));
        imageStore(colorimg5, ivec2(pixel), relaxPackFast(fast));
#if DEBUG_VIEW == 12 || (DEBUG_VIEW >= 38 && DEBUG_VIEW <= 40)
        writeReflLight(pixel, slow.radiance, slow.hitDistance,
            slow.historyLength);
#elif DEBUG_VIEW == 14
        // N=1.zw are overwritten below by the endpoint moments. Keep the
        // diagnostic scalar in the packed color words (N=1.xy) instead.
        writeReflLight(pixel, vec3(0.0), slow.hitDistance, 0.0);
#endif
        writeReflEndpointMoments(pixel, emptyRelaxEndpointMoments());
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
    float lobeAngle = max(atan(relaxSpecLobeTanHalfAngle(currentRoughness, 0.75)),
        1.5 / 255.0);
    float surfaceViewWeight = 0.0;
    if (surface.found) {
        vec3 previousV = -relaxSafeNormalize(currentPos + cameraDelta, -V);
        float angle = acos(clamp(dot(V, previousV), -1.0, 1.0));
        surfaceViewWeight = clamp(1.0 - angle / max(lobeAngle * max(NoV, 0.05), 1e-4), 0.0, 1.0);
        surfaceViewWeight *= surface.footprintQuality;
    }

    vec3 previousSurfacePosition = currentPos + cameraDelta;
    RelaxEndpointMoments accumulatedEndpoint =
        relaxAccumulateEndpointMoments(
            surface, noisy.endpoint, surfaceViewWeight);
    vec3 currentMacroNormal =
        readMicroNormal(GEO_N_MICRONORMAL, pixel);
    RelaxEndpointProjection endpointProjection =
        relaxBuildEndpointProjection(
            previousSurfacePosition, accumulatedEndpoint,
            currentAlpha, currentMacroNormal, V);
    // A finite endpoint model cannot represent an infinity/sky component.
    // Retain the previous finite moments in storage, but do not use them to
    // reproject the current sky sample.
    endpointProjection.valid = endpointProjection.valid &&
        relaxEndpointMomentsValid(noisy.endpoint);
    RelaxReprojectedHistory virtualHistory =
        relaxLoadEllipseHistory(
            endpointProjection, pixel, currentPos, currentNormal,
            currentMaterial, cameraDelta);
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
        float footprintConfidence = 1.0 - smoothstep(
            8.0, 32.0, endpointProjection.sigmaMajorPixels);
        virtualAccumulationConfidence =
            endpointProjection.confidence * roughnessWeight *
            footprintConfidence;
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
    relaxAccumulatePath(surface, noisy.radiance, noisyM2, noisy.hitDistance,
        surfaceViewWeight, surfaceViewWeight,
        surfaceSlow, surfaceFast,
        surfaceHistoryContribution);
    relaxAccumulatePath(virtualHistory, noisy.radiance, noisyM2,
        noisy.hitDistance, virtualAccumulationConfidence,
        virtualResponsiveConfidence,
        virtualSlow, virtualFast,
        virtualHistoryContribution);

    RelaxSlowSignal outputSlow;
    RelaxFastSignal outputFast;
    outputSlow.radiance = mix(surfaceSlow.radiance, virtualSlow.radiance, virtualAmount);
    outputSlow.secondMoment = mix(surfaceSlow.secondMoment, virtualSlow.secondMoment, virtualAmount);
    outputSlow.hitDistance = mix(surfaceSlow.hitDistance, virtualSlow.hitDistance, virtualAmount);
    outputSlow.historyLength = mix(surfaceSlow.historyLength, virtualSlow.historyLength, virtualAmount);
    outputSlow.historyLength *= sqrt(max(surface.footprintQuality, 1.0 / max(outputSlow.historyLength, 1.0)));
    outputSlow.historyLength = clamp(outputSlow.historyLength, 1.0, float(RELAX_SPEC_MAX_HISTORY));
    outputSlow.confidence = mix(surfaceViewWeight,
        virtualAccumulationConfidence, virtualAmount);
    outputFast.radiance = mix(surfaceFast.radiance, virtualFast.radiance, virtualAmount);
    outputFast.hitDistance = mix(surfaceFast.hitDistance, virtualFast.hitDistance, virtualAmount);
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
    writeReflLight(pixel, noisy.radiance, noisy.hitDistance, 0.0);
#elif DEBUG_VIEW == 39
    // The history value actually fetched through surface reprojection.
    writeReflLight(pixel,
        surface.found ? relaxFiniteColor(surface.slowRadiance) : vec3(0.0),
        surface.hitDistance, surface.found ? 1.0 : 0.0);
#elif DEBUG_VIEW == 40
    // History fetched at the endpoint-moment mean with GGX covariance taps.
    writeReflLight(pixel,
        virtualHistory.found
            ? relaxFiniteColor(virtualHistory.slowRadiance) : vec3(0.0),
        virtualHistory.hitDistance, virtualHistory.found ? 1.0 : 0.0);
#elif DEBUG_VIEW == 12
    // Preserve the temporal-only result in ReflectBuffer N=1.  The later
    // RELAX passes may still execute, but resolve leaves this value untouched
    // in temporal diagnostic views, so no spatial stage contributes.
    writeReflLight(pixel, relaxFiniteColor(outputSlow.radiance),
        outputSlow.hitDistance, outputHistoryContribution);
#elif DEBUG_VIEW == 14
    // N=1.zw must carry the four endpoint moments into history clamp, so the
    // contribution cannot live in the usual accumWeight field (N=1.z).
    // Store it as grayscale in N=1.xy; writeReflEndpointMoments only replaces
    // z/w and therefore cannot corrupt this diagnostic value.
    writeReflLight(pixel, vec3(outputHistoryContribution),
        outputSlow.hitDistance, 0.0);
#endif
    writeReflEndpointMoments(pixel, accumulatedEndpoint);
}
