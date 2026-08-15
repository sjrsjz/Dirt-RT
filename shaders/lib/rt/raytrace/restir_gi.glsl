#ifndef DIRT_RT_RAYTRACE_RESTIR_GI_GLSL
#define DIRT_RT_RAYTRACE_RESTIR_GI_GLSL

// Low-history ReSTIR GI path-guiding prewarm resolve and commit.

// ===========================================================================
// Low-history ReSTIR GI path-guiding prewarm
// ===========================================================================
//
// ray1 draws one canonical first-direction proposal at every primary surface.
// ray4 treats compatible current-frame neighbours as a deliberately biased
// prewarm distribution: it selects one direction in proportion to the donor's
// luminance, but assigns that direction the compatible set's mean luminance.
// This drops reciprocal-selection, transformed-density and suffix-replay
// weights. The estimator is therefore not an unbiased GI estimator; it is a
// bounded, low-variance directional prior for MaxEnt path guiding. Only the
// selected shifted donor receives one geometry-only visibility query.
//
// The biased atom is used only while geometrically validated history is low,
// then fades out and is cleared. Mature pixels always use the canonical path
// estimator, so this pass cannot accumulate a permanent high-history bias or
// preserve stale ReSTIR visibility.

#if defined(RESTIR_GI_RESOLVE_PASS)

const int RESTIR_GI_MAX_DOMAINS = 9;
const vec2 RESTIR_GI_POISSON[8] = vec2[8](
    vec2(-0.7071068, -0.7071068),
    vec2( 0.0000000, -1.0000000),
    vec2( 0.9238795, -0.3826834),
    vec2( 0.7071068,  0.7071068),
    vec2( 0.0000000,  1.0000000),
    vec2(-0.9238795,  0.3826834),
    vec2( 0.3826834, -0.9238795),
    vec2(-0.3826834,  0.9238795));

struct RestirGIPrimaryDomain {
    uvec2 pixel;
    vec3 position;
    vec3 geometryNormal;
    vec3 macroNormal;
    bool valid;
};

bool restirGIIsFinite(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

bool restirGISafeNormalize(vec3 value, out vec3 normalizedValue) {
    float length2 = dot(value, value);
    if (!restirGIIsFinite(value) || isnan(length2) || isinf(length2)
            || length2 <= 1e-12) {
        normalizedValue = vec3(0.0, 1.0, 0.0);
        return false;
    }
    normalizedValue = value * inversesqrt(length2);
    if (!restirGIIsFinite(normalizedValue)) {
        normalizedValue = vec3(0.0, 1.0, 0.0);
        return false;
    }
    return true;
}

vec3 restirGIDecodeRGB(MaxEntEncoding encoded) {
    float Y = encoded.maxEntY.w;
    float Co = encoded.CoCg.x;
    float Cg = encoded.CoCg.y;
    float t = Y - Cg;
    return max(vec3(t + Co, Y + Cg, t - Co), vec3(0.0));
}

RestirGIPrimaryDomain restirGILoadDomain(uvec2 pixel, vec3 cameraOrigin) {
    RestirGIPrimaryDomain domain;
    domain.pixel = pixel;
    vec3 positionRelative;
    float distance;
    readDiffusePrimaryGeometry(pixel, positionRelative, distance);
    domain.position = cameraOrigin + positionRelative;
    vec3 diffuseAlbedoUnused;
    float roughnessUnused;
    readDiffuseSurface(pixel, domain.geometryNormal, domain.macroNormal,
        diffuseAlbedoUnused, roughnessUnused);
    vec3 geometryNormal = domain.geometryNormal;
    vec3 macroNormal = domain.macroNormal;
    bool geometryNormalValid = restirGISafeNormalize(
        geometryNormal, domain.geometryNormal);
    bool macroNormalValid = restirGISafeNormalize(
        macroNormal, domain.macroNormal);
    domain.valid = distance >= 0.0
        && restirGIIsFinite(domain.position)
        && geometryNormalValid && macroNormalValid;
    return domain;
}

vec3 restirGITraceOrigin(RestirGIPrimaryDomain domain) {
    // Match ray1's geometric-normal offset exactly when reconnecting the
    // selected donor endpoint from the target surface.
    return domain.position + domain.geometryNormal * 0.001;
}

bool restirGIDomainAlreadyPresent(
        RestirGIPrimaryDomain domains[RESTIR_GI_MAX_DOMAINS],
        int domainCount, uvec2 pixel) {
    for (int i = 0; i < RESTIR_GI_MAX_DOMAINS; ++i) {
        if (i >= domainCount) break;
        if (all(equal(domains[i].pixel, pixel))) return true;
    }
    return false;
}

int restirGIBuildDomains(RestirGIPrimaryDomain target, vec3 cameraOrigin,
        out RestirGIPrimaryDomain domains[RESTIR_GI_MAX_DOMAINS]) {
    uvec2 targetPixel = target.pixel;
    domains[0] = target;
    // Resolve validates the target before entering the neighbour builder.
    int count = 1;

    float angle = 2.0 * PI * hash12(vec2(targetPixel)
        + vec2(float(cam.frameId) * 0.754877666,
            float(cam.frameId) * 0.569840291));
    mat2 rotation = mat2(cos(angle), -sin(angle),
                         sin(angle),  cos(angle));
    ivec2 resolution = ivec2(resolution_global);
    int requested = min(RESTIR_GI_SPATIAL_SAMPLES, 8);
    for (int i = 0; i < 8; ++i) {
        if (i >= requested || count >= RESTIR_GI_MAX_DOMAINS) break;
        ivec2 offset = ivec2(round(rotation * RESTIR_GI_POISSON[i]
            * RESTIR_GI_SPATIAL_RADIUS));
        if (all(equal(offset, ivec2(0)))) continue;
        ivec2 neighbourI = ivec2(targetPixel) + offset;
        if (any(lessThan(neighbourI, ivec2(0)))
                || any(greaterThanEqual(neighbourI, resolution))) continue;
        uvec2 neighbourPixel = uvec2(neighbourI);
        if (restirGIDomainAlreadyPresent(
                domains, count, neighbourPixel)) continue;

        RestirGIPrimaryDomain neighbour = restirGILoadDomain(
            neighbourPixel, cameraOrigin);
        if (!neighbour.valid) continue;

        // Both tests use only current primary geometry, never the proposal's
        // radiance. They prevent donors from crossing a block corner or a
        // nearby parallel layer before luminance resampling begins.
        vec3 separation = neighbour.position - target.position;
        bool samePlane = dot(target.geometryNormal,
                neighbour.geometryNormal) >= RESTIR_GI_NORMAL_COS
            && abs(dot(separation, target.geometryNormal))
                <= RESTIR_GI_PLANE_DISTANCE
            && abs(dot(separation, neighbour.geometryNormal))
                <= RESTIR_GI_PLANE_DISTANCE;
        if (!samePlane) continue;
        domains[count++] = neighbour;
    }
    return count;
}

bool restirGIGeometrySegmentClear(vec3 rayOrigin, vec3 direction,
        float traceDistance) {
    if (!restirGIIsFinite(rayOrigin) || !restirGIIsFinite(direction)
            || isnan(traceDistance) || isinf(traceDistance)) return false;

    const float geometryTMin = 1e-4;
    float geometryTMax = min(max(traceDistance, 0.0),
        RESTIR_GI_VISIBILITY_MAX_DISTANCE);
    if (geometryTMax <= geometryTMin) return true;

    // This is a dedicated geometry-only support channel. Opaque suppresses
    // any-hit (alpha/transmission/material sampling), SkipClosestHit suppresses
    // closest-hit payload construction, and TerminateOnFirstHit makes the
    // first triangle sufficient. The regular miss shader changes the positive
    // sentinel to -1; an opaque hit leaves it untouched.
    Payload savedPayload = payload;
    payload_packHitDistance(payload.data, 1.0);
    traceRayEXT(acc,
        gl_RayFlagsOpaqueEXT
            | gl_RayFlagsTerminateOnFirstHitEXT
            | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 0, rayOrigin, geometryTMin, direction,
        geometryTMax, 6);
    bool clear = payload_unpackHitDistance(payload.data) < -0.5;
    payload = savedPayload;
    return clear;
}

bool restirGIProposalCanReach(RestirGIPrimaryDomain source,
        RestirGIFreshCandidate candidate, vec3 endpoint,
        vec3 direction) {
    if (!restirGIIsFinite(endpoint) || !restirGIIsFinite(direction))
        return false;
    vec3 rayOrigin = restirGITraceOrigin(source);
    if (candidate.environment)
        return restirGIGeometrySegmentClear(rayOrigin, direction,
            RESTIR_GI_VISIBILITY_MAX_DISTANCE);

    float endpointDistance = length(endpoint - rayOrigin);
    if (isnan(endpointDistance) || isinf(endpointDistance)) return false;
    // Stop just before the reused vertex: its own triangle is the destination,
    // not an occluder. The dedicated cap bounds traversal through large empty
    // regions; geometry beyond the cap is deliberately not queried.
    float endpointGuard = max(0.0025, endpointDistance * 0.0005);
    return restirGIGeometrySegmentClear(rayOrigin, direction,
        max(endpointDistance - endpointGuard, 0.0));
}

bool restirGILoadCandidateAtTarget(RestirGIPrimaryDomain source,
        RestirGIPrimaryDomain target, vec3 cameraOrigin,
        out RestirGIFreshCandidate candidate, out vec3 endpoint,
        out vec3 targetDirection) {
    candidate = readRestirGIFreshCandidate(source.pixel);
    if (!restirGIFreshCandidateValid(candidate)) return false;

    vec3 endpointDirection;
    if (candidate.environment) {
        if (!restirGISafeNormalize(
                candidate.endpointRelative, endpointDirection)) return false;
        endpoint = cameraOrigin + endpointDirection * VPROJDIST_SKY;
        targetDirection = endpointDirection;
    } else {
        endpoint = cameraOrigin + candidate.endpointRelative;
        if (!restirGIIsFinite(endpoint)
                || !restirGISafeNormalize(endpoint - restirGITraceOrigin(
                    target), targetDirection)) return false;
    }
    // The MaxEnt mixture has support over the complete upper hemisphere. A
    // candidate outside either geometric or macro hemisphere is unusable at
    // the target and is rejected without evaluating a guide PDF.
    return dot(target.geometryNormal, targetDirection) > 1e-6
        && dot(target.macroNormal, targetDirection) > 1e-6;
}

MaxEntEncoding restirGILoadCanonicalIndirect(uvec2 pixel) {
    MaxEntEncoding canonical;
    float unusedMeanY2;
    readDiffuseLightRT(pixel, canonical, unusedMeanY2);
    if (any(isnan(canonical.maxEntY)) || any(isinf(canonical.maxEntY))
            || any(isnan(canonical.CoCg)) || any(isinf(canonical.CoCg)))
        return init_maxent();
    return canonical;
}

vec3 restirGIReproject(vec3 worldPosition) {
    vec3 previousRelativePosition = worldPosition - prevRaytracingCamPos;
    vec4 clip = rtPrevViewProjection
        * vec4(previousRelativePosition, 1.0);
    if (!(abs(clip.w) > 1e-8) || isnan(clip.w) || isinf(clip.w))
        return vec3(-1.0);
    return clip.xyz / clip.w * 0.5 + 0.5;
}

float restirGIReprojectedHistoryWeight(RestirGIPrimaryDomain target) {
    vec3 previousUv = restirGIReproject(target.position);
    if (!restirGIIsFinite(previousUv)
            || any(lessThan(previousUv.xy, vec2(0.0)))
            || any(greaterThanEqual(previousUv.xy, vec2(1.0)))) return 0.0;

    ivec2 previousPixel = clamp(ivec2(previousUv.xy
        * vec2(resolution_global)), ivec2(0),
        ivec2(resolution_global) - 1);
    MaxEntEncoding unusedHistory;
    float historyWeight, unusedMeanY2;
    readDiffuseHist(uvec2(previousPixel), unusedHistory,
        historyWeight, unusedMeanY2);
    if (historyWeight <= 0.0 || isnan(historyWeight)
            || isinf(historyWeight)) return 0.0;

    vec3 historyPosition, historyNormal;
    readDiffuseHistGeo(uvec2(previousPixel), historyPosition, historyNormal);
    if (!restirGIIsFinite(historyPosition)
            || !restirGIIsFinite(historyNormal)) return 0.0;
    vec3 normalizedHistoryNormal;
    if (!restirGISafeNormalize(
            historyNormal, normalizedHistoryNormal)) return 0.0;

    // N3 positions are stored relative to the camera of their frame.
    vec3 expectedPreviousPosition = target.position - prevRaytracingCamPos;
    vec3 historyDelta = historyPosition - expectedPreviousPosition;
    bool sameSurface = dot(target.geometryNormal,
            normalizedHistoryNormal) >= RESTIR_GI_NORMAL_COS
        && abs(dot(historyDelta, target.geometryNormal))
            <= RESTIR_GI_PLANE_DISTANCE
        && abs(dot(historyDelta, normalizedHistoryNormal))
            <= RESTIR_GI_PLANE_DISTANCE;
    return sameSurface ? historyWeight : 0.0;
}

float restirGIHistoryActivity(RestirGIPrimaryDomain target) {
    float fadeStart = min(RESTIR_GI_HISTORY_FADE_START,
        RESTIR_GI_HISTORY_FADE_END);
    float fadeEnd = max(RESTIR_GI_HISTORY_FADE_END,
        fadeStart + 1e-3);
    float historyWeight = restirGIReprojectedHistoryWeight(target);
    return 1.0 - smoothstep(fadeStart, fadeEnd, historyWeight);
}

void ResolveFirstBounceRestirGI(uvec2 coord, vec3 ro) {
    #if RESTIR_GI_ENABLED && EON_ENABLED
    RestirGIPrimaryDomain target = restirGILoadDomain(coord, ro);
    if (!target.valid) {
        writeRestirGIPrewarm(coord, init_maxent());
        return;
    }

    // A mature pixel does not merely blend toward canonical: it clears the
    // transient atom. This lets both ray5 and the path-guide pass distinguish
    // "no prewarm" from a current-frame biased prior without another plane.
    float restirActivity = restirGIHistoryActivity(target);
    if (restirActivity <= 1e-4) {
        writeRestirGIPrewarm(coord, init_maxent());
        return;
    }

    // Mature pixels only clear the transient atom, so they need no canonical
    // light-buffer read. Defer it until a fallback or mixture actually uses it.
    MaxEntEncoding canonicalIndirect = restirGILoadCanonicalIndirect(coord);

    RestirGIPrimaryDomain domains[RESTIR_GI_MAX_DOMAINS];
    int domainCount = restirGIBuildDomains(target, ro, domains);
    int eligibleCount = 0;
    int selectedIndex = -1;
    float luminanceSum = 0.0;
    MaxEntEncoding selectedIndirect = init_maxent();
    RestirGIFreshCandidate selectedCandidate;
    vec3 selectedEndpoint = vec3(0.0);
    vec3 selectedDirection = target.geometryNormal;

    // One buffer-only reservoir scan. Bright canonical atoms are deliberately
    // more likely to provide the direction, but zero atoms still count toward
    // the local mean. Unlike an unbiased RIS estimator, no 1/P(select) factor
    // is applied later; that is the intentional variance-reducing bias.
    for (int j = 0; j < RESTIR_GI_MAX_DOMAINS; ++j) {
        if (j >= domainCount) break;
        RestirGIFreshCandidate candidate;
        vec3 endpoint, targetDirection;
        if (!restirGILoadCandidateAtTarget(domains[j], target, ro,
                candidate, endpoint, targetDirection)) continue;

        ++eligibleCount;
        MaxEntEncoding sourceIndirect = restirGILoadCanonicalIndirect(
            domains[j].pixel);
        float sourceY = max(sourceIndirect.maxEntY.w, 0.0);
        if (!(sourceY > 1e-8) || isnan(sourceY) || isinf(sourceY)) continue;

        float newLuminanceSum = luminanceSum + sourceY;
        float selectionRandom = hash12(vec2(coord)
            + vec2(float(cam.frameId) * 0.618033989
                    + float(j) * 17.371,
                float(cam.frameId) * 0.414213562
                    + float(j) * 31.719));
        if (selectionRandom * newLuminanceSum < sourceY) {
            selectedIndex = j;
            selectedIndirect = sourceIndirect;
            selectedCandidate = candidate;
            selectedEndpoint = endpoint;
            selectedDirection = targetDirection;
        }
        luminanceSum = newLuminanceSum;
    }

    if (eligibleCount == 0 || selectedIndex < 0
            || !(luminanceSum > 0.0) || isnan(luminanceSum)
            || isinf(luminanceSum)) {
        writeRestirGIPrewarm(coord, canonicalIndirect);
        return;
    }

    // The target's own canonical path is already known visible. A shifted
    // donor gets exactly one opaque, terminate-on-first-hit geometry query.
    // Material, alpha, texture and suffix evaluation are intentionally absent.
    if (selectedIndex != 0 && !restirGIProposalCanReach(target,
            selectedCandidate, selectedEndpoint, selectedDirection)) {
        writeRestirGIPrewarm(coord, canonicalIndirect);
        return;
    }

    float selectedY = max(selectedIndirect.maxEntY.w, 1e-8);
    float localMeanY = luminanceSum / float(max(eligibleCount, 1));
    vec3 biasedRadiance = restirGIDecodeRGB(selectedIndirect)
        * (localMeanY / selectedY);
    if (!restirGIIsFinite(biasedRadiance)) biasedRadiance = vec3(0.0);

    // Preserve the selected direction/chroma, but bound its energy by the
    // compatible local mean and the existing GI hard cap. This is the core
    // low-variance trade: bright donors steer the guide without producing a
    // reciprocal-probability firefly.
    vec3 storageRadiance = clamp(
        biasedRadiance, vec3(0.0), vec3(GI_CLAMP_MAX));
    vec3 encodingDirection;
    if (!restirGISafeNormalize(selectedDirection, encodingDirection))
        encodingDirection = target.geometryNormal;
    MaxEntEncoding biasedPrewarm = radiance_to_maxent(
        storageRadiance, encodingDirection);
    writeRestirGIPrewarm(coord, mix_maxent(
        canonicalIndirect, biasedPrewarm, restirActivity));
    #endif
}

#elif defined(RESTIR_GI_FINAL_PASS)

void FinalizeFirstBounceRestirGI(uvec2 coord) {
    #if RESTIR_GI_ENABLED && EON_ENABLED
    MaxEntEncoding canonicalIndirect;
    float canonicalMeanY2;
    readDiffuseLightRT(coord, canonicalIndirect, canonicalMeanY2);

    MaxEntEncoding prewarmIndirect;
    MaxEntEncoding direct;
    readRestirGIPrewarm(coord, prewarmIndirect);
    readRestirGIDirect(coord, direct);

    // A zero prewarm plane means the low-history system has retired (or had no
    // safe candidate). Keep the canonical estimator as the authoritative
    // fallback; the biased atom never becomes persistent history on its own.
    bool usePrewarm = prewarmIndirect.maxEntY.w > 1e-8
        && !any(isnan(prewarmIndirect.maxEntY))
        && !any(isinf(prewarmIndirect.maxEntY));
    MaxEntEncoding indirect = usePrewarm
        ? prewarmIndirect : canonicalIndirect;
    indirect.maxEntY += direct.maxEntY;
    indirect.CoCg += direct.CoCg;
    float meanY2 = indirect.maxEntY.w * indirect.maxEntY.w;
    writeDiffuseLightRT(coord, indirect, meanY2);
    #endif
}

#endif

#endif // DIRT_RT_RAYTRACE_RESTIR_GI_GLSL
