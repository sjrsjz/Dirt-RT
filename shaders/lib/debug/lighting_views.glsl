#ifndef LIGHTING_DEBUG_VIEWS_GLSL
#define LIGHTING_DEBUG_VIEWS_GLSL

// Display-only helpers. Diagnostic selection remains in the composite pass.
// ---------------------------------------------------------------------------
// Jet/rainbow colormap: [0,1] → blue→cyan→green→yellow→red
// ---------------------------------------------------------------------------
vec3 jetColormap(float t) {
    return vec3(
        clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0),
        clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0),
        clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0));
}

// Log-scale normalize for ray-segment distance (0.01m..~160m → [0,1])
float logDistNorm(float d) {
    return clamp(log2(max(d, 0.01) * 100.0 + 1.0) / 14.0, 0.0, 1.0);
}

#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_NOISE_ONLY_CURRENT_WEIGHT || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_NOISE_ONLY_CURRENT_WEIGHT
vec3 debugNoiseOnlyCurrentWeight(float currentWeight) {
    if (!(currentWeight >= 0.0) || isnan(currentWeight) || isinf(currentWeight))
        return vec3(1.0, 0.0, 1.0);
    return jetColormap(clamp(currentWeight, 0.0, 1.0));
}
#endif

#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_PREPARED_MONTE_CARLO_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_FILTERED_MONTE_CARLO_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_PREPARED_MONTE_CARLO_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_FILTERED_MONTE_CARLO_VARIANCE
vec3 debugMonteCarloVariance(float standardDeviation) {
    // Neutral gray is valid light without a usable uncertainty estimate.
    if (standardDeviation == -2.0) return vec3(0.35);
    if (!(standardDeviation >= 0.0) || isnan(standardDeviation)
            || isinf(standardDeviation))
        return vec3(1.0, 0.0, 1.0);
    float variance = standardDeviation * standardDeviation;
    // Log2 display of the actual trace variance. V=1 occupies 1/16 of the
    // scale and V=65535 reaches red; larger HDR variances saturate.
    float normalizedVariance = clamp(log2(1.0 + variance) / 16.0,
        0.0, 1.0);
    return jetColormap(normalizedVariance);
}
#endif

#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_KISH_EFFECTIVE_SAMPLES || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_KISH_EFFECTIVE_SAMPLES
vec3 debugKishEffectiveSamples(float effectiveSamples) {
    if (!(effectiveSamples >= 1.0) || isnan(effectiveSamples) || isinf(effectiveSamples)) return vec3(1.0, 0.0, 1.0);
    return jetColormap(clamp(log2(effectiveSamples) / log2(65504.0), 0.0, 1.0));
}
#endif

#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_FINAL_VIRTUAL_NORMAL
bool debugReadFinalDenoisedVirtualPosition(ivec2 pixel,
        out vec3 position) {
    if (any(lessThan(pixel, ivec2(0)))
            || any(greaterThanEqual(pixel, ivec2(resolution_global)))) {
        position = vec3(0.0);
        return false;
    }

    SpecularMaxEnt unusedSignal;
    float virtualDistance, validWeight;
    readReflMaxEnt(uvec2(pixel), unusedSignal, virtualDistance, validWeight);
    if (!(validWeight > 0.0) || !(virtualDistance > 0.0)
            || isnan(virtualDistance) || isinf(virtualDistance)) {
        position = vec3(0.0);
        return false;
    }

    position = reconstructPrimaryRay(uvec2(pixel)) * virtualDistance;
    return !any(isnan(position)) && !any(isinf(position));
}

bool debugFinalDenoisedVirtualNormal(ivec2 pixel, out vec3 normal) {
    vec3 centerPosition;
    if (!debugReadFinalDenoisedVirtualPosition(pixel, centerPosition)) {
        normal = vec3(0.0);
        return false;
    }

    // Match the denoiser's one-pixel central-difference reconstruction.
    // Missing image-edge or invalid neighbors collapse to the center point.
    vec3 left = centerPosition;
    vec3 right = centerPosition;
    vec3 down = centerPosition;
    vec3 up = centerPosition;
    vec3 candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(-1, 0), candidate)) left = candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(1, 0), candidate)) right = candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(0, -1), candidate)) down = candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(0, 1), candidate)) up = candidate;

    vec3 virtualTangentX = right - left;
    vec3 virtualTangentY = up - down;
    vec3 unnormalizedNormal = cross(virtualTangentX, virtualTangentY);
    float normalLength2 = dot(unnormalizedNormal, unnormalizedNormal);
    vec3 fallback = normalize(reconstructPrimaryRay(uvec2(pixel)));
    normal = normalLength2 > 1e-20
        ? unnormalizedNormal * inversesqrt(normalLength2) : fallback;
    return !any(isnan(normal)) && !any(isinf(normal));
}
#endif

#endif
