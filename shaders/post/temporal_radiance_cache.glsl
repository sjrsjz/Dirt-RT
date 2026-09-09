#version 430 core
#ifndef TEMPORAL_RADIANCE_CACHE_GLSL
#define TEMPORAL_RADIANCE_CACHE_GLSL

// In-place sparse temporal processing. Each pool voxel has exactly one
// invocation, so neither the RIS reservoir nor the radiance-field accumulator
// needs atomics or additional SSBO storage.
#include "/lib/buffers/radiance_cache.glsl"
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
const ivec3 workGroups = ivec3(16, 16, 16);

bool isFiniteRadianceCacheSample(RadianceCache c) {
    return !any(isnan(c.maxent.maxentR)) && !any(isinf(c.maxent.maxentR))
        && !any(isnan(c.maxent.maxentG)) && !any(isinf(c.maxent.maxentG))
        && !any(isnan(c.maxent.maxentB)) && !any(isinf(c.maxent.maxentB))
        && !isnan(c.W) && !isinf(c.W)
        && !isnan(c.M) && !isinf(c.M);
}
bool isValidRadianceCacheReservoir(RadianceCache c) {
    return c.W > 0.0 && c.M > 0.0 && isFiniteRadianceCacheSample(c);
}

uint radianceCacheRisHash(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}
float radianceCacheRisRandom(ivec3 worldVoxel, uint frameStamp) {
    uvec3 v = uvec3(worldVoxel);
    uint seed = v.x * 0x8da6b343u ^ v.y * 0xd8163841u
        ^ v.z * 0xcb1ab31fu ^ frameStamp * 0x9e3779b9u;
    // Exact FP32 grid in [0,1): converting all 32 bits can round up to 1.
    return float(radianceCacheRisHash(seed) >> 8u) * (1.0 / 16777216.0);
}

float radianceCacheRisTarget(RadianceCache candidate) {
    // The cached sample already contains p_uniform / p_mixture. Luminance of
    // that vector contribution is therefore a valid scalar RIS target.
    float target = rgb_maxent_luminance(candidate.maxent).w;
    return (!isnan(target) && !isinf(target)) ? max(target, 0.0) : 0.0;
}

struct RadianceCacheRisReservoir {
    RadianceCache selected;
    float weightSum;
    float M;
    float selectedTarget;
};

void radianceCacheRisInit(out RadianceCacheRisReservoir r) {
    r.selected = emptyCache();
    r.weightSum = 0.0;
    r.M = 0.0;
    r.selectedTarget = 0.0;
}

void radianceCacheRisMerge(inout RadianceCacheRisReservoir r,
        RadianceCache candidate, float randomValue) {
    if (!isValidRadianceCacheReservoir(candidate)) return;

    float candidateM = min(candidate.M, max(float(RADIANCE_CACHE_MAX_HIST), 1.0));
    float target = radianceCacheRisTarget(candidate);
    float selectionWeight = target * candidate.W * candidateM;
    if (isnan(selectionWeight) || isinf(selectionWeight)) return;

    r.M += candidateM;
    // A black sample has zero selection probability but still contributes to
    // M, allowing an illuminated history reservoir to decay toward darkness.
    if (!(selectionWeight > 0.0) || !(target > 0.0)) return;

    float newWeightSum = r.weightSum + selectionWeight;
    if (isnan(newWeightSum) || isinf(newWeightSum)) return;
    if (randomValue * newWeightSum < selectionWeight) {
        r.selected = candidate;
        r.selectedTarget = target;
    }
    r.weightSum = newWeightSum;
}

RadianceCache radianceCacheRisFinalize(inout RadianceCacheRisReservoir r) {
    float maxM = max(float(RADIANCE_CACHE_MAX_HIST), 1.0);
    if (r.M > maxM) {
        float scale = maxM / r.M;
        r.weightSum *= scale;
        r.M = maxM;
    }

    float denominator = r.M * r.selectedTarget;
    if (!(r.weightSum > 0.0) || !(denominator > 0.0)) {
        // Zero-valued candidates are still samples. Preserve their M so a
        // later bright candidate is normalized by every preceding miss.
        RadianceCache black = emptyCache();
        if (r.M > 0.0) {
            black.W = 1.0;
            black.M = r.M;
        }
        return black;
    }
    float W = r.weightSum / denominator;
    if (!(W > 0.0) || isnan(W) || isinf(W)) return emptyCache();

    RadianceCache result = r.selected;
    result.W = W;
    result.M = r.M;
    return result;
}

RadianceCache resampleTemporalRadiance(RadianceCache current,
        RadianceCache history, ivec3 worldVoxel, uint frameStamp) {
    RadianceCacheRisReservoir r;
    radianceCacheRisInit(r);

    // The first positive candidate is selected with probability one. A random
    // draw here is redundant, and a rounded value of one could lose it.
    float historyRandom = radianceCacheRisRandom(worldVoxel, frameStamp * 2u + 1u);
    radianceCacheRisMerge(r, current, 0.0);
    radianceCacheRisMerge(r, history, historyRandom);
    return radianceCacheRisFinalize(r);
}

RadianceCache radianceCacheRisEstimate(RadianceCache reservoir,
        bool currentSampleAvailable) {
    RadianceCache estimate = emptyCache();
    if (isValidRadianceCacheReservoir(reservoir)) {
        estimate.maxent.maxentR = reservoir.maxent.maxentR * reservoir.W;
        estimate.maxent.maxentG = reservoir.maxent.maxentG * reservoir.W;
        estimate.maxent.maxentB = reservoir.maxent.maxentB * reservoir.W;
    } else if (!currentSampleAvailable) {
        return estimate;
    }

    // A valid all-black probe produces an empty RIS selection but is still a
    // valid zero estimate; preserving it is required for history fade-out.
    estimate.W = 1.0;
    estimate.M = 1.0;
    return estimate;
}

RadianceCache radianceCacheCurrentEstimate(RadianceCache current) {
    if (!isValidRadianceCacheReservoir(current)) return emptyCache();

    // CURRENT is a one-sample unbiased estimate of all four MaxEnt moments. Its
    // importance correction is normally already baked into maxent by ray5, but
    // canonicalizing W here keeps the filtered planes well-defined if the
    // producer representation changes later.
    current.maxent.maxentR *= current.W;
    current.maxent.maxentG *= current.W;
    current.maxent.maxentB *= current.W;
    current.W = 1.0;
    current.M = 1.0;
    return current;
}

RadianceCache smoothTemporalRadiance(RadianceCache currentEstimate,
        RadianceCache previousFiltered) {
    bool currentValid = isValidRadianceCacheReservoir(currentEstimate);
    bool previousValid = isValidRadianceCacheReservoir(previousFiltered);
    if (!currentValid) return previousValid ? previousFiltered : emptyCache();
    if (!previousValid) return currentEstimate;

    float maxHistory = max(float(RADIANCE_CACHE_FILTER_MAX_HIST), 1.0);
    float previousM = min(previousFiltered.M, max(maxHistory - 1.0, 0.0));
    float combinedM = previousM + 1.0;
    float alpha = 1.0 / combinedM;

    // Filtered planes are stored with W=1, but applying W here also makes a
    // stale or migrated valid value canonical before it is written back.
    RadianceCache result;
    result.maxent.maxentR = mix(previousFiltered.maxent.maxentR * previousFiltered.W,
        currentEstimate.maxent.maxentR * currentEstimate.W, alpha);
    result.maxent.maxentG = mix(previousFiltered.maxent.maxentG * previousFiltered.W,
        currentEstimate.maxent.maxentG * currentEstimate.W, alpha);
    result.maxent.maxentB = mix(previousFiltered.maxent.maxentB * previousFiltered.W,
        currentEstimate.maxent.maxentB * currentEstimate.W, alpha);
    result.W = 1.0;
    result.M = combinedM;
    return isFiniteRadianceCacheSample(result) ? result : currentEstimate;
}

void main() {
    uint linearIndex = gl_GlobalInvocationID.x
        + gl_GlobalInvocationID.y * 64u
        + gl_GlobalInvocationID.z * 4096u;
    ivec3 worldVoxel;
    RadianceCacheAddress address = radianceCacheAddressForPoolVoxel(
        linearIndex, camPos, worldVoxel);
    if (address.token >= RC_LOCKED_TOKEN) return;
    uint frameStamp = uint(max(frame_id, 0));
    bool hasHistory = radianceCacheResolvedPoolAddressHasHistory(
        address, frameStamp);
    if (hasHistory && !radianceCacheShouldUpdate(worldVoxel, frameStamp)) return;
    RadianceCache current = loadRadianceCachePlanes(
        address, RC_PLANE_CURRENT_0, RC_PLANE_CURRENT_1);
    RadianceCache history = hasHistory
        ? loadRadianceCachePlanes(address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1)
        : emptyCache();
    RadianceCache previousFiltered = hasHistory
        ? loadRadianceCachePlanes(address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1)
        : emptyCache();

    RadianceCache reservoir = resampleTemporalRadiance(
        current, history, worldVoxel, frameStamp);
    // Do not feed the selected RIS sample into the MaxEnt field accumulator.
    // A temporal reservoir retains one direction for O(M) frames; averaging
    // that correlated selection makes |E[L*w]| / E[L] approach one and turns
    // a multi-directional field into an artificial sharp lobe. Accumulating
    // the raw per-frame moment estimate preserves every sampled direction and
    // is the unbiased estimator required by MaxEnt's nonlinear reconstruction.
    RadianceCache filtered = smoothTemporalRadiance(
        radianceCacheCurrentEstimate(current), previousFiltered);

    storeRadianceCachePlanes(address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1,
        reservoir);
    storeRadianceCachePlanes(address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1,
        filtered);
}
#endif
