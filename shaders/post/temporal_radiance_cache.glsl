#version 430 compatibility
#ifndef TEMPORAL_RADIANCE_CACHE_GLSL
#define TEMPORAL_RADIANCE_CACHE_GLSL

// In-place sparse temporal RIS. Each pool voxel has exactly one invocation, so
// reservoir merging needs no atomics and does not increase SSBO allocation.
#include "/lib/buffers/radiance_cache.glsl"
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
const ivec3 workGroups = ivec3(16, 16, 16);

bool isFiniteRadianceCacheSample(RadianceCache c) {
    return !any(isnan(c.alice.aliceR)) && !any(isinf(c.alice.aliceR))
        && !any(isnan(c.alice.aliceG)) && !any(isinf(c.alice.aliceG))
        && !any(isnan(c.alice.aliceB)) && !any(isinf(c.alice.aliceB))
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
    return float(radianceCacheRisHash(seed)) * (1.0 / 4294967296.0);
}

float radianceCacheRisTarget(RadianceCache candidate) {
    // The cached sample already contains p_uniform / p_mixture. Luminance of
    // that vector contribution is therefore a valid scalar RIS target.
    float target = rgb_alice_luminance(candidate.alice).w;
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
    if (!(r.weightSum > 0.0) || !(denominator > 0.0)) return emptyCache();
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

    // Two independent hashes avoid order-correlated replacement decisions.
    float currentRandom = radianceCacheRisRandom(worldVoxel, frameStamp * 2u);
    float historyRandom = radianceCacheRisRandom(worldVoxel, frameStamp * 2u + 1u);
    radianceCacheRisMerge(r, current, currentRandom);
    radianceCacheRisMerge(r, history, historyRandom);
    return radianceCacheRisFinalize(r);
}

RadianceCache radianceCacheRisEstimate(RadianceCache reservoir,
        bool currentSampleAvailable) {
    RadianceCache estimate = emptyCache();
    if (isValidRadianceCacheReservoir(reservoir)) {
        estimate.alice.aliceR = reservoir.alice.aliceR * reservoir.W;
        estimate.alice.aliceG = reservoir.alice.aliceG * reservoir.W;
        estimate.alice.aliceB = reservoir.alice.aliceB * reservoir.W;
    } else if (!currentSampleAvailable) {
        return estimate;
    }

    // A valid all-black probe produces an empty RIS selection but is still a
    // valid zero estimate; preserving it is required for history fade-out.
    estimate.W = 1.0;
    estimate.M = 1.0;
    return estimate;
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
    result.alice.aliceR = mix(previousFiltered.alice.aliceR * previousFiltered.W,
        currentEstimate.alice.aliceR * currentEstimate.W, alpha);
    result.alice.aliceG = mix(previousFiltered.alice.aliceG * previousFiltered.W,
        currentEstimate.alice.aliceG * currentEstimate.W, alpha);
    result.alice.aliceB = mix(previousFiltered.alice.aliceB * previousFiltered.W,
        currentEstimate.alice.aliceB * currentEstimate.W, alpha);
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
    if (!validateRadianceCacheAddress(address)) return;
    RadianceCache current = loadRadianceCachePlanes(
        address, RC_PLANE_CURRENT_0, RC_PLANE_CURRENT_1);
    uint frameStamp = uint(max(frame_id, 0));
    bool hasHistory = radianceCacheAddressHasHistory(address, frameStamp);
    RadianceCache history = hasHistory
        ? loadRadianceCachePlanes(address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1)
        : emptyCache();
    RadianceCache previousFiltered = hasHistory
        ? loadRadianceCachePlanes(address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1)
        : emptyCache();

    RadianceCache reservoir = resampleTemporalRadiance(
        current, history, worldVoxel, frameStamp);
    RadianceCache filtered = smoothTemporalRadiance(
        radianceCacheRisEstimate(reservoir,
            isValidRadianceCacheReservoir(current)),
        previousFiltered);

    storeRadianceCachePlanes(address, RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1,
        reservoir);
    storeRadianceCachePlanes(address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1,
        filtered);
}
#endif
