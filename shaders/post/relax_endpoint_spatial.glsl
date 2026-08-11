#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

#define REFLECT_BUFFER
#include "/lib/denoise/relax_specular_common.glsl"

layout(rgba32ui) uniform writeonly uimage2D colorimg6;

void storeSpatialEndpoint(ivec2 pixel, RelaxEndpointMoments endpoint) {
    uvec2 packedEndpoint = relaxPackEndpointMoments(endpoint);
    imageStore(colorimg6, pixel, uvec4(packedEndpoint, 0u, 0u));
}

#define ENDPOINT_GROUP_SIZE 16
#define ENDPOINT_FILTER_RADIUS 3
#define ENDPOINT_TILE_SIZE (ENDPOINT_GROUP_SIZE + 2 * ENDPOINT_FILTER_RADIUS) // 22
#define ENDPOINT_TILE_AREA (ENDPOINT_TILE_SIZE * ENDPOINT_TILE_SIZE)           // 484

shared vec4 sm_moments[ENDPOINT_TILE_AREA]; // xyz: mean, w: secondMoment
shared vec4 sm_pos_rough[ENDPOINT_TILE_AREA]; // xyz: position, w: perceptualRoughness
shared vec3 sm_normal[ENDPOINT_TILE_AREA]; // xyz: normalized normal

bool isFiniteVec3(vec3 v) {
    return all(lessThan(abs(v), vec3(1e30)));
}
bool isFiniteFloat(float f) {
    return abs(f) < 1e30;
}

void main() {
    ivec2 size = ivec2(resolution_global);
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy) * ENDPOINT_GROUP_SIZE - ivec2(ENDPOINT_FILTER_RADIUS);

    for (uint i = gl_LocalInvocationIndex; i < uint(ENDPOINT_TILE_AREA);
            i += uint(ENDPOINT_GROUP_SIZE * ENDPOINT_GROUP_SIZE)) {
        uint tx = i % uint(ENDPOINT_TILE_SIZE);
        uint ty = i / uint(ENDPOINT_TILE_SIZE);
        ivec2 q = clamp(tileOrigin + ivec2(tx, ty), ivec2(0), size - ivec2(1));

        RelaxEndpointMoments moments = readReflEndpointMoments(uvec2(q));
        vec3 position;
        float primaryDistance;
        readGeo0(GEO_N_GEO, uvec2(q), position, primaryDistance);

        vec4 surfaceData = geomBuffer.data[addr(GEO_N_NORMALS, uvec2(q))];
        vec3 normal = decodeNormal(surfaceData.x);
        float alpha = surfaceData.y;

        bool finiteGeometry = isFiniteVec3(position) && isFiniteFloat(primaryDistance) &&
                isFiniteVec3(normal) && isFiniteFloat(alpha);

        if (!finiteGeometry) {
            moments = emptyRelaxEndpointMoments();
            position = vec3(0.0);
            normal = vec3(0.0, 1.0, 0.0);
            alpha = 1.0;
        } else {
            moments = sanitizeRelaxEndpointMoments(moments);
            normal = relaxSafeNormalize(normal, vec3(0.0, 1.0, 0.0));
        }

        float perceptualRoughness = relaxPerceptualRoughness(alpha);

        sm_moments[i] = vec4(moments.mean, moments.secondMoment);
        sm_pos_rough[i] = vec4(position, perceptualRoughness);
        sm_normal[i] = normal;
    }

    barrier();

    if (!relaxInBounds(pixel, size)) return;

    ivec2 centerTile = ivec2(gl_LocalInvocationID.xy) + ivec2(ENDPOINT_FILTER_RADIUS);
    int centerIndex = centerTile.y * ENDPOINT_TILE_SIZE + centerTile.x;

    vec4 centerPacked = sm_moments[centerIndex];
    RelaxEndpointMoments center;
    center.mean = centerPacked.xyz;
    center.secondMoment = centerPacked.w;

    if (!relaxEndpointMomentsValid(center)) {
        storeSpatialEndpoint(pixel, emptyRelaxEndpointMoments());
        return;
    }

    vec4 centerPosRough = sm_pos_rough[centerIndex];
    vec3 centerPosition = centerPosRough.xyz;
    float centerRoughness = centerPosRough.w;
    vec3 centerNormal = sm_normal[centerIndex];

    float spatialSigma = 0.12 + 2.88 * centerRoughness;
    float invTwoSpatialSigma2 = 0.5 / max(spatialSigma * spatialSigma, 1e-8);

    float planeSigma = max(RELAX_DEPTH_THRESHOLD * max(length(centerPosition), 1.0), 1e-5);
    float invPlaneSigma = 1.0 / planeSigma;

    float normalSigma = 0.02 + 0.35 * centerRoughness;
    float invNormalSigma = 1.0 / normalSigma;

    float roughnessSigma = 0.03 + 0.25 * centerRoughness;
    float invTwoRoughnessSigma2 = 0.5 / max(roughnessSigma * roughnessSigma, 1e-8);

    float invEndpointScale = 1.0 / relaxEndpointDistanceScale();
    float centerPlaneDistance = dot(centerPosition, centerNormal);

    // Ignore taps whose spatial-only Gaussian contribution is below 1e-3.
    // Smooth surfaces use the support their narrow kernel actually needs;
    // rough surfaces retain the complete 7x7 footprint.
    int filterRadius = clamp(int(ceil(3.7169221888 * spatialSigma)),
        1, ENDPOINT_FILTER_RADIUS);

    vec3 sumMean = vec3(0.0);
    float sumSecondMoment = 0.0;
    float sumWeight = 0.0;

    for (int oy = -filterRadius; oy <= filterRadius; ++oy) {
        for (int ox = -filterRadius; ox <= filterRadius; ++ox) {
            int sampleIndex = centerIndex + (oy * ENDPOINT_TILE_SIZE + ox);

            vec4 sampleMomentsPacked = sm_moments[sampleIndex];
            vec3 sampleMean = sampleMomentsPacked.xyz;
            float sampleSecondMoment = sampleMomentsPacked.w;

            float endpointPresence = step(1e-20, sampleSecondMoment);
            if (endpointPresence <= 0.0) continue; // 快速跳过无效/空样本

            vec4 samplePosRough = sm_pos_rough[sampleIndex];
            vec3 samplePosition = samplePosRough.xyz;
            float sampleRoughness = samplePosRough.w;
            vec3 sampleNormal = sm_normal[sampleIndex];

            vec3 originDelta = (samplePosition - centerPosition) * invEndpointScale;
            vec3 rebasedMean = sampleMean + originDelta;
            float rebasedSecondMoment = sampleSecondMoment + dot(originDelta, sampleMean + rebasedMean);

            float radiusSquared = float(ox * ox + oy * oy);
            float planeDistance = abs(dot(samplePosition, centerNormal) -
                centerPlaneDistance);
            float normalDifference = 1.0 - clamp(dot(centerNormal, sampleNormal), -1.0, 1.0);
            float roughnessDifference = sampleRoughness - centerRoughness;

            float exponent = radiusSquared * invTwoSpatialSigma2 +
                    planeDistance * invPlaneSigma +
                    normalDifference * invNormalSigma +
                    (roughnessDifference * roughnessDifference) * invTwoRoughnessSigma2;

            float weight = exp(-exponent);

            sumMean += rebasedMean * weight;
            sumSecondMoment += rebasedSecondMoment * weight;
            sumWeight += weight;
        }
    }

    RelaxEndpointMoments filtered;
    float inverseWeight = 1.0 / max(sumWeight, 1e-20);
    filtered.mean = sumMean * inverseWeight;
    filtered.secondMoment = sumSecondMoment * inverseWeight;
    filtered = sanitizeRelaxEndpointMoments(filtered);

    storeSpatialEndpoint(pixel, filtered);
}
