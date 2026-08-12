#version 430 core

// Shared-memory 3x3 implementation for the 1/2/4-pixel A-trous passes.
layout(local_size_x = 16, local_size_y = 16) in;

#include "/lib/denoise/relax_specular_atrous_common.glsl"

#define RELAX_TILE_SIZE (16 + 2 * RELAX_ATROUS_STEP)
#define RELAX_TILE_AREA (RELAX_TILE_SIZE * RELAX_TILE_SIZE)

shared vec4 relaxTileGeometry[RELAX_TILE_AREA];
shared uvec4 relaxTileSignalWords[RELAX_TILE_AREA];
shared uint relaxTileBuresStddev[RELAX_TILE_AREA];
shared uint relaxTileRoughness[RELAX_TILE_AREA];

const ivec2 RELAX_GRID_8[8] = ivec2[](
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0),
    ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));
const float RELAX_GRID_WEIGHT[8] = float[](
    0.44445, 0.66667, 0.44445, 0.66667,
    0.66667, 0.44445, 0.66667, 0.44445);

RelaxAtrousBuresData relaxTileMakeBuresData(
        RelaxSpatialSignal inputSignal, uint stddevWord) {
    RelaxAtrousBuresData data;
    data.stddev = unpackHalf2x16(stddevWord);
    vec2 variance = data.stddev * data.stddev;
    data.trace = 2.0 * variance.x + variance.y;
    data.anisotropy = variance.y - variance.x;
    float meanLength2 = dot(inputSignal.signal.aliceY.xyz,
        inputSignal.signal.aliceY.xyz);
    data.invMeanLength2 = meanLength2 > 1e-16
        ? 1.0 / meanLength2 : 0.0;
    return data;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    uvec2 localId = gl_LocalInvocationID.xy;
    ivec2 size = ivec2(resolution_global);
    ivec2 tileOrigin = ivec2(gl_WorkGroupID.xy * 16u)
        - ivec2(RELAX_ATROUS_STEP);

    // No invocation may return before this cooperative load and barrier.
    for (uint tileY = localId.y; tileY < uint(RELAX_TILE_SIZE);
            tileY += 16u) {
        for (uint tileX = localId.x; tileX < uint(RELAX_TILE_SIZE);
                tileX += 16u) {
            uint tileIndex = tileY * uint(RELAX_TILE_SIZE) + tileX;
            ivec2 sourcePixel = tileOrigin + ivec2(tileX, tileY);
            if (relaxInBounds(sourcePixel, size)) {
                vec4 geometry = texelFetch(colortex9, sourcePixel, 0);
                uvec4 signalWords = relaxFetchAtrousWords(sourcePixel);
                relaxTileGeometry[tileIndex] = geometry;
                relaxTileSignalWords[tileIndex] = signalWords;

                RelaxSpatialSignal tileSignal = relaxUnpackSpatial(signalWords);
                RelaxAtrousBuresData bures = relaxMakeAtrousBuresData(
                    tileSignal.signal.aliceY);
                relaxTileBuresStddev[tileIndex] = packHalf2x16(clamp(
                    bures.stddev, vec2(0.0), vec2(65504.0)));

                vec3 unusedNormal;
                float alpha, unusedPathRoughness;
                int unusedMaterial;
                readGeo1(GEO_N_NORMALS, uvec2(sourcePixel), unusedNormal,
                    alpha, unusedMaterial, unusedPathRoughness);
                relaxTileRoughness[tileIndex] = packHalf2x16(vec2(
                    relaxPerceptualRoughness(alpha), 0.0));
            } else {
                relaxTileGeometry[tileIndex] = vec4(0.0);
                relaxTileSignalWords[tileIndex] = uvec4(0u);
                relaxTileBuresStddev[tileIndex] = 0u;
                relaxTileRoughness[tileIndex] = 0u;
            }
        }
    }

    barrier();
    if (!relaxInBounds(pixel, size)) return;

    uint centerX = localId.x + uint(RELAX_ATROUS_STEP);
    uint centerY = localId.y + uint(RELAX_ATROUS_STEP);
    uint centerIndex = centerY * uint(RELAX_TILE_SIZE) + centerX;
    vec4 centerGeometry = relaxTileGeometry[centerIndex];
    RelaxSpatialSignal center = relaxUnpackSpatial(
        relaxTileSignalWords[centerIndex]);
    if (!relaxAtrousGeometryValid(centerGeometry)) {
        relaxStoreAtrous(pixel, center);
        return;
    }

    vec3 centerNormal;
    uint centerMaterial;
    relaxUnpackNormalMaterial(floatBitsToUint(centerGeometry.w),
        centerNormal, centerMaterial);
    float centerRoughness = unpackHalf2x16(
        relaxTileRoughness[centerIndex]).x;
    RelaxAtrousBuresData centerBures = relaxTileMakeBuresData(center,
        relaxTileBuresStddev[centerIndex]);

    const float centerWeight = 1.0;
    float sumWeight = centerWeight;
    vec4 sumAliceY = center.signal.aliceY * centerWeight;
    vec2 sumCoCg = center.signal.CoCg * centerWeight;
    vec2 varianceEnergy = vec2(center.variance * centerWeight,
        center.variance * centerWeight * centerWeight);

    for (int i = 0; i < 8; ++i) {
        int sampleX = int(centerX)
            + RELAX_GRID_8[i].x * RELAX_ATROUS_STEP;
        int sampleY = int(centerY)
            + RELAX_GRID_8[i].y * RELAX_ATROUS_STEP;
        uint sampleIndex = uint(sampleY * RELAX_TILE_SIZE + sampleX);
        vec4 sampleGeometry = relaxTileGeometry[sampleIndex];
        if (!relaxAtrousGeometryValid(sampleGeometry)) continue;

        vec3 sampleNormal;
        uint sampleMaterial;
        relaxUnpackNormalMaterial(floatBitsToUint(sampleGeometry.w),
            sampleNormal, sampleMaterial);
        if (sampleMaterial != centerMaterial) continue;

        RelaxSpatialSignal sampleSignal = relaxUnpackSpatial(
            relaxTileSignalWords[sampleIndex]);
        float sampleRoughness = unpackHalf2x16(
            relaxTileRoughness[sampleIndex]).x;
        RelaxAtrousBuresData sampleBures = relaxTileMakeBuresData(sampleSignal,
            relaxTileBuresStddev[sampleIndex]);
        float weight = relaxAtrousSpecularWeight(center, centerBures,
            centerGeometry.xyz, centerNormal, centerRoughness,
            sampleSignal, sampleBures, sampleGeometry.xyz, sampleRoughness,
            RELAX_GRID_WEIGHT[i]);
        if (weight <= 1e-6) continue;

        sumAliceY += sampleSignal.signal.aliceY * weight;
        sumCoCg += sampleSignal.signal.CoCg * weight;
        float weightedVariance = weight * sampleSignal.variance;
        varianceEnergy += vec2(weightedVariance,
            weight * weightedVariance);
        sumWeight += weight;
    }

    float invWeight = 1.0 / max(sumWeight, 1e-6);
    RelaxSpatialSignal outputSignal = center;
    outputSignal.signal.aliceY = sumAliceY * invWeight;
    outputSignal.signal.CoCg = sumCoCg * invWeight;
    outputSignal.signal = sanitizeSpecularMaxEnt(outputSignal.signal);
    outputSignal.variance = relaxAtrousFilteredVariance(
        varianceEnergy, sumWeight);
    relaxFinishAtrous(pixel, outputSignal);
}
