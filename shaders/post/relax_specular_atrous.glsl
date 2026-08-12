#version 430 core

// Non-shared-memory large-kernel implementation. The last two passes use a
// rotated Poisson disk because their 8/16-pixel halo is unsuitable for LDS.
layout(local_size_x = 8, local_size_y = 8) in;

#include "/lib/denoise/relax_specular_atrous_common.glsl"

const vec4 RELAX_POISSON_8[8] = vec4[](
    vec4(-0.4706069, -0.4427112, 0.6461146, 0.81170),
    vec4(-0.9057375,  0.3003471, 0.9542373, 0.63422),
    vec4(-0.3487388,  0.4037880, 0.5335386, 0.86734),
    vec4( 0.1023042,  0.6439373, 0.6520134, 0.80847),
    vec4( 0.5699277,  0.3513750, 0.6695386, 0.79925),
    vec4( 0.2939128, -0.1131226, 0.3149309, 0.95161),
    vec4( 0.7836658, -0.4208784, 0.8895339, 0.67328),
    vec4( 0.1564120, -0.8198990, 0.8346850, 0.70589));

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(resolution_global);
    if (!relaxInBounds(pixel, size)) return;

    vec4 centerGeometry = texelFetch(colortex9, pixel, 0);
    RelaxSpatialSignal center = relaxUnpackSpatial(
        relaxFetchAtrousWords(pixel));
    if (!relaxAtrousGeometryValid(centerGeometry)) {
        relaxStoreAtrous(pixel, center);
#if defined(RELAX_ATROUS_RESOLVE)
        writeReflMaxEnt(uvec2(pixel), emptySpecularMaxEnt(), 0.0, 0.0);
#endif
        return;
    }

    vec3 centerNormal;
    uint centerMaterial;
    relaxUnpackNormalMaterial(floatBitsToUint(centerGeometry.w),
        centerNormal, centerMaterial);
    vec3 unusedNormal;
    float centerAlpha, unusedPathRoughness;
    int unusedMaterial;
    readGeo1(GEO_N_NORMALS, uvec2(pixel), unusedNormal, centerAlpha,
        unusedMaterial, unusedPathRoughness);
    float centerRoughness = relaxPerceptualRoughness(centerAlpha);
    RelaxAtrousBuresData centerBures = relaxMakeAtrousBuresData(
        center.signal.aliceY);

    float rotationAngle = 2.0 * PI * relaxHash2(
        uvec2(pixel), uint(RELAX_ATROUS_STEP)).x;
    float cs = cos(rotationAngle);
    float sn = sin(rotationAngle);
    mat2 rotation = mat2(cs, -sn, sn, cs)
        * (float(RELAX_ATROUS_STEP) * 1.75);

    float sumWeight = 1.0;
    vec4 sumAliceY = center.signal.aliceY;
    vec2 sumCoCg = center.signal.CoCg;
    vec2 varianceEnergy = vec2(max(center.variance, 0.0));

    for (int i = 0; i < 8; ++i) {
        ivec2 samplePixel = pixel
            + ivec2(round(rotation * RELAX_POISSON_8[i].xy));
        if (!relaxInBounds(samplePixel, size)) continue;

        vec4 sampleGeometry = texelFetch(colortex9, samplePixel, 0);
        if (!relaxAtrousGeometryValid(sampleGeometry)) continue;
        vec3 sampleNormal;
        uint sampleMaterial;
        relaxUnpackNormalMaterial(floatBitsToUint(sampleGeometry.w),
            sampleNormal, sampleMaterial);
        if (sampleMaterial != centerMaterial) continue;

        RelaxSpatialSignal sampleSignal = relaxUnpackSpatial(
            relaxFetchAtrousWords(samplePixel));
        float sampleAlpha;
        readGeo1(GEO_N_NORMALS, uvec2(samplePixel), unusedNormal,
            sampleAlpha, unusedMaterial, unusedPathRoughness);
        float sampleRoughness = relaxPerceptualRoughness(sampleAlpha);
        RelaxAtrousBuresData sampleBures = relaxMakeAtrousBuresData(
            sampleSignal.signal.aliceY);
        float weight = relaxAtrousSpecularWeight(center, centerBures,
            centerGeometry.xyz, centerNormal, centerRoughness,
            sampleSignal, sampleBures, sampleGeometry.xyz, sampleRoughness,
            RELAX_POISSON_8[i].w);
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
    // The center hit distance is deliberately retained as the virtual-depth
    // descriptor; neighbor hit distances are edge-stopping data, not signal.
    relaxFinishAtrous(pixel, outputSignal);
}
