#version 430 core
// Vertical 33-tap Gaussian, LOD-aware, 1x256 workgroup + shared memory
layout(local_size_x = 1, local_size_y = 256) in;
layout(rgba32f) uniform image2D bloomBlur;
layout(rgba32f) uniform writeonly image2D bloomAtlas;

#include "/lib/post_processing/bloom.glsl"

const uint GROUP_SIZE = 256u;
const int  RADIUS  = 16;
const uint TILE_H  = GROUP_SIZE + 2u * uint(RADIUS); // 288

shared vec3 sm_rgb[TILE_H];

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint ly = gl_LocalInvocationID.y;
    ivec2 atlasSize = imageSize(bloomBlur);
    ivec2 coord = ivec2(gx, gy);

    // --- Coop load: absolute group-origin mapping ---
    uint group_origin_y = gl_WorkGroupID.y * GROUP_SIZE;

    for (uint i = ly; i < TILE_H; i += GROUP_SIZE) {
        int sample_y = int(group_origin_y) - RADIUS + int(i);
        ivec2 sc = ivec2(int(gx), sample_y);
        if (sc.x >= 0 && sc.x < atlasSize.x && sc.y >= 0 && sc.y < atlasSize.y)
            sm_rgb[i] = bloomSafeFloat(imageLoad(bloomBlur, sc).rgb);
        else
            sm_rgb[i] = vec3(0);
    }
    barrier();
    memoryBarrierShared();

    // --- Center LOD check ---
    if (coord.x >= atlasSize.x || coord.y >= atlasSize.y) return;
    int level; ivec2 regionMin, regionMax;
    bloomFindLOD(coord, atlasSize, level, regionMin, regionMax);
    if (level < 0) { imageStore(bloomAtlas, coord, vec4(0)); return; }

    // Match the horizontal pass: L0 is the exact sigma -> 0 center sample.
    if (level == 0) {
        imageStore(bloomAtlas, coord, vec4(bloomSafeFloat(sm_rgb[ly + RADIUS]), 1.0));
        return;
    }

    // Use the same continuous all-LOD sigma schedule as the horizontal pass.
    float greenSigma = 0.5 * float(level);
    float greenSigmaCoefficient = LOG2_E / (2.0 * greenSigma * greenSigma);
    vec3 rgbSigma = greenSigma * bloomDiffusionSigmaScale();
    int kernelRadius = clamp(int(ceil(3.0 * max(rgbSigma.r, max(rgbSigma.g, rgbSigma.b)))), 1, RADIUS);

    vec3 sum = vec3(0);
    vec3 weightSum = vec3(0.0);
    for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
        vec3 w = bloomGaussianWeight(float(dy * dy), greenSigmaCoefficient);
        weightSum += w;

        ivec2 sampleCoord = coord + ivec2(0, dy);
        if (sampleCoord.x >= regionMin.x && sampleCoord.x <= regionMax.x
         && sampleCoord.y >= regionMin.y && sampleCoord.y <= regionMax.y)
            sum += sm_rgb[ly + RADIUS + dy] * w;
    }
    vec3 result = sum / max(weightSum, vec3(1e-5));
    imageStore(bloomAtlas, coord, vec4(bloomSafeFloat(result), 1.0));
}
