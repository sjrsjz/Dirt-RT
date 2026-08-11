#version 430 core
// Horizontal 33-tap Gaussian, LOD-aware, 256x1 workgroup + shared memory
layout(local_size_x = 256, local_size_y = 1) in;
layout(rgba32f) uniform image2D bloomAtlas;
layout(rgba32f) uniform writeonly image2D bloomBlur;

#include "/lib/post_processing/bloom.glsl"

const uint GROUP_SIZE = 256u;
const int  RADIUS  = 16;
const uint TILE_W  = GROUP_SIZE + 2u * uint(RADIUS); // 288

shared vec3 sm_rgb[TILE_W];

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint lx = gl_LocalInvocationID.x;
    ivec2 atlasSize = imageSize(bloomAtlas);
    ivec2 coord = ivec2(gx, gy);

    // --- Coop load: absolute group-origin mapping ---
    uint group_origin_x = gl_WorkGroupID.x * GROUP_SIZE;

    for (uint i = lx; i < TILE_W; i += GROUP_SIZE) {
        int sample_x = int(group_origin_x) - RADIUS + int(i);
        ivec2 sc = ivec2(sample_x, int(gy));
        if (sc.x >= 0 && sc.x < atlasSize.x && sc.y >= 0 && sc.y < atlasSize.y)
            sm_rgb[i] = bloomSafeFloat(imageLoad(bloomAtlas, sc).rgb);
        else
            sm_rgb[i] = vec3(0);
    }
    barrier();
    memoryBarrierShared();

    // --- Center LOD check ---
    if (coord.x >= atlasSize.x || coord.y >= atlasSize.y) return;
    int level; ivec2 regionMin, regionMax;
    bloomFindLOD(coord, atlasSize, level, regionMin, regionMax);
    if (level < 0) { imageStore(bloomBlur, coord, vec4(0)); return; }

    // L0 carries the highest-frequency bloom signal. Its spatial Gaussian is
    // the sigma -> 0 limit: an exact center-tap copy with no 33-wide support.
    if (level == 0) {
        imageStore(bloomBlur, coord, vec4(bloomSafeFloat(sm_rgb[lx + RADIUS]), 1.0));
        return;
    }

    // Sigma grows continuously from the exact-zero L0 limit. The previous
    // sqrt(2.5 * (level + 1)) schedule made even L1 excessively broad.
    float greenSigma = 0.5 * float(level);
    float greenSigmaCoefficient = LOG2_E / (2.0 * greenSigma * greenSigma);
    vec3 rgbSigma = greenSigma * bloomDiffusionSigmaScale();
    int kernelRadius = clamp(int(ceil(3.0 * max(rgbSigma.r, max(rgbSigma.g, rgbSigma.b)))), 1, RADIUS);

    vec3 sum = vec3(0);
    vec3 weightSum = vec3(0.0);
    for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
        vec3 w = bloomGaussianWeight(float(dx * dx), greenSigmaCoefficient);
        weightSum += w;

        ivec2 sampleCoord = coord + ivec2(dx, 0);
        if (sampleCoord.x >= regionMin.x && sampleCoord.x <= regionMax.x
         && sampleCoord.y >= regionMin.y && sampleCoord.y <= regionMax.y)
            sum += sm_rgb[lx + RADIUS + dx] * w;
    }
    vec3 result = sum / max(weightSum, vec3(1e-5));
    imageStore(bloomBlur, coord, vec4(bloomSafeFloat(result), 1.0));
}
