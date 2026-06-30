#version 430 compatibility
// Horizontal 33-tap Gaussian, LOD-aware, 256x1 workgroup + shared memory
layout(local_size_x = 256, local_size_y = 1) in;
layout(rgba16f) uniform image2D bloomAtlas;
layout(rgba16f) uniform writeonly image2D bloomBlur;

#include "/lib/bloom.glsl"

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
            sm_rgb[i] = imageLoad(bloomAtlas, sc).rgb;
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

    // --- 33-tap filter from shared memory ---
    vec3 sum = vec3(0);
    float weightSum = 0.0;
    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
        float w = exp(-float(dx * dx) * 0.05);
        weightSum += w;

        ivec2 sampleCoord = coord + ivec2(dx, 0);
        if (sampleCoord.x >= regionMin.x && sampleCoord.x <= regionMax.x
         && sampleCoord.y >= regionMin.y && sampleCoord.y <= regionMax.y)
            sum += sm_rgb[lx + RADIUS + dx] * w;
    }
    vec3 result = sum / max(weightSum, 1e-5);
    imageStore(bloomBlur, coord, vec4(result, 1.0));
}
