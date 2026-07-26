#ifndef TEMPORAL_RADIANCE_CACHE_GLSL
#define TEMPORAL_RADIANCE_CACHE_GLSL
#include "/lib/buffers/radiance_cache.glsl"
layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
const ivec3 workGroups = ivec3(RADIANCE_CACHE_W / 4, RADIANCE_CACHE_H / 4, RADIANCE_CACHE_D / 4);

struct HistRC {
    RadianceCache hist;
    bool valid;
}

shared HistRC sharedHistRC[6][6][6];

// 我们约定辐射率缓存相对于玩家有如下形式：
// O --- O --- O --- O
// |     |     |     |
// O --- O --- O --- O
// |     |  c  |     |
// O --- O --- O --- O
// |     |     |     |
// O --- O --- O --- O
// 其中 c 是摄像机坐标，O 是 RC 采样点。也就是说，它并非采取纹理中心对齐，而是采取 RC 采样点对齐。
// 这样做的好处是，RC 采样点的坐标是整数，便于计算和索引，并且能完美做到中心对称
// 需要注意的是，无论如何都应当用 VOXEL_SIZE 来计算 RC 采样点的坐标，而不是用 MC 的方块大小来计算，因为 VOXEL_SIZE 可以是任意值，甚至可以小于 1.0

#endif // TEMPORAL_RADIANCE_CACHE_GLSL