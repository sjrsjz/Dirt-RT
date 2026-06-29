#version 430 compatibility
// 横向 17px 高斯, 逐LOD独立, 边界外取0, 非图集区域填0
layout(local_size_x=16, local_size_y=16) in;
layout(rgba16f) uniform image2D bloomAtlas;
layout(rgba16f) uniform writeonly image2D bloomBlur;
#include "/lib/bloom.glsl"

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 atlasSize = imageSize(bloomAtlas);
    if (coord.x >= atlasSize.x || coord.y >= atlasSize.y) return;

    int level;
    ivec2 regionMin, regionMax;
    bloomFindLOD(coord, atlasSize, level, regionMin, regionMax);
    if (level < 0) { imageStore(bloomBlur, coord, vec4(0)); return; }

    vec3 sum = vec3(0);
    float weightSum = 0.0;
    for (int dx = -8; dx <= 8; dx++) {
        ivec2 sampleCoord = coord + ivec2(dx, 0);
        float gaussWeight = exp(-float(dx * dx) * 0.06);
        weightSum += gaussWeight;
        if (sampleCoord.x >= regionMin.x && sampleCoord.x <= regionMax.x
         && sampleCoord.y >= regionMin.y && sampleCoord.y <= regionMax.y)
            sum += imageLoad(bloomAtlas, sampleCoord).rgb * gaussWeight;
    }
    vec3 result = sum / max(weightSum, 1e-5);
    imageStore(bloomBlur, coord, vec4(result, 1.0));
}
