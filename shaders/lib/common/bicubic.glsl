#ifndef BICUBIC_GLSL
#define BICUBIC_GLSL

// 三次B样条核（Mitchell-Netravali, B=1, C=0）
// 该核始终非负，不产生振铃/过冲，但比Catmull-Rom稍模糊
float bspline(float x) {
    x = abs(x);
    if (x < 1.0) {
        float t = 2.0 - x;
        float s = 1.0 - x;
        return (t*t*t - 4.0 * s*s*s) / 6.0;
    } else if (x < 2.0) {
        float t = 2.0 - x;
        return t*t*t / 6.0;
    } else {
        return 0.0;
    }
}

vec4 textureBicubic(sampler2D tex, vec2 uv, vec4 box, vec2 texSize) {
    // uv 已经是 atlas 绝对坐标 （由 computeParallaxUV 返回）
    vec2 texPos = uv * texSize - 0.5;
    ivec2 base = ivec2(floor(texPos));
    vec2 f = texPos - vec2(base);

    // atlas 格式: box.xy = 左上角 UV， box.zw = 宽度/高度
    ivec2 boxMin = ivec2(round(box.xy * texSize));
    ivec2 boxSize = ivec2(round(box.zw * texSize));
    ivec2 boxMax = boxMin + boxSize - 1;

    // 全局纹理边界钳制
    boxMin = clamp(boxMin, ivec2(0), ivec2(texSize) - 1);
    boxMax = clamp(boxMax, ivec2(0), ivec2(texSize) - 1);

    vec4 result = vec4(0.0);
    float weightSum = 0.0;

    for (int j = -1; j <= 2; ++j) {
        float wy = bspline(f.y - float(j));
        if (wy == 0.0) continue;
        for (int i = -1; i <= 2; ++i) {
            float wx = bspline(f.x - float(i));
            if (wx == 0.0) continue;
            float w = wx * wy;

            ivec2 samplePos = base + ivec2(i, j);
            // 钳制到当前方块的像素范围，避免跨方块渗漏
            samplePos = clamp(samplePos, boxMin, boxMax);
            vec4 samp = texelFetch(tex, samplePos, 0);
            result += samp * w;
            weightSum += w;
        }
    }

    return (weightSum > 1e-8) ? (result / weightSum) : vec4(0.0);
}
#endif
