#ifndef BICUBIC_HERMITE_GLSL
#define BICUBIC_HERMITE_GLSL

// 辅助函数：安全采样，防止 Atlas 渗漏
vec4 getTexel(sampler2D tex, ivec2 coord, ivec2 boxMin, ivec2 boxMax) {
    return texelFetch(tex, clamp(coord, boxMin, boxMax), 0);
}

// 1D Hermite 插值基函数
// 返回: vec4(h00, h10, h01, h11)
vec4 hermiteBasis(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
         2.0 * t3 - 3.0 * t2 + 1.0, // h00: 起点值权重
               t3 - 2.0 * t2 + t,   // h10: 起点导数权重
        -2.0 * t3 + 3.0 * t2,       // h01: 终点值权重
               t3 - t2              // h11: 终点导数权重
    );
}

vec4 textureBicubic(sampler2D tex, vec2 uv, vec4 box, vec2 texSize) {
    // 1. 计算纹理空间坐标与 fractional 偏移
    vec2 texPos = uv * texSize - 0.5;
    ivec2 base = ivec2(floor(texPos));
    vec2 f = texPos - vec2(base);

    // 2. 计算并钳制 Atlas 边界
    ivec2 boxMin = ivec2(round(box.xy * texSize));
    ivec2 boxSize = ivec2(round(box.zw * texSize));
    ivec2 boxMax = boxMin + boxSize - 1;
    boxMin = clamp(boxMin, ivec2(0), ivec2(texSize) - 1);
    boxMax = clamp(boxMax, ivec2(0), ivec2(texSize) - 1);

    // 3. 一次性读取 4x4 区域的所有 16 个像素（无循环，便于编译器并行流水线化）
    vec4 p00 = getTexel(tex, base + ivec2(-1, -1), boxMin, boxMax);
    vec4 p10 = getTexel(tex, base + ivec2( 0, -1), boxMin, boxMax);
    vec4 p20 = getTexel(tex, base + ivec2( 1, -1), boxMin, boxMax);
    vec4 p30 = getTexel(tex, base + ivec2( 2, -1), boxMin, boxMax);

    vec4 p01 = getTexel(tex, base + ivec2(-1,  0), boxMin, boxMax);
    vec4 p11 = getTexel(tex, base + ivec2( 0,  0), boxMin, boxMax); // 中心2x2 左上
    vec4 p21 = getTexel(tex, base + ivec2( 1,  0), boxMin, boxMax); // 中心2x2 右上
    vec4 p31 = getTexel(tex, base + ivec2( 2,  0), boxMin, boxMax);

    vec4 p02 = getTexel(tex, base + ivec2(-1,  1), boxMin, boxMax);
    vec4 p12 = getTexel(tex, base + ivec2( 0,  1), boxMin, boxMax); // 中心2x2 左下
    vec4 p22 = getTexel(tex, base + ivec2( 1,  1), boxMin, boxMax); // 中心2x2 右下
    vec4 p32 = getTexel(tex, base + ivec2( 2,  1), boxMin, boxMax);

    vec4 p03 = getTexel(tex, base + ivec2(-1,  2), boxMin, boxMax);
    vec4 p13 = getTexel(tex, base + ivec2( 0,  2), boxMin, boxMax);
    vec4 p23 = getTexel(tex, base + ivec2( 1,  2), boxMin, boxMax);
    vec4 p33 = getTexel(tex, base + ivec2( 2,  2), boxMin, boxMax);

    // 4. 利用 3x3 邻域为中心 2x2 控制点计算 Sobel 梯度
    // 水平梯度 dx (使用权重 [1, 2, 1] / 8)
    vec4 dx11 = ((p20 - p00) + 2.0 * (p21 - p01) + (p22 - p02)) * 0.125; // 点 p11 的 dx
    vec4 dx21 = ((p30 - p10) + 2.0 * (p31 - p11) + (p32 - p12)) * 0.125; // 点 p21 的 dx
    vec4 dx12 = ((p21 - p01) + 2.0 * (p22 - p02) + (p23 - p03)) * 0.125; // 点 p12 的 dx
    vec4 dx22 = ((p31 - p11) + 2.0 * (p32 - p12) + (p33 - p13)) * 0.125; // 点 p22 的 dx

    // 垂直梯度 dy
    vec4 dy11 = ((p02 - p00) + 2.0 * (p12 - p10) + (p22 - p20)) * 0.125; // 点 p11 的 dy
    vec4 dy21 = ((p12 - p10) + 2.0 * (p22 - p20) + (p32 - p30)) * 0.125; // 点 p21 的 dy
    vec4 dy12 = ((p03 - p01) + 2.0 * (p13 - p11) + (p23 - p21)) * 0.125; // 点 p12 的 dy
    vec4 dy22 = ((p13 - p11) + 2.0 * (p23 - p21) + (p33 - p31)) * 0.125; // 点 p22 的 dy

    // 混合偏导数 dxy (双线性交叉差分估计，使 Bicubic 表面更为连续光滑)
    vec4 dxy11 = ((p22 - p02) - (p20 - p00)) * 0.25;
    vec4 dxy21 = ((p32 - p12) - (p30 - p10)) * 0.25;
    vec4 dxy12 = ((p23 - p03) - (p21 - p01)) * 0.25;
    vec4 dxy22 = ((p33 - p13) - (p31 - p11)) * 0.25;

    // 5. 计算插值权重向量
    vec4 Hx = hermiteBasis(f.x);
    vec4 Hy = hermiteBasis(f.y);

    // 6. 沿 Y 方向执行一维 Hermite 插值（得到 4 个沿 X 方向的混合控制元素）
    // Row0: 左侧（x=0）的值插值
    vec4 Row0 = p11  * Hy.x + dy11  * Hy.y + p12  * Hy.z + dy12  * Hy.w;
    // Row1: 左侧（x=0）的 X 导数插值
    vec4 Row1 = dx11 * Hy.x + dxy11 * Hy.y + dx12 * Hy.z + dxy12 * Hy.w;
    // Row2: 右侧（x=1）的值插值
    vec4 Row2 = p21  * Hy.x + dy21  * Hy.y + p22  * Hy.z + dy22  * Hy.w;
    // Row3: 右侧（x=1）的 X 导数插值
    vec4 Row3 = dx21 * Hy.x + dxy21 * Hy.y + dx22 * Hy.z + dxy22 * Hy.w;

    // 7. 沿 X 方向执行最后一维 Hermite 插值
    return Row0 * Hx.x + Row1 * Hx.y + Row2 * Hx.z + Row3 * Hx.w;
}

#endif