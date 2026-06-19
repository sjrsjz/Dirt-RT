#version 430

// ===========================================================================
// Pass bloom_sampler: 绽放采样 (Bloom Dithering / Ring Sampling)
// ===========================================================================
// 功能: 使用泊松盘状采样 / 环形抖动来泛光采样高亮区域。
//
// 算法:
//   1. 生成随机角度偏移 (基于 texCoord 的 hash)
//   2. 在半径 R 处做 sampleN 次环形采样 (旋转 sampleN 次取平均)
//   3. 使用自定义 kernel() 函数进行半径加权 (中心亮、外围衰减)
//
// kernel(x) = exp(-7.5·x^0.25 - 1.5·x^0.025) · sqrt(1-x²)
//   该核在 x≈0 附近陡峭衰减，适合捕获紧凑的高光绽放
// ===========================================================================

#include "/lib/common.glsl"

in vec2 texCoord;

uniform sampler2D colortex0;
uniform sampler2D colortex5;  // 待采样的高亮目标纹理

/* RENDERTARGETS: 1 */
layout(location = 0) out vec4 fragColor;

// 绽放径向权重核 — 中心权重高，边缘权重低
float kernel(float x) {
    return exp(-7.5 * pow(x + 0.00125, 0.25) - 1.5 * pow(x, 0.025))
         * sqrt(1.0 - x * x);
}

void main() {
    const int sampleN = 32;      // 环形采样数
    vec3 sumX = vec3(0.0);

    // 基于像素位置的伪随机角度偏移 (使相邻像素有不同的采样模式)
    float angleOffset = rand(texCoord - 400.0) * 32.0 * PI;
    const float angleShift = 2.0 * PI / float(sampleN);  // 每次旋转步长

    // 半径: 使用 log(0.01 + 0.99·rand)² 将均匀采样映射为偏小半径的分布
    float R = pow(log(0.01 + 0.99 * rand(texCoord * 100.0 + 300.0)), 2.0);

    // 权重: 由核函数确定
    float weight = kernel(R * R) / kernel(0.0);

    // 将半径映射为像素单位 (使用 colortex5 的尺寸)
    R = max(pow(R + 1.0, 4.0) - 1.125, 0.0) * 0.1;

    // 旋转矩阵 + 初始方向
    mat2 rotM = mat2(cos(angleShift), sin(angleShift),
                     -sin(angleShift), cos(angleShift));
    vec2 v = vec2(cos(angleOffset), sin(angleOffset))
           * R * textureSize(colortex5, 0).x;

    // ---- 环形采样 (32 次旋转) ---------------------------------------------
    for (int i = 0; i < sampleN; i++) {
        v *= rotM;  // 每次旋转 angleShift 弧度
        vec3 A = texelFetch(colortex5, ivec2(gl_FragCoord.xy + v), 0).xyz;
        sumX += A;
    }

    // ---- 归一化: 除以样本数 × (R² + 0.5) 防止过曝 ------------------------
    sumX /= float(sampleN) * (R * R + 0.5);

    fragColor.xyz = sumX * weight;

    if (any(isnan(fragColor.xyz))) fragColor.xyz = vec3(0.0);
}
