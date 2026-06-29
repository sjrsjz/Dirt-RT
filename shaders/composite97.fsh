#version 430
// 上采样合成: bloomAtlas L0..L8 -> colortex1 (双三次 Catmull-Rom, C1 连续)
#include "/lib/bloom.glsl"
in vec2 texCoord;
uniform sampler2D bloomAtlas_Sampler;
uniform vec2 resolution;
/* RENDERTARGETS: 1 */
layout(location = 0) out vec4 fragColor;

float cubicWeight(float x) {
    // Catmull-Rom: C1 连续, weights sum to 1
    float x2 = x * x;
    float x3 = x2 * x;
    if (x <= 1.0)
        return 1.5 * x3 - 2.5 * x2 + 1.0;
    else if (x < 2.0)
        return -0.5 * x3 + 2.5 * x2 - 4.0 * x + 2.0;
    else
        return 0.0;
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    ivec2 atlasSize = textureSize(bloomAtlas_Sampler, 0);
    vec2 screenUV = vec2(pix) / max(resolution - 1.0, vec2(1e-6));
    vec3 bloom = vec3(0);
    float totalWeight = 0.0;

    for (int level = 0; level <= 8; level++) {
        ivec2 origin = bloomOrigin(level, atlasSize);
        ivec2 levelSize = bloomSize(level, atlasSize);

        vec2 fLevelCoord = screenUV * max(vec2(levelSize) - 1.0, vec2(0.0));
        ivec2 tBase = ivec2(floor(fLevelCoord));
        vec2 frac = fLevelCoord - vec2(tBase);

        // 4×4 双三次核，锚点 t00 = tBase - 1
        vec4 wx = vec4(
            cubicWeight(1.0 + frac.x),
            cubicWeight(frac.x),
            cubicWeight(1.0 - frac.x),
            cubicWeight(2.0 - frac.x)
        );
        vec4 wy = vec4(
            cubicWeight(1.0 + frac.y),
            cubicWeight(frac.y),
            cubicWeight(1.0 - frac.y),
            cubicWeight(2.0 - frac.y)
        );

        ivec2 t00 = tBase - ivec2(1, 1);
        vec3 sum = vec3(0);

        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 4; dx++) {
                ivec2 tc = t00 + ivec2(dx, dy);
                if (tc.x >= 0 && tc.x < levelSize.x && tc.y >= 0 && tc.y < levelSize.y)
                    sum += texelFetch(bloomAtlas_Sampler, origin + tc, 0).rgb * (wx[dx] * wy[dy]);
            }
        }

        float levelWeight = 1.0 / pow(1.0 + level, 1.5); // 可调节不同 LOD 的权重
        bloom += sum * levelWeight;
        totalWeight += levelWeight;
    }
    fragColor = vec4(bloom / max(totalWeight, 1e-5), 1.0);
}
