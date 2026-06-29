#version 430
// 上采样合成: bloomAtlas L0..L8 -> colortex1 (手动双线性, pix/scale 与下采样 dp*S 精确互逆)
#include "/lib/bloom.glsl"
in vec2 texCoord;
uniform sampler2D bloomAtlas_Sampler;
/* RENDERTARGETS: 1 */
layout(location = 0) out vec4 fragColor;

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    ivec2 atlasSize = textureSize(bloomAtlas_Sampler, 0);
    vec3 bloom = vec3(0);
    float totalWeight = 0.0;

    for (int level = 0; level <= 8; level++) {
        ivec2 origin = bloomOrigin(level, atlasSize);
        ivec2 levelSize = bloomSize(level, atlasSize);

        int scale = 1 << (level + 1);
        // clamp 保证 texelFetch 不越界; 中心偏移由下采样侧统一修正
        vec2 fLevelCoord = clamp(vec2(pix) / float(scale), vec2(0.0), max(vec2(levelSize) - 0.001, vec2(0.0)));
        ivec2 t00 = ivec2(floor(fLevelCoord));
        ivec2 t11 = min(t00 + ivec2(1, 1), levelSize - ivec2(1, 1));
        vec2 frac = fLevelCoord - vec2(t00);

        float w00 = (1.0 - frac.x) * (1.0 - frac.y);
        float w10 = frac.x * (1.0 - frac.y);
        float w01 = (1.0 - frac.x) * frac.y;
        float w11 = frac.x * frac.y;

        vec3 sum = vec3(0);

        sum += texelFetch(bloomAtlas_Sampler, origin + t00, 0).rgb * w00;
        sum += texelFetch(bloomAtlas_Sampler, origin + ivec2(t11.x, t00.y), 0).rgb * w10;
        sum += texelFetch(bloomAtlas_Sampler, origin + ivec2(t00.x, t11.y), 0).rgb * w01;
        sum += texelFetch(bloomAtlas_Sampler, origin + t11, 0).rgb * w11;

        float levelWeight = exp(-float(level) * 0.7);
        bloom += sum * levelWeight;
        totalWeight += levelWeight;
    }
    fragColor = vec4(bloom / max(totalWeight, 1e-5), 1.0);
}
