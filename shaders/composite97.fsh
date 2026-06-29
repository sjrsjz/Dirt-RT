#version 430
// 上采样合成: bloomAtlas L0..L8 -> colortex1 (每级读1次texelFetch)
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
        ivec2 maxCoord = origin + levelSize - ivec2(1, 1);

        int scale = 1 << (level + 1);
        vec2 sampleUV = (vec2(origin) + (vec2(pix) + 0.5) / float(scale)) / vec2(atlasSize);
        float weight = 1.0 / (1.0 + level);//exp(-float(level) * 0.7);
        vec2 rMin = (vec2(origin) + 0.5) / vec2(atlasSize);
        vec2 rMax = (vec2(maxCoord) + 0.5) / vec2(atlasSize);
        bloom += texture(bloomAtlas_Sampler, clamp(sampleUV, rMin, rMax)).rgb * weight;
        totalWeight += weight;
    }
    fragColor = vec4(bloom / max(totalWeight, 1e-5), 1.0);
}
