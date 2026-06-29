#version 430
// 最终合成: bloom(colortex1) + scene(colortex0) -> mix + tonemap + gamma
#include "/lib/bloom.glsl"
#include "/lib/tonemap.glsl"
in vec2 texCoord;
uniform sampler2D colortex0;
uniform sampler2D colortex1;
#ifdef BLOOM_DEBUG_ATLAS
uniform sampler2D bloomAtlas_Sampler;
#endif
/* RENDERTARGETS: 0,1 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 bloomOut;
void main() {
#ifdef BLOOM_DEBUG_ATLAS
    // 直接输出 bloomAtlas 图集 (映射 screen UV -> atlas UV)
    fragColor = vec4(texture(bloomAtlas_Sampler, texCoord).rgb, 1.0);
    bloomOut = vec4(0.0);
#else
    vec3 scene = texture(colortex0, texCoord).rgb;
    vec3 bloom = texture(colortex1, texCoord).rgb;
    bloomOut = vec4(bloom, 1.0);
    vec3 hdr = mix(scene, bloom, BLOOM_MIX);
    vec3 mapped = tonemap_NeuralNetwork(hdr.zyx / (1.0 + hdr.zyx)) - tonemap_NeuralNetwork(vec3(0.0));
    fragColor = vec4(pow(max(mapped, 0.0), vec3(1.0 / 2.2)), 1.0);
    if (any(isnan(fragColor.xyz))) fragColor.xyz = vec3(0.0);
#endif
}
