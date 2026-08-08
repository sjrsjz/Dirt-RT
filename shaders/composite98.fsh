#version 430
// 最终合成: bloom(colortex1) + scene(colortex0) -> mix + tonemap + gamma
#include "/lib/buffers/frame_data.glsl"
#include "/lib/post_processing/bloom.glsl"
#include "/lib/post_processing/tonemap.glsl"
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
    fragColor = vec4(bloomSafeFloat(texture(bloomAtlas_Sampler, texCoord).rgb), 1.0);
    bloomOut = vec4(0.0);
#else
    // composite94 writes one reconstructed bloom value per native ray-grid
    // pixel. Fetch that exact texel here: an interpolated UV lookup needlessly
    // reintroduces sampler/filter state into an otherwise integer-aligned
    // composite path.
    ivec2 pix = ivec2(gl_FragCoord.xy);
    vec3 scene = bloomSafeFloat(texelFetch(colortex0, pix, 0).rgb);
    vec3 bloom = bloomSafeFloat(texelFetch(colortex1, pix, 0).rgb);
    bloomOut = vec4(bloom, 1.0);
    vec3 hdr = mix(scene, bloom, BLOOM_MIX) * avgExposure;
    vec3 mapped = TonyMcMapface_Tiny(apply_shadow_toe(hdr, EXPOSURE_CURVE_K));
    fragColor = vec4(linear_to_srgb(mapped), 1.0);
    if (any(isnan(fragColor.xyz))) fragColor.xyz = vec3(0.0);
#endif
}
