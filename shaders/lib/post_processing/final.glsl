#version 430 core
// Fused bloom upsample + final scene composite.
#include "/lib/buffers/frame_data.glsl"
#include "/lib/post_processing/bloom.glsl"
#include "/lib/post_processing/tonemap.glsl"
in vec2 texCoord;
uniform sampler2D colortex0;
uniform sampler2D bloomAtlas_Sampler;
/* RENDERTARGETS: 0,1 */
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 bloomOut;

// Same C2 cubic B-spline as the old 16-fetch reconstruction. Positive cubic
// taps are paired so hardware bilinear filtering evaluates each 2x2 group in
// one lookup: four texture reads per level instead of sixteen.
// xy = paired weights, zw = paired bilinear offsets relative to floor(x).
// Returning the final pair representation shortens the live range compared
// with carrying four cubic taps plus a second pair vector per axis.
vec4 bloomCubicPairs(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float oneMinusX = 1.0 - x;
    float w0 = oneMinusX * oneMinusX * oneMinusX * (1.0 / 6.0);
    float w1 = (3.0 * x3 - 6.0 * x2 + 4.0) * (1.0 / 6.0);
    float w3 = x3 * (1.0 / 6.0);
    float pair0 = w0 + w1;
    float pair1 = 1.0 - pair0;
    return vec4(pair0, pair1,
        -1.0 + w1 / pair0, 1.0 + w3 / pair1);
}

vec3 sampleBloomCubic4(vec2 screenUV, ivec2 origin, ivec2 levelSize,
        vec2 invAtlasSize) {
    vec2 source = screenUV * vec2(levelSize);
    ivec2 base = ivec2(floor(source));
    vec2 f = fract(source);
    vec4 pairX = bloomCubicPairs(f.x);
    vec4 pairY = bloomCubicPairs(f.y);
    vec2 sampleX = pairX.zw + float(base.x);
    vec2 sampleY = pairY.zw + float(base.y);

    // Clamp inside the packed atlas rectangle before filtering. This exactly
    // reproduces repeated edge texels without leaking into an adjacent LOD.
    sampleX = clamp(sampleX, 0.0, float(levelSize.x - 1));
    sampleY = clamp(sampleY, 0.0, float(levelSize.y - 1));
    vec2 atlasOrigin = vec2(origin) + 0.5;

    vec3 row0 =
        textureLod(bloomAtlas_Sampler,
            (atlasOrigin + vec2(sampleX.x, sampleY.x)) * invAtlasSize,
            0.0).rgb * pairX.x
      + textureLod(bloomAtlas_Sampler,
            (atlasOrigin + vec2(sampleX.y, sampleY.x)) * invAtlasSize,
            0.0).rgb * pairX.y;
    vec3 row1 =
        textureLod(bloomAtlas_Sampler,
            (atlasOrigin + vec2(sampleX.x, sampleY.y)) * invAtlasSize,
            0.0).rgb * pairX.x
      + textureLod(bloomAtlas_Sampler,
            (atlasOrigin + vec2(sampleX.y, sampleY.y)) * invAtlasSize,
            0.0).rgb * pairX.y;
    return row0 * pairY.x + row1 * pairY.y;
}

vec3 reconstructBloom(ivec2 pixel, ivec2 screenSize) {
    const float levelWeight[9] = float[9](
        1.0, 0.3535533906, 0.1924500897,
        0.125, 0.0894427191, 0.0680413817,
        0.0539949247, 0.0441941738, 0.0370370370);

    ivec2 atlasSize = textureSize(bloomAtlas_Sampler, 0);
    vec2 invAtlasSize = 1.0 / vec2(atlasSize);
    vec2 screenUV = vec2(pixel) / max(vec2(screenSize), vec2(1.0));
    vec3 sum = vec3(0.0);
    float weightSum = 0.0;

    for (int level = 0; level <= 8; ++level) {
        ivec2 levelSize = bloomSize(level, atlasSize);
        if (any(lessThanEqual(levelSize, ivec2(0)))) continue;
        float weight = levelWeight[level];
        sum += sampleBloomCubic4(screenUV, bloomOrigin(level, atlasSize),
            levelSize, invAtlasSize) * weight;
        weightSum += weight;
    }
    return bloomSafeFloat(sum / max(weightSum, 1e-5));
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    ivec2 screenSize = textureSize(colortex0, 0);
    vec3 scene = bloomSafeFloat(texelFetch(colortex0, pix, 0).rgb);
    vec3 bloom = reconstructBloom(pix, screenSize);
    bloomOut = vec4(bloom, 1.0);
    vec3 hdr = mix(scene, bloom, BLOOM_MIX) * avgExposure;
    vec3 mapped = TonyMcMapface_Tiny(apply_shadow_toe(hdr, EXPOSURE_CURVE_K));
    fragColor = vec4(linear_to_srgb(mapped), 1.0);
    if (any(isnan(fragColor.xyz))) fragColor.xyz = vec3(0.0);
}
