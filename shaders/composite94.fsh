#version 430
// 上采样合成: bloomAtlas L0..L8 -> colortex1 
#include "/lib/post_processing/bloom.glsl"

in vec2 texCoord;
uniform sampler2D bloomAtlas_Sampler;
uniform vec2 resolution;

/* RENDERTARGETS: 1 */
layout(location = 0) out vec4 fragColor;

// ===========================================================================
// 无分支 B-Spline 权重
// ===========================================================================
vec4 getBSplineWeights(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    return vec4(
        -x3 + 3.0*x2 - 3.0*x + 1.0,
        3.0*x3 - 6.0*x2 + 4.0,
        -3.0*x3 + 3.0*x2 + 3.0*x + 1.0,
        x3
    ) / 6.0;
}

// ===========================================================================
// 高质量 16-Tap B-Spline
// ===========================================================================
vec3 sampleBloom4x4_Exact(sampler2D tex, vec2 screenUV, ivec2 origin, ivec2 levelSize) {
    // screenUV and every bloom level use the ray tracer's native i/N grid.
    // This is the exact inverse of mappedSource() in composite91.
    vec2 scf = screenUV * vec2(levelSize);
    ivec2 tBase = ivec2(floor(scf));
    vec2 frac = scf - vec2(tBase);

    vec4 wx = getBSplineWeights(frac.x);
    vec4 wy = getBSplineWeights(frac.y);

    ivec2 t00 = tBase - ivec2(1);
    vec3 sum = vec3(0.0);

    for (int dy = 0; dy < 4; dy++) {
        int ty = clamp(t00.y + dy, 0, levelSize.y - 1);
        for (int dx = 0; dx < 4; dx++) {
            int tx = clamp(t00.x + dx, 0, levelSize.x - 1);
            float baseWeight = wx[dx] * wy[dy];
            sum += bloomSafeFloat(
                texelFetch(tex, origin + ivec2(tx, ty), 0).rgb) * baseWeight;
        }
    }
    return sum;
}

// ===========================================================================
// 主入口
// ===========================================================================
void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    ivec2 atlasSize = textureSize(bloomAtlas_Sampler, 0);
    
    vec2 screenUV = vec2(pix) / max(resolution, vec2(1.0));

    vec3 bloom = vec3(0.0);
    float totalWeight = 0.0;

    // Every level uses C2-continuous cubic reconstruction. Bilinear sampling
    // exposed the texel grid of coarse LODs as JPEG-like macroblocks once the
    // Gaussian sigma was corrected and no longer hid it.
    for (int level = 0; level <= 8; level++) {
        ivec2 origin = bloomOrigin(level, atlasSize);
        ivec2 levelSize = bloomSize(level, atlasSize);
        // A level may collapse to zero on a small render target. Never form
        // negative clamp bounds or issue a texelFetch for such a region.
        if (any(lessThanEqual(levelSize, ivec2(0)))) continue;
        float weight = 1.0 / pow(1.0 + float(level), 1.5);

        bloom += sampleBloom4x4_Exact(bloomAtlas_Sampler, screenUV, origin, levelSize) * weight;
        totalWeight += weight;
    }

    fragColor = vec4(bloomSafeFloat(bloom / max(totalWeight, 1e-5)), 1.0);
}
