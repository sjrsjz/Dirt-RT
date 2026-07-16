#version 430
// 上采样合成: bloomAtlas L0..L8 -> colortex1 
// 特性: 亚像素中心精准对齐 (size-1.0) + 越界严格零截断 + B-Spline无分支滤波
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
// 高质量 16-Tap B-Spline (严格使用 size - 1.0 亚像素对齐 + 边界截断)
// ===========================================================================
vec3 sampleBloom4x4_Exact(sampler2D tex, vec2 screenUV, ivec2 origin, ivec2 levelSize) {
    // 严格匹配你的 align_corners 坐标映射 (使用 max 防止负数或NaN)
    vec2 scf = screenUV * max(vec2(levelSize) - 1.0, vec2(0.0));
    ivec2 tBase = ivec2(floor(scf));
    vec2 frac = scf - vec2(tBase);

    vec4 wx = getBSplineWeights(frac.x);
    vec4 wy = getBSplineWeights(frac.y);

    ivec2 t00 = tBase - ivec2(1);
    vec3 sum = vec3(0.0);

    for (int dy = 0; dy < 4; dy++) {
        int ty = t00.y + dy;
        // 越界自动变 0，完美符合狄利克雷边界
        float weightY = wy[dy] * float(ty >= 0 && ty < levelSize.y);
        
        if (weightY > 1e-6) { 
            for (int dx = 0; dx < 4; dx++) {
                int tx = t00.x + dx;
                float w = wx[dx] * weightY * float(tx >= 0 && tx < levelSize.x);
                
                if (w > 1e-6) {
                    sum += texelFetch(tex, origin + ivec2(tx, ty), 0).rgb * w;
                }
            }
        }
    }
    return sum;
}

// ===========================================================================
// 降级 4-Tap Bilinear (同等严苛的亚像素对齐)
// ===========================================================================
vec3 sampleBloom2x2_Exact(sampler2D tex, vec2 screenUV, ivec2 origin, ivec2 levelSize) {
    vec2 scf = screenUV * max(vec2(levelSize) - 1.0, vec2(0.0));
    ivec2 tBase = ivec2(floor(scf));
    vec2 frac = scf - vec2(tBase);

    vec2 wx = vec2(1.0 - frac.x, frac.x);
    vec2 wy = vec2(1.0 - frac.y, frac.y);

    vec3 sum = vec3(0.0);

    for (int dy = 0; dy < 2; dy++) {
        int ty = tBase.y + dy;
        float weightY = wy[dy] * float(ty >= 0 && ty < levelSize.y);
        
        if (weightY > 1e-6) {
            for (int dx = 0; dx < 2; dx++) {
                int tx = tBase.x + dx;
                float w = wx[dx] * weightY * float(tx >= 0 && tx < levelSize.x);
                
                if (w > 1e-6) {
                    sum += texelFetch(tex, origin + ivec2(tx, ty), 0).rgb * w;
                }
            }
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
    
    // 【关键修复】：screenUV 同样使用 resolution - 1.0 对齐屏幕像素中心范式
    vec2 screenUV = vec2(pix) / max(resolution - 1.0, vec2(1e-6));

    vec3 bloom = vec3(0.0);
    float totalWeight = 0.0;

    // L0~L2：对视觉极其重要的高频层，使用 16-Tap B-Spline
    for (int level = 0; level <= 2; level++) {
        ivec2 origin = bloomOrigin(level, atlasSize);
        ivec2 levelSize = bloomSize(level, atlasSize);
        float weight = 1.0 / pow(1.0 + float(level), 1.5);

        bloom += sampleBloom4x4_Exact(bloomAtlas_Sampler, screenUV, origin, levelSize) * weight;
        totalWeight += weight;
    }

    // L3~L8：能量极其扩散的低频层，降维到 4-Tap Bilinear 榨取性能
    for (int level = 3; level <= 8; level++) {
        ivec2 origin = bloomOrigin(level, atlasSize);
        ivec2 levelSize = bloomSize(level, atlasSize);
        float weight = 1.0 / pow(1.0 + float(level), 1.5);

        bloom += sampleBloom2x2_Exact(bloomAtlas_Sampler, screenUV, origin, levelSize) * weight;
        totalWeight += weight;
    }

    fragColor = vec4(bloom / max(totalWeight, 1e-5), 1.0);
}