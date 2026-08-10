#ifndef POM_GLSL
#define POM_GLSL

// ===========================================================================
// Parallax Occlusion Mapping — extracted from rchit, callable from rgen.
//
// computeParallaxUV() returns the POM-corrected UV (in global atlas space).
// Only use for the first hit (bounce == 0); skip for secondary bounces.
// ===========================================================================

#ifndef POM_STEPS
#define POM_STEPS 32
#endif
#ifndef POM_DEPTH
#define POM_DEPTH 0.25
#endif
#ifndef BINARY_SEARCH_STEPS
#define BINARY_SEARCH_STEPS 6
#endif

#define LINEAR_SAMPLING 1

// ---------------------------------------------------------------------------
// Atlas helpers (same as raytrace_rchit.glsl)
// ---------------------------------------------------------------------------
vec2 pom_getTexCoord(vec2 coord, vec4 atlas) {
    return atlas.xy + fract(coord) * atlas.zw;
}

vec2 pom_localUV(vec2 uv, vec4 atlas) {
    return (uv - atlas.xy) / atlas.zw;
}

// ---------------------------------------------------------------------------
// Height-map sample — texture resolution is queried once by the caller.
// ---------------------------------------------------------------------------
float pom_sampleHeight(sampler2D heightTex, vec2 coord, vec4 atlas,
        vec2 res, float mipLevel, int fetchMip) {
    #if LINEAR_SAMPLING == 1
    // The old path issued four already-filtered texture() calls and blended
    // them a second time. A hardware linear lookup is exact away from a sprite
    // boundary. Only the one-texel wrap seam needs a four-texel fallback.
    vec2 local = fract(coord);
    ivec2 spriteOrigin = ivec2(round(atlas.xy * res));
    ivec2 spriteSize = max(ivec2(round(atlas.zw * res)), ivec2(1));
    vec2 spritePos = local * vec2(spriteSize) - 0.5;
    ivec2 base = ivec2(floor(spritePos));
    bool interior = all(greaterThanEqual(base, ivec2(0)))
        && all(lessThan(base + ivec2(1), spriteSize));

    float height;
    if (interior) {
        vec2 uv = atlas.xy + local * atlas.zw;
        height = textureLod(heightTex, uv, mipLevel).a;
    } else {
        vec2 f = fract(spritePos);
        ivec2 p0 = ivec2(
            (base.x % spriteSize.x + spriteSize.x) % spriteSize.x,
            (base.y % spriteSize.y + spriteSize.y) % spriteSize.y);
        ivec2 p1 = (p0 + ivec2(1)) % spriteSize;
        float h00 = texelFetch(heightTex, spriteOrigin + p0, fetchMip).a;
        float h10 = texelFetch(heightTex,
            spriteOrigin + ivec2(p1.x, p0.y), fetchMip).a;
        float h01 = texelFetch(heightTex,
            spriteOrigin + ivec2(p0.x, p1.y), fetchMip).a;
        float h11 = texelFetch(heightTex,
            spriteOrigin + p1, fetchMip).a;
        height = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    }
    return height * POM_DEPTH - POM_DEPTH;
    #else
    return textureLod(heightTex, pom_getTexCoord(coord, atlas),
        mipLevel).a * POM_DEPTH - POM_DEPTH;
    #endif
}

// ===========================================================================
// Main entry point: computeParallaxUV
//
// Parameters:
//   heightTex  — texture with height in .a channel (blockTexNormal)
//   localUV    — fragment UV in local atlas space [0,1]
//   atlasBox   — {origin.x, origin.y, width, height} in global texture space
//   viewDir    — world-space view direction
//   tbn        — tangent-to-world matrix
//
// Returns:    — global UV (atlas-space) after POM correction
// ===========================================================================
vec2 computeParallaxUV(
    sampler2D heightTex,
    vec2 localUV,
    vec4 atlasBox,
    vec3 viewDir,
    mat3 tbn,
    float mipLevel
) {
    int fetchMip = int(floor(mipLevel + 0.5));
    vec2 res = vec2(textureSize(heightTex, fetchMip));
    int activeSteps = max(POM_STEPS >> min(fetchMip, 3), 4);

    vec3 V = normalize(transpose(tbn) * viewDir);

    if (V.z >= 0.0) {
        return pom_getTexCoord(localUV, atlasBox);
    }

    vec2 currentTexCoord = localUV;
    vec2 dtex = V.xy * POM_DEPTH / (-V.z * float(activeSteps));
    float currentHeight = 0.0;
    float stepSize = POM_DEPTH / float(activeSteps);

    float heightFromTexture = pom_sampleHeight(heightTex, currentTexCoord,
        atlasBox, res, mipLevel, fetchMip);
    int steps = 0;

    while (currentHeight > heightFromTexture && steps < activeSteps) {
        currentTexCoord += dtex;
        heightFromTexture = pom_sampleHeight(heightTex, currentTexCoord,
            atlasBox, res, mipLevel, fetchMip);
        currentHeight -= stepSize;
        steps++;
    }

    vec2 prevTexCoord = currentTexCoord - dtex;
    float prevHeight = currentHeight + stepSize;

    for (int i = 0; i < BINARY_SEARCH_STEPS; i++) {
        dtex *= 0.5;
        stepSize *= 0.5;

        vec2 midTexCoord = prevTexCoord + dtex;
        float midHeight = prevHeight - stepSize;
        float hft = pom_sampleHeight(heightTex, midTexCoord, atlasBox,
            res, mipLevel, fetchMip);

        if (hft > midHeight) {
            currentTexCoord = midTexCoord;
            currentHeight = midHeight;
        } else {
            prevTexCoord = midTexCoord;
            prevHeight = midHeight;
        }
    }

    return pom_getTexCoord(currentTexCoord, atlasBox);
}

#endif // POM_GLSL
