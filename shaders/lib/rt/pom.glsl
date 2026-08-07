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
// Height-map sample — res and rcpRes passed from caller (computed ONCE).
// ---------------------------------------------------------------------------
float pom_sampleHeight(sampler2D heightTex, vec2 coord, vec4 atlas, vec2 res, vec2 rcpRes) {
    #if LINEAR_SAMPLING == 1
    vec2 pixel = (atlas.xy + coord * atlas.zw) * res;
    vec2 i = floor(pixel);
    vec2 f = pixel - i;

    vec2 base_uv = (i * rcpRes - atlas.xy) / atlas.zw;
    vec2 dx = vec2(rcpRes.x / atlas.z, 0.0);
    vec2 dy = vec2(0.0, rcpRes.y / atlas.w);

    float h00 = texture(heightTex, pom_getTexCoord(base_uv, atlas)).a;
    float h10 = texture(heightTex, pom_getTexCoord(base_uv + dx, atlas)).a;
    float h01 = texture(heightTex, pom_getTexCoord(base_uv + dy, atlas)).a;
    float h11 = texture(heightTex, pom_getTexCoord(base_uv + dx + dy, atlas)).a;

    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y) * POM_DEPTH - POM_DEPTH;
    #else
    return texture(heightTex, pom_getTexCoord(coord, atlas)).a * POM_DEPTH - POM_DEPTH;
    #endif
}

// ---------------------------------------------------------------------------
// Height-map derivatives (for normal reconstruction)
// ---------------------------------------------------------------------------
vec2 pom_computeDerivatives(sampler2D heightTex, vec2 coord, vec4 atlas, vec2 res, vec2 rcpRes) {
    const float offset = 0.00025;
    float x_h_L = pom_sampleHeight(heightTex, coord + vec2(-offset * 2.0, 0.0), atlas, res, rcpRes);
    float x_h_R = pom_sampleHeight(heightTex, coord + vec2(offset * 2.0, 0.0), atlas, res, rcpRes);
    float y_h_L = pom_sampleHeight(heightTex, coord + vec2(0.0, -offset), atlas, res, rcpRes);
    float y_h_R = pom_sampleHeight(heightTex, coord + vec2(0.0, offset), atlas, res, rcpRes);

    return vec2((x_h_L - x_h_R) / (2.0 * offset),
        (y_h_L - y_h_R) / (2.0 * offset));
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
//   derivatives — (out) UV derivatives for normal reconstruction
// ===========================================================================
vec2 computeParallaxUV(
    sampler2D heightTex,
    vec2 localUV,
    vec4 atlasBox,
    vec3 viewDir,
    mat3 tbn,
    out vec2 derivatives
) {
    vec2 res = vec2(textureSize(heightTex, 0)); // ONE query total
    vec2 rcpRes = 1.0 / res;

    vec3 V = normalize(transpose(tbn) * viewDir);

    if (V.z >= 0.0) {
        vec2 realCoord = pom_getTexCoord(localUV, atlasBox);
        derivatives = pom_computeDerivatives(heightTex, localUV, atlasBox, res, rcpRes);
        return realCoord;
    }

    vec2 currentTexCoord = localUV;
    vec2 dtex = V.xy * POM_DEPTH / (-V.z * float(POM_STEPS));
    float currentHeight = 0.0;
    float stepSize = POM_DEPTH / float(POM_STEPS);

    float heightFromTexture = pom_sampleHeight(heightTex, currentTexCoord, atlasBox, res, rcpRes);
    int steps = 0;

    while (currentHeight > heightFromTexture && steps < POM_STEPS) {
        currentTexCoord += dtex;
        heightFromTexture = pom_sampleHeight(heightTex, currentTexCoord, atlasBox, res, rcpRes);
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
        float hft = pom_sampleHeight(heightTex, currentTexCoord, atlasBox, res, rcpRes);

        if (hft > midHeight) {
            currentTexCoord = midTexCoord;
            currentHeight = midHeight;
        } else {
            prevTexCoord = midTexCoord;
            prevHeight = midHeight;
        }
    }

    vec2 realCoord = pom_getTexCoord(currentTexCoord, atlasBox);
    derivatives = pom_computeDerivatives(heightTex, currentTexCoord, atlasBox, res, rcpRes);
    return realCoord;
}

#endif // POM_GLSL
