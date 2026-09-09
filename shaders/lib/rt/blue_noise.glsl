#ifndef RT_BLUE_NOISE_GLSL
#define RT_BLUE_NOISE_GLSL

// Vulkanite binds descriptor sets in this order:
//   0 = common, 1 = geometry, 2 = sorted Iris custom textures, 3 = SSBOs.
// shaders.properties deliberately names this sampler aaaRtBlueNoise so it is
// the first entry in Vulkanite's lexicographically sorted custom-texture set.
layout(set = 2, binding = 0) uniform sampler2D aaaRtBlueNoise;

const uint RT_BLUE_NOISE_TILE_SIZE = 64u;
const uint RT_BLUE_NOISE_ATLAS_COLUMNS = 4u;
const uvec2 RT_BLUE_NOISE_FRAME_ROTATION =
    uvec2(0xc13fa9a9u, 0x91e10da5u);

uint rtBlueNoisePassIndex() {
#if defined(FIRST_LOBE_DIFFUSE)
    return 0u;
#elif defined(FIRST_LOBE_REFLECTION)
    return 1u;
#elif defined(FIRST_LOBE_REFRACTION)
    return 2u;
#else
    return 3u;
#endif
}

// Each pass/dimension pair owns one independent 64x64 RG tile. A global
// Cranley-Patterson/Weyl rotation changes the sample every frame without
// destroying the screen-space blue-noise structure inside a frame.
vec2 rtBlueNoise2D(uvec2 pixel, uint dimensionPair) {
    uint slot = (rtBlueNoisePassIndex() * 2u + dimensionPair) & 7u;
    uvec2 spatialOffset = uvec2(slot * 37u, slot * 17u);
    uvec2 localPixel = (pixel + spatialOffset) & (RT_BLUE_NOISE_TILE_SIZE - 1u);
    uvec2 tile = uvec2(slot & (RT_BLUE_NOISE_ATLAS_COLUMNS - 1u),
        slot / RT_BLUE_NOISE_ATLAS_COLUMNS);
    ivec2 atlasPixel = ivec2(localPixel + tile * RT_BLUE_NOISE_TILE_SIZE);

    vec2 encoded = texelFetch(aaaRtBlueNoise, atlasPixel, 0).rg;
    // Decode normalized RG8 at texel centers so the sample stays in [0, 1).
    vec2 base = encoded * (255.0 / 256.0) + (0.5 / 256.0);
    uvec2 phase = cam.frameId * RT_BLUE_NOISE_FRAME_ROTATION;
    vec2 rotation = vec2(phase >> 8u) * (1.0 / 16777216.0);
    return fract(base + rotation);
}

#endif
