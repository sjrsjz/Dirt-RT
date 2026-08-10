#ifndef RT_MIPMAP_GLSL
#define RT_MIPMAP_GLSL

// Ray-tracing stages have no implicit screen-space derivatives. Approximate
// the angular width of one camera pixel from the asymmetric near-plane corners.
float rtPixelConeSpread(vec3 corner0, vec3 corner1, vec3 corner2,
        vec2 resolution) {
    vec2 safeResolution = max(resolution, vec2(1.0));
    float nearDistance = max(abs(corner0.z), 1e-4);
    vec2 nearPixelSize = vec2(
        abs(corner1.x - corner0.x) / safeResolution.x,
        abs(corner2.y - corner0.y) / safeResolution.y);
    return max(nearPixelSize.x, nearPixelSize.y) / nearDistance;
}

// Convert a ray cone footprint to an atlas-safe explicit mip level. Atlas
// extent is the current sprite in normalized texture coordinates. Clamping to
// the sprite's smallest dimension prevents selecting a mip where it vanishes.
float rtTextureLod(ivec2 baseTextureSize, vec4 atlas, float hitDistance,
        vec3 rayDirection, vec3 geometryNormal, uint bounce,
        float pixelConeSpread) {
    ivec2 spriteTexelSize = max(ivec2(round(
        atlas.zw * vec2(baseTextureSize))), ivec2(1));
    vec2 spriteTexels = vec2(spriteTexelSize);
    float maxSpriteLod = float(findMSB(min(spriteTexelSize.x,
        spriteTexelSize.y)));

    // Callers provide normalized ray directions and geometric normals.
    float NoV = max(abs(dot(rayDirection, geometryNormal)), 0.125);
    // Later path vertices inherit the preceding cone and scattering broadens
    // it. This conservative factor avoids mip-0 aliasing on indirect rays
    // without requiring extra payload words for cone state.
    float bounceGrowth = 1.0 + 0.75 * float(bounce);
    float surfaceFootprint = max(hitDistance, 0.0) * pixelConeSpread
        * bounceGrowth / NoV;
    float footprintTexels = surfaceFootprint
        * max(spriteTexels.x, spriteTexels.y);
    float lod = log2(max(footprintTexels, 1.0));
    return clamp(lod, 0.0, maxSpriteLod);
}

#endif // RT_MIPMAP_GLSL
