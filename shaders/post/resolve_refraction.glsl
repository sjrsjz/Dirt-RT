#ifndef COMPOSITE_REFRACTION_RESOLVE_GLSL
#define COMPOSITE_REFRACTION_RESOLVE_GLSL

// Resolve PSR endpoints from the screen, cache or environment. The caller
// provides projectDiffuseLighting and the scene-buffer/lighting includes.
float primaryTransmissionIor(float transmissionCode) {
    if (transmissionCode < 0.999) return 1.0;
    int mediumClass = int(transmissionCode + 0.5);
    if (mediumClass == 1) return REFRACTIVE_INDEX;
    if (mediumClass == 2) return GLASS_REFRACTIVE_INDEX;
    if (mediumClass == 3) return 1.31;
    return 1.0;
}

vec3 resolvePSRRefraction(uvec2 xy, out bool reusedScreen) {
    PSRResolveData psr = readPSRResolve(xy);
    reusedScreen = false;

    if (psr.environment) {
        setSkyVars();
        vec3 sky = sampleSky(camPos.y, psr.refractedDirection,
            -lightDir_global);
        return psr.transmittance * max(sky, vec3(0.0));
    }
    if (!psr.endpointValid) return vec3(0.0);

    vec3 resolved = vec3(0.0);
    if (psr.screenCandidate) {
        vec4 clip = rtViewProjection * vec4(psr.endpointRelative, 1.0);
        if (clip.w > 1e-6) {
            vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
            if (all(greaterThanEqual(uv, vec2(0.0)))
                    && all(lessThan(uv, vec2(1.0)))) {
                // RT rays use integer pixel/resolution, including the current
                // projection phase. A raster-style -0.5 shifts every reuse tap.
                vec2 samplePixel = uv * vec2(resolution_global);
                ivec2 basePixel = ivec2(floor(samplePixel));
                vec2 fraction = fract(samplePixel);
                float pixelFootprint = max(length(psr.endpointRelative)
                    / max(float(resolution_global.y), 1.0), 0.025);
                float positionTolerance = max(0.12,
                    6.0 * pixelFootprint);
                MaxEntEncoding background = init_maxent();
                float acceptedWeight = 0.0;
                for (int y = 0; y < 2; ++y) {
                    for (int x = 0; x < 2; ++x) {
                        ivec2 tap = basePixel + ivec2(x, y);
                        if (any(lessThan(tap, ivec2(0))) || any(greaterThanEqual(
                                tap, ivec2(resolution_global)))) continue;
                        float weight = (x == 0 ? 1.0 - fraction.x : fraction.x)
                            * (y == 0 ? 1.0 - fraction.y : fraction.y);
                        if (weight <= 0.0) continue;
                        vec3 position;
                        float distance;
                        readDiffusePrimaryGeometry(uvec2(tap), position, distance);
                        if (!(distance >= 0.0) || isinf(distance)) continue;
                        if (length(position - psr.endpointRelative) > positionTolerance
                                || dot(readDiffuseGeometryNormal(uvec2(tap)),
                                    psr.geometryNormal) <= 0.75) continue;
                        MaxEntEncoding light;
                        float effectiveSamples, rms;
                        readDiffuseSwap(uvec2(tap), light, effectiveSamples, rms);
                        background.maxEntY += weight * light.maxEntY;
                        background.CoCg += weight * light.CoCg;
                        acceptedWeight += weight;
                    }
                }
                if (acceptedWeight > 1e-6) {
                    background.maxEntY /= acceptedWeight;
                    background.CoCg /= acceptedWeight;
                    resolved = projectDiffuseLighting(background,
                        psr.macroNormal, psr.refractedDirection,
                        psr.roughness, psr.diffuseAlbedo);
                    reusedScreen = true;
                }
            }
        }
    }

    if (!reusedScreen) {
        vec3 cachePosition = camPos + psr.endpointRelative
            + psr.geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
        RadianceCache cache = loadRadianceCacheHistWorld(cachePosition);
        if (radianceCacheValueValid(cache)) {
            resolved = radianceCacheDiffuseIncident(cache,
                psr.macroNormal) * psr.diffuseAlbedo;
        }
    }

    return psr.transmittance * (resolved + psr.surfaceLight);
}

vec3 resolvePSRRefraction(uvec2 xy) {
    bool reusedScreen;
    return resolvePSRRefraction(xy, reusedScreen);
}

#endif
