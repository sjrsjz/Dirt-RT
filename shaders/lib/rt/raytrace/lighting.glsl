#ifndef DIRT_RT_RAYTRACE_LIGHTING_GLSL
#define DIRT_RT_RAYTRACE_LIGHTING_GLSL

// Sun NEE and sparse-radiance-cache lighting evaluation.

#if defined(PRIMARY_GBUFFER_PASS) || defined(FIRST_LOBE_DIFFUSE)
void markRadianceCacheGeometryHit(uvec2 pixel, vec3 hitPosition,
        vec3 geometryNormal) {
    uvec2 tileMask = uvec2(RADIANCE_CACHE_MARK_TILE_SIZE - 1u);
    if (any(notEqual(pixel & tileMask, uvec2(0u)))) return;
    vec3 cameraPosition = cam.viewInverse[3].xyz;
    vec3 cacheCoord = radianceCacheWorldToVoxel(hitPosition, cameraPosition);
    if (!isRadianceCacheSampleInBounds(cacheCoord)) return;
    vec3 solidPosition = hitPosition
        - geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    vec3 airPosition = hitPosition
        + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    ivec3 solidBrick = radianceCacheWorldBrick(
        radianceCacheWorldVoxel(solidPosition));
    ivec3 airBrick = radianceCacheWorldBrick(
        radianceCacheWorldVoxel(airPosition));
    markRadianceCacheBlockHasGeometry(solidBrick, cameraPosition);
    if (any(notEqual(airBrick, solidBrick)))
        markRadianceCacheBlockHasGeometry(airBrick, cameraPosition);
}
#endif

// ===========================================================================
// NEE: Direct Sunlight
// ===========================================================================

bool sampleDirectSun(vec3 ro, vec3 geometryNormal, vec3 shadingNormal,
    vec3 lightDir, bool inside, vec2 xi,
    out vec3 wi, out vec3 Li, out float lightPdf) {
    wi = -lightDir;
    Li = vec3(0.0);
    lightPdf = 1.0 / sunSolidAngle();
    vec3 X, Y, Z;
    XYZ(lightDir, X, Y, Z);
    float r1 = xi.x;
    float alpha = xi.y * 2.0 * PI;
    float cosbeta = 1.0 - r1 * (1.0 - cosD_S);
    vec3 sampleDir = cosbeta * Y + sqrt(1.0 - cosbeta * cosbeta) * (cos(alpha) * X + sin(alpha) * Z);

    wi = -sampleDir;
    // Reject directions that no surface lobe can consume before paying for
    // an RT visibility traversal.
    if (dot(wi, geometryNormal) <= 0.0
            || dot(wi, shadingNormal) <= 0.0) return false;

    ro += (dot(lightDir, geometryNormal) > 0.15 ? lightDir : geometryNormal) * 0.001;

    vec3 ro_o, rd_o;
    float t = raycast(ro, wi, ro_o, rd_o, !inside, true);
    if (t > -0.5) return false;

    // NEE owns only the solar disc. Atmospheric scattering remains in the
    // BSDF-sampled no-disc environment and is therefore never double counted.
    Li = sampleSkySunDisc(ro.y, wi, lightDir).xyz
            * payload_unpackShadow(tmp_Payload.data);
    return !any(isnan(Li)) && !any(isinf(Li));
}

bool sampleDirectSun(vec3 ro, vec3 geometryNormal, vec3 shadingNormal,
    vec3 lightDir, bool inside,
    out vec3 wi, out vec3 Li, out float lightPdf) {
    return sampleDirectSun(ro, geometryNormal, shadingNormal, lightDir, inside,
        vec2(getRandom(), getRandom()), wi, Li, lightPdf);
}

vec3 evalDirectDiffuse(vec3 ro, vec3 geometryNormal, vec3 shadingNormal,
    vec3 diffuseAlbedo, vec3 rd_i, vec3 lightDir, bool inside) {
    vec3 wi, Li;
    float lightPdf;
    if (!sampleDirectSun(ro, geometryNormal, shadingNormal,
            lightDir, inside, wi, Li, lightPdf))
        return vec3(0.0);

    // sampleDirectSun already rejected both invalid hemispheres.
    float OiN = dot(wi, shadingNormal);
    return max(vec3(0.0), diffuseAlbedo * Li
            * (OiN / max(PI * lightPdf, 1e-20)));
}

vec3 evalDirectDiffuseIncident(vec3 ro, vec3 geometryNormal,
    vec3 shadingNormal, vec3 rd_i, vec3 lightDir, bool inside) {
    // Unit Lambertian albedo makes the result E/pi without a fragile
    // component-wise divide by the material base color.
    return evalDirectDiffuse(ro, geometryNormal, shadingNormal, vec3(1.0),
        rd_i, lightDir, inside);
}

vec3 misLightContribution(
    vec3 fTimesNoL, vec3 Li, float lightPdf, float bsdfStrategyPdf
) {
    float misWeight = powerHeuristic(lightPdf, bsdfStrategyPdf);
    vec3 result = fTimesNoL * Li
            * (misWeight / max(lightPdf, 1e-20));
    return (!any(isnan(result)) && !any(isinf(result)))
    ? max(result, vec3(0.0)) : vec3(0.0);
}

bool loadSecondaryRadianceCache(
    vec3 surfacePosition,
    vec3 geometryNormal,
    out RadianceCache cache
) {
    cache = emptyCache();
    vec3 samplePosition = surfacePosition
            + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
    vec3 currentCameraPosition = cam.viewInverse[3].xyz;
    vec3 voxelCoord = radianceCacheWorldToVoxel(
            samplePosition, currentCameraPosition);
    if (!isRadianceCacheSampleInBounds(voxelCoord)) return false;

    RadianceCacheAddress address = findRadianceCacheAddress(samplePosition);
    if (!radianceCacheAddressHasHistory(address, cam.frameId)) return false;
    cache = loadRadianceCachePlanes(
            address, RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
    return radianceCacheValueValid(cache);
}

vec3 evaluateCachedDiffuseLighting(
    RadianceCache cache,
    vec3 macroNormal,
    LobeProbs lobes
) {
    // diffWeight already contains diffuseAlbedo / P_diff.
    return radianceCacheDiffuseIncident(cache, macroNormal) * lobes.diffWeight;
}

vec3 evaluateCachedRoughSpecularLighting(
    RadianceCache cache,
    vec3 rd_i,
    vec3 macroNormal,
    material surf,
    LobeProbs lobes
) {
    // Broad GGX is approximated by a cosine convolution around its dominant
    // direction, then modulated by the integrated GGX DFG response.
    vec3 dominantDirection = GetSpecularDominantDirection(
            macroNormal, rd_i, sqrt(clamp(surf.R.x, 0.0, 1.0)));
    vec3 incidentResponse = radianceCacheDiffuseIncident(
            cache, dominantDirection);
    vec3 specularAlbedo = evaluateSpecularAlbedo(
            surf, rd_i, macroNormal,
            1.0 / transportIorFromMaterial(surf));
    return incidentResponse * specularAlbedo / max(lobes.P_spec, 1e-5);
}


#endif // DIRT_RT_RAYTRACE_LIGHTING_GLSL
