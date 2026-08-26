#version 430 core

// ===========================================================================
// Pass fog: 最终合成 — 雾效、天空、光照组合 (Final Composite)
// ===========================================================================
// 这是管线末端的合成 pass，负责将各光照分量合成为最终像素颜色。
//
// 输入分量:
//   - diffuse  MaxEntEncoding 入射辐射率场 → Lambert 或 EON 投影 × 漫反射率
//   - refract  折射颜色 → 直接加到漫反射上
//   - reflect  反射颜色 → × albedo (金属/镜面度)
//   - light    直接光照 → 直接加入
//   - emission 发光     → 直接加入
//   - absorption 大气透射率 → × 整体颜色
//
// 天空:
//   - distance < -0.5 → 使用 sampleSky() 计算大气散射颜色
//   - 时域历史已随统一 diffuseIlluminationBuffer (binding 2) 流转，无需额外写回
// ===========================================================================

#define DIFFUSE_BUFFER_MIN2
#define REFLECT_BUFFER_MIN2
#define REFRACT_BUFFER_MIN2

#include "/lib/buffers/frame_data.glsl"
#include "/lib/buffers/buffer_io.glsl"
#include "/lib/buffers/radiance_cache.glsl"
#include "/lib/common.glsl"
#include "/lib/sky.glsl"
#include "/lib/lighting/maxent.glsl"
#include "/lib/lighting/specular_maxent.glsl"
#if EON_ENABLED
#include "/lib/lighting/eon.glsl"
#endif

in vec2 texCoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

// ---------------------------------------------------------------------------
// Jet/rainbow colormap: [0,1] → blue→cyan→green→yellow→red
// ---------------------------------------------------------------------------
vec3 jetColormap(float t) {
    return vec3(
        clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0),
        clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0),
        clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0));
}

// Log-scale normalize for ray-segment distance (0.01m..~160m → [0,1])
float logDistNorm(float d) {
    return clamp(log2(max(d, 0.01) * 100.0 + 1.0) / 14.0, 0.0, 1.0);
}

#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_NOISE_ONLY_CURRENT_WEIGHT || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_NOISE_ONLY_CURRENT_WEIGHT
vec3 debugNoiseOnlyCurrentWeight(float currentWeight) {
    if (!(currentWeight >= 0.0) || isnan(currentWeight) || isinf(currentWeight))
        return vec3(1.0, 0.0, 1.0);
    return jetColormap(clamp(currentWeight, 0.0, 1.0));
}
#endif

#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_PREPARED_ESTIMATOR_VARIANCE || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_PREPARED_ESTIMATOR_VARIANCE
vec3 debugPreparedEstimatorVariance(float standardDeviation) {
    if (!(standardDeviation >= 0.0) || isnan(standardDeviation)
            || isinf(standardDeviation))
        return vec3(1.0, 0.0, 1.0);
    float variance = standardDeviation * standardDeviation;
    // Log2 display of the actual trace variance. V=1 occupies 1/16 of the
    // scale and V=65535 reaches red; larger HDR variances saturate.
    float normalizedVariance = clamp(log2(1.0 + variance) / 16.0,
        0.0, 1.0);
    return jetColormap(normalizedVariance);
}
#endif

#if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_KISH_EFFECTIVE_SAMPLES || DEBUG_VIEW == DEBUG_VIEW_SPECULAR_KISH_EFFECTIVE_SAMPLES
vec3 debugKishEffectiveSamples(float effectiveSamples) {
    if (!(effectiveSamples >= 1.0) || isnan(effectiveSamples) || isinf(effectiveSamples)) return vec3(1.0, 0.0, 1.0);
    return jetColormap(clamp(log2(effectiveSamples) / log2(65504.0), 0.0, 1.0));
}
#endif

#if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_FINAL_VIRTUAL_NORMAL
bool debugReadFinalDenoisedVirtualPosition(ivec2 pixel,
        out vec3 position) {
    if (any(lessThan(pixel, ivec2(0)))
            || any(greaterThanEqual(pixel, ivec2(resolution_global)))) {
        position = vec3(0.0);
        return false;
    }

    SpecularMaxEnt unusedSignal;
    float virtualDistance, validWeight;
    readReflMaxEnt(uvec2(pixel), unusedSignal, virtualDistance, validWeight);
    if (!(validWeight > 0.0) || !(virtualDistance > 0.0)
            || isnan(virtualDistance) || isinf(virtualDistance)) {
        position = vec3(0.0);
        return false;
    }

    position = reconstructPrimaryRay(uvec2(pixel)) * virtualDistance;
    return !any(isnan(position)) && !any(isinf(position));
}

bool debugFinalDenoisedVirtualNormal(ivec2 pixel, out vec3 normal) {
    vec3 centerPosition;
    if (!debugReadFinalDenoisedVirtualPosition(pixel, centerPosition)) {
        normal = vec3(0.0);
        return false;
    }

    // Match the denoiser's one-pixel central-difference reconstruction.
    // Missing image-edge or invalid neighbors collapse to the center point.
    vec3 left = centerPosition;
    vec3 right = centerPosition;
    vec3 down = centerPosition;
    vec3 up = centerPosition;
    vec3 candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(-1, 0), candidate)) left = candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(1, 0), candidate)) right = candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(0, -1), candidate)) down = candidate;
    if (debugReadFinalDenoisedVirtualPosition(
            pixel + ivec2(0, 1), candidate)) up = candidate;

    vec3 virtualTangentX = right - left;
    vec3 virtualTangentY = up - down;
    vec3 unnormalizedNormal = cross(virtualTangentX, virtualTangentY);
    float normalLength2 = dot(unnormalizedNormal, unnormalizedNormal);
    vec3 fallback = normalize(reconstructPrimaryRay(uvec2(pixel)));
    normal = normalLength2 > 1e-20
        ? unnormalizedNormal * inversesqrt(normalLength2) : fallback;
    return !any(isnan(normal)) && !any(isinf(normal));
}
#endif

vec3 projectDiffuseLighting(MaxEntEncoding encoded, vec3 normal,
        vec3 primaryRay, float ggxAlpha, vec3 diffuseAlbedo) {
    #if EON_ENABLED
    // EON owns the nonlinear albedo response, so diffuseAlbedo must be passed
    // into the BRDF rather than multiplied after projection. Dirt RT stores
    // GGX alpha; EON's roughness parameter is its square root.
    return eon_project_maxent(encoded.maxEntY, encoded.CoCg,
        normal, -normalize(primaryRay),
        sqrt(clamp(ggxAlpha, 0.0, 1.0)), diffuseAlbedo);
    #else
    return project_maxent_irradiance(encoded, normal) * diffuseAlbedo;
    #endif
}

float primaryTransmissionIor(float transmissionCode) {
    if (transmissionCode < 0.999) return 1.0;
    int mediumClass = int(transmissionCode + 0.5);
    if (mediumClass == 1) return REFRACTIVE_INDEX;
    if (mediumClass == 2) return GLASS_REFRACTIVE_INDEX;
    if (mediumClass == 3) return 1.31;
    return 1.0;
}

vec3 resolvePSRRefraction(uvec2 xy) {
    PSRResolveData psr = readPSRResolve(xy);

    if (psr.environment) {
        setSkyVars();
        vec3 sky = sampleSky(camPos.y, psr.refractedDirection,
            -lightDir_global);
        return psr.transmittance * max(sky, vec3(0.0));
    }
    if (!psr.endpointValid) return vec3(0.0);

    vec3 resolved = vec3(0.0);
    bool reusedScreen = false;
    if (psr.screenCandidate) {
        vec4 clip = rtViewProjection * vec4(psr.endpointRelative, 1.0);
        if (clip.w > 1e-6) {
            vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
            if (all(greaterThanEqual(uv, vec2(0.0)))
                    && all(lessThan(uv, vec2(1.0)))) {
                vec2 samplePixel = uv * vec2(resolution_global) - 0.5;
                ivec2 nearestPixel = clamp(
                    ivec2(floor(samplePixel + 0.5)), ivec2(0),
                    ivec2(resolution_global) - 1);
                vec3 backgroundPosition;
                float backgroundDistance;
                readDiffusePrimaryGeometry(uvec2(nearestPixel),
                    backgroundPosition, backgroundDistance);
                vec3 backgroundGeometryNormal =
                    readPrimaryGeometryNormal(uvec2(nearestPixel));

                float pixelFootprint = max(length(psr.endpointRelative)
                    / max(float(resolution_global.y), 1.0), 0.025);
                float positionTolerance = max(0.12,
                    6.0 * pixelFootprint);
                bool geometryMatches = backgroundDistance >= 0.0
                    && length(backgroundPosition - psr.endpointRelative)
                        <= positionTolerance
                    && dot(backgroundGeometryNormal,
                        psr.geometryNormal) > 0.75;
                if (geometryMatches) {
                    diffuseIlluminationData background =
                        sampleDiffuse(samplePixel);
                    resolved = projectDiffuseLighting(background.data_swap,
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

void main() {
    uvec2 xy = uvec2(gl_FragCoord.xy);
    ivec2 pix = ivec2(gl_FragCoord.xy);
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);

    // Primary visibility remains authoritative. The diffuse domain may be sky
    // behind water/glass even though the primary pixel itself is a surface.
    uvec4 primaryGeometryWords = readPrimaryGeometryWords(xy);
    float primaryDistance = uintBitsToFloat(primaryGeometryWords.w);
    float surfaceMask = primaryDistance >= 0.0 ? 1.0 : 0.0;

    #if DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_NOISE_ONLY_CURRENT_WEIGHT
    fragColor.xyz = surfaceMask < 0.5 ? vec3(0.0)
        : debugNoiseOnlyCurrentWeight(debugReadDiffuseNoiseOnlyCurrentWeight(xy));
    return;
    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_NOISE_ONLY_CURRENT_WEIGHT
    fragColor.xyz = surfaceMask < 0.5 ? vec3(0.0)
        : debugNoiseOnlyCurrentWeight(debugReadSpecularNoiseOnlyCurrentWeight(xy));
    return;
    #elif DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_PREPARED_ESTIMATOR_VARIANCE
    fragColor.xyz = surfaceMask < 0.5 ? vec3(0.0)
        : debugPreparedEstimatorVariance(
            debugReadDiffusePreparedEstimatorStandardDeviation(xy));
    return;
    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_PREPARED_ESTIMATOR_VARIANCE
    fragColor.xyz = surfaceMask < 0.5 ? vec3(0.0)
        : debugPreparedEstimatorVariance(
            debugReadSpecularPreparedEstimatorStandardDeviation(xy));
    return;
    #endif

    #if DEBUG_VIEW == DEBUG_VIEW_SPECULAR_FINAL_VIRTUAL_NORMAL
    if (surfaceMask < 0.5) {
        fragColor.xyz = vec3(0.0);
        return;
    }
    vec3 virtualNormal;
    bool virtualNormalValid = debugFinalDenoisedVirtualNormal(
        pix, virtualNormal);
    fragColor.xyz = virtualNormalValid
        ? virtualNormal * 0.5 + 0.5 : vec3(1.0, 0.0, 1.0);
    return;
    #endif

    #if DEBUG_VIEW >= DEBUG_VIEW_MOTION_MATCH_STATUS && DEBUG_VIEW <= DEBUG_VIEW_MOTION_PRIMARY_SCREEN
    // Sky pixels do not contain a surface motion record.
    if (surfaceMask < 0.5) {
        fragColor.xyz = vec3(0.0);
        return;
    }
    #endif

    // Sky has no surface-domain denoiser, material, cache, or motion record.
    // Only diagnostics with an explicit sky meaning produce a non-black value.
    if (surfaceMask < 0.5) {
        #if DEBUG_VIEW == DEBUG_VIEW_OUTPUT_COMPOSITE
        vec3 emisVal, rdVal, absorptionVal;
        vec3 transAlbedo_unused, lightVal_unused;
        readMiscTransport(GEO_N_MISC, xy, transAlbedo_unused, emisVal);
        rdVal = reconstructPrimaryRay(xy);
        readLightAbs(GEO_N_LIGHTABS, xy, lightVal_unused, absorptionVal);

        setSkyVars();
        vec3 sky = sampleSky(camPos.y, rdVal, -lightDir_global);
        // The ray buffer is intentionally sampled on its native grid rather
        // than assuming texel centres. Derivatives describe that grid's pixel
        // footprint and are used only to area-filter the finite solar disc.
        vec3 pointSunDisc = sampleSkySunDisc(camPos.y, rdVal, -lightDir_global);
        vec3 filteredSunDisc = sampleSkySunDiscFiltered(
            camPos.y,
            rdVal,
            -lightDir_global,
            dFdx(rdVal),
            dFdy(rdVal));
        sky = max(sky - pointSunDisc + filteredSunDisc, vec3(0.0));
        fragColor.xyz = absorptionVal * sky + emisVal;
        #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_MEDIUM_EMISSION
        vec3 transmissionAlbedoUnused;
        readMiscTransport(GEO_N_MISC, xy, transmissionAlbedoUnused, fragColor.xyz);
        #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_ABSORPTION_TRANSMITTANCE
        vec3 surfaceLightUnused;
        readLightAbs(GEO_N_LIGHTABS, xy, surfaceLightUnused, fragColor.xyz);
        #else
        fragColor.xyz = vec3(0.0);
        #endif
        writeDiffuseLightRTSky(xy);
        return;
    }

    // =========================================================================
    // 分支 2: 表面像素 — 跳过 Geo0，读 Geo1 + 所有光照
    // =========================================================================
    vec3 geometryNormal, specAlbedo, diffAlbedo, transAlbedo, emisVal, lightVal, absorptionVal, rdVal;
    float rough, pathR;
    int illumType;
    geometryNormal = decodeNormalU(primaryGeometryWords.x);
    rough = unpackHalf2x16(primaryGeometryWords.y).x;
    pathR = rough;
    illumType = int(primaryGeometryWords.y >> 16u);
    vec3 textureNormal = decodeNormalU(primaryGeometryWords.z);
    readAlbedosPath(GEO_N_ALBEDOS, xy, specAlbedo, diffAlbedo);
    readMiscTransport(GEO_N_MISC, xy, transAlbedo, emisVal);
    rdVal = reconstructPrimaryRay(xy);
    readLightAbs(GEO_N_LIGHTABS, xy, lightVal, absorptionVal);
    vec3 primaryCs, primaryCd;
    vec2 primaryS;
    readPrimaryMaterial(xy, primaryCs, primaryCd, primaryS);

    diffuseIlluminationData tmp = fetchDiffuse(pix);
    vec3 refractionLighting = resolvePSRRefraction(xy);
    SpecularMaxEnt reflectionMaxEnt;
    float reflectionVirtualDistance, reflectionDebugWeight;
    readReflMaxEnt(xy, reflectionMaxEnt, reflectionVirtualDistance,
        reflectionDebugWeight);


    float primaryIor = primaryTransmissionIor(primaryS.y);
    float primaryEtaRatio = eye_medium_global != 0u
        ? primaryIor : (1.0 / primaryIor);
    vec3 specularLighting = projectSpecularMaxEnt(reflectionMaxEnt,
        rdVal, textureNormal, geometryNormal, rough,
        primaryCs, primaryS, primaryEtaRatio);
    vec3 diffuseLighting = projectDiffuseLighting(
        tmp.data_swap, textureNormal, rdVal, rough, diffAlbedo);

    // MaxEnt 漫反射投影使用微法线；EON 开启时同时恢复粗糙漫反射响应。

    // ---------------------------------------------------------------------
    // Final lighting decomposition (0-7).
    // ---------------------------------------------------------------------
    #if DEBUG_VIEW == DEBUG_VIEW_OUTPUT_COMPOSITE
    fragColor.xyz = absorptionVal
            * (diffuseLighting
                + refractionLighting * transAlbedo
                + specularLighting
                + lightVal)
            + emisVal;

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_DIFFUSE_LIGHTING
    // Material-applied diffuse transport. First-hit NEE is already included.
    fragColor.xyz = diffuseLighting;

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_SPECULAR_LIGHTING
    fragColor.xyz = specularLighting;

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_REFRACTION_LIGHTING
    fragColor.xyz = refractionLighting * transAlbedo;

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_DIFFUSE_IRRADIANCE
    // Diffuse projection without the material diffuse-albedo multiplier.
    fragColor.xyz = projectDiffuseLighting(
        tmp.data_swap, textureNormal, rdVal, rough, vec3(1.0));

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_PRIMARY_SURFACE_LIGHT
    fragColor.xyz = lightVal;

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_MEDIUM_EMISSION
    fragColor.xyz = emisVal;

    #elif DEBUG_VIEW == DEBUG_VIEW_OUTPUT_ABSORPTION_TRANSMITTANCE
    fragColor.xyz = absorptionVal;

    // ---------------------------------------------------------------------
    // Primary material and geometry (10-13).
    // ---------------------------------------------------------------------
    #elif DEBUG_VIEW == DEBUG_VIEW_MATERIAL_SHADING_NORMAL
    fragColor.xyz = textureNormal * 0.5 + 0.5;

    #elif DEBUG_VIEW == DEBUG_VIEW_MATERIAL_DIFFUSE_ALBEDO
    fragColor.xyz = diffAlbedo;

    #elif DEBUG_VIEW == DEBUG_VIEW_MATERIAL_SPECULAR_ALBEDO
    // Reflection material multiplier produced by the primary material model.
    fragColor.xyz = specAlbedo;

    #elif DEBUG_VIEW == DEBUG_VIEW_MATERIAL_GGX_ALPHA
    fragColor.xyz = vec3(rough);

    // ---------------------------------------------------------------------
    // Diffuse denoiser (20-24). Views 23-24 return before this branch.
    // ---------------------------------------------------------------------
    #elif DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_TEMPORAL_HISTORY_SIGNAL
    {
        MaxEntEncoding temporalHistory;
        float effectiveSamples, rootMeanY2Unused;
        readDiffuseHist(xy, temporalHistory, effectiveSamples, rootMeanY2Unused);
        fragColor.xyz = projectDiffuseLighting(temporalHistory, geometryNormal, rdVal, rough, vec3(1.0));
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_FINAL_NORMALIZED_FIRST_MOMENT
    // 2*abs(E[R*u]/max(E[R], |E[R*u]|)); this is directionality, not RGB light.
    fragColor.xyz = 2.0 * abs(tmp.data_swap.maxEntY.xyz / max(max(tmp.data_swap.maxEntY.w, length(tmp.data_swap.maxEntY.xyz)), 1e-6));

    #elif DEBUG_VIEW == DEBUG_VIEW_DIFFUSE_KISH_EFFECTIVE_SAMPLES
    // Blue is N_eff=1; red approaches the FP16 metadata storage limit.
    fragColor.xyz = debugKishEffectiveSamples(tmp.weight);

    // ---------------------------------------------------------------------
    // Specular denoiser (30-37). Views 33, 36, and 37 return above.
    // ---------------------------------------------------------------------
    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_SAMPLED_DIRECTION
    fragColor.xyz = debugReadReflectionSampleDirection(xy) * 0.5 + 0.5;

    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_CURRENT_TRACKING_HIT_DISTANCE
    {
        uvec3 debugSignalWords;
        float trackingHitDistance, resolvedHistoryContribution;
        debugReadSpecularTemporalState(xy, debugSignalWords, trackingHitDistance, resolvedHistoryContribution);
        fragColor.xyz = trackingHitDistance >= VPROJDIST_SKY * 0.99 ? vec3(1.0) : jetColormap(logDistNorm(trackingHitDistance));
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_TEMPORAL_HISTORY_SIGNAL
    // Committed Raw RT temporal history, projected without the spatial result.
    {
        uvec3 debugSignalWords;
        float trackingHitDistance, resolvedHistoryContribution;
        debugReadSpecularTemporalState(xy, debugSignalWords, trackingHitDistance, resolvedHistoryContribution);
        fragColor.xyz = projectSpecularMaxEnt(
            unpackSpecularMaxEnt(debugSignalWords), rdVal, textureNormal,
            geometryNormal, rough, primaryCs, primaryS, primaryEtaRatio);
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_KISH_EFFECTIVE_SAMPLES
    // Read the committed current-frame history, not the final blend weight.
    fragColor.xyz = debugKishEffectiveSamples(readMaxEntSpecularHistory(xy).historyEffectiveSamples);

    #elif DEBUG_VIEW == DEBUG_VIEW_SPECULAR_RESOLVED_HISTORY_CONTRIBUTION
    // Exact history coefficient used by the Raw RT temporal commit: 1-alpha.
    {
        uvec3 debugSignalWords;
        float trackingHitDistance, resolvedHistoryContribution;
        debugReadSpecularTemporalState(xy, debugSignalWords, trackingHitDistance, resolvedHistoryContribution);
        fragColor.xyz = jetColormap(clamp(resolvedHistoryContribution, 0.0, 1.0));
    }

    // ---------------------------------------------------------------------
    // Refraction and PSR (40-41).
    // ---------------------------------------------------------------------
    #elif DEBUG_VIEW == DEBUG_VIEW_REFRACTION_PSR_ROUTE
    // PSR route: green=screen reuse, blue=environment, orange=cache fallback.
    {
        PSRResolveData psr = readPSRResolve(xy);
        fragColor.xyz = psr.environment ? vec3(0.0, 0.35, 1.0)
            : psr.screenCandidate ? vec3(0.0, 1.0, 0.0)
            : psr.endpointValid ? vec3(1.0, 0.35, 0.0)
            : vec3(0.0);
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_REFRACTION_VIRTUAL_ENDPOINT_DISTANCE
    {
        PSRResolveData psr = readPSRResolve(xy);
        float d = length(psr.endpointRelative);
        fragColor.xyz = psr.environment ? vec3(1.0)
            : jetColormap(logDistNorm(d));
    }

    // ---------------------------------------------------------------------
    // Path guiding and radiance cache (50-55).
    // ---------------------------------------------------------------------
    #elif DEBUG_VIEW == DEBUG_VIEW_PATH_GUIDE_FINAL_DIRECTION
    {
        vec4 guideY;
        float guideW;
        float guideM;
        readPathGuide(xy, guideY, guideW, guideM);
        if (guideM < 1e-6) {
            fragColor.xyz = vec3(0.0);
        } else {
            vec3 dir = guideY.xyz / max(length(guideY.xyz), 1e-6);
            fragColor.xyz = dir * 0.5 + 0.5;
        }
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_RADIANCE_CACHE_SURFACE_HISTORY_SAMPLE
    // Trilinear history-cache irradiance at the visible surface plus its
    // separately stored primary-surface light.
    {
        vec3 relativePos;
        float distance;
        readPrimaryPosition(xy, relativePos, distance);
        vec3 surfaceWorldPos = camPos + relativePos;
        vec3 cacheSamplePos = surfaceWorldPos
                + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
        vec3 cacheCoord = radianceCacheWorldToVoxel(cacheSamplePos, camPos);
        if (isRadianceCacheSampleInBounds(cacheCoord)) {
            RadianceCache cache = sampleRadianceCacheHist(cacheCoord, camPos);
            fragColor.xyz = radianceCacheValueValid(cache)
                ? radianceCacheDiffuseIncident(cache, textureNormal) : vec3(0.0);
        } else {
            fragColor.xyz = vec3(0.0);
        }
        fragColor.xyz += lightVal;
    }

    #elif DEBUG_VIEW >= DEBUG_VIEW_RADIANCE_CACHE_CURRENT_IRRADIANCE && DEBUG_VIEW <= DEBUG_VIEW_RADIANCE_CACHE_FILTERED_IRRADIANCE
    {
        vec3 relativePos;
        float distance;
        readPrimaryPosition(xy, relativePos, distance);
        vec3 worldPos = camPos + relativePos
            + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
        RadianceCacheAddress address = findRadianceCacheAddress(worldPos);
        if (!validateRadianceCacheAddress(address)) {
            fragColor.xyz = vec3(1.0, 0.0, 1.0);
        } else {
            #if DEBUG_VIEW == DEBUG_VIEW_RADIANCE_CACHE_CURRENT_IRRADIANCE
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_CURRENT_0, RC_PLANE_CURRENT_1);
                fragColor.xyz = radianceCacheValueValid(cache)
                    ? radianceCacheDiffuseIncident(cache, textureNormal) : vec3(0.0);
            #elif DEBUG_VIEW == DEBUG_VIEW_RADIANCE_CACHE_HISTORY_IRRADIANCE
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1);
                fragColor.xyz = radianceCacheValueValid(cache)
                    ? radianceCacheDiffuseIncident(cache, textureNormal) : vec3(0.0);
            #elif DEBUG_VIEW == DEBUG_VIEW_RADIANCE_CACHE_HISTORY_SAMPLE_COUNT
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1);
                fragColor.xyz = jetColormap(clamp(cache.M
                    / max(float(RADIANCE_CACHE_MAX_HIST), 1.0), 0.0, 1.0));
            #else
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
                fragColor.xyz = radianceCacheValueValid(cache)
                    ? radianceCacheDiffuseIncident(cache, textureNormal) : vec3(0.0);
            #endif
        }
    }

    // ---------------------------------------------------------------------
    // Entity and screen-space motion (60-62).
    // ---------------------------------------------------------------------
    #elif DEBUG_VIEW == DEBUG_VIEW_MOTION_MATCH_STATUS
    {
        vec3 surfaceMotion;
        float motionClass;
        readSurfaceMotion(xy, surfaceMotion, motionClass);

        // Gray: static scene. Green: entity matched to the previous frame.
        // Failure colors: red=no key, orange=vertex count, cyan=topology,
        // blue=non-finite or excessive world motion.
        if (motionClass > 1.5) {
            fragColor.xyz = vec3(0.18);
        } else if (motionClass > 0.5) {
            fragColor.xyz = vec3(0.0, 1.0, 0.0);
        } else if (motionClass > -0.5) {
            fragColor.xyz = vec3(1.0, 0.0, 0.0);
        } else if (motionClass > -1.5) {
            fragColor.xyz = vec3(1.0, 0.35, 0.0);
        } else if (motionClass > -2.5) {
            fragColor.xyz = vec3(0.0, 1.0, 1.0);
        } else {
            fragColor.xyz = vec3(0.0, 0.2, 1.0);
        }
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_MOTION_ENTITY_WORLD
    {
        vec3 surfaceMotion;
        float motionClass;
        readSurfaceMotion(xy, surfaceMotion, motionClass);

        if (motionClass > 1.5) {
            fragColor.xyz = vec3(0.0);
        } else if (motionClass < 0.5) {
            fragColor.xyz = vec3(1.0, 0.0, 1.0);
        } else {
            // Neutral gray is zero. +/-0.0625 block/frame reaches the range ends.
            fragColor.xyz = clamp(vec3(0.5) + surfaceMotion * 8.0, 0.0, 1.0);
        }
    }

    #elif DEBUG_VIEW == DEBUG_VIEW_MOTION_PRIMARY_SCREEN
    {
        vec3 surfaceMotion;
        float motionClass;
        readSurfaceMotion(xy, surfaceMotion, motionClass);

        if (motionClass < 0.5) {
            fragColor.xyz = vec3(1.0, 0.0, 1.0);
        } else {
            vec3 relativePos;
            float distance;
            readPrimaryPosition(xy, relativePos, distance);
            vec3 previousRelativePos = relativePos
                + camPos - prevRaytracingCamPos - surfaceMotion;
            vec4 previousClip = rtPrevViewProjection
                * vec4(previousRelativePos, 1.0);
            if (abs(previousClip.w) < 1e-6) {
                fragColor.xyz = vec3(1.0, 0.0, 1.0);
            } else {
                vec2 currentUv = vec2(xy) / vec2(resolution_global);
                vec2 previousUv = previousClip.xy / previousClip.w * 0.5 + 0.5;
                vec2 velocityPixels = (currentUv - previousUv)
                    * vec2(resolution_global);
                // R/G: signed X/Y at 16 px/frame. B: magnitude.
                fragColor.xyz = vec3(
                    0.5 + 0.5 * clamp(velocityPixels.x / 16.0, -1.0, 1.0),
                    0.5 + 0.5 * clamp(velocityPixels.y / 16.0, -1.0, 1.0),
                    clamp(length(velocityPixels) / 16.0, 0.0, 1.0));
            }
        }
    }

    #endif
}
