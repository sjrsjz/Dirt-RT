#version 430 core

// ===========================================================================
// Pass fog: 最终合成 — 雾效、天空、光照组合 (Final Composite)
// ===========================================================================
// 这是管线末端的合成 pass，负责将各光照分量合成为最终像素颜色。
//
// 输入分量:
//   - diffuse  AliceEncoding 入射辐射率场 → project_alice_irradiance() 投影为辐照度 × albedo2
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
#include "/lib/sky.glsl"
#include "/lib/lighting/alice.glsl"

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

void main() {
    uvec2 xy = uvec2(gl_FragCoord.xy);
    ivec2 pix = ivec2(gl_FragCoord.xy);
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);

    // 用 N=1 surfaceMask 判定天空/表面（surfaceMask 不再在 N=0 重复存储）
    float surfaceMask = readDiffuseSurfaceMask(xy);

    #if DEBUG_VIEW >= 35 && DEBUG_VIEW <= 37
    // Sky pixels do not contain a surface motion record.
    if (surfaceMask < 0.5) {
        fragColor.xyz = vec3(0.0);
        return;
    }
    #endif

    // =========================================================================
    // 分支 1: 天空像素 — 读 Geo0 + N=3 + N=4
    // =========================================================================
    if (surfaceMask < 0.5) {
        vec3 worldPos;
        float dist;
        readGeo0(GEO_N_GEO, xy, worldPos, dist);

        vec3 emisVal, rdVal, absorptionVal;
        vec3 transAlbedo_unused, lightVal_unused;
        readMisc(GEO_N_MISC, xy, transAlbedo_unused, emisVal, rdVal);
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
        writeDiffuseLightRTSky(xy);
        return;
    }

    // =========================================================================
    // 分支 2: 表面像素 — 跳过 Geo0，读 Geo1 + 所有光照
    // =========================================================================
    vec3 geometryNormal, specAlbedo, diffAlbedo, transAlbedo, emisVal, lightVal, absorptionVal, rdVal;
    float rough, pathR;
    int illumType;
    readGeo1(GEO_N_NORMALS, xy, geometryNormal, rough, illumType, pathR);
    vec3 microN;
    readAlbedosPathMicroNormal(GEO_N_ALBEDOS, xy, specAlbedo,
        diffAlbedo, microN);
    readMisc(GEO_N_MISC, xy, transAlbedo, emisVal, rdVal);
    readLightAbs(GEO_N_LIGHTABS, xy, lightVal, absorptionVal);

    diffuseIlluminationData tmp = fetchDiffuse(pix);
    vec3IlluminationData tmp2 = fetchReflect(pix);
    vec3IlluminationData tmp3 = fetchRefract(pix);

    // ALICE 辐照度投影使用微法线 (microN, N=5) — 恢复法线贴图细节

    #if DEBUG_VIEW == 0
    fragColor.xyz = absorptionVal
            * (project_alice_irradiance(tmp.data_swap, microN)
                * diffAlbedo
                + tmp3.data_swap * transAlbedo
                + tmp2.data_swap * specAlbedo
                + lightVal)
            + emisVal;

    #elif DEBUG_VIEW == 1
    // Diffuse transport only. First-hit NEE is already part of this signal.
    fragColor.xyz = project_alice_irradiance(tmp.data_swap, microN) * diffAlbedo;

    #elif DEBUG_VIEW == 2
    // Refract only
    fragColor.xyz = tmp3.data_swap * transAlbedo;

    #elif DEBUG_VIEW == 3
    // Reflect only
    fragColor.xyz = tmp2.data_swap * specAlbedo;

    #elif DEBUG_VIEW == 4
    // White model: diffuse irradiance only, no albedo
    fragColor.xyz = project_alice_irradiance(tmp.data_swap, microN);

    #elif DEBUG_VIEW == 5
    // Light field: ALICE normalized dominant direction × energy
    fragColor.xyz = 2.0 * abs(tmp.data_swap.aliceY.xyz / max(max(tmp.data_swap.aliceY.w, length(tmp.data_swap.aliceY.xyz)), 1e-6));

    #elif DEBUG_VIEW == 6
    // Normals: world-space geometryNormal as RGB
    fragColor.xyz = microN * 0.5 + 0.5;

    #elif DEBUG_VIEW == 7
    // Absorption / atmospheric transmission
    fragColor.xyz = absorptionVal;

    #elif DEBUG_VIEW == 8
    // Actually sampled GGX reflection direction as RGB (oct-decoded)
    {
        vec3 Rpos, R;
        readReflGeo(xy, Rpos, R);
        fragColor.xyz = R * 0.5 + 0.5;
    }

    #elif DEBUG_VIEW == 9
    // Representative RT endpoint after the 7x7 moment-space filter:
    // length(E[X]). The fourth moment is deliberately not visualized here.
    {
        RelaxEndpointMoments endpoint =
            readReflEndpointMoments(xy);
        float d = relaxEndpointMomentsValid(endpoint)
            ? length(endpoint.mean) *
                clamp(VPROJDIST_SKY, 1.0, 65504.0)
            : 0.0;
        fragColor.xyz = (d >= VPROJDIST_SKY * 0.99) ? vec3(1.0) : jetColormap(logDistNorm(d));
    }

    #elif DEBUG_VIEW == 10
    // Specular albedo (rC.rgb * S.x) — the reflection material multiplier
    fragColor.xyz = specAlbedo;

    #elif DEBUG_VIEW == 11
    // Roughness as grayscale
    fragColor.xyz = vec3(rough);

    #elif DEBUG_VIEW == 12
    // Temporally accumulated reflection incident, before every spatial stage
    // and before specularAlbedo modulation.
    fragColor.xyz = tmp2.data_swap;

    #elif DEBUG_VIEW == 13
    // Diffuse temporal accumulation weight (heatmap) — N_eff / TEMPORAL_MAX_HISTORY
    fragColor.xyz = jetColormap(clamp(tmp.weight / TEMPORAL_MAX_HISTORY, 0.0, 1.0));

    #elif DEBUG_VIEW == 14
    // Actual temporal history contribution to the reflection radiance.
    fragColor.xyz = jetColormap(clamp(tmp2.data_swap.r, 0.0, 1.0));

    #elif DEBUG_VIEW == 38
    // Current reflection input before temporal accumulation.
    fragColor.xyz = tmp2.data_swap;

    #elif DEBUG_VIEW == 39
    // Previous reflection fetched at the surface-reprojected address.
    fragColor.xyz = tmp2.data_swap;

    #elif DEBUG_VIEW == 40
    // Previous reflection fetched at the virtual-motion address.
    fragColor.xyz = tmp2.data_swap;

    #elif DEBUG_VIEW == 15
    // Refract temporal accumulation weight (heatmap)
    fragColor.xyz = jetColormap(clamp(tmp3.weight / ACCUMULATION_LENGTH, 0.0, 1.0));

    #elif DEBUG_VIEW == 16
    // First-surface material emission
    fragColor.xyz = lightVal;

    #elif DEBUG_VIEW == 17
    // Emission accumulated along the primary medium segment
    fragColor.xyz = emisVal;

    #elif DEBUG_VIEW == 18
    // Diffuse albedo — per-pixel diffuse material multiplier
    fragColor.xyz = diffAlbedo;

    #elif DEBUG_VIEW == 19
    // Refraction virtual projection distance (IOR-adjusted, rainbow colormap, log scale)
    {
        vec3 dummyColor;
        float d, dummyW;
        readRefrLight(xy, dummyColor, d, dummyW);
        fragColor.xyz = (d >= VPROJDIST_SKY * 0.99) ? vec3(1.0) : jetColormap(logDistNorm(d));
    }

    #elif DEBUG_VIEW == 20
    // Path guide ALICE direction as RGB (蓄水池+降噪投票结果, N=5)
    {
        vec4 guideY;
        float guideW;
        float guideM;
        readPathGuide(xy, guideY, guideW, guideM);
        if (guideM < 1e-6) {
            fragColor.xyz = vec3(0.0); // 无效/天空 → 黑
        } else {
            vec3 dir = guideY.xyz / max(length(guideY.xyz), 1e-6);
            fragColor.xyz = dir * 0.5 + 0.5; // 方向→RGB
        }
    }

    #elif DEBUG_VIEW == 21
    // 原始时域累积白模 (N=2 hist ALICE × geometryNormal, 降噪前)
    {
        AliceEncoding raw;
        float w, meanY2_unused;
        readDiffuseHist(xy, raw, w, meanY2_unused);
        fragColor.xyz = project_alice_irradiance(raw, geometryNormal);
    }

    #elif DEBUG_VIEW == 22
    // 世界格点辐射率缓存：可见表面处的白模辐照度
    {
        vec3 relativePos;
        float distance;
        readGeo0(GEO_N_GEO, xy, relativePos, distance);
        vec3 surfaceWorldPos = camPos + relativePos;
        vec3 cacheSamplePos = surfaceWorldPos
                + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
        vec3 cacheCoord = radianceCacheWorldToVoxel(cacheSamplePos, camPos);
        if (isRadianceCacheSampleInBounds(cacheCoord)) {
            RadianceCache cache = sampleRadianceCacheHist(cacheCoord, camPos);
            fragColor.xyz = radianceCacheValueValid(cache)
                ? radianceCacheDiffuseIncident(cache, microN) : vec3(0.0);
        } else {
            fragColor.xyz = vec3(0.0);
        }
        fragColor.xyz += lightVal;
    }

    #elif DEBUG_VIEW >= 23 && DEBUG_VIEW <= 30
    // Reflection spatial-pipeline stage selected by the corresponding pass.
    fragColor.xyz = tmp2.data_swap;

    #elif DEBUG_VIEW >= 31 && DEBUG_VIEW <= 34
    {
        vec3 relativePos;
        float distance;
        readGeo0(GEO_N_GEO, xy, relativePos, distance);
        vec3 worldPos = camPos + relativePos
            + geometryNormal * RADIANCE_CACHE_SURFACE_EPSILON;
        RadianceCacheAddress address = findRadianceCacheAddress(worldPos);
        if (!validateRadianceCacheAddress(address)) {
            fragColor.xyz = vec3(1.0, 0.0, 1.0);
        } else {
            #if DEBUG_VIEW == 31
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_CURRENT_0, RC_PLANE_CURRENT_1);
                fragColor.xyz = radianceCacheValueValid(cache)
                    ? radianceCacheDiffuseIncident(cache, microN) : vec3(0.0);
            #elif DEBUG_VIEW == 32
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1);
                fragColor.xyz = radianceCacheValueValid(cache)
                    ? radianceCacheDiffuseIncident(cache, microN) : vec3(0.0);
            #elif DEBUG_VIEW == 33
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_HISTORY_0, RC_PLANE_HISTORY_1);
                fragColor.xyz = jetColormap(clamp(cache.M
                    / max(float(RADIANCE_CACHE_MAX_HIST), 1.0), 0.0, 1.0));
            #else
                RadianceCache cache = loadRadianceCachePlanes(address,
                    RC_PLANE_FILTERED_0, RC_PLANE_FILTERED_1);
                fragColor.xyz = radianceCacheValueValid(cache)
                    ? radianceCacheDiffuseIncident(cache, microN) : vec3(0.0);
            #endif
        }
    }

    #elif DEBUG_VIEW == 35
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

    #elif DEBUG_VIEW == 36
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

    #elif DEBUG_VIEW == 37
    {
        vec3 surfaceMotion;
        float motionClass;
        readSurfaceMotion(xy, surfaceMotion, motionClass);

        if (motionClass < 0.5) {
            fragColor.xyz = vec3(1.0, 0.0, 1.0);
        } else {
            vec3 relativePos;
            float distance;
            readGeo0(GEO_N_GEO, xy, relativePos, distance);
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
