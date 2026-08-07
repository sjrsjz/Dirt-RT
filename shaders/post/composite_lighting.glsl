#version 430 compatibility

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
        fragColor.xyz = absorptionVal * sampleSky(camPos.y, rdVal, -lightDir_global) + emisVal;
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
    vec3 microN = readMicroNormal(GEO_N_MICRONORMAL, xy);
    readAlbedosPath(GEO_N_ALBEDOS, xy, specAlbedo, diffAlbedo);
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
    // Sampled reflection ray hit distance (rainbow colormap, log scale)
    {
        vec3 dummyColor;
        float d, dummyW;
        readReflLight(xy, dummyColor, d, dummyW);
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
    // Reflect temporal accumulation weight (heatmap)
    fragColor.xyz = jetColormap(clamp(tmp2.weight / RELAX_SPEC_MAX_HISTORY, 0.0, 1.0));

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

    #endif
}
