#version 430 compatibility

// ===========================================================================
// Pass fog: 最终合成 — 雾效、天空、光照组合 (Final Composite)
// ===========================================================================
// 这是管线末端的合成 pass，负责将各光照分量合成为最终像素颜色。
//
// 输入分量:
//   - diffuse  AliceEncoding 辐照度 → project_alice_irradiance() 解码为 RGB × albedo2
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
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky.glsl"
#include "/lib/lighting/alice.glsl"

in vec2 texCoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    uvec2 xy = uvec2(gl_FragCoord.xy);
    ivec2 pix = ivec2(gl_FragCoord.xy);

    // 只读 dist 判定天空（pos 顺带读出不浪费）
    vec3 worldPos; float dist;
    readGeo0(GEO_N_GEO, xy, worldPos, dist);

    // =========================================================================
    // 分支 1: 天空像素 — 只读 N=3 + N=4（跳过 N=1,N=2 省 3 次 vec4 读）
    // =========================================================================
    if (dist < -0.5) {
        vec3 emisVal, rdVal, absorptionVal; // ignore transAlbedo, lightVal
        vec3 transAlbedo_unused, lightVal_unused;
        readMisc(GEO_N_MISC, xy, transAlbedo_unused, emisVal, rdVal);
        readLightAbs(GEO_N_LIGHTABS, xy, lightVal_unused, absorptionVal);

        setSkyVars();
        fragColor.xyz = absorptionVal * sampleSky(camPos.y, rdVal, -lightDir_global) + emisVal;
        writeDiffuseLightRTSky(xy);
        return;
    }

    // =========================================================================
    // 分支 2: 表面像素 — 读剩余 G-Buffer + 所有光照
    // =========================================================================
    vec3 macroN, specAlbedo, diffAlbedo, transAlbedo, emisVal, lightVal, absorptionVal, rdVal;
    float rough, pathR; int illumType;
    readGeo1(GEO_N_NORMALS, xy, macroN, rough, illumType, pathR);
    readAlbedosPath(GEO_N_ALBEDOS, xy, specAlbedo, diffAlbedo);
    readMisc(GEO_N_MISC, xy, transAlbedo, emisVal, rdVal);
    readLightAbs(GEO_N_LIGHTABS, xy, lightVal, absorptionVal);

    diffuseIlluminationData tmp = fetchDiffuse(pix);
    vec3IlluminationData tmp2 = fetchReflect(pix);
    vec3IlluminationData tmp3 = fetchRefract(pix);

    AliceEncoding dummy; vec3 n2;
    readDiffuseLightRT(xy, dummy, n2);

    #if DEBUG_VIEW == 0
    fragColor.xyz = absorptionVal
            * (project_alice_irradiance(tmp.data_swap, n2)
                * diffAlbedo
                + tmp3.data_swap * transAlbedo
                + tmp2.data_swap * specAlbedo
                + lightVal)
            + emisVal;

        #elif DEBUG_VIEW == 1
        // Diffuse only: irradiance × diffuseAlbedo
        fragColor.xyz = project_alice_irradiance(tmp.data_swap, n2) * diffAlbedo;

        #elif DEBUG_VIEW == 2
        // Refract only
        fragColor.xyz = tmp3.data_swap * transAlbedo;

        #elif DEBUG_VIEW == 3
        // Reflect only
        fragColor.xyz = tmp2.data_swap * specAlbedo;

        #elif DEBUG_VIEW == 4
        // White model: diffuse irradiance only, no albedo
        fragColor.xyz = project_alice_irradiance(tmp.data_swap, n2);

        #elif DEBUG_VIEW == 5
        // Light field: ALICE normalized dominant direction × energy
        fragColor.xyz = 2.0 * abs(tmp.data_swap.aliceY.xyz / max(max(tmp.data_swap.aliceY.w, length(tmp.data_swap.aliceY.xyz)), 1e-6));

        #elif DEBUG_VIEW == 6
        // Normals: world-space normal as RGB
        vec3 dbg_n = n2;
        fragColor.xyz = dbg_n * 0.5 + 0.5;

        #elif DEBUG_VIEW == 7
        // Absorption / atmospheric transmission
        fragColor.xyz = absorptionVal;

        #elif DEBUG_VIEW == 8
        // Reflection dominant direction R as RGB (oct-decoded)
        {
            vec3 Rpos, R;
            readReflGeo(xy, Rpos, R);
            fragColor.xyz = R * 0.5 + 0.5;
        }

        #elif DEBUG_VIEW == 9
        // Reflection virtual projection distance (rainbow colormap, log scale)
        {
            vec3 dummyColor; float d, dummyW;
            readReflLight(xy, dummyColor, d, dummyW);
            if (d >= VPROJDIST_SKY * 0.99) {
                fragColor.xyz = vec3(1.0, 1.0, 1.0);  // sky / no hit → white
            } else {
                // log-scale normalize: 0.01 .. 100m → [0, 1]
                float t = clamp(log2(max(d, 0.01) * 100.0 + 1.0) / 14.0, 0.0, 1.0);
                // Jet / rainbow colormap
                float r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
                float g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
                float b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
                fragColor.xyz = vec3(r, g, b);
            }
        }

        #elif DEBUG_VIEW == 10
        // Specular albedo (rC.rgb * S.x) — the reflection material multiplier
        fragColor.xyz = specAlbedo;

        #elif DEBUG_VIEW == 11
        // Roughness as grayscale
        fragColor.xyz = vec3(rough);

        #elif DEBUG_VIEW == 12
        // Raw reflection incident (before specularAlbedo modulation)
        // The denoised signal before composite multiplications — pure light field
        fragColor.xyz = tmp2.data_swap;

        #elif DEBUG_VIEW == 13
        // Diffuse temporal accumulation weight (heatmap: blue→cyan→green→yellow→red)
        // N_eff / ACCUMULATION_LENGTH — cold = few frames, hot = fully converged
        {
            float t = clamp(tmp.weight / ACCUMULATION_LENGTH, 0.0, 1.0);
            fragColor.xyz = vec3(
                clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0),
                clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0),
                clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0));
        }

        #elif DEBUG_VIEW == 14
        // Reflect temporal accumulation weight (heatmap)
        {
            float t = clamp(tmp2.weight / ACCUMULATION_LENGTH, 0.0, 1.0);
            fragColor.xyz = vec3(
                clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0),
                clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0),
                clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0));
        }

        #elif DEBUG_VIEW == 15
        // Refract temporal accumulation weight (heatmap)
        {
            float t = clamp(tmp3.weight / ACCUMULATION_LENGTH, 0.0, 1.0);
            fragColor.xyz = vec3(
                clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0),
                clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0),
                clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0));
        }

        #elif DEBUG_VIEW == 16
        // Direct light only — raw direct illumination component
        fragColor.xyz = lightVal;

        #elif DEBUG_VIEW == 17
        // Emission only — self-illuminating surfaces (glowstone, lava, etc.)
        fragColor.xyz = emisVal;

        #elif DEBUG_VIEW == 18
        // Diffuse albedo — per-pixel diffuse material multiplier
        fragColor.xyz = diffAlbedo;

        #elif DEBUG_VIEW == 19
        // Refraction virtual projection distance (IOR-adjusted, rainbow colormap, log scale)
        {
            vec3 dummyColor; float d, dummyW;
            readRefrLight(xy, dummyColor, d, dummyW);
            if (d >= VPROJDIST_SKY * 0.99) {
                fragColor.xyz = vec3(1.0, 1.0, 1.0);  // sky / no hit → white
            } else {
                float t = clamp(log2(max(d, 0.01) * 100.0 + 1.0) / 14.0, 0.0, 1.0);
                float r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
                float g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
                float b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
                fragColor.xyz = vec3(r, g, b);
            }
        }

        #elif DEBUG_VIEW == 20
        // Path guide ALICE direction as RGB (蓄水池+降噪投票结果, N=5)
        {
            vec4 guideY; float guideEnergy;
            readPathGuide(xy, guideY, guideEnergy);
            if (guideEnergy < 1e-6) {
                fragColor.xyz = vec3(0.0); // 无效/天空 → 黑
            } else {
                vec3 dir = guideY.xyz / max(length(guideY.xyz), 1e-6);
                fragColor.xyz = dir * 0.5 + 0.5; // 方向→RGB
            }
        }

        #endif
}
