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
    uint idx = getIndex(uvec2(gl_FragCoord.xy));
    bufferData data = denoiseBuffer.data[idx];

    // =========================================================================
    // 分支 1: 天空像素 (无几何体命中)
    // =========================================================================
    if (data.distance < -0.5) {
        setSkyVars(); // 初始化天空散射参数 (原由 init_sky 预计算, 现改为分析式)
        // 天空颜色 = 大气散射 × 透射率 + 发光项
        fragColor.xyz = data.absorption * sampleSky(camPos.y, data.rd, -lightDir_global) + data.emission;

        // 重置漫反射历史 (避免天空像素使用上一帧地面数据)
        diffuseIlluminationBuffer.data[idx].rt_aliceY_xy = 0.0;
        diffuseIlluminationBuffer.data[idx].rt_aliceY_zw = 0.0;
        diffuseIlluminationBuffer.data[idx].rt_CoCg = 0.0;
    }
    // =========================================================================
    // 分支 2: 表面像素 — 组合所有光照分量
    // =========================================================================
    else {
        ivec2 pix = ivec2(gl_FragCoord.xy);

        // 读取各光照类型的数据
        diffuseIlluminationData tmp = fetchDiffuse(pix);
        vec3IlluminationData tmp2 = fetchReflect(pix);
        vec3IlluminationData tmp3 = fetchRefract(pix);

        // 保存当前漫反射数据到历史缓冲区 (供下一帧 temporal_diffuse.glsl 使用)
        // Temporal history now lives in unified diffuseIlluminationBuffer (binding 2).
        // swap3 already wrote the final filtered AliceEncoding + weight to the swap fields;
        // ray0.rgen reads from there via samplePrevDiffuse. No additional write needed.

        // ---- Debug view selector ----
        // Full disentanglement: each denoised buffer holds a material-independent incident
        // light field (divided by stable per-lobe albedo in rgen using normal, not microNormal).
        // Composite multiplies each back by its stable material multiplier from denoiseBuffer.
        // Stable divisor = no temporal noise amplification (unlike microNormal-based division).
        #if DEBUG_VIEW == 0
        // Normal: full composition
        //   color = absorption × [
        //       diffuse_irradiance × diffuseAlbedo
        //     + refract_incident   × transmissionAlbedo
        //     + reflect_incident   × specularAlbedo
        //     + direct_light
        //   ] + emission
        fragColor.xyz = data.absorption
                * (project_alice_irradiance(tmp.data_swap, decodeNormal(diffuseIlluminationBuffer.data[idx].oct_n2))
                    * data.diffuseAlbedo
                    + tmp3.data_swap * data.transmissionAlbedo
                    + tmp2.data_swap * data.specularAlbedo
                    + data.light)
                + data.emission;

        #elif DEBUG_VIEW == 1
        // Diffuse only: irradiance × diffuseAlbedo
        fragColor.xyz = project_alice_irradiance(tmp.data_swap, decodeNormal(diffuseIlluminationBuffer.data[idx].oct_n2)) * data.diffuseAlbedo;

        #elif DEBUG_VIEW == 2
        // Refract only
        fragColor.xyz = tmp3.data_swap * data.transmissionAlbedo;

        #elif DEBUG_VIEW == 3
        // Reflect only
        fragColor.xyz = tmp2.data_swap * data.specularAlbedo;

        #elif DEBUG_VIEW == 4
        // White model: diffuse irradiance only, no albedo
        fragColor.xyz = project_alice_irradiance(tmp.data_swap, decodeNormal(diffuseIlluminationBuffer.data[idx].oct_n2));

        #elif DEBUG_VIEW == 5
        // Light field: ALICE normalized dominant direction × energy
        fragColor.xyz = 2.0 * abs(tmp.data_swap.aliceY.xyz / max(max(tmp.data_swap.aliceY.w, length(tmp.data_swap.aliceY.xyz)), 1e-6));

        #elif DEBUG_VIEW == 6
        // Normals: world-space normal as RGB
        vec3 dbg_n = decodeNormal(diffuseIlluminationBuffer.data[idx].oct_n2);
        fragColor.xyz = dbg_n * 0.5 + 0.5;

        #elif DEBUG_VIEW == 7
        // Absorption / atmospheric transmission
        fragColor.xyz = data.absorption;

        #elif DEBUG_VIEW == 8
        // Reflection dominant direction R as RGB (oct-decoded)
        {
            SpecularRTElement re = reflectIlluminationBuffer.data[getIndex(uvec2(gl_FragCoord.xy))];
            vec3 R = decodeNormal(re.oct_dir);
            fragColor.xyz = R * 0.5 + 0.5;
        }

        #elif DEBUG_VIEW == 9
        // Reflection virtual projection distance (rainbow colormap, log scale)
        //   blue → cyan → green → yellow → red → white(sky)
        //   near 0              20              60      VPROJDIST_SKY
        {
            SpecularRTElement re = reflectIlluminationBuffer.data[getIndex(uvec2(gl_FragCoord.xy))];
            float d = re.virtualProjDist;
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
        fragColor.xyz = data.specularAlbedo;

        #elif DEBUG_VIEW == 11
        // Roughness as grayscale
        fragColor.xyz = vec3(data.roughness);

        #elif DEBUG_VIEW == 12
        // Raw reflection incident (before specularAlbedo modulation)
        // The denoised signal before composite multiplications — pure light field
        fragColor.xyz = tmp2.data_swap;

        #endif
    }
}
