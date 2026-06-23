#version 430 compatibility

// ===========================================================================
// Pass fog: 最终合成 — 雾效、天空、光照组合 (Final Composite)
// ===========================================================================
// 这是管线末端的合成 pass，负责将各光照分量合成为最终像素颜色。
//
// 输入分量:
//   - diffuse  SH 辐照度 → project_SH_irradiance() 解码为 RGB × albedo2
//   - refract  折射颜色 → 直接加到漫反射上
//   - reflect  反射颜色 → × albedo (金属/镜面度)
//   - light    直接光照 → 直接加入
//   - emission 发光     → 直接加入
//   - absorption 大气透射率 → × 整体颜色
//
// 天空:
//   - distance < -0.5 → 使用 SampleSky() 计算大气散射颜色
//   - 同时写入 prevDiffuseIllumiantionBuffer (为下一帧时域累积做准备)
// ===========================================================================

#define DIFFUSE_BUFFER_MIN2
#define REFLECT_BUFFER_MIN2
#define REFRACT_BUFFER_MIN2

#include "/lib/buffers/frame_data.glsl"
#include "/lib/tonemap.glsl"
#include "/lib/buffers/denoise.glsl"
#include "/lib/sky_color.glsl"
#include "/lib/lighting/alice.glsl"

in vec2 texCoord;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 fragColor;

void main() {
    uint idx = getIdx(uvec2(gl_FragCoord.xy));
    bufferData data = denoiseBuffer.data[idx];

    // =========================================================================
    // 分支 1: 天空像素 (无几何体命中)
    // =========================================================================
    if (data.distance < -0.5) {
        // 天空颜色 = 大气散射 × 透射率 + 发光项
        fragColor.xyz = data.absorption * SampleSky(data.rd) + data.emission;

        // 重置漫反射历史 (避免天空像素使用上一帧地面数据)
        diffuseIllumiantionBuffer.data[idx].data_swap = init_SH();
    }
    // =========================================================================
    // 分支 2: 表面像素 — 组合所有光照分量
    // =========================================================================
    else {
        ivec2 pix = ivec2(gl_FragCoord.xy);

        // 读取各光照类型的数据
        diffuseIllumiantionData tmp   = fetchDiffuse(pix);
        vec3IllumiantionData tmp2     = fetchReflect(pix);
        vec3IllumiantionData tmp3     = fetchRefract(pix);

        // 保存当前漫反射数据到历史缓冲区 (供下一帧 100.glsl 使用)
        prevDiffuseIllumiantionBuffer.data[idx].data_swap = tmp.data_swap;
        prevDiffuseIllumiantionBuffer.data[idx].weight    = max(tmp.weight, 0.0);

        // ---- 最终颜色合成 --------------------------------------------------
        // 公式:
        //   color = absorption × [
        //       (diffuse_irradiance + refract_color) × albedo2
        //     + reflect_color × albedo
        //     + direct_light
        //   ] + emission
        //
        // albedo2: 漫反射/折射反照率 (非金属分量)
        // albedo:  镜面反射反照率 (金属/镜面分量)
        fragColor.xyz = data.absorption
                      * ((project_SH_irradiance(tmp.data_swap, diffuseIllumiantionBuffer.data[idx].normal2)
                          + tmp3.data_swap) * data.albedo2
                         + tmp2.data_swap * data.albedo
                         + data.light)
                      + data.emission;
        // // 调试输出: 直接输出各分量的线性组合，验证时域累积效果
        // fragColor.xyz = project_SH_irradiance(tmp.data_swap, diffuseIllumiantionBuffer.data[idx].normal);
        // fragColor.xyz = normalize(abs(tmp.data_swap.shY.xyz));
        // fragColor.xyz = vec3(alice_estimator_variance(tmp.data_swap.shY, tmp.weight));
    }
}
