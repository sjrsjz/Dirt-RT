# Dirt RT

[English](README.md) | 简体中文

Dirt RT 是面向 Minecraft 的实时路径追踪光影。项目基于 Vulkanite，将多次弹射光线追踪、物理材质、方向光照重建、路径引导和时空降噪组合成一套完整的实时渲染管线。

> [!IMPORTANT]
> Dirt RT 需要 [Vulkanite modified 26.2-v0.0.6-lut](https://github.com/sjrsjz/vulkanite-modified/releases/tag/26.2-v0.0.6-lut)、兼容的 NVIDIA RTX 显卡和 LabPBR 资源包。

## 画面展示

![Dirt RT 室外场景](image/README/1784211218150.png)

![Dirt RT 室内光照](image/README/1784211225727.png)

![Dirt RT 材质与反射](image/README/1784211234505.png)

## 主要特性

- 多次弹射实时路径追踪，计算直接光和间接光。
- GGX 微表面反射、能量守恒粗糙漫反射、透射、介质吸收和自发光材质。
- 使用 MaxEnt 方向光照矩进行紧凑的辐射重建与引导采样。
- 漫反射和镜面共享统一降噪架构，包括时域重投影、方差跟踪和六级 A-Trous 滤波。
- 稀疏辐射率缓存、时域 RIS 以及与路径引导的联合采样。
- 用于稳定反射和折射历史的屏幕空间重建。
- LabPBR 材质、视差遮蔽映射、湿润表面、自动曝光、辉光和显示色调映射。
- 覆盖光照、历史、方差、几何和缓存状态的内置调试视图。

## 安装

1. 安装 Minecraft 26.2 和 Fabric Loader 0.19.3。
2. 安装 [Vulkanite modified 26.2-v0.0.6-lut](https://github.com/sjrsjz/vulkanite-modified/releases/tag/26.2-v0.0.6-lut)，或使用兼容的自行构建版本。
3. 将 `Dirt RT` 目录复制到 Minecraft 的 `shaderpacks` 目录。
4. 在光影包菜单中启用 Dirt RT。
5. 使用包含 LabPBR 材质数据的资源包。

游戏内可以调整光影选项。默认值和说明位于 [`shaders/lib/settings.glsl`](shaders/lib/settings.glsl)。

## 渲染结构

```text
光栅 G-buffer
      │
      ▼
主可见性 ──► 漫反射 / 镜面 / 透射路径
      │                    │
      ├──────► 稀疏辐射率缓存与路径引导
      │                    │
      ▼                    ▼
时域重建 ──► 方差准备 ──► A-Trous 滤波
      │
      ▼
光照合成 ──► 辉光 ──► 曝光与色调映射
```

| 系统 | 主要源码位置 |
| --- | --- |
| 光线追踪入口与调度 | [`shaders/ray0.rgen`](shaders/ray0.rgen)–[`ray5.rgen`](shaders/ray5.rgen)、[`shaders/lib/rt/raytrace_rgen.glsl`](shaders/lib/rt/raytrace_rgen.glsl) |
| 路径积分与弹射策略 | [`shaders/lib/rt/raytrace/path_trace.glsl`](shaders/lib/rt/raytrace/path_trace.glsl)、[`bounces.glsl`](shaders/lib/rt/raytrace/bounces.glsl) |
| BSDF、lobe 选择与折射 | [`bsdf.glsl`](shaders/lib/rt/raytrace/bsdf.glsl)、[`lobe_selection.glsl`](shaders/lib/rt/raytrace/lobe_selection.glsl)、[`refraction.glsl`](shaders/lib/rt/raytrace/refraction.glsl) |
| GGX、Fresnel 与材质模型 | [`shaders/lib/pbr/`](shaders/lib/pbr/)、[`shaders/lib/lighting/eon.glsl`](shaders/lib/lighting/eon.glsl) |
| 路径引导 | [`shaders/lib/rt/raytrace/guiding.glsl`](shaders/lib/rt/raytrace/guiding.glsl) |
| MaxEnt 光照表示 | [`shaders/lib/lighting/maxent.glsl`](shaders/lib/lighting/maxent.glsl)、[`maxent_encode.glsl`](shaders/lib/lighting/maxent_encode.glsl) |
| 镜面方向重建 | [`shaders/lib/lighting/specular_maxent.glsl`](shaders/lib/lighting/specular_maxent.glsl)、[`shaders/lib/lighting/specular_cdf/`](shaders/lib/lighting/specular_cdf/) |
| 统一降噪器核心 | [`shaders/lib/lighting/denoiser/`](shaders/lib/lighting/denoiser/) |
| 漫反射与反射降噪 pass | [`shaders/post/denoiser/`](shaders/post/denoiser/) |
| 稀疏辐射率缓存 | [`shaders/lib/buffers/radiance_cache/`](shaders/lib/buffers/radiance_cache/)、[`shaders/post/temporal_radiance_cache.glsl`](shaders/post/temporal_radiance_cache.glsl) |
| 缓冲布局与 GPU 数据接口 | [`shaders/lib/buffers/`](shaders/lib/buffers/)、[`shaders/lib/rt/payload_pack.glsl`](shaders/lib/rt/payload_pack.glsl) |
| 光照合成与折射解析 | [`shaders/post/composite_lighting.glsl`](shaders/post/composite_lighting.glsl)、[`resolve_refraction.glsl`](shaders/post/resolve_refraction.glsl) |
| 辉光、曝光与色调映射 | [`shaders/lib/post_processing/`](shaders/lib/post_processing/)、[`shaders/post/auto_exposure.glsl`](shaders/post/auto_exposure.glsl) |
| 光影设置与资源绑定 | [`shaders/lib/settings.glsl`](shaders/lib/settings.glsl)、[`shaders/shaders.properties`](shaders/shaders.properties) |

## 许可证

Dirt RT 使用 [GNU General Public License v3.0](LICENSE)。项目内单独标注来源或许可证的资源继续遵循各自声明。
