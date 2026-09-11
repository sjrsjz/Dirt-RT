# Dirt RT

English | [简体中文](README.zh-CN.md)

Dirt RT is a real-time path-tracing shader pack for Minecraft. Built for Vulkanite, it combines multi-bounce ray tracing, physically based materials, directional light reconstruction, path guiding, and spatiotemporal denoising in a complete real-time rendering pipeline.

> [!IMPORTANT]
> Dirt RT requires [Vulkanite modified 26.2-v0.0.6-lut](https://github.com/sjrsjz/vulkanite-modified/releases/tag/26.2-v0.0.6-lut), a compatible NVIDIA RTX GPU, and a LabPBR resource pack.

## Gallery

![Dirt RT outdoor scene](image/README/1784211218150.png)

![Dirt RT interior lighting](image/README/1784211225727.png)

![Dirt RT materials and reflections](image/README/1784211234505.png)

## Features

- Multi-bounce real-time path tracing for direct and indirect illumination.
- GGX microfacet reflection, energy-preserving rough diffuse shading, transmission, volumetric absorption, and emissive materials.
- MaxEnt directional light moments for compact radiance reconstruction and guided sampling.
- Unified diffuse and specular denoising with temporal reprojection, variance tracking, and six-stage A-Trous filtering.
- Sparse radiance cache with temporal RIS and path-guiding integration.
- Screen-space reconstruction for stable reflection and refraction history.
- LabPBR materials, parallax occlusion mapping, wet surfaces, automatic exposure, bloom, and display tone mapping.
- Built-in diagnostic views for lighting, history, variance, geometry, and cache state.

## Installation

1. Install Minecraft 26.2 with Fabric Loader 0.19.3.
2. Install [Vulkanite modified 26.2-v0.0.6-lut](https://github.com/sjrsjz/vulkanite-modified/releases/tag/26.2-v0.0.6-lut), or a compatible source build.
3. Copy the `Dirt RT` directory into Minecraft's `shaderpacks` directory.
4. Enable Dirt RT in the shader-pack menu.
5. Use a resource pack containing LabPBR material data.

Shader options are available in game. Their defaults and descriptions live in [`shaders/lib/settings.glsl`](shaders/lib/settings.glsl).

## Rendering architecture

```text
Raster G-buffer
      │
      ▼
Primary visibility ──► diffuse / specular / transmission paths
      │                              │
      ├────────► sparse radiance cache and path guiding
      │                              │
      ▼                              ▼
Temporal reconstruction ──► variance preparation ──► A-Trous filtering
      │
      ▼
Lighting composition ──► bloom ──► exposure and tone mapping
```

| System | Main source locations |
| --- | --- |
| Ray-tracing entry points and scheduling | [`shaders/ray0.rgen`](shaders/ray0.rgen)–[`ray5.rgen`](shaders/ray5.rgen), [`shaders/lib/rt/raytrace_rgen.glsl`](shaders/lib/rt/raytrace_rgen.glsl) |
| Path integration and bounce policy | [`shaders/lib/rt/raytrace/path_trace.glsl`](shaders/lib/rt/raytrace/path_trace.glsl), [`bounces.glsl`](shaders/lib/rt/raytrace/bounces.glsl) |
| BSDF, lobe selection, and refraction | [`bsdf.glsl`](shaders/lib/rt/raytrace/bsdf.glsl), [`lobe_selection.glsl`](shaders/lib/rt/raytrace/lobe_selection.glsl), [`refraction.glsl`](shaders/lib/rt/raytrace/refraction.glsl) |
| GGX, Fresnel, and material models | [`shaders/lib/pbr/`](shaders/lib/pbr/), [`shaders/lib/lighting/eon.glsl`](shaders/lib/lighting/eon.glsl) |
| Path guiding | [`shaders/lib/rt/raytrace/guiding.glsl`](shaders/lib/rt/raytrace/guiding.glsl) |
| MaxEnt light representation | [`shaders/lib/lighting/maxent.glsl`](shaders/lib/lighting/maxent.glsl), [`maxent_encode.glsl`](shaders/lib/lighting/maxent_encode.glsl) |
| Specular directional reconstruction | [`shaders/lib/lighting/specular_maxent.glsl`](shaders/lib/lighting/specular_maxent.glsl), [`shaders/lib/lighting/specular_cdf/`](shaders/lib/lighting/specular_cdf/) |
| Unified denoiser core | [`shaders/lib/lighting/denoiser/`](shaders/lib/lighting/denoiser/) |
| Diffuse and reflection denoiser passes | [`shaders/post/denoiser/`](shaders/post/denoiser/) |
| Sparse radiance cache | [`shaders/lib/buffers/radiance_cache/`](shaders/lib/buffers/radiance_cache/), [`shaders/post/temporal_radiance_cache.glsl`](shaders/post/temporal_radiance_cache.glsl) |
| Buffer layouts and GPU data interfaces | [`shaders/lib/buffers/`](shaders/lib/buffers/), [`shaders/lib/rt/payload_pack.glsl`](shaders/lib/rt/payload_pack.glsl) |
| Lighting composition and refraction resolve | [`shaders/post/composite_lighting.glsl`](shaders/post/composite_lighting.glsl), [`resolve_refraction.glsl`](shaders/post/resolve_refraction.glsl) |
| Bloom, exposure, and tone mapping | [`shaders/lib/post_processing/`](shaders/lib/post_processing/), [`shaders/post/auto_exposure.glsl`](shaders/post/auto_exposure.glsl) |
| Shader settings and resource bindings | [`shaders/lib/settings.glsl`](shaders/lib/settings.glsl), [`shaders/shaders.properties`](shaders/shaders.properties) |

## License

Dirt RT is released under the [GNU General Public License v3.0](LICENSE). Assets carrying their own source or license notice remain subject to that notice.
