# Tony McMapface

## What Tony McMapface is

**Tony McMapface** is a tone-mapping **LUT** (look-up table) by Tomasz Stachowiak
(GitHub: h3r2tic). The upstream project provides the LUT itself — it is **not** a neural
network.

- **Upstream repository**: https://github.com/h3r2tic/tony-mc-mapface
- **Author**: Tomasz Stachowiak (h3r2tic)
- **License**: dual Apache 2.0 / MIT — see `LICENSE-APACHE` and `LICENSE-MIT`

This folder retains the upstream license files for attribution. No LUT data file from
upstream is checked in here.

## How Dirt RT uses it

Dirt RT does **not** ship the LUT directly. Instead, the project designed its **own
neural network** to fit / approximate the Tony McMapface LUT's tone curve, so the
tonemapper can run in real time on the GPU.

The neural-network implementation lives in
[`shaders/lib/post_processing/tonemap.glsl`](../../shaders/lib/post_processing/tonemap.glsl)
and is invoked from the final composite pass
[`shaders/composite98.fsh`](../../shaders/composite98.fsh).

### Boundary: upstream vs. our work

| Aspect | Source |
|---|---|
| Tone-mapping LUT (the reproduced target) | Tony McMapface (upstream) |
| Neural network architecture + weights (`TonyMcMapface_Tiny`, MLP 4→32→3) | Dirt RT (this project's own design), fit to the LUT |
| `linear_to_srgb`, `apply_shadow_toe`, `ACESFilm` helpers | Dirt RT (this project) |
| `LICENSE-APACHE`, `LICENSE-MIT` | Tony McMapface (upstream), retained for attribution |

### The neural network (our work)

- **Type**: Multi-Layer Perceptron (MLP)
- **Architecture**: 4 → 32 → 3
- **Parameters**: 259
- **Activations**: SiLU (hidden) + Softplus (output)
- **Inputs**: normalized HDR color + Reinhard luminance approximation
- **Purpose**: approximate the Tony McMapface LUT's tone curve at real-time GPU cost

All weights and biases are baked in as compile-time `const` values — no runtime allocation,
minimal branching.

### Usage

```glsl
#include "/lib/post_processing/tonemap.glsl"

vec3 hdr        = texture(colortex0, texCoord).rgb;
vec3 bloom      = texture(colortex1, texCoord).rgb;
vec3 combined   = mix(hdr, bloom, BLOOM_MIX);

// Our NN approximating the Tony McMapface LUT, plus a shadow-toe for darks
vec3 tonemapped = TonyMcMapface_Tiny(apply_shadow_toe(combined, EXPOSURE_CURVE_K));
fragColor = vec4(linear_to_srgb(tonemapped), 1.0);
```

Function signature:

```glsl
vec3 TonyMcMapface_Tiny(vec3 hdrColor)
```

## Why the upstream license is retained

The LUT's tonemapping curve is being reproduced (via approximation) by this project's own
network. Both the Apache 2.0 and MIT licenses require retaining the copyright notice and
license text, so they are kept in this folder for attribution. The neural network code
itself is this project's own work and is not derived from upstream source code.
