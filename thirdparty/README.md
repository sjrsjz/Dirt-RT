# Third Party Libraries

This directory holds the licenses and attribution for third-party material referenced by
the Dirt RT shader pack. Each subdirectory contains the original license file(s) and a
README describing how the material is used.

> Note: the licensed material here is **referenced / reproduced** by this project. Where
> this project has written its own reimplementation, that reimplementation is the project's
> own code (see each README for the exact boundary between upstream material and our work).

## Overview

### 1. Tony McMapface (tonemapping LUT)

- **Location**: `./TonyMcMapface/`
- **Upstream**: https://github.com/h3r2tic/tony-mc-mapface
- **Author**: Tomasz Stachowiak (h3r2tic)
- **License**: Dual-licensed under Apache 2.0 and MIT (see individual files)
- **What upstream provides**: A tone-mapping **LUT** (look-up table). It is *not* a neural
  network.
- **How Dirt RT uses it**: Dirt RT ships its **own specially designed neural network**
  (an MLP, 4 → 32 → 3, in `shaders/lib/post_processing/tonemap.glsl`) that is fit to
  approximate the Tony McMapface LUT's tone curve. The LUT's behavior is reproduced; the
  network itself is this project's work. The upstream license is retained here for
  attribution because the LUT's tonemapping curve is being approximated.

### 2. VulkaniteDemoPack

- **Location**: `./VulkaniteDemoPack/`
- **Upstream**: https://github.com/BalintCsala/VulkaniteDemoPack
- **Author**: Bálint Csala
- **License**: MIT License
- **How Dirt RT uses it**: Only the license is currently retained; no code from this
  project is used in the current implementation.

## Legal Notice

This project is not affiliated with or endorsed by the original authors. For issues
related to the upstream material, refer to the original repositories.
