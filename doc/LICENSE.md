# Documentation License

This directory contains the technical documentation for the Dirt RT shader pack
(`tech.typ` / `tech_en.typ` and the generated `tech.pdf` / `tech_en.pdf`).

## Document body — CC BY 4.0

The **document body** — i.e. all textual content, figures, and theoretical
exposition — is licensed under

> **Creative Commons Attribution 4.0 International (CC BY 4.0)**  
> https://creativecommons.org/licenses/by/4.0/  
> Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode

Under CC BY 4.0 you are free to:

- **Share** — copy and redistribute the document in any medium or format
- **Adapt** — remix, transform, and build upon it for any purpose, including
  commercial use

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the
  license, and indicate if changes were made. You may do so in any reasonable
  manner, but not in a way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological
  measures that legally restrict others from doing anything the license permits.

### Required attribution

When redistributing or adapting this document, please retain:

- **Author**: sjrsjz — https://github.com/sjrsjz / sjrsjz@gmail.com
- **Title**: *Asymmetric Laplace Isomorphic Conic Encoding — Technical Document*
- **Source**: https://github.com/sjrsjz (project repository)
- **License**: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- **Modifications**: indicate clearly if the document was modified

## Accompanying code — MIT

The following code is released under the **MIT License**
(https://opensource.org/license/mit):

- **Code listings embedded in this documentation** (GLSL / HLSL shader
  implementation snippets shown in `tech.typ` / `tech_en.typ`).
- **The core ALICE implementation** at
  [`shaders/lib/lighting/alice.glsl`](../shaders/lib/lighting/alice.glsl), which
  carries its own per-file MIT license header (`Copyright (c) 2026 sjrsjz`).

The remainder of the shader pack's source code at the repository root is governed
by its own top-level [`LICENSE`](../LICENSE) file (GPLv3). Where a file carries
an explicit per-file license header (such as `alice.glsl` above), that per-file
license takes precedence for the file in question.

## Figures

All figures under `assets/` are renderings produced by the author's own shader
implementation and are part of the document body, hence licensed under CC BY 4.0.

## References

Academic citations (`references.bib`) are bibliographic references to third-party
works and are not part of the licensed document body. Cited algorithms and ideas
(SVGF, NRD, ReSTIR, MIS, maximum-entropy / Maxwell–Jüttner distribution,
information geometry, etc.) are referenced for attribution only; no verbatim text
or figures from those works are reproduced herein.

## No warranty

The document and the MIT-licensed code are provided "as is", without warranty of
any kind. See the CC BY 4.0 license and the MIT License for their respective
full disclaimers.
