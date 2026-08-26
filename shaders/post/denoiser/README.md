# MaxEnt denoiser pipeline

The encoder stores linear moment state `(E[R u], E[R], sqrt(E[R²]), N_eff)`. Spatial passes operate only in that moment space. MaxEnt decoding belongs to lighting composition and is not an assumption of the denoiser.

## Scheduling and transient images

| Composite | Domain | Operation | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | Diffuse | Temporal reprojection/proposal | Raw and previous histories | swap state, filtered reprojection, `c6` raw reprojection |
| 50 | Diffuse | Variance preparation | swap proposal, Raw RT, geometry | `c3` geometry, `c4` signal, scratch A |
| 51–56 | Diffuse | A-Trous steps 1–32 | `c3`, `c4/c5`, scratch A/B | opposite `c4/c5`, scratch A/B; final signal is `c4` |
| 57 | Diffuse | ReSTIR path guide | final `c4` | `c5` path-guide scratch |
| 58 | Diffuse | Resolve | `c4`, `c5`, `c6`, scratch A, reprojection scratch | all accepted diffuse histories |
| 59 | Reflection | Raw input adapter | RT reflection and geometry | `c6` Raw reflection |
| 61 | Reflection | Temporal reprojection/proposal | `c6`, previous histories | `c4` proposal, `c5` raw reprojection, filtered reprojection scratch |
| 63 | Reflection | History staging | `c5` and reprojection scratch | reflection N1/N2 staging |
| 65 | Reflection | Variance preparation | `c4`, `c6`, geometry | `c3` geometry, `c5` signal, scratch A |
| 66–71 | Reflection | A-Trous steps 1–32 | `c3`, `c4/c5`, scratch A/B | opposite `c4/c5`, scratch A/B; final signal is `c5` |
| 72 | Reflection | Resolve | `c5`, `c6`, scratch A, staged histories | all accepted reflection histories and public reflection output |

`c3` and `c6` are reused only after the diffuse domain has completed. Scratch A/B are two transient planes in the diffuse SSBO; diffuse and reflection execute serially. No pass reads a neighborhood from the same image it writes.

## Ownership and statistical invariants

- Diffuse resolve (58) is the only accepted diffuse-history commit point; temporal swap data is provisional.
- Reflection resolve (72) is the only accepted reflection-history commit point. Composite 63 temporarily stages raw reprojection in N1/N2, and composite 71 has no hidden history publication.
- Raw RT updates raw moments, `sqrt(E[R²])`, and Kish `N_eff`; the visible result always comes from the spatially filtered branch.
- `currentAlpha = 1` publishes the independently spatially filtered current result and updates raw Kish `N_eff` to 1.
- `estimatorStdDev` always means the standard deviation of the estimator in the four-moment metric. Packing never performs a hidden square root.
- `rootMeanY2` always means `sqrt(E[R²])`.
- `E[R u]`, `E[R]`, `CoCg`, and `E[R²]` are linear encoder/latent moments. Every temporal, reprojection, branch, and spatial moment combination is a normalized weighted sum; only Kish `N_eff` has a nonlinear update.
- Estimator variance is derived read-only from `(E[R u], E[R], E[R²], N_eff)`. It must never be inverted to reconstruct `E[R²]`, and moment-cone or RGB-feasibility projection is confined to decoding rather than history or latent filtering.
- Temporal reprojection reconstructs moments and Kish effective samples; it never linearly interpolates `N_eff`.
- Diffuse reprojection pulls a normalized tent kernel through the local current-to-previous tangent-plane Jacobian. The same accepted weights reconstruct moments, filtered variance, and Kish `N_eff`; a Jacobian determinant never scales moment amplitudes.
- Reflection hit distance is a per-frame virtual-motion tracking guide. It is never mixed with the lighting alpha.
- Exact delta mirrors accept only virtual-motion history; failed virtual reprojection resets history instead of falling back to surface motion.
- The tuned temporal response in `temporal_response.glsl` is part of the current behavior and must not be silently changed.
- Fixed estimator, provisional-pipeline and kernel-correlation constants live in `internal_constants.glsl`; they are code invariants, not shader-pack options.

The first refactor intentionally keeps workgroup sizes, image formats, SSBO sizes, and A-Trous pass count unchanged.
