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
- `currentAlpha = 1` publishes the final independently A-Trous-filtered center result and updates raw Kish `N_eff` to 1.
- `standardDeviation` is the local MC observation deviation at every A-Trous boundary. Each spatial pass linearly filters its variance with the signal weights; it neither propagates weighted-mean estimator uncertainty nor assumes that the supplied MC variance shares an empirical sample set or `N_eff` with the signal moments. Packing never performs a hidden square root.
- A-Trous light rejection uses `(V_MC,center+V_MC,sample)/N_eff,center`. The neighbor contributes its MC variance but never its `N_eff`; center `N_eff` remains the sole owner of the center pixel's convergence strength.
- Each A-Trous step also applies a fixed cumulative spatial-confidence scale derived from the preceding fixed kernels. This restores progressive late-pass rejection without converting the stored MC variance into estimator variance or folding difference covariance into the scale.
- Short-history variance preparation linearly reconstructs neighboring temporal `E[R u]`, `E[R]` and `E[R²]` with the geometry kernel, while Kish `N_eff` is reconstructed separately as `W²/sum(w_i²/N_i)`. The reconstructed `N_eff` removes finite-sample bias and the output remains MC observation variance. Raw RT supplies only the independent-current signal; it is not the spatial variance population.
- Temporal response is the only point that converts filtered MC variance into estimator variance: `V_est,H=V_MC,H/N_eff,H` and `V_est,C=V_MC,C/N_eff,C`. The current moment, CoCg, `V_MC,C`, and `N_eff,C` are read from the same final A-Trous center record without a second neighborhood reconstruction.
- Filtered history is updated directly as `H_filtered <- mix(H_filtered, M3x3(C), currentAlpha)`. Raw history independently mixes reprojected raw moments with Raw RT using the same alpha, so spatial filtering never contaminates `E[R²]` or raw Kish `N_eff`.
- Spatial geometry stores center Kish `N_eff` instead of material identity. Spatial light-difference rejection multiplies its configured coefficient by `sqrt(N_eff)`; material boundaries are not a denoiser rejection condition.
- `rootMeanY2` always means `sqrt(E[R²])`.
- `E[R u]`, `E[R]`, `CoCg`, and `E[R²]` are linear encoder/latent moments. Every temporal, reprojection, branch, and spatial moment combination is a normalized weighted sum; only Kish `N_eff` has a nonlinear update.
- MC observation variance is derived read-only from `(E[R u], E[R], E[R²], N_eff)`. It must never be inverted to reconstruct `E[R²]`, and moment-cone or RGB-feasibility projection is confined to decoding rather than history or latent filtering.
- Temporal reprojection reconstructs moments and Kish effective samples; it never linearly interpolates `N_eff`.
- Diffuse reprojection uses the hand-tuned thick tangent-plane footprint for geometry acceptance and ordinary normalized bilinear weights for history fields. Its per-tap projected-area Jacobian `d_cur²|NoV_hist|/(d_prev²|NoV_center|)` scales only that tap's `N_eff` before Kish reconstruction; it never scales `E[R u]`, `E[R]`, `CoCg`, `E[R²]`, or filtered MC variance.
- Reflection hit distance is a per-frame virtual-motion tracking guide. It is never mixed with the lighting alpha.
- Exact delta mirrors accept only virtual-motion history; failed virtual reprojection resets history instead of falling back to surface motion.
- The tuned temporal response in `temporal_response.glsl` is part of the current behavior and must not be silently changed.
- Fixed estimator and provisional-pipeline constants live in `internal_constants.glsl`; they are code invariants, not shader-pack options.

This correction keeps workgroup sizes, image formats, and the A-Trous pass count unchanged. The diffuse SSBO grows by one RGBA32UI plane for spatial-filter N_eff ping-pong and diffuse filtered-history N_eff reprojection.

## TODO

- Calibrate variance preparation. Spatial reconstruction of temporal moments still assumes local stationarity and can confuse spatial signal variation with Monte Carlo variance.
