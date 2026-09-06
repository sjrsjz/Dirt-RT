# MaxEnt denoiser pipeline

The encoder stores linear moment state `(E[R u], E[R], sqrt(E[R²]), N_eff)`. The filtered mean remains linear. Light differences use the Bures--Wasserstein geometry of its `2x2` PSD embedding, and MC variance uses the alpha=1 `g^-3` family only to close the missing `R²`-weighted angular moments. Lighting decoding remains a separate operation.

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

Independent-current Kish propagation uses plane-calibrated effective
correlations (0, 0.09184833, 0.12613998, 0.13781854, 0.14262104, 0.14655028).
Calibration retains runtime Bures signal weights and measures their frozen
operator overlap using independent error probes. A separate repeated-MC audit
measures the full adaptive filter's centered variance. These constants depend
on the calibrated rejection and variance-preparation settings.

- Diffuse resolve (58) is the only accepted diffuse-history commit point; temporal swap data is provisional.
- Reflection resolve (72) is the only accepted reflection-history commit point. Composite 63 temporarily stages raw reprojection in N1/N2, and composite 71 has no hidden history publication.
- Raw RT updates raw moments, `sqrt(E[R²])`, and Kish `N_eff`; the visible result always comes from the spatially filtered branch.
- `currentAlpha = 1` publishes the final independently A-Trous-filtered center result and updates raw Kish `N_eff` to 1.
- `standardDeviation` is the square root of the local `g^-3`-closed Bures MC observation variance at every A-Trous boundary. Each spatial pass linearly filters its variance with the signal weights; it neither propagates weighted-mean estimator uncertainty nor assumes that the supplied MC variance shares an empirical sample set or `N_eff` with the signal moments. Packing never performs a hidden square root.
- A-Trous light rejection uses squared Bures distance and `V_B,center/N_eff,center + V_B,sample/N_eff,sample`, so each tap converts its own MC observation variance into estimator variance before the difference test. Both numerator and denominator scale linearly with radiance.
- Each A-Trous step also applies a fixed cumulative spatial-confidence scale derived from the preceding fixed kernels. This restores progressive late-pass rejection without converting the stored MC variance into estimator variance or folding difference covariance into the scale.
- Short-history variance preparation linearly reconstructs neighboring temporal `E[R u]`, `E[R]` and `E[R²]` with the geometry kernel, then evaluates the Bures variance closure once on the pooled population. This ordering preserves between-pixel directional spread. Kish `N_eff` is reconstructed separately as `W²/sum(w_i²/N_i)` and supplies the finite-history correction. Raw RT supplies only the independent-current signal; it is not the spatial variance population.
- Temporal response converts filtered MC variance into estimator variance for history blending: `V_est,H=V_MC,H/N_eff,H` and `V_est,C=V_MC,C/N_eff,C`. A-Trous performs the same conversion read-only for its center-neighbor difference test. The current moment, CoCg, `V_MC,C`, and `N_eff,C` are read from the same final A-Trous center record without a second neighborhood reconstruction.
- Filtered history is updated directly as `H_filtered <- mix(H_filtered, M3x3(C), currentAlpha)`. Raw history independently mixes reprojected raw moments with Raw RT using the same alpha, so spatial filtering never contaminates `E[R²]` or raw Kish `N_eff`.
- Spatial geometry stores each pixel's Kish `N_eff` instead of material identity. Spatial light-difference rejection normalizes the center and neighbor variance by their respective `N_eff`; material boundaries are not a denoiser rejection condition.
- `rootMeanY2` always means `sqrt(E[R²])`.
- `E[R u]`, `E[R]`, `CoCg`, and `E[R²]` are linear encoder/latent moments. Every temporal, reprojection, branch, and spatial moment combination is a normalized weighted sum; only Kish `N_eff` and the read-only Bures variance closure are nonlinear.
- MC observation variance is derived read-only from `(E[R u], E[R], E[R²], N_eff)`. The empirical `E[R²]` retains radial firefly energy; the `g^-3` joint model supplies `E[R²u]` and `E[R²uu^T]`. The result must never be inverted to reconstruct `E[R²]`, and realizability projection remains local to distance/closure evaluation rather than modifying history.
- Temporal reprojection reconstructs moments and Kish effective samples; it never linearly interpolates `N_eff`.
- Diffuse reprojection uses the hand-tuned thick tangent-plane footprint for geometry acceptance and ordinary normalized bilinear weights for history fields. Its per-tap projected-area Jacobian `d_cur²|NoV_hist|/(d_prev²|NoV_center|)` scales only that tap's `N_eff` before Kish reconstruction; it never scales `E[R u]`, `E[R]`, `CoCg`, `E[R²]`, or filtered MC variance.
- Reflection hit distance is a per-frame virtual-motion tracking guide. It is never mixed with the lighting alpha.
- Exact delta mirrors accept only virtual-motion history; failed virtual reprojection resets history instead of falling back to surface motion.
- The tuned temporal response in `temporal_response.glsl` is part of the current behavior and must not be silently changed.
- Fixed estimator and provisional-pipeline constants live in `internal_constants.glsl`; they are code invariants, not shader-pack options.

This correction keeps workgroup sizes, image formats, and the A-Trous pass count unchanged. The diffuse SSBO grows by one RGBA32UI plane for spatial-filter N_eff ping-pong and diffuse filtered-history N_eff reprojection.

## Numerical audit

The six Kish propagation constants are calibrated with the runtime Bures
weights on a homogeneous front-facing plane. Run
python -B tools/calibrate_bures_pass_correlations.py to reproduce the
[calibration and independent MC holdout](../../../doc/calibration/README.md).
The fixed-kernel overlap script remains the fully accepted-kernel baseline.

Run python tools/audit_bures_g3_denoiser.py from the shader-pack root. The
audit compares the elementary distance with an independent 2x2 matrix-square-
root implementation, checks the g^-4 angular moments by quadrature, verifies
the local-metric contraction and exposure homogeneity, exercises the spatial
pooling order, and stress-tests the cancellation-free FP32 variance form.

## TODO

- Calibrate the `g^-3` Bures closure against independent temporal pairs. Spatial reconstruction of temporal moments still assumes local stationarity and can confuse spatial signal variation with Monte Carlo variance.
