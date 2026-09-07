# MaxEnt denoiser pipeline

Diffuse resolve publishes its final temporally resolved denoised direction/
energy moments directly into the next frame's screen-space guide cache.
The ReSTIR PG reservoir, compute stage, and two RT prewarm stages are removed.
Sampling, PDF evaluation, guiding strength and radiance-cache probe RIS remain
unchanged. See [the guide contract](../../../doc/denoised_path_guiding.md).

The existing sigma lane now distinguishes invalid light (-1) from valid light
with unknown uncertainty (-2). Unknown values never enter variance arithmetic;
known neighbors supply missing sigma estimates before actual-weight propagation; all-unknown neighborhoods
retain their light while bypassing variance-based rejection. Geometry rejection
remains active. Temporal response falls back to the ordinary history update.
Prepared/Filtered debug views show unknown uncertainty in neutral gray.
See [ingress guard and validation](../../../doc/uncertainty_ingress_guard.md).

Temporal resolve uses direct Bures estimator-variance propagation over the
final A-Trous current estimate and reprojected denoised history. See
[the current trial and validation](../../../doc/estimator_variance_trial.md).

The encoder stores linear moment state `(E[R u], E[R], sqrt(E[R²]), N_eff)`. The filtered mean remains linear. Light differences use the Bures--Wasserstein geometry of its `2x2` PSD embedding, and MC variance uses the alpha=1 `g^-3` family only to close the missing `R²`-weighted angular moments. Lighting decoding remains a separate operation.

## Scheduling and transient images

| Composite | Domain | Operation | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | Diffuse | Temporal reprojection/proposal | Raw and previous histories | swap state, filtered reprojection, `c6` raw reprojection |
| 50 | Diffuse | Variance preparation | swap proposal, Raw RT, geometry | `c3` geometry, `c4` signal, scratch A |
| 51–56 | Diffuse | A-Trous steps 1–32 | `c3`, `c4/c5`, scratch A/B | opposite `c4/c5`, scratch A/B; final signal is `c4` |
| 58 | Diffuse | Resolve | `c4`, `c6`, scratch A, reprojection scratch | accepted diffuse histories and denoised guide |
| 59 | Reflection | Raw input adapter | RT reflection and geometry | `c6` Raw reflection |
| 61 | Reflection | Temporal reprojection/proposal | `c6`, previous histories | `c4` proposal, `c5` raw reprojection, filtered reprojection scratch |
| 63 | Reflection | History staging | `c5` and reprojection scratch | reflection N1/N2 staging |
| 65 | Reflection | Variance preparation | `c4`, `c6`, geometry | `c3` geometry, `c5` signal, scratch A |
| 66–71 | Reflection | A-Trous steps 1–32 | `c3`, `c4/c5`, scratch A/B | opposite `c4/c5`, scratch A/B; final signal is `c5` |
| 72 | Reflection | Resolve | `c5`, `c6`, scratch A, staged histories | all accepted reflection histories and public reflection output |

`c3` and `c6` are reused only after the diffuse domain has completed. Scratch A/B are two transient planes in the diffuse SSBO; diffuse and reflection execute serially. No pass reads a neighborhood from the same image it writes.

## Ownership and statistical invariants

Spatial estimator variance uses the existing overlap correlations
(0, 0.09184833, 0.12613998, 0.13781854, 0.14262104, 0.14655028) as trial
initial values. They were calibrated for the earlier variance policy and
require recalibration for the new adaptive weights.

- Diffuse resolve (58) is the only accepted diffuse-history commit point; temporal swap data is provisional.
- Reflection resolve (72) is the only accepted reflection-history commit point. Composite 63 temporarily stages raw reprojection in N1/N2, and composite 71 has no hidden history publication.
- Raw RT updates raw moments, `sqrt(E[R²])`, and Kish `N_eff`; the visible result always comes from the spatially filtered branch.
- `currentAlpha = 1` publishes the final independently A-Trous-filtered center result and updates raw Kish `N_eff` to 1.
- `standardDeviation` stores sqrt(estimator variance) at every A-Trous boundary. Preparation divides the selected observation variance by the center raw temporal count once; independent-current starts with N=1. Each pass propagates actual squared weights with a constant-correlation overlap closure.
- A-Trous light rejection uses squared Bures distance divided by `V_est,center + V_est,sample`. Both numerator and denominator scale linearly with radiance; endpoint cross-covariance is approximated as zero.
- Each A-Trous step uses the same configured light sensitivity. Fixed cumulative confidence multipliers and filtered/spatial Neff propagation have been removed.
- Short-history variance preparation linearly reconstructs neighboring temporal `E[R u]`, `E[R]` and `E[R²]` with the geometry kernel, then evaluates the Bures variance closure once on the pooled population. This ordering preserves between-pixel directional spread. Kish `N_eff` is reconstructed separately as `W²/sum(w_i²/N_i)` and supplies the finite-history correction. Raw RT supplies only the independent-current signal; it is not the spatial variance population.
- Temporal response consumes the final independent-current estimator and reprojected filtered history, including their stored estimator sigmas directly. Temporal sigma propagates squared blend weights under an independence approximation. Raw temporal Kish count controls the baseline alpha only.
- Filtered history is updated directly as `H_filtered <- mix(H_filtered, C_final_atrous, currentAlpha)`. Raw history independently mixes reprojected raw moments with Raw RT using the same alpha, so spatial filtering never contaminates `E[R²]` or raw Kish `N_eff`.
- Spatial geometry retains raw Kish count in its existing ABI. Light rejection no longer divides uncertainty by this count; material boundaries are not a denoiser rejection condition.
- `rootMeanY2` always means `sqrt(E[R²])`.
- `E[R u]`, `E[R]`, `CoCg`, and `E[R²]` are linear encoder/latent moments. Every temporal, reprojection, branch, and spatial moment combination is a normalized weighted sum; uncertainty propagation, Kish `N_eff`, and the read-only Bures variance closure are nonlinear.
- MC observation variance is derived read-only from `(E[R u], E[R], E[R²], N_eff)`. The empirical `E[R²]` retains radial firefly energy; the `g^-3` joint model supplies `E[R²u]` and `E[R²uu^T]`. The result must never be inverted to reconstruct `E[R²]`, and realizability projection remains local to distance/closure evaluation rather than modifying history.
- Temporal reprojection reconstructs moments and Kish effective samples; it never linearly interpolates `N_eff`.
- Diffuse reprojection uses the tangent-plane footprint and normalized bilinear moment weights. Its area Jacobian reduces raw Neff and inflates stored estimator variance for footprint loss. Filtered-history reconstruction uses correlation 1, as does the surface/virtual branch mixture, to account conservatively for overlap in a fixed metric.
- Reflection hit distance is a per-frame virtual-motion tracking guide. It is never mixed with the lighting alpha.
- Exact delta mirrors accept only virtual-motion history; failed virtual reprojection resets history instead of falling back to surface motion.
- The tuned temporal response in `temporal_response.glsl` is part of the current behavior and must not be silently changed.
- Fixed estimator and provisional-pipeline constants live in `internal_constants.glsl`; they are code invariants, not shader-pack options.

The estimator-variance trial preserves workgroup sizes, signal formats and A-Trous pass count. The subsequent PG removal and shared diffuse-history reprojection compact diffuse storage to ten RGBA32UI planes (160 bytes per allocated pixel), releasing the obsolete guiding plane, three PG scratch planes, and the unused Neff metadata plane.

## Numerical audit

Run python -B tools/audit_estimator_variance.py --compile for the current
covariance, FP16, multikernel and shader checks. The previous
tools/calibrate_bures_pass_correlations.py reproduces the historical policy,
not the current estimator-variance runtime. Its
[results](../../../doc/calibration/README.md) remain archived.

Run python tools/audit_bures_g3_denoiser.py from the shader-pack root. The
audit compares the elementary distance with an independent 2x2 matrix-square-
root implementation, checks the g^-4 angular moments by quadrature, verifies
the local-metric contraction and exposure homogeneity, exercises the spatial
pooling order, and stress-tests the cancellation-free FP32 variance form.

## TODO

- Calibrate the `g^-3` Bures closure against independent temporal pairs. Spatial reconstruction of temporal moments still assumes local stationarity and can confuse spatial signal variation with Monte Carlo variance.
