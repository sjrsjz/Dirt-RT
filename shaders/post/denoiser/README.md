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

The normalized constant-correlation estimator is bounded by its endpoint
sigmas, including the existing known-sigma donor. Its final conversion permits
a narrow FP32 overshoot at the FP16 sigma cap; the general variance ingress
remains strict. See [sigma/ray measurements and the cap proof](../../../doc/bench_sigma_ray_analysis.md).

Temporal resolve uses direct Bures estimator-variance propagation over the
final A-Trous current estimate and reprojected denoised history. See
[the current trial and validation](../../../doc/estimator_variance_trial.md).

The encoder stores linear moment state `(E[R u], E[R], sqrt(E[R²]), N_eff)`. The filtered mean remains linear. Light differences use the Bures--Wasserstein geometry of its `2x2` PSD embedding, and MC variance uses the alpha=1 `g^-3` family only to close the missing `R²`-weighted angular moments. Lighting decoding remains a separate operation.

## Scheduling and transient images

| Composite | Domain | Operation | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | Diffuse | Temporal reprojection/proposal | Raw and previous histories | swap state, filtered reprojection, `c6` raw reprojection |
| 50 | Diffuse | Variance preparation | swap proposal, Raw RT, geometry | `c3` geometry, `c4` signal, scratch A |
| 51–55 | Diffuse | A-Trous steps 1–16 | `c3`, `c4/c5` proposal, scratch A/B | opposite `c4/c5` proposal and scratch A/B |
| 56 | Diffuse | A-Trous step 32 | `c3`, `c5` proposal, scratch B | final independent-current estimator in scratch A; selected filtered-variance diagnostic |
| 58 | Diffuse | Resolve | `c6`, scratch A, Raw RT and reprojection scratch | accepted diffuse histories and denoised guide |
| 59 | Reflection | Raw input adapter | RT reflection and geometry | `c6` Raw reflection |
| 61 | Reflection | Temporal reprojection/proposal | `c6`, previous histories | `c4` proposal, `c5` raw reprojection, filtered reprojection scratch |
| 63 | Reflection | History staging | `c5` and reprojection scratch | reflection N1/N2 staging |
| 65 | Reflection | Variance preparation | `c4`, `c6`, geometry | `c3` geometry, `c5` signal, scratch A |
| 66–70 | Reflection | A-Trous steps 1–16 | `c3`, `c4/c5` proposal, scratch A/B | opposite `c4/c5` proposal and scratch A/B |
| 71 | Reflection | A-Trous step 32 | `c3`, `c4` proposal, scratch B | final independent-current estimator in scratch A; selected filtered-variance diagnostic |
| 72 | Reflection | Resolve | `c6`, scratch A, geometry and staged histories | all accepted reflection histories and public reflection output |

`c3` and `c6` are reused only after the diffuse domain has completed. Scratch A
is the existing RGBA32F `bloomAtlas` image; scratch B is `bloomBlur`. Both domains
use this same mapping and execute serially. Variance preparation initializes A,
the six spatial steps alternate A/B, and resolve consumes the final A. No pass
reads a neighborhood from the image it writes. The host must make image writes
visible to subsequent texture fetches at each pass boundary.

`scratch_io.glsl` stores signal words through `imageStore` and retrieves them
with `texelFetch`, bypassing texture filtering. Each word holds two finite FP16
values. Adding `0x00800000u` before `uintBitsToFloat` maps it to a finite normal
FP32 representation; `floatBitsToUint` followed by integer subtraction restores
the original bits. This prevents subnormal flushing from destroying the signal.
These image values are transport bits during denoising: float arithmetic,
filtered sampling and color conversion would violate this storage contract.

Both streams retain the generic 16-byte `uvec4` signal layout, but carry only
the fields required by their spatial role. This is common to both domains:

| Spatial stream | Live fields | Canonical zero fields | Consumer |
| --- | --- | --- | --- |
| Proposal | `maxEntY`, estimator sigma, `virtualDistance` | `CoCg` | Shared tap weights, virtual geometry and the next proposal |
| Independent current | `maxEntY`, `CoCg`, estimator sigma | `virtualDistance` | Next current estimate and final temporal resolve |

Variance preparation (50/65) writes these zeros before the first spatial step;
each subsequent step preserves them. The proposal accumulator has no chroma,
and the current accumulator has no virtual-distance statistic. This removes
unused arithmetic and live values without shrinking the signal ABI or changing
persistent history layouts. Resolve derives its published distance from the
existing geometry/history contract; it does not consume current scratch distance.
Step 32 publishes only current, as shown in the scheduling table.

After resolve 72, neither scratch image has a denoiser consumer. Bloom 90/91
overwrites its valid LOD rectangles in `bloomAtlas`; 92/93 then use the two images
for ordinary FP32 Gaussian filtering. Bloom's valid filter contributions must
come from those overwritten rectangles, with the existing per-LOD boundary rules
excluding residual scratch outside them. The image handoff requires no extra
allocation or copy. See [the 1080p profile changes](../../../doc/bench1080_optimization.md).

## Ownership and statistical invariants

Spatial estimator variance uses the existing overlap correlations
(0, 0.09184833, 0.12613998, 0.13781854, 0.14262104, 0.14655028) as trial
initial values. They were calibrated for the earlier variance policy and
require recalibration for the new adaptive weights.

- Diffuse resolve (58) is the only accepted diffuse-history commit point; temporal swap data is provisional.
- Reflection resolve (72) is the only accepted reflection-history commit point. Composite 63 temporarily stages raw reprojection in N1/N2, and composite 71 has no hidden history publication.
- Both spatial estimators share tap acceptance and validity. The final independent-current record supplies resolve validity as well as its estimate; neither resolve reads a final proposal image.
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

The estimator-variance trial preserves workgroup sizes, signal formats and
A-Trous pass count. Following the earlier PG removal, moving independent-current
scratch to the existing bloom images reduces the diffuse SSBO from ten to eight
`uvec4` planes: 160 to 128 bytes per allocated pixel. Planes N0–N7 retain their
raw light, history, surface and denoised-history layouts; the two transient
SSBO planes are removed. Allocation dimensions still follow `shaders.properties`.

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

## Unified spatial execution

Both domains use the same virtual-plane reconstruction, tap processing and
estimator propagation. Steps 1/2/4 use 16x16 groups; steps 8/16/32 use 8x8 groups
with a shared virtual-position tile. Roughness is input data, not a compile-time
domain switch. Storage bindings and diagnostic outputs belong to the adapters.

Small groups cooperatively reconstruct an FP32 primary-ray tile with side
`16 + 2 * step`: 18, 20 or 24 pixels. The halo covers both spatial taps and
the four virtual-normal neighbors. All 256 invocations populate the tile and
reach its barrier before any bounds or validity return. A `vec2` plus a float
stores each ray without vec3 padding. `geometry.glsl` reads this tile for small
steps and retains direct reconstruction for large sparse Poisson gathers.
Projection jitter, normalization and world rotation use the existing formula;
there is no new direction quantization or persistent geometry field.

The small kernel encodes each offset component plus one in two bits per tap
inside integer constants, replacing dynamically indexed offset/weight arrays.
It preserves the order `(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1),
(0,1), (1,1)` and the original diagonal/axis weights `0.44445` / `0.66667`.
This changes lookup representation, not tap positions, accumulation order or
the filter's rounded coefficients.

`atrous_policy.glsl` expresses the common output lifetime. Steps 1–16 accumulate
and publish the live fields of both estimators. Step 32 still loads the proposal
and uses it for every tap's validity and weight, but publishes only the independent-current
result. Its proposal accumulation is retained only for filtered-variance debug
views 25/38; no final proposal image is written in any view. This policy depends
on the consumer and step, and applies to both signal domains.

`denoiserSpatialPreparedSignalWordsValid` is restricted to the closed chain
from variance preparation through spatial ping-pong. Preparation sanitizes
inputs, spatial resolve sanitizes its outputs, and invalid stores use the
canonical sigma -1. Sigma at these boundaries is therefore finite nonnegative,
-1 (invalid), or -2 (valid light with unknown variance). The spatial check may
reject the FP16 -1 word directly, followed by trusted unpacking; both streams
and the separate geometry checks still participate in tap acceptance. It is
not a validator for arbitrary bits or unsanitized external input. History,
reprojection and final resolve retain their checked entry points and full
validation contracts. Adding another spatial producer requires establishing
the same sanitization boundary before using the compact predicate.

`debug_buffer.glsl` produces only fields required by the compile-time
`DEBUG_VIEW`. Inactive fields are unspecified; they must not feed renderer state
or be interpreted as outputs of the current frame.

`atrous_tap.glsl` is the common neighborhood loop body for both schedules.
It retains loop rejection directly, avoiding the driver control-flow regression
observed when passing the guide and two accumulators through a helper function.
`variance_prepare.glsl` owns the moment closure and pass orchestration;
`variance_tile.glsl` owns cooperative staging and spatial moment pooling.

Variance preparation encodes tile validity in the sign bit of its nonnegative
surface distance (negative zero is canonicalized). This avoids 484 validity uints;
invalid tile entries are rejected before their uninitialized moment lanes are read.
The tile stores a decoded ray and distance in vec4. Octahedral quantization is
preserved, but each ray is normalized once per tile entry instead of repeatedly
in every 7x7 neighborhood. This uses two extra shared words per entry; gfx1100
reports 13,824 LDS bytes versus 9,728 for the encoded-ray intermediate version.
The persistent geometry and signal formats are unchanged. Distance comparisons
prepare the center PSD state once, retaining the stable rationalized metric.

`tools/audit_post_pipeline.py` executes production compute shaders, with explicit
OpenGL image bindings and synthetic inputs written through production buffer APIs.
It compares live intermediate signals and accepted histories across temporal
preparation, variance, all six spatial iterations and resolve. Image scratch is
decoded back to signal words for comparison with older SSBO storage. Removed
final proposal images and inactive diagnostic fields have no comparison contract.
Role-zeroed spatial lanes are checked as zero rather than compared with the
discarded baseline statistic; live signal fields retain their comparison contract.
Selected diagnostics, history and geometry metadata retain their checks. A
pre-edit shader tree is required by `--baseline-dir`. Equal synthetic inputs also
require bitwise agreement between both domain adapters at every spatial step.
Results are written beneath `temp/<label>/`.

The full-chain comparison defaults to bitwise equality. The explicit
`--signal-half-ulp 1` option admits one FP16 step only in spatial lighting
payloads and the final reflection lighting payloads; sigma, geometry, counts,
distances and raw histories still require exact equality. This is a measured
rounding allowance, not a proof of a global error bound. The separate six-step
stress fixtures also expose a few one-ULP sigma differences, which this strict
check rejects and the [capture 11 report](../../../doc/bench11_joint_optimization.md)
records without relaxing the check.

`tools/benchmark_spatial_chain.py` measures five or six consecutive spatial
dispatches with real proposal/current ping-pong, Iris barrier bits and
READ_WRITE image bindings. GPU copies restore inputs outside each query;
AB/BA query results are collected after the batch. It reports raw and consumed
field differences separately. Its coherent and mixed fixtures measure synthetic
chain costs; game frame times and cross-frame bloom reuse require game captures.
