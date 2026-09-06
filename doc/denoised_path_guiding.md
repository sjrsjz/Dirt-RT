# Denoised screen-space path guiding

Diffuse resolve publishes the exact FP16 direction/energy moments of its final
temporally resolved light into DIF_N_PATHGUIDE. The next frame's continuation
sampler and sun-NEE competing-PDF evaluation read this same cache. The existing
bilinear guide interpolation, strength, kappa conversion and sampling/PDF
formulas are unchanged from the direct-guide test.

The cache stores two packed moment words and two fixed validity tags. It has
no independent reservoir weights, candidate identity or temporal sample count.
Invalid and black guide records are cleared. The direction debug view reads
the same cache; it displays the axis, not concentration or sampling confidence.

The user reported that direct denoised guiding significantly reduced variance
for a nighttime single-source DI test and removed the observed super-bright
fireflies. This is an in-game qualitative A/B observation, not a measured
variance ratio or a claim that guiding always improves every scene.

## Removed pipeline

- The former composite57 reservoir build and its implementation.
- The two dedicated RT PG prewarm resolve/commit stages and their helpers.
- Raw proposal/endpoint/prewarm scratch writes and dedicated settings.
- Three PG scratch planes and the already-unused spatial-Neff metadata plane.

Diffuse storage is now eleven RGBA32UI layers, 176 bytes per allocated pixel,
down from 240. RT stages are contiguous: ray0 primary geometry; ray1..ray3
continuations; ray4 radiance-cache allocation; ray5 radiance-cache tracing.
Execution groups are 0,1,1,1,2,3. Radiance-cache RIS itself is retained.

Canonical sun MIS handles both NEE and continuation hits after removal of the
prewarm-only branches. No additional throughput clamp or variance tuning was
introduced. Reload shaders to rebuild allocations and pipelines; the diffuse
history signature was changed for the compacted layout.

## Validation

    python -B tools/audit_denoised_path_guide.py
    python -B tools/audit_estimator_variance.py --compile

The first checks stage ordering, unique buffer indices/allocation, removed
symbols, 10,000 exact FP16 guide copies, and canonical sun-MIS source wiring.
The second runs estimator arithmetic checks and compiles 51 shader variants,
including the guide debug view and representative RT hit/miss stages.
Compilation and CPU checks do not replace an in-game reload test.
