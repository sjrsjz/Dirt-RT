# Primary specular path guiding

The primary non-delta GGX reflection pass can mix its native visible-normal
distribution function (VNDF) proposal with the previous final denoised
specular state. The stored state represents the measure

```text
dnu(wi) = q_vndf(wi) Li(wi) dwi.
```

Its MaxEnt reconstruction is therefore a directly usable importance proposal
for the incident radiance already weighted by the native GGX proposal. The
runtime mixture is

```text
p(wi) = (1 - beta) q_vndf(wi) + beta g_qLi(wi),   beta <= 0.75.
```

`beta` is the user strength multiplied by directional concentration and valid
reprojection coverage. Geometry, versioned material identity, motion validity,
plane distance, and the current-frame surface footprint are checked per
bilinear history tap. Invalid history gives `beta = 0`, exactly recovering the
original VNDF sampler. Delta reflection also remains on its analytic path.

The estimator keeps two first-bounce factors separate:

```text
transport weight = f NoL / p
qLi response     = f NoL / q = F (G2 / G1)
stored atom      = (transport contribution) / (qLi response) = q Li / p.
```

The response uses the cancelled analytic form, avoiding a numerical `0/0`
when the guide samples outside a narrow VNDF peak. Sampling the atom under
`p` has expectation `q Li`, so the existing qLi denoiser and two-fetch GGX
decoder retain their measure. Sun next-event estimation uses `p` as its
competing continuation PDF while injecting `q Li` into the same cache state.

This implementation guides only the primary reflection continuation. It adds
no buffer or history allocation. Ray0 reprojects the previous N3 final
denoised state once after producing the current primary surface and publishes
the complete six-component state, estimator sigma, and valid coverage in the
existing N4 scratch plane. Ray2 consumes it as the guide; reflection temporal
reuses the same surface reprojection for confidence clamping, then overwrites
N4 with its normal downstream scratch layout. All transforms stay in the
Vulkanite camera convention. `SPECULAR_PATH_GUIDING_STRENGTH = 0` provides the
VNDF-only control.

Run `python -B tools/audit_specular_path_guiding.py` for the algebra and source
contract checks. `python -B tools/audit_estimator_variance.py --compile`
compiles the complete shader variant set.
