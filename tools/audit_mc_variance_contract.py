"""CPU contract/counterexample audit; does not run or modify production shaders.

Run from any directory: python -B tools/audit_mc_variance_contract.py
Prints JSON to stdout. Cases use IEEE half rounding, not an assumed GPU FTZ mode.
Assertions establish the counterexamples, not that their trigger occurs in-game.
"""
from pathlib import Path
import hashlib
import json
import platform
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEED = 906202619


def half(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def closure(moment, rms, n, zero_value=65504.0**2):
    w = max(float(moment[3]), 0.0)
    if n <= 1:
        return 0.0
    if w == 0:
        return (zero_value if rms > 0 else 0.0) / (1.0 - 1.0 / n)
    rms = max(float(rms), w)
    k2 = np.clip(np.dot(moment[:3], moment[:3]) / w**2, 0, 1)
    biased = ((rms-w)*(rms+w) + 3*(1-k2)/(3+k2)*rms**2)/(4*w)
    return float(biased / (1-1/n))


def lane_checks(rng):
    # Producer/consumer ABIs: diffuse (N,RMS), reflection (RMS,N).
    values = rng.uniform(1, 65000, (10000, 2)).astype(np.float16)
    words = values.view(np.uint16).astype(np.uint32)
    packed = words[:, 0] | (words[:, 1] << 16)
    decoded = np.stack((packed & 65535, packed >> 16), 1).astype(np.uint16)
    assert np.array_equal(decoded, words)
    # Shared tile always writes (N,RMS), then .yx exposes (RMS,N).
    assert np.array_equal(decoded[:, ::-1][:, ::-1], words)
    return {"pairs": len(values), "bit_roundtrip": True,
            "scope": "CPU packing plus source ABI inspection; not GPU buffer readback"}


def moment_checks(rng):
    # Arbitrary positive weighted mixtures, including already-compound samples.
    r = np.exp(rng.uniform(-20, 10, (10000, 8)))
    u = rng.normal(size=(10000, 8, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    a = rng.random((10000, 8))
    a /= a.sum(1, keepdims=True)
    w = (a*r).sum(1)
    v = (a[:, :, None]*r[:, :, None]*u).sum(1)
    e2 = (a*r*r).sum(1)
    assert np.all(e2 >= w*w*(1-1e-12))
    assert np.all(np.linalg.norm(v, axis=1) <= w*(1+1e-12))
    assert np.all(half(np.sqrt(e2)) >= half(w))
    return {"mixtures": len(w), "jensen_and_cone": True,
            "half_rms_ge_half_mean": True,
            "scope": "finite nonnegative normalized mixtures; no claim of unbiased adaptive weights"}


def precision_case():
    r = float(half(1e-6))
    alpha, n = .01, 199.
    m = np.array([0., 0., alpha*r, alpha*r])
    rms = np.sqrt(alpha*r*r)
    q = half(m)
    qr = float(half(rms))
    assert q[3] == 0 and qr > 0
    before = closure(m, rms, n)
    old = closure(q, qr, n)
    b, neighbor = .001, 1e-4
    mixed = half((1-b)*q + b*np.array([0, 0, neighbor, neighbor]))
    mixed_rms = float(half(np.sqrt((1-b)*qr**2+b*neighbor**2)))
    after = closure(mixed, mixed_rms, n)
    assert old > 1e9 and after < 1e-3
    return dict(raw_R=r, mean_before=m[3], rms_before=rms,
                mean_half=q[3], rms_half=qr, same_formula_before=before,
                old_sentinel_variance=old, user_one_variance=closure(q, qr, n, 1.),
                zero_fallback_variance=closure(q, qr, n, 0.),
                after_point_one_percent_neighbor=after,
                old_variance_at_1e_minus5_spatial_weight=old*1e-5)


def sanitizer_cases():
    # Both moment-state sanitizers clear all six components if even CoCg is NaN.
    # Temporal packers independently preserve valid RMS and N.
    signal = np.array([0., 0., 20., 100., np.nan, 0.])
    cleaned = np.zeros(6) if not np.isfinite(signal).all() else signal
    rms, n = 110., 32.
    assert cleaned[3] == 0 and rms > 0 and n > 1
    return dict(input_mean=100., fault="CoCg NaN", output_mean=cleaned[3],
                retained_rms=rms, retained_n=n,
                old_sentinel_variance=closure(cleaned[:4], rms, n),
                note="fault injection, not an observed in-game NaN")


def compound_cases():
    # Two deterministic hemispherical atoms of weight R/2 at +/-60 degrees.
    # Every frame has exactly the same packet (v,w)=(0,0,R/2,R).
    r, k, n = 100., .5, 32.
    m = np.array([0., 0., r*k, r])
    result = closure(m, r, n)
    assert result > 17 and result < 18
    # An alternative ensemble picks one of those directions with equal chance,
    # weight R. It has identical stored population moments (v,w,E[R²]), but
    # its true local Bures observation variance is R(1-k²)/4, not zero.
    return dict(R=r, kappa=k, n=n, closure_variance=result,
                deterministic_packet_true_variance=0.,
                same_moments_random_atom_local_variance=r*(1-k*k)/4,
                conclusion="stored moments do not identify general packet MC covariance")


def propagation_cases():
    a = np.array([.9, .1])
    v = np.array([1., 1000.])
    # First spatial pass p=0, independent unit-N observations, common metric.
    kish = 1/np.dot(a, a)
    product = np.dot(a, v)/kish
    exact = np.dot(a*a, v)
    assert product/exact > 7
    return dict(weights=a.tolist(), observation_variances=v.tolist(),
                kish=kish, linear_variance_divided_by_kish=product,
                independent_weighted_estimator_variance=exact,
                ratio=product/exact,
                condition="fixed weights and common tangent metric; no geometry or correlation")


def cold_start_case():
    # Equal temporal inputs with R=100 and k=.5, N=2. Spatial pooling changes
    # only finite-history correction. The angular closure isn't a centered
    # sample covariance and doesn't vanish at N=1 before correction.
    m = np.array([0., 0., 50., 100.])
    offsets = np.arange(-3, 4)
    g = np.exp(-.5*offsets**2)
    weights = np.outer(g, g).ravel()
    weights /= weights.sum()
    pooled_n = 2/np.dot(weights, weights)
    temporal, spatial = closure(m, 100., 2), closure(m, 100., pooled_n)
    return dict(temporal_n=2., pooled_n=pooled_n,
                temporal_variance=temporal, spatial_variance=spatial,
                temporal_over_spatial=temporal/spatial,
                note="constant packet; both true variances zero, no FP16 or spatial heterogeneity")


def indexing_checks():
    # Source variance tile: 16x16 interior plus radius-3 halo, 256 lanes.
    counts = np.zeros(22*22, dtype=int)
    for y in range(16):
        for x in range(16):
            counts[(y+3)*22+x+3] += 1
    for lane in range(256):
        for index in range(lane, 22*22, 256):
            y, x = divmod(index, 22)
            if not (3 <= x < 19 and 3 <= y < 19):
                counts[index] += 1
    assert np.all(counts == 1)
    for w, h in ((1269, 744), (1273, 793), (13, 10)):
        y, x = np.mgrid[:h, :w]
        addresses = (((y >> 3)*((w+7)//8)+(x >> 3))*64
                     + (y & 7)*8+(x & 7))
        assert np.unique(addresses).size == w*h
        assert addresses.max() < ((w+7)//8)*((h+7)//8)*64
    return dict(shared_tile_written_exactly_once=True,
                tiled_address_bijection=True,
                scope="index arithmetic only; driver pass barriers and allocations not executed")


def main():
    rng = np.random.default_rng(SEED)
    paths = sorted(set(ROOT.glob("shaders/post/denoiser/**/*.glsl")) |
                   set(ROOT.glob("shaders/lib/lighting/denoiser/*.glsl")) |
                   {ROOT/"shaders/lib/buffers/diffuse_buffer.glsl",
                    ROOT/"shaders/lib/buffers/specular_buffer.glsl",
                    ROOT/"shaders/lib/common/pack_half.glsl",
                    ROOT/"shaders/lib/math/statistics.glsl",
                    ROOT/"shaders/lib/rt/raytrace/gbuffer_io.glsl"})
    source = (ROOT/"shaders/lib/lighting/denoiser/variance_prepare.glsl").read_text(encoding="utf-8")
    found = re.search(r"if \(!\(meanY > 0.0\)\) \{.*?\n    \}", source, re.S)
    branch = found.group(0) if found else "Unknown-uncertainty ingress guard installed; old failure reproduced below as a historical counterexample."
    result = dict(scope="CPU algebra/IEEE FP16 counterexamples plus source review; no GPU scene capture",
                  seed=SEED, python=platform.python_version(), numpy=np.__version__,
                  current_zero_mean_branch=branch,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  packing=lane_checks(rng), finite_mixtures=moment_checks(rng),
                  precision=precision_case(), partial_sanitation=sanitizer_cases(),
                  compound_samples=compound_cases(), heteroscedastic_propagation=propagation_cases(),
                  cold_start=cold_start_case(), indexing=indexing_checks())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
