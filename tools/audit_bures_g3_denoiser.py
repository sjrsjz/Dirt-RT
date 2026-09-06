#!/usr/bin/env python3
"""Independent numerical audit of the Bures and g^-3 denoiser formulas."""

from __future__ import annotations

import math

import numpy as np


def project_state(state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=np.float64).copy()
    result[3] = max(result[3], 0.0)
    norm = np.linalg.norm(result[:3])
    if norm > result[3] and norm > 0.0:
        result[:3] *= result[3] / norm
    return result


def bures_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    a = project_state(a)
    b = project_state(b)
    va, wa = a[:3], a[3]
    vb, wb = b[:3], b[3]
    qa = math.sqrt(max(wa * wa - float(va @ va), 0.0))
    qb = math.sqrt(max(wb * wb - float(vb @ vb), 0.0))
    affinity = math.sqrt(
        max(0.5 * (wa * wb + float(va @ vb) + qa * qb), 0.0)
    )
    numerator = float((va - vb) @ (va - vb)) + (qa - qb) ** 2
    denominator = wa + wb + 2.0 * affinity
    return numerator / denominator if denominator > 0.0 else 0.0


def state_matrix(state: np.ndarray) -> np.ndarray:
    x, y, z, w = project_state(state)
    return 0.5 * np.array(
        [[w + z, x - 1j * y], [x + 1j * y, w - z]],
        dtype=np.complex128,
    )


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.conj().T


def bures_matrix_reference(a: np.ndarray, b: np.ndarray) -> float:
    matrix_a = state_matrix(a)
    matrix_b = state_matrix(b)
    root_a = psd_sqrt(matrix_a)
    middle = root_a @ matrix_b @ root_a
    root_fidelity = float(np.sqrt(np.maximum(np.linalg.eigvalsh(middle), 0.0)).sum())
    result = float(np.trace(matrix_a).real + np.trace(matrix_b).real - 2.0 * root_fidelity)
    return max(result, 0.0)


def g3_bures_variance(w: float, kappa: float, e2: float) -> float:
    if w <= 0.0:
        return math.inf if e2 > 0.0 else 0.0
    e2 = max(e2, w * w)
    kappa2 = min(max(kappa * kappa, 0.0), 1.0)
    radial = max(e2 - w * w, 0.0)
    angular_scale = 3.0 * (1.0 - kappa2) / (3.0 + kappa2)
    return (radial + angular_scale * e2) / (4.0 * w)


def expanded_local_metric_variance(w: float, kappa: float, e2: float) -> float:
    first_mu = 4.0 * kappa / (3.0 + kappa * kappa)
    second_mu = (1.0 + 3.0 * kappa * kappa) / (3.0 + kappa * kappa)
    transverse = e2 * (1.0 - second_mu) / (4.0 * w)
    plus = (
        e2 * (1.0 + 2.0 * first_mu + second_mu)
        - w * w * (1.0 + kappa) ** 2
    ) / (8.0 * w * (1.0 + kappa))
    minus = (
        e2 * (1.0 - 2.0 * first_mu + second_mu)
        - w * w * (1.0 - kappa) ** 2
    ) / (8.0 * w * (1.0 - kappa))
    return transverse + plus + minus


def audit_distance(rng: np.random.Generator) -> tuple[float, float]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for _ in range(4096):
        states = []
        for _ in range(2):
            w = 10.0 ** rng.uniform(-4.0, 4.0)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            kappa = rng.uniform(0.0, 1.0)
            states.append(np.r_[direction * (w * kappa), w])
        closed = bures_distance_squared(states[0], states[1])
        reference = bures_matrix_reference(states[0], states[1])
        absolute = abs(closed - reference)
        relative = absolute / max(reference, 1.0e-12)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)

    invalid = np.array([2.0, 0.0, 0.0, 1.0])
    projected = np.array([1.0, 0.0, 0.0, 1.0])
    target = np.array([-0.2, 0.3, 0.1, 0.8])
    assert bures_distance_squared(invalid, target) == bures_distance_squared(projected, target)
    assert maximum_relative < 2.0e-9
    return maximum_absolute, maximum_relative


def audit_g4_moments() -> float:
    nodes, weights = np.polynomial.legendre.leggauss(1024)
    maximum_error = 0.0
    for kappa in (0.0, 0.1, 0.5, 0.9, 0.99, 0.999):
        density = np.power(1.0 - kappa * nodes, -4.0)
        normalization = float(weights @ density)
        first = float(weights @ (nodes * density)) / normalization
        second = float(weights @ (nodes * nodes * density)) / normalization
        expected_first = 4.0 * kappa / (3.0 + kappa * kappa)
        expected_second = (1.0 + 3.0 * kappa * kappa) / (3.0 + kappa * kappa)
        maximum_error = max(
            maximum_error,
            abs(first - expected_first),
            abs(second - expected_second),
        )
    assert maximum_error < 2.0e-9
    return maximum_error


def audit_variance_contraction(rng: np.random.Generator) -> float:
    maximum_relative = 0.0
    for _ in range(4096):
        w = 10.0 ** rng.uniform(-4.0, 4.0)
        kappa = rng.uniform(0.0, 0.9999)
        e2 = w * w * (1.0 + 10.0 ** rng.uniform(-4.0, 3.0))
        stable = g3_bures_variance(w, kappa, e2)
        expanded = expanded_local_metric_variance(w, kappa, e2)
        relative = abs(stable - expanded) / max(abs(stable), 1.0e-12)
        maximum_relative = max(maximum_relative, relative)
    assert maximum_relative < 2.0e-8
    return maximum_relative


def audit_boundaries_and_scaling(rng: np.random.Generator) -> float:
    w = 1.7
    e2 = 4.2
    assert abs(g3_bures_variance(w, 0.0, e2) - (2.0 * e2 - w * w) / (4.0 * w)) < 1.0e-14
    assert abs(g3_bures_variance(w, 1.0, e2) - (e2 - w * w) / (4.0 * w)) < 1.0e-14

    maximum_relative = 0.0
    for _ in range(1024):
        w = 10.0 ** rng.uniform(-3.0, 3.0)
        kappa = rng.uniform(0.0, 1.0)
        e2 = w * w * (1.0 + 10.0 ** rng.uniform(-3.0, 2.0))
        scale = 10.0 ** rng.uniform(-3.0, 3.0)
        base = g3_bures_variance(w, kappa, e2)
        scaled = g3_bures_variance(scale * w, kappa, scale * scale * e2)
        maximum_relative = max(
            maximum_relative,
            abs(scaled - scale * base) / max(scale * base, 1.0e-12),
        )
    assert maximum_relative < 2.0e-14
    return maximum_relative


def audit_spatial_pooling_order() -> float:
    positive = np.array([1.0, 0.0, 0.0, 1.0])
    negative = np.array([-1.0, 0.0, 0.0, 1.0])
    individual = 0.5 * (
        g3_bures_variance(positive[3], 1.0, 1.0)
        + g3_bures_variance(negative[3], 1.0, 1.0)
    )
    pooled = 0.5 * (positive + negative)
    pooled_kappa = np.linalg.norm(pooled[:3]) / pooled[3]
    pooled_variance = g3_bures_variance(pooled[3], pooled_kappa, 1.0)
    assert individual == 0.0
    assert abs(pooled_variance - 0.25) < 1.0e-15
    return pooled_variance


def audit_float32_stability(rng: np.random.Generator) -> float:
    count = 65536
    w = np.power(10.0, rng.uniform(-7.0, math.log10(65504.0), size=count)).astype(np.float32)
    rms_scale = np.power(10.0, rng.uniform(0.0, 3.0, size=count)).astype(np.float32)
    rms = np.minimum(w * rms_scale, np.float32(65504.0))
    e2 = (rms * rms).astype(np.float32)
    w2 = (w * w).astype(np.float32)
    e2 = np.maximum(e2, w2)
    tail = np.power(10.0, rng.uniform(-7.0, 0.0, size=count)).astype(np.float32)
    kappa2 = np.maximum(np.float32(1.0) - tail, np.float32(0.0))

    radial = (rms - w) * (rms + w)
    angular = (
        np.float32(3.0) * (np.float32(1.0) - kappa2)
        / (np.float32(3.0) + kappa2)
    )
    actual = (
        (radial + angular * e2) / (np.float32(4.0) * w)
    ).astype(np.float32)

    w64 = w.astype(np.float64)
    rms64 = rms.astype(np.float64)
    e264 = rms64 * rms64
    k264 = kappa2.astype(np.float64)
    expected = (
        np.maximum((rms64 - w64) * (rms64 + w64), 0.0)
        + 3.0 * (1.0 - k264) * e264 / (3.0 + k264)
    ) / (4.0 * w64)
    relative = np.abs(actual.astype(np.float64) - expected) / np.maximum(expected, 1.0e-30)
    assert np.isfinite(actual).all()
    maximum_relative = float(relative[expected > 1.0e-20].max(initial=0.0))
    assert maximum_relative < 4.0e-6
    return maximum_relative


def main() -> None:
    rng = np.random.default_rng(0xB0E5)
    distance_absolute, distance_relative = audit_distance(rng)
    moment_error = audit_g4_moments()
    contraction_error = audit_variance_contraction(rng)
    scaling_error = audit_boundaries_and_scaling(rng)
    pooled_variance = audit_spatial_pooling_order()
    float32_error = audit_float32_stability(rng)

    print("Bures matrix reference max absolute error:", f"{distance_absolute:.3e}")
    print("Bures matrix reference max relative error:", f"{distance_relative:.3e}")
    print("g^-4 quadrature moment max error:", f"{moment_error:.3e}")
    print("local-metric contraction max relative error:", f"{contraction_error:.3e}")
    print("exposure homogeneity max relative error:", f"{scaling_error:.3e}")
    print("opposite-direction pooled spatial variance:", f"{pooled_variance:.6f}")
    print("FP32 stable-form max relative error:", f"{float32_error:.3e}")
    print("audit: PASS")


if __name__ == "__main__":
    main()
