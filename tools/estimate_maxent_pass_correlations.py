#!/usr/bin/env python3
"""Derive the moment-error correlations induced by the fixed spatial kernel.

Start with independent, equal-variance errors at every pixel. If the linear
response at output pixel x is h_x, then the trace-error correlation is

    rho(x, y) = <h_x, h_y> / (||h_x|| ||h_y||).

The three grid passes are translation invariant and are evaluated exactly.
The rotating Poisson passes use the shader's integer hash and rounding. Their
spatial average is estimated by sampling pixel coordinates, but every sampled
kernel overlap is evaluated exactly; no light samples, MaxEnt decoder, phi,
geometry weight, signal-dependent weight, or N_eff enters the calculation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from typing import Dict, Mapping, Sequence, Tuple

import torch


STEPS = (1, 2, 4, 8, 16, 32)
GRID_OFFSETS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)
GRID_WEIGHTS = (
    0.44445,
    0.66667,
    0.44445,
    0.66667,
    0.66667,
    0.44445,
    0.66667,
    0.44445,
)
POISSON = (
    (-0.4706069, -0.4427112, 0.81170),
    (-0.9057375, 0.3003471, 0.63422),
    (-0.3487388, 0.4037880, 0.86734),
    (0.1023042, 0.6439373, 0.80847),
    (0.5699277, 0.3513750, 0.79925),
    (0.2939128, -0.1131226, 0.95161),
    (0.7836658, -0.4208784, 0.67328),
    (0.1564120, -0.8198990, 0.70589),
)
UINT_MASK = 0xFFFFFFFF

Point = Tuple[int, int]
SparseKernel = Dict[Point, float]


def shifted_kernel_covariance(
    kernel: Mapping[Point, float], a: Point, b: Point
) -> float:
    """Return <h(.-a), h(.-b)> for one translation-invariant kernel."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sum(
        value * kernel.get((point[0] + dx, point[1] + dy), 0.0)
        for point, value in kernel.items()
    )


def exact_grid_pass_statistics(
    kernel: Mapping[Point, float], step: int
) -> Tuple[float, float]:
    points = ((0, 0),) + tuple(
        (step * dx, step * dy) for dx, dy in GRID_OFFSETS
    )
    weights = (1.0,) + GRID_WEIGHTS
    covariance = tuple(
        tuple(shifted_kernel_covariance(kernel, a, b) for b in points)
        for a in points
    )
    variance = tuple(covariance[i][i] for i in range(9))

    difference_numerator = sum(
        weights[i] * covariance[0][i] for i in range(1, 9)
    )
    difference_denominator = sum(
        weights[i] * math.sqrt(variance[0] * variance[i])
        for i in range(1, 9)
    )
    propagation_numerator = 0.0
    propagation_denominator = 0.0
    for i in range(9):
        for j in range(i + 1, 9):
            pair_weight = weights[i] * weights[j]
            propagation_numerator += pair_weight * covariance[i][j]
            propagation_denominator += pair_weight * math.sqrt(
                variance[i] * variance[j]
            )
    return (
        difference_numerator / difference_denominator,
        propagation_numerator / propagation_denominator,
    )


def apply_grid_pass(kernel: Mapping[Point, float], step: int) -> SparseKernel:
    weight_sum = 1.0 + sum(GRID_WEIGHTS)
    result: defaultdict[Point, float] = defaultdict(float)
    for point, value in kernel.items():
        result[point] += value / weight_sum
        for (dx, dy), weight in zip(GRID_OFFSETS, GRID_WEIGHTS):
            shifted = (point[0] + step * dx, point[1] + step * dy)
            result[shifted] += value * weight / weight_sum
    return dict(result)


def build_grid_kernel_and_statistics() -> Tuple[SparseKernel, Dict[int, Tuple[float, float]]]:
    kernel: SparseKernel = {(0, 0): 1.0}
    statistics: Dict[int, Tuple[float, float]] = {}
    for step in STEPS[:3]:
        statistics[step] = exact_grid_pass_statistics(kernel, step)
        kernel = apply_grid_pass(kernel, step)
    return kernel, statistics


def autocorrelation_table(
    kernel: Mapping[Point, float], device: torch.device
) -> Tuple[torch.Tensor, int]:
    values: defaultdict[Point, float] = defaultdict(float)
    for (x0, y0), weight0 in kernel.items():
        for (x1, y1), weight1 in kernel.items():
            values[(x1 - x0, y1 - y0)] += weight0 * weight1
    radius = max(max(abs(x), abs(y)) for x, y in values)
    table = torch.zeros(
        (2 * radius + 1, 2 * radius + 1),
        dtype=torch.float64,
        device=device,
    )
    for (x, y), value in values.items():
        table[y + radius, x + radius] = value
    return table, radius


def whash(seed: torch.Tensor) -> torch.Tensor:
    seed = ((seed ^ 61) ^ (seed >> 16)) & UINT_MASK
    seed = (seed * 9) & UINT_MASK
    seed = (seed ^ (seed >> 4)) & UINT_MASK
    seed = (seed * 0x27D4EB2D) & UINT_MASK
    return (seed ^ (seed >> 15)) & UINT_MASK


def sample_points(points: torch.Tensor, step: int) -> torch.Tensor:
    """Return center plus the shader's eight rounded Poisson taps."""
    x = points[..., 0].to(torch.int64)
    y = points[..., 1].to(torch.int64)
    seed = (
        ((x * 0x9E3779B9) & UINT_MASK)
        ^ ((y * 0x85EBCA6B) & UINT_MASK)
        ^ ((step * 0xC2B2AE35) & UINT_MASK)
    )
    angle = (2.0 * math.pi) * (
        (whash(seed) >> 8).to(torch.float32) * (1.0 / 16777216.0)
    )
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    poisson_x = points.new_tensor(
        [point[0] for point in POISSON], dtype=torch.float32
    )
    poisson_y = points.new_tensor(
        [point[1] for point in POISSON], dtype=torch.float32
    )
    scale = float(step) * 1.75
    rotated_x = scale * (
        cosine[..., None] * poisson_x + sine[..., None] * poisson_y
    )
    rotated_y = scale * (
        -sine[..., None] * poisson_x + cosine[..., None] * poisson_y
    )

    # GLSL round ties are irrelevant here in practice, but this implements
    # nearest integer with halves away from zero explicitly.
    dx = torch.sign(rotated_x) * torch.floor(torch.abs(rotated_x) + 0.5)
    dy = torch.sign(rotated_y) * torch.floor(torch.abs(rotated_y) + 0.5)
    taps = torch.stack(
        (x[..., None] + dx.to(torch.int64), y[..., None] + dy.to(torch.int64)),
        dim=-1,
    )
    return torch.cat((points[..., None, :].to(torch.int64), taps), dim=-2)


def response_components(
    points: torch.Tensor,
    completed_large_steps: Sequence[int],
    pass_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Represent a response as shifted copies of the exact step-4 kernel."""
    centers = points[..., None, :].to(torch.int64)
    weights = torch.ones(1, dtype=torch.float64, device=points.device)
    for step in reversed(completed_large_steps):
        old_shape = centers.shape
        centers = sample_points(centers, step).reshape(
            *old_shape[:-2], -1, 2
        )
        weights = (weights[:, None] * pass_weights[None, :]).reshape(-1)
    return centers, weights


def response_covariance(
    centers_a: torch.Tensor,
    weights_a: torch.Tensor,
    centers_b: torch.Tensor,
    weights_b: torch.Tensor,
    table: torch.Tensor,
    radius: int,
    target_entries: int,
) -> torch.Tensor:
    """Evaluate exact kernel dot products for corresponding response batches."""
    pair_weights = (weights_a[:, None] * weights_b[None, :])[None, ...]
    pair_count = weights_a.numel() * weights_b.numel()
    chunk_size = max(1, target_entries // pair_count)
    side = 2 * radius + 1
    results = []
    for begin in range(0, len(centers_a), chunk_size):
        delta = (
            centers_b[begin : begin + chunk_size, None, :, :]
            - centers_a[begin : begin + chunk_size, :, None, :]
        )
        supported = (
            (torch.abs(delta[..., 0]) <= radius)
            & (torch.abs(delta[..., 1]) <= radius)
        )
        table_x = torch.clamp(delta[..., 0] + radius, 0, side - 1)
        table_y = torch.clamp(delta[..., 1] + radius, 0, side - 1)
        covariance = table[table_y, table_x] * supported
        results.append((covariance * pair_weights).sum(dim=(-2, -1)))
    return torch.cat(results)


def ratio_with_block_standard_error(
    numerator: torch.Tensor, denominator: torch.Tensor, blocks: int = 16
) -> Tuple[float, float]:
    estimate = float(numerator.sum() / denominator.sum())
    usable_blocks = min(blocks, len(numerator))
    if usable_blocks < 2:
        return estimate, 0.0
    block_estimates = []
    for indices in torch.tensor_split(
        torch.arange(len(numerator), device=numerator.device), usable_blocks
    ):
        block_estimates.append(
            numerator[indices].sum() / denominator[indices].sum()
        )
    values = torch.stack(block_estimates)
    standard_error = float(values.std(unbiased=True) / math.sqrt(usable_blocks))
    return estimate, standard_error


def large_pass_statistics(
    step: int,
    completed_large_steps: Sequence[int],
    sample_count: int,
    coordinate_min: int,
    coordinate_max: int,
    seed: int,
    pass_weights: torch.Tensor,
    table: torch.Tensor,
    radius: int,
    target_entries: int,
) -> Dict[str, float]:
    generator = torch.Generator(device=table.device).manual_seed(seed)
    points = torch.randint(
        coordinate_min,
        coordinate_max,
        (sample_count, 2),
        generator=generator,
        device=table.device,
    )
    signal_points = sample_points(points, step)
    centers, component_weights = response_components(
        signal_points.reshape(-1, 2), completed_large_steps, pass_weights
    )
    centers = centers.reshape(sample_count, 9, -1, 2)
    covariance = torch.empty(
        (sample_count, 9, 9), dtype=torch.float64, device=table.device
    )
    for i in range(9):
        for j in range(i, 9):
            value = response_covariance(
                centers[:, i],
                component_weights,
                centers[:, j],
                component_weights,
                table,
                radius,
                target_entries,
            )
            covariance[:, i, j] = value
            covariance[:, j, i] = value

    variance = covariance.diagonal(dim1=1, dim2=2)
    stddev_product = torch.sqrt(
        variance[:, :, None] * variance[:, None, :]
    )
    pair_weights = torch.triu(
        pass_weights[:, None] * pass_weights[None, :], diagonal=1
    )
    difference = ratio_with_block_standard_error(
        (covariance[:, 0, 1:] * pass_weights[1:]).sum(dim=1),
        (stddev_product[:, 0, 1:] * pass_weights[1:]).sum(dim=1),
    )
    propagation = ratio_with_block_standard_error(
        (covariance * pair_weights).sum(dim=(1, 2)),
        (stddev_product * pair_weights).sum(dim=(1, 2)),
    )
    return {
        "difference_correlation": difference[0],
        "difference_standard_error": difference[1],
        "propagation_correlation": propagation[0],
        "propagation_standard_error": propagation[1],
        "sampled_pixels": sample_count,
    }


def final_adjacent_statistics(
    sample_count: int,
    coordinate_min: int,
    coordinate_max: int,
    seed: int,
    pass_weights: torch.Tensor,
    table: torch.Tensor,
    radius: int,
    target_entries: int,
) -> Dict[str, object]:
    generator = torch.Generator(device=table.device).manual_seed(seed)
    points = torch.randint(
        coordinate_min,
        coordinate_max,
        (sample_count, 2),
        generator=generator,
        device=table.device,
    )
    centers, component_weights = response_components(
        points, (8, 16, 32), pass_weights
    )
    variance = response_covariance(
        centers,
        component_weights,
        centers,
        component_weights,
        table,
        radius,
        target_entries,
    )
    correlations: Dict[str, torch.Tensor] = {}
    for name, displacement in (
        ("positive_x", (1, 0)),
        ("positive_y", (0, 1)),
        ("positive_diagonal", (1, 1)),
        ("negative_diagonal", (1, -1)),
    ):
        displaced = points + points.new_tensor(displacement)
        displaced_centers, _ = response_components(
            displaced, (8, 16, 32), pass_weights
        )
        displaced_variance = response_covariance(
            displaced_centers,
            component_weights,
            displaced_centers,
            component_weights,
            table,
            radius,
            target_entries,
        )
        cross_covariance = response_covariance(
            centers,
            component_weights,
            displaced_centers,
            component_weights,
            table,
            radius,
            target_entries,
        )
        correlations[name] = cross_covariance / torch.sqrt(
            variance * displaced_variance
        )

    axial = 0.5 * (
        correlations["positive_x"] + correlations["positive_y"]
    )
    diagonal = 0.5 * (
        correlations["positive_diagonal"]
        + correlations["negative_diagonal"]
    )
    bilinear = 0.8 * axial + 0.2 * diagonal

    def summarize(values: torch.Tensor) -> Dict[str, float]:
        return {
            "correlation": float(values.mean()),
            "standard_error": float(
                values.std(unbiased=True) / math.sqrt(len(values))
            ),
        }

    return {
        "axial": summarize(axial),
        "diagonal": summarize(diagonal),
        "uniform_phase_bilinear": summarize(bilinear),
        "sampled_pixels": sample_count,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-8-samples", type=int, default=16384)
    parser.add_argument("--step-16-samples", type=int, default=8192)
    parser.add_argument("--step-32-samples", type=int, default=1024)
    parser.add_argument("--final-samples", type=int, default=1024)
    parser.add_argument("--coordinate-min", type=int, default=256)
    parser.add_argument("--coordinate-max", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=330127)
    parser.add_argument("--target-entries", type=int, default=4_000_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    counts = (
        args.step_8_samples,
        args.step_16_samples,
        args.step_32_samples,
        args.final_samples,
    )
    if min(counts) <= 0:
        raise SystemExit("all sample counts must be positive")
    if args.coordinate_min < 128 or args.coordinate_max <= args.coordinate_min:
        raise SystemExit("coordinate range must be a nonempty interior region")
    if args.target_entries <= 0:
        raise SystemExit("--target-entries must be positive")

    device = torch.device(args.device)
    grid_kernel, exact_statistics = build_grid_kernel_and_statistics()
    table, radius = autocorrelation_table(grid_kernel, device)
    pass_weights = torch.tensor(
        (1.0,) + tuple(point[2] for point in POISSON),
        dtype=torch.float64,
        device=device,
    )
    pass_weights /= pass_weights.sum()

    measurements: Dict[str, Dict[str, float]] = {}
    for step in STEPS[:3]:
        difference, propagation = exact_statistics[step]
        measurements[str(step)] = {
            "difference_correlation": difference,
            "difference_standard_error": 0.0,
            "propagation_correlation": propagation,
            "propagation_standard_error": 0.0,
            "sampled_pixels": 0,
        }

    large_config = (
        (8, (), args.step_8_samples),
        (16, (8,), args.step_16_samples),
        (32, (8, 16), args.step_32_samples),
    )
    for step, completed, sample_count in large_config:
        measurements[str(step)] = large_pass_statistics(
            step,
            completed,
            sample_count,
            args.coordinate_min,
            args.coordinate_max,
            args.seed + step * 1009,
            pass_weights,
            table,
            radius,
            args.target_entries,
        )

    final_adjacent = final_adjacent_statistics(
        args.final_samples,
        args.coordinate_min,
        args.coordinate_max,
        args.seed + 99991,
        pass_weights,
        table,
        radius,
        args.target_entries,
    )
    result = {
        "model": {
            "input_errors": "independent_equal_variance",
            "weights": "fixed_sampling_kernel_only",
            "depends_on_phi": False,
            "depends_on_maxent": False,
            "grid_kernel_coefficients": len(grid_kernel),
            "grid_kernel_energy": float(table[radius, radius]),
            "large_pass_hash": "shader_whash_per_pixel",
        },
        "configuration": {
            "coordinate_min": args.coordinate_min,
            "coordinate_max": args.coordinate_max,
            "seed": args.seed,
            "device": str(device),
        },
        "passes": measurements,
        "final_output_adjacent": final_adjacent,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
