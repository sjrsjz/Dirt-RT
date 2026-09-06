#!/usr/bin/env python3
"""Heuristically tune pass-specific moment-correlation closures.

Each training tile has one analytic, spatially constant MaxEnt target.  Its
input pixels are independent N-spp importance-sampling estimates of the same
discrete incident-radiance measure.  The model shares the shader's 2x2 PSD
Bures distance, alpha=1 g^-3 MC-variance closure, six A-Trous levels, and
per-pass equicorrelation covariance expansion.  Geometry weights are
deliberately omitted.

This legacy optimization model propagates estimator variance directly and
uses a difference-correlation term, unlike the current shader's separate
MC-variance/Kish fields and cumulative rejection scales. Use
calibrate_bures_pass_correlations.py for current runtime coefficients.
The objective here includes filtering bias and only provides an experimental
engineering tuning result for this legacy operator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


STEPS = (1, 2, 4, 8, 16, 32)
HISTORIES = (1, 2, 4, 8, 12, 24, 32)
DEFAULT_DIFFERENCE_CORRELATIONS = (
    0.0, 0.1633, 0.2171, 0.2508, 0.2585, 0.2665
)
DEFAULT_PROPAGATION_CORRELATIONS = (
    0.0, 0.09184833, 0.12613998, 0.13781854, 0.14262104, 0.14655028
)
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


@dataclass
class Batch:
    mean: torch.Tensor
    mean_y2: torch.Tensor
    target: torch.Tensor
    history: int


def quantize_half(value: torch.Tensor) -> torch.Tensor:
    return value.clamp(-65504.0, 65504.0).to(torch.float16).to(torch.float32)


def safe_normalize(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=dim, keepdim=True).clamp_min(1.0e-20))


def exact_forward_smooth_sqrt(
    value: torch.Tensor, epsilon: float = 1.0e-12
) -> torch.Tensor:
    """Exact nonnegative sqrt values with finite optimization gradients at zero."""
    value = value.clamp_min(0.0)
    exact = value.sqrt()
    differentiable = (value + epsilon).sqrt() - math.sqrt(epsilon)
    return exact.detach() + differentiable - differentiable.detach()


def canonicalize_mean(mean: torch.Tensor) -> torch.Tensor:
    direction = mean[:, :3]
    energy = mean[:, 3:4].clamp_min(0.0)
    direction_length = direction.square().sum(dim=1, keepdim=True).clamp_min(1.0e-30).sqrt()
    scale = torch.where(direction_length > energy, energy / direction_length.clamp_min(1.0e-20), 1.0)
    return torch.cat((direction * scale, energy), dim=1)


def bures_distance_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable 2x2 PSD Bures--Wasserstein distance for channel-first states."""
    a = canonicalize_mean(a)
    b = canonicalize_mean(b)
    va, wa = a[:, :3], a[:, 3:4]
    vb, wb = b[:, :3], b[:, 3:4]

    def cone_radius(vector: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        slack = (energy.square() - vector.square().sum(dim=1, keepdim=True)).clamp_min(0.0)
        exact = slack.sqrt()
        epsilon = (energy.square() * 1.0e-12).clamp_min(1.0e-30)
        differentiable = (slack + epsilon).sqrt() - epsilon.sqrt()
        # Keep the exact shader value in the forward pass and use the smooth
        # continuation only for training gradients at rank-one states.
        return exact.detach() + differentiable - differentiable.detach()

    qa = cone_radius(va, wa)
    qb = cone_radius(vb, wb)
    root_affinity = exact_forward_smooth_sqrt(
        0.5 * (wa * wb + (va * vb).sum(dim=1, keepdim=True) + qa * qb)
    )
    numerator = (va - vb).square().sum(dim=1, keepdim=True) + (qa - qb).square()
    denominator = wa + wb + 2.0 * root_affinity
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(1.0e-30),
        torch.zeros_like(numerator),
    ).squeeze(1)


def g3_bures_biased_observation_variance(
    mean: torch.Tensor, mean_r2: torch.Tensor
) -> torch.Tensor:
    """Plug-in Bures observation variance from empirical R^2 and g^-3 angles."""
    mean = canonicalize_mean(mean)
    energy = mean[:, 3:4]
    energy2 = energy.square()
    mean_r2 = torch.maximum(mean_r2.clamp_min(0.0), energy2)
    root_mean_r2 = mean_r2.sqrt()
    kappa2 = (
        mean[:, :3].square().sum(dim=1, keepdim=True)
        / energy2.clamp_min(1.0e-30)
    ).clamp(0.0, 1.0)
    radial_variance = (root_mean_r2 - energy) * (root_mean_r2 + energy)
    angular_scale = 3.0 * (1.0 - kappa2) / (3.0 + kappa2)
    variance = (
        radial_variance + angular_scale * mean_r2
    ) / (4.0 * energy.clamp_min(1.0e-30))
    return torch.where(energy > 0.0, variance.clamp_min(0.0), torch.zeros_like(variance))


def circular_filter(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = value.shape[1]
    radius_y = kernel.shape[-2] // 2
    radius_x = kernel.shape[-1] // 2
    padded = F.pad(value, (radius_x, radius_x, radius_y, radius_y), mode="circular")
    expanded_kernel = kernel.expand(channels, 1, -1, -1)
    return F.conv2d(padded, expanded_kernel, groups=channels)


def tile_local_variance_blur(variance: torch.Tensor) -> torch.Tensor:
    """Mirror the shader's 3x3 blur, which does not cross 16x16 workgroups."""
    batch, channels, height, width = variance.shape
    if channels != 1 or height % 16 != 0 or width % 16 != 0:
        raise ValueError("variance tiles require one channel and dimensions divisible by 16")
    tiles_y = height // 16
    tiles_x = width // 16
    tiles = variance.reshape(batch, 1, tiles_y, 16, tiles_x, 16)
    tiles = tiles.permute(0, 2, 4, 1, 3, 5).reshape(-1, 1, 16, 16)
    one_d = variance.new_tensor((0.6065306597, 1.0, 0.6065306597))
    kernel = one_d[:, None] * one_d[None, :]
    kernel = kernel.reshape(1, 1, 3, 3)
    numerator = F.conv2d(tiles, kernel, padding=1)
    denominator = F.conv2d(torch.ones_like(tiles), kernel, padding=1)
    tiles = numerator / denominator.clamp_min(1.0e-8)
    result = tiles.reshape(batch, tiles_y, tiles_x, 1, 16, 16)
    return result.permute(0, 3, 1, 4, 2, 5).reshape(batch, 1, height, width)


def prepare_variance(batch: Batch, simulate_fp16: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = canonicalize_mean(batch.mean)
    mean_y2 = batch.mean_y2.clamp_min(0.0)
    if simulate_fp16:
        mean = canonicalize_mean(quantize_half(mean))
        mean_y2 = quantize_half(mean_y2.sqrt()).square()

    mean_y2 = torch.maximum(mean_y2, mean[:, 3:4].square())
    population = g3_bures_biased_observation_variance(mean, mean_y2)
    if batch.history > 1:
        temporal = population / float(batch.history - 1)
    else:
        temporal = torch.zeros_like(population)

    sigma = 1.25
    positions = torch.arange(-3, 4, device=mean.device, dtype=mean.dtype)
    one_d = torch.exp(-0.5 * positions.square() / (sigma * sigma))
    pool_kernel = (one_d[:, None] * one_d[None, :])
    pool_kernel = (pool_kernel / pool_kernel.sum()).reshape(1, 1, 7, 7)
    pooled_mean = canonicalize_mean(circular_filter(mean, pool_kernel))
    pooled_y2 = circular_filter(mean_y2, pool_kernel)
    pooled_y2 = torch.maximum(pooled_y2, pooled_mean[:, 3:4].square())
    pooled_population = g3_bures_biased_observation_variance(
        pooled_mean, pooled_y2
    )
    spatial = pooled_population / float(max(batch.history, 1))

    trust_linear = min(max((batch.history - 2.0) / 10.0, 0.0), 1.0)
    temporal_trust = trust_linear * trust_linear * (3.0 - 2.0 * trust_linear)
    if batch.history <= 1:
        temporal_trust = 0.0
    estimator_sigma = (1.0 - temporal_trust) * spatial.sqrt() + temporal_trust * temporal.sqrt()
    variance = tile_local_variance_blur(estimator_sigma.square())
    return mean, variance


class AtrousModel:
    def __init__(self, height: int, width: int, device: torch.device, phi: float = 0.35):
        self.height = height
        self.width = width
        self.device = device
        self.phi = phi
        self._indices: Dict[Tuple[int, int], torch.Tensor] = {}
        self._build_small_indices()

    def _indices_from_offsets(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij",
        )
        sample_x = (xx.unsqueeze(0) + dx) % self.width
        sample_y = (yy.unsqueeze(0) + dy) % self.height
        return (sample_y * self.width + sample_x).reshape(dx.shape[0], -1).long()

    def _build_small_indices(self) -> None:
        base = torch.tensor(GRID_OFFSETS, device=self.device, dtype=torch.long)
        for step in STEPS[:3]:
            dx = (base[:, 0] * step).reshape(-1, 1, 1)
            dy = (base[:, 1] * step).reshape(-1, 1, 1)
            self._indices[(step, 0)] = self._indices_from_offsets(dx, dy)

    def _large_indices(self, step: int) -> torch.Tensor:
        key = (step, 0)
        if key in self._indices:
            return self._indices[key]

        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.int64),
            torch.arange(self.width, device=self.device, dtype=torch.int64),
            indexing="ij",
        )
        mask = 0xFFFFFFFF
        seed = (
            ((xx * 0x9E3779B9) & mask)
            ^ ((yy * 0x85EBCA6B) & mask)
            ^ ((step * 0xC2B2AE35) & mask)
        )
        seed = ((seed ^ 61) ^ (seed >> 16)) & mask
        seed = (seed * 9) & mask
        seed = (seed ^ (seed >> 4)) & mask
        seed = (seed * 0x27D4EB2D) & mask
        seed = (seed ^ (seed >> 15)) & mask
        angle = 2.0 * math.pi * (
            (seed >> 8).to(torch.float32) * (1.0 / 16777216.0)
        )
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        poisson = torch.tensor(POISSON, device=self.device, dtype=torch.float32)
        source_x = poisson[:, 0, None, None]
        source_y = poisson[:, 1, None, None]
        scale = float(step) * 1.75
        # GLSL mat2 is column-major.  This matches mat2(cs,-sn,sn,cs) * p.
        rotated_x = scale * (
            cosine.unsqueeze(0) * source_x + sine.unsqueeze(0) * source_y
        )
        rotated_y = scale * (
            -sine.unsqueeze(0) * source_x + cosine.unsqueeze(0) * source_y
        )
        dx = torch.sign(rotated_x) * torch.floor(torch.abs(rotated_x) + 0.5)
        dy = torch.sign(rotated_y) * torch.floor(torch.abs(rotated_y) + 0.5)
        self._indices[key] = self._indices_from_offsets(dx.long(), dy.long())
        return self._indices[key]

    @staticmethod
    def _gather_neighbors(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        taps, pixels = indices.shape
        flat = value.reshape(batch, channels, 1, height * width).expand(-1, -1, taps, -1)
        gather_indices = indices.reshape(1, 1, taps, pixels).expand(batch, channels, -1, -1)
        return torch.gather(flat, 3, gather_indices).reshape(batch, channels, taps, height, width)

    def _filter_level(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        difference_correlation: torch.Tensor,
        propagation_correlation: torch.Tensor,
        step: int,
        frame: int,
        simulate_fp16: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._indices[(step, 0)] if step <= 4 else self._large_indices(step)
        kernel_weights = GRID_WEIGHTS if step <= 4 else tuple(row[2] for row in POISSON)
        kernel = mean.new_tensor(kernel_weights).reshape(1, 1, -1, 1, 1)
        neighbor_mean = self._gather_neighbors(mean, indices)
        neighbor_variance = self._gather_neighbors(variance, indices)

        distance2 = bures_distance_squared(mean.unsqueeze(2), neighbor_mean)

        center_variance = variance[:, 0].unsqueeze(1)
        sample_variance = neighbor_variance[:, 0]
        variance_sum = (
            center_variance
            + sample_variance
            - 2.0 * difference_correlation
            * (center_variance * sample_variance).clamp_min(0.0).sqrt()
        ).clamp_min(0.0)
        normalized_distance = exact_forward_smooth_sqrt(
            distance2 / variance_sum.clamp_min(1.0e-12)
        )
        weight = kernel[:, 0] * torch.exp(-self.phi * normalized_distance)
        weight = torch.where(weight > 1.0e-6, weight, 0.0)
        weight_sum = 1.0 + weight.sum(dim=1, keepdim=True)
        inverse_weight = weight_sum.reciprocal()

        output_mean = (
            mean + (neighbor_mean * weight.unsqueeze(1)).sum(dim=2)
        ) * inverse_weight
        squared_weight_variance = variance + (
            neighbor_variance * weight.square().unsqueeze(1)
        ).sum(dim=2)
        weighted_stddev = exact_forward_smooth_sqrt(variance) + (
            exact_forward_smooth_sqrt(neighbor_variance) * weight.unsqueeze(1)
        ).sum(dim=2)
        output_variance = (
            (1.0 - propagation_correlation) * squared_weight_variance
            + propagation_correlation * weighted_stddev.square()
        ) * inverse_weight.square()
        output_mean = canonicalize_mean(output_mean)
        if simulate_fp16:
            output_mean = canonicalize_mean(quantize_half(output_mean))
        return output_mean, output_variance.clamp(0.0, 1.0e30)

    def run(
        self,
        batch: Batch,
        difference_correlations: torch.Tensor,
        propagation_correlations: torch.Tensor,
        frame: int,
        simulate_fp16: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        mean, variance = prepare_variance(batch, simulate_fp16)
        levels: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for level, step in enumerate(STEPS):
            mean, variance = self._filter_level(
                mean, variance, difference_correlations[level],
                propagation_correlations[level], step, frame, simulate_fp16
            )
            levels.append((mean, variance))
        return mean, variance, levels


def make_batch(
    batch_size: int,
    height: int,
    width: int,
    history: int,
    atoms: int,
    device: torch.device,
    generator: torch.Generator,
) -> Batch:
    # A shared lobe axis per tile, with concentration spanning isotropic to
    # almost directional targets.
    axis = safe_normalize(
        torch.randn(batch_size, 3, device=device, generator=generator), dim=1
    )
    concentration_choices = torch.tensor(
        (0.0, 0.25, 0.75, 2.0, 6.0, 20.0), device=device
    )
    concentration_index = torch.randint(
        0, len(concentration_choices), (batch_size,), device=device, generator=generator
    )
    concentration = concentration_choices[concentration_index].reshape(batch_size, 1, 1)
    directions = safe_normalize(
        torch.randn(batch_size, atoms, 3, device=device, generator=generator)
        + concentration * axis.unsqueeze(1),
        dim=2,
    )

    spread = 0.15 + 2.35 * torch.rand(
        batch_size, 1, device=device, generator=generator
    )
    contribution = torch.softmax(
        torch.randn(batch_size, atoms, device=device, generator=generator) * spread,
        dim=1,
    )
    proposal_exponent = -0.75 + 2.0 * torch.rand(
        batch_size, 1, device=device, generator=generator
    )
    raw_proposal = contribution.clamp_min(1.0e-8).pow(proposal_exponent)
    raw_proposal = raw_proposal / raw_proposal.sum(dim=1, keepdim=True)
    uniform_floor = torch.exp(
        math.log(0.02)
        + (math.log(0.30) - math.log(0.02))
        * torch.rand(batch_size, 1, device=device, generator=generator)
    )
    proposal = (1.0 - uniform_floor) * raw_proposal + uniform_floor / float(atoms)

    sample_count = height * width * history
    indices = torch.multinomial(
        proposal, sample_count, replacement=True, generator=generator
    )
    sampled_contribution = torch.gather(contribution, 1, indices)
    sampled_probability = torch.gather(proposal, 1, indices)
    sample_y = sampled_contribution / sampled_probability
    sampled_direction = torch.gather(
        directions,
        1,
        indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    sample_directional_moment = sampled_direction * sample_y.unsqueeze(-1)
    sample_directional_moment = sample_directional_moment.reshape(
        batch_size, height, width, history, 3
    ).mean(dim=3)
    sample_energy = sample_y.reshape(batch_size, height, width, history).mean(dim=3)
    mean_y2 = sample_y.square().reshape(
        batch_size, height, width, history
    ).mean(dim=3)
    mean = torch.cat((sample_directional_moment, sample_energy.unsqueeze(-1)), dim=-1)
    mean = mean.permute(0, 3, 1, 2).contiguous()
    mean_y2 = mean_y2.unsqueeze(1)

    target_direction = (contribution.unsqueeze(-1) * directions).sum(dim=1)
    target = torch.cat(
        (target_direction, torch.ones(batch_size, 1, device=device)), dim=1
    ).reshape(batch_size, 4, 1, 1)
    return Batch(mean=mean, mean_y2=mean_y2, target=target, history=history)


def normalized_level_metrics(
    batch: Batch,
    levels: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch.target
    input_error = bures_distance_squared(batch.mean, target).mean().detach().clamp_min(1.0e-12)
    errors = torch.stack(
        [bures_distance_squared(mean, target).mean() for mean, _ in levels]
    )
    predicted = torch.stack([variance.mean() for _, variance in levels])
    mse_match = (torch.log(predicted.clamp_min(1.0e-12)) - torch.log(errors.clamp_min(1.0e-12))).square().mean()
    final_residual = levels[-1][0] - target
    patch_loss = final_residual.new_zeros(())
    for size in (4, 8, 16):
        pooled = F.avg_pool2d(final_residual, kernel_size=size, stride=size)
        patch_loss = patch_loss + pooled.square().sum(dim=1).mean() / input_error
    signal_loss = errors[-1] / input_error
    return signal_loss, mse_match, patch_loss / 3.0


def decode_correlations(raw: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(raw)


def evaluate(
    model: AtrousModel,
    difference_correlations: torch.Tensor,
    propagation_correlations: torch.Tensor,
    histories: Iterable[int],
    seeds: Iterable[int],
    batch_size: int,
    atoms: int,
    simulate_fp16: bool,
) -> dict:
    per_history = {}
    all_signal: List[float] = []
    all_patch: List[float] = []
    all_predicted = [[] for _ in STEPS]
    all_errors = [[] for _ in STEPS]
    with torch.no_grad():
        for history in histories:
            history_signal = []
            history_patch = []
            for seed in seeds:
                generator = torch.Generator(device=model.device).manual_seed(
                    seed + history * 100003
                )
                batch = make_batch(
                    batch_size,
                    model.height,
                    model.width,
                    history,
                    atoms,
                    model.device,
                    generator,
                )
                _, _, levels = model.run(
                    batch, difference_correlations, propagation_correlations,
                    frame=seed % 8, simulate_fp16=simulate_fp16
                )
                signal, _, patch = normalized_level_metrics(batch, levels)
                history_signal.append(float(signal))
                history_patch.append(float(patch))
                for level, (mean, variance) in enumerate(levels):
                    error = bures_distance_squared(mean, batch.target).mean()
                    all_errors[level].append(float(error))
                    all_predicted[level].append(float(variance.mean()))
            per_history[str(history)] = {
                "final_mse_over_input": sum(history_signal) / len(history_signal),
                "low_frequency_loss": sum(history_patch) / len(history_patch),
            }
            all_signal.extend(history_signal)
            all_patch.extend(history_patch)

    variance_ratios = [
        sum(predicted) / max(sum(errors), 1.0e-20)
        for predicted, errors in zip(all_predicted, all_errors)
    ]
    return {
        "mean_final_mse_over_input": sum(all_signal) / len(all_signal),
        "mean_low_frequency_loss": sum(all_patch) / len(all_patch),
        "variance_over_empirical_mse_by_step": dict(zip(map(str, STEPS), variance_ratios)),
        "per_history": per_history,
    }


def measure_bias(
    model: AtrousModel,
    difference_correlations: torch.Tensor,
    propagation_correlations: torch.Tensor,
    histories: Iterable[int],
    seeds: Iterable[int],
    batch_size: int,
    atoms: int,
    simulate_fp16: bool,
) -> dict:
    """Estimate ensemble bias after aligning each field to its target axis."""

    def mean_and_standard_error(values: torch.Tensor) -> Tuple[float, float]:
        count = values.numel()
        mean = values.mean()
        if count <= 1:
            return float(mean), 0.0
        standard_error = values.std(unbiased=True) / math.sqrt(float(count))
        return float(mean), float(standard_error)

    result = {}
    with torch.no_grad():
        for history in histories:
            energy_biases = []
            axial_biases = []
            moment_length_biases = []
            field_mean_residual2 = []
            pixel_mses = []
            input_mses = []
            for seed in seeds:
                generator = torch.Generator(device=model.device).manual_seed(
                    seed + history * 100003
                )
                batch = make_batch(
                    batch_size,
                    model.height,
                    model.width,
                    history,
                    atoms,
                    model.device,
                    generator,
                )
                output, _, _ = model.run(
                    batch, difference_correlations, propagation_correlations,
                    frame=seed % 8,
                    simulate_fp16=simulate_fp16
                )
                target = batch.target[:, :, 0, 0]
                field_mean = output.mean(dim=(2, 3))
                residual = field_mean - target
                target_axis = safe_normalize(target[:, :3], dim=1)
                energy_biases.append(residual[:, 3])
                axial_biases.append((residual[:, :3] * target_axis).sum(dim=1))
                moment_length_biases.append(
                    field_mean[:, :3].norm(dim=1) - target[:, :3].norm(dim=1)
                )
                field_mean_residual2.append(residual.square().sum(dim=1))
                pixel_mses.append(
                    (output - batch.target).square().sum(dim=1).mean(dim=(1, 2))
                )
                input_mses.append(
                    (batch.mean - batch.target).square().sum(dim=1).mean(dim=(1, 2))
                )

            energy = torch.cat(energy_biases).cpu()
            axial = torch.cat(axial_biases).cpu()
            moment_length = torch.cat(moment_length_biases).cpu()
            field_residual2 = torch.cat(field_mean_residual2).cpu()
            pixel_mse = torch.cat(pixel_mses).cpu()
            input_mse = torch.cat(input_mses).cpu()
            energy_mean, energy_se = mean_and_standard_error(energy)
            axial_mean, axial_se = mean_and_standard_error(axial)
            length_mean, length_se = mean_and_standard_error(moment_length)
            systematic_bias2 = energy_mean * energy_mean + axial_mean * axial_mean
            result[str(history)] = {
                "energy_bias": energy_mean,
                "energy_bias_standard_error": energy_se,
                "axial_moment_bias": axial_mean,
                "axial_moment_bias_standard_error": axial_se,
                "moment_length_bias": length_mean,
                "moment_length_bias_standard_error": length_se,
                "aligned_maxent_bias_norm": math.sqrt(systematic_bias2),
                "rms_field_mean_residual": float(field_residual2.mean().sqrt()),
                "bias_squared_over_output_mse": systematic_bias2
                / max(float(pixel_mse.mean()), 1.0e-20),
                "output_mse_over_input": float(pixel_mse.mean() / input_mse.mean()),
                "fields": int(energy.numel()),
            }
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.008)
    parser.add_argument("--signal-weight", type=float, default=0.0)
    parser.add_argument("--patch-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=991337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-fp16-simulation", action="store_true")
    parser.add_argument("--validation-seeds", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.size % 16 != 0:
        raise SystemExit("--size must be divisible by 16")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    simulate_fp16 = not args.no_fp16_simulation
    model = AtrousModel(args.size, args.size, device, phi=args.phi)

    initial_difference = torch.tensor(
        DEFAULT_DIFFERENCE_CORRELATIONS, device=device)
    initial_propagation = torch.tensor(
        DEFAULT_PROPAGATION_CORRELATIONS, device=device)
    raw_difference = torch.nn.Parameter(
        torch.logit(initial_difference.clamp(1.0e-4, 1.0 - 1.0e-4))
    )
    raw_propagation = torch.nn.Parameter(
        torch.logit(initial_propagation.clamp(1.0e-4, 1.0 - 1.0e-4))
    )
    optimizer = torch.optim.Adam(
        (raw_difference, raw_propagation), lr=args.learning_rate)

    for iteration in range(1, args.iterations + 1):
        history = HISTORIES[(iteration - 1) % len(HISTORIES)]
        batch = make_batch(
            args.batch_size,
            args.size,
            args.size,
            history,
            args.atoms,
            device,
            generator,
        )
        difference_correlations = decode_correlations(raw_difference)
        propagation_correlations = decode_correlations(raw_propagation)
        _, _, levels = model.run(
            batch,
            difference_correlations,
            propagation_correlations,
            frame=iteration % 8,
            simulate_fp16=simulate_fp16,
        )
        # This MSE-matching term contains estimator bias.  It can tune the
        # engineering default but must not be reported as an estimate of the
        # trace-correlation coefficient itself.
        signal_loss, mse_match_loss, patch_loss = normalized_level_metrics(batch, levels)
        loss = mse_match_loss + args.signal_weight * signal_loss \
            + args.patch_weight * patch_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_difference, raw_propagation), 5.0)
        optimizer.step()

        if iteration == 1 or iteration % 25 == 0 or iteration == args.iterations:
            difference_values = ",".join(
                f"{value:.6f}"
                for value in difference_correlations.detach().cpu().tolist()
            )
            propagation_values = ",".join(
                f"{value:.6f}"
                for value in propagation_correlations.detach().cpu().tolist()
            )
            print(
                f"iter={iteration:4d} N={history:2d} loss={float(loss):.6f} "
                f"signal={float(signal_loss):.6f} mse_match={float(mse_match_loss):.6f} "
                f"patch={float(patch_loss):.6f} "
                f"p_diff=[{difference_values}] p_prop=[{propagation_values}]",
                flush=True,
            )

    trained_difference = decode_correlations(raw_difference).detach()
    trained_propagation = decode_correlations(raw_propagation).detach()
    validation_seeds = tuple(args.seed + 900001 + i * 7919 for i in range(args.validation_seeds))
    original_metrics = evaluate(
        model,
        initial_difference,
        initial_propagation,
        HISTORIES,
        validation_seeds,
        args.batch_size,
        args.atoms,
        simulate_fp16,
    )
    trained_metrics = evaluate(
        model,
        trained_difference,
        trained_propagation,
        HISTORIES,
        validation_seeds,
        args.batch_size,
        args.atoms,
        simulate_fp16,
    )
    result = {
        "configuration": {
            "iterations": args.iterations,
            "size": args.size,
            "batch_size": args.batch_size,
            "atoms": args.atoms,
            "histories": HISTORIES,
            "phi": model.phi,
            "simulate_fp16": simulate_fp16,
            "seed": args.seed,
            "validation_seeds": validation_seeds,
            "signal_weight": args.signal_weight,
            "patch_weight": args.patch_weight,
        },
        "initial_difference_correlations": dict(zip(
            map(str, STEPS), initial_difference.cpu().tolist())),
        "initial_propagation_correlations": dict(zip(
            map(str, STEPS), initial_propagation.cpu().tolist())),
        "trained_difference_correlations": dict(zip(
            map(str, STEPS), trained_difference.cpu().tolist())),
        "trained_propagation_correlations": dict(zip(
            map(str, STEPS), trained_propagation.cpu().tolist())),
        "original_metrics": original_metrics,
        "trained_metrics": trained_metrics,
    }
    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
