#!/usr/bin/env python3
"""Calibrate frozen Bures-weight overlap on a homogeneous front-facing plane.

The pilot moments generate exactly the runtime signal weights. Independent
Rademacher fields measure the resulting linear operator's variance. Fit the
runtime recursive Kish closure, with disjoint pilot/probe seeds for holdout.
Actual current-sample errors are reported separately, including selection bias.
No geometry rejection, extra runtime state, or changes to signal weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from estimate_maxent_pass_correlations import STEPS, GRID_OFFSETS, GRID_WEIGHTS, POISSON, sample_points

ROOT = Path(__file__).resolve().parents[1]
# Frozen comparison from estimate_maxent_pass_correlations.py, seed 330127.
# Keep this baseline stable when the runtime constants adopt calibrated values.
FIXED_KERNEL_REFERENCE = (0., .10600873, .14116979, .13941258, .14705113, .15253123)


def settings():
    paths = [ROOT / 'shaders/lib/settings.glsl',
             ROOT / 'shaders/lib/lighting/denoiser/internal_constants.glsl']
    source = '\n'.join(p.read_text(encoding='utf-8') for p in paths)

    def number(name):
        match = re.search(r'(?:#define\s+|const float\s+)' + name
                          + r'\s*(?:=\s*)?([0-9.eE+-]+)', source)
        if match is None:
            raise ValueError(name)
        return float(match[1])

    return {
        'phi': number('MAXENT_SPATIAL_DIFFUSE_LIGHT_FIELD_SENSITIVITY'),
        'specular_phi': number('MAXENT_SPATIAL_SPECULAR_LIGHT_FIELD_SENSITIVITY'),
        'sigma': number('MAXENT_VARIANCE_KERNEL_SIGMA'),
        'spatial_only': number('MAXENT_VARIANCE_SPATIAL_ONLY_SAMPLES'),
        'transition': number('MAXENT_VARIANCE_TRANSITION_SAMPLES'),
        'confidence': [number(f'MAXENT_SPATIAL_REJECTION_CONFIDENCE_STEP_{s}') for s in STEPS],
        'old_p': list(FIXED_KERNEL_REFERENCE),
    }


def half(x):
    return x.clamp(-65504, 65504).half().float()


def bures(a, b):
    def parts(x):
        w = x[3].clamp_min(0)
        v = x[:3]
        length = v.square().sum(0).sqrt()
        v = v * torch.minimum(torch.ones_like(w), w / length.clamp_min(1e-20))
        q = (w.square() - v.square().sum(0)).clamp_min(0).sqrt()
        return v, w, q
    va, wa, qa = parts(a)
    vb, wb, qb = parts(b)
    affinity = (.5 * (wa * wb + (va * vb).sum(0) + qa * qb)).clamp_min(0).sqrt()
    return ((va-vb).square().sum(0) + (qa-qb).square()) / (wa+wb+2*affinity).clamp_min(1e-20)


def variance(mean, rms, neff):
    w = mean[3].clamp_min(1e-20)
    rms = torch.maximum(rms, w)
    k2 = (mean[:3].square().sum(0)/w.square()).clamp(0, 1)
    value = ((rms-w)*(rms+w) + 3*(1-k2)/(3+k2)*rms.square())/(4*w)
    if isinstance(neff, (int, float)) and neff <= 1:
        return torch.zeros_like(w)
    return value / (1-1/neff)


def pool(x, kernel, side):
    channels = x.shape[0]
    x = x.reshape(1, channels, side, side)
    return F.conv2d(F.pad(x, (3, 3, 3, 3), mode='circular'),
                    kernel.expand(channels, 1, 7, 7), groups=channels).reshape(channels, -1)


def generate(kappa, history, seed, args, config, phi):
    device = args.device
    gen = torch.Generator(device=device).manual_seed(seed)
    count = args.side**2
    # alpha=1 joint density: U has g^-2 angular law; R|U is Gamma(2,1/(a*g)).
    # a=2/(1-k^2) gives E[R]=1 and E[Ru]=(0,0,k).
    mean = torch.zeros(4, count, device=device)
    e2 = torch.zeros(count, device=device)
    for _ in range(history):
        t = 2*torch.rand(count, device=device, generator=gen)-1
        mu = (t+kappa)/(1+kappa*t)
        azimuth = 2*math.pi*torch.rand(count, device=device, generator=gen)
        sine = (1-mu.square()).clamp_min(0).sqrt()
        uniforms = torch.rand(2, count, device=device, generator=gen).clamp_min(1e-20)
        r = -uniforms.log().sum(0)*(1-kappa*kappa)/(2*(1-kappa*mu))
        current = torch.stack((sine*azimuth.cos()*r, sine*azimuth.sin()*r, mu*r, r))
        mean += current/history
        e2 += r.square()/history
    mean = half(mean)
    rms = half(e2.sqrt())
    current = half(current)
    positions = torch.arange(-3, 4, device=device)
    one = torch.exp(-.5*positions.square()/config['sigma']**2)
    kernel = one[:, None]*one[None, :]
    kernel = (kernel/kernel.sum()).reshape(1, 1, 7, 7)
    moments = pool(torch.cat((mean, rms.square()[None]), 0), kernel, args.side)
    pooled_n = history/float(kernel.square().sum())
    spatial = variance(moments[:4], moments[4].clamp_min(0).sqrt(), pooled_n)
    temporal = variance(mean, rms, history)
    trust = min(max((history-config['spatial_only'])/config['transition'], 0), 1)
    trust = trust*trust*(3-2*trust)
    prepared = half(((1-trust)*spatial+trust*temporal).sqrt()).square()
    current_var = half(spatial.sqrt()).square()
    # Probe RNG is independent of pilot/current samples. No probe quantization:
    # quantization is model error, not MC kernel-overlap correlation.
    pg = torch.Generator(device=device).manual_seed(seed+100000003)
    probes = (torch.randint(0, 2, (args.probes, count), device=device, generator=pg)*2-1).float()
    return dict(mean=mean, variance=prepared, current=current, current_var=current_var,
                probes=probes, fitted_inv_n=torch.ones(count, device=device),
                old_inv_n=torch.ones(count, device=device), kappa=kappa,
                history=history, seed=seed, phi=phi)


def indices(side, step, device):
    y, x = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing='ij')
    points = torch.stack((x, y), -1).reshape(-1, 2)
    if step <= 4:
        taps = points[:, None] + points.new_tensor(GRID_OFFSETS)[None]*step
    else:
        taps = sample_points(points, step)[:, 1:]
    return ((taps[..., 1] % side)*side + taps[..., 0] % side).T.contiguous()


def tangent_square(delta, kappa):
    return (delta[:2].square().sum(0)/4
            + (delta[3]+delta[2]).square()/(8*(1+kappa))
            + (delta[3]-delta[2]).square()/(8*(1-kappa)))


def advance(state, taps, level, config):
    mean, v = state['mean'], state['variance']
    kernel = GRID_WEIGHTS if level < 3 else tuple(p[2] for p in POISSON)
    weights = [torch.ones_like(v)]
    n = min(state['history'], 16)
    for i in range(8):
        neighbor = taps[i]
        distance = bures(mean, mean[:, neighbor])
        denominator = (v+v[neighbor])/n
        weights.append(kernel[i]*torch.exp(-state['phi']*config['confidence'][level]
                                           *(distance/denominator.clamp_min(1e-20)).sqrt()))
    weights = torch.stack(weights)
    weights /= weights.sum(0)
    w0 = weights[0]
    out_mean = mean*w0
    out_var = v*w0
    out_current = state['current']*w0
    out_current_var = state['current_var']*w0
    out_probe = state['probes']*w0
    d_fit = state['fitted_inv_n']*w0.square()
    b_fit = state['fitted_inv_n'].sqrt()*w0
    d_old = state['old_inv_n']*w0.square()
    b_old = state['old_inv_n'].sqrt()*w0
    for i in range(8):
        ix, wi = taps[i], weights[i+1]
        out_mean += mean[:, ix]*wi
        out_var += v[ix]*wi
        out_current += state['current'][:, ix]*wi
        out_current_var += state['current_var'][ix]*wi
        out_probe += state['probes'][:, ix]*wi
        d_fit += state['fitted_inv_n'][ix]*wi.square()
        b_fit += state['fitted_inv_n'][ix].sqrt()*wi
        d_old += state['old_inv_n'][ix]*wi.square()
        b_old += state['old_inv_n'][ix].sqrt()*wi
    cross_fit = (b_fit.square()-d_fit).clamp_min(0)
    cross_old = (b_old.square()-d_old).clamp_min(0)
    # For pass 1 independent unit-variance probes have an exact analytic answer.
    oracle = weights.square().sum(0) if level == 0 else out_probe.square().mean(0)
    state.update(mean=half(out_mean), variance=half(out_var.sqrt()).square(),
                 current=half(out_current), current_var=half(out_current_var.sqrt()).square(),
                 probes=out_probe)
    return dict(d_fit=d_fit, cross_fit=cross_fit, d_old=d_old,
                cross_old=cross_old, oracle=oracle)


def run(split, args, config, coefficients=None):
    cases = []
    seed_offset = 0 if split == 'train' else 1000000
    repetitions = args.train_repeats if split == 'train' else args.holdout_repeats
    phis = sorted(set((config['phi'], config['specular_phi'])))
    for phi in phis:
        for k in (0., .7, .98):
            for n in (1, 4, 16, 64):
                for rep in range(repetitions):
                    seed = args.seed+seed_offset+len(cases)*1009
                    cases.append(generate(k, n, seed, args, config, phi))
    measured, rows, ensembles = [], [], []
    for level, step in enumerate(STEPS):
        taps = indices(args.side, step, args.device)
        pending = [advance(case, taps, level, config) for case in cases]
        numerator = sum(float((p['oracle']-p['d_fit']).double().mean()) for p in pending)
        denominator = sum(float(p['cross_fit'].double().mean()) for p in pending)
        raw_p = numerator/denominator
        chosen = (0.0 if level == 0 else min(max(raw_p, 0), 1)) if coefficients is None else coefficients[level]
        measured.append(chosen)
        for case, p in zip(cases, pending):
            case['fitted_inv_n'] = (p['d_fit']+chosen*p['cross_fit']).clamp(1/65504, 1)
            case['old_inv_n'] = (p['d_old']+config['old_p'][level]*p['cross_old']).clamp(1/65504, 1)
            # Match the runtime FP16 Kish field.
            case['fitted_inv_n'] = half(case['fitted_inv_n'].reciprocal()).reciprocal()
            case['old_inv_n'] = half(case['old_inv_n'].reciprocal()).reciprocal()
            target = case['current'].new_tensor((0., 0., case['kappa'], 1.))[:, None]
            residual = case['current']-target
            oracle = float(p['oracle'].double().mean())
            rows.append(dict(split=split, step=step, kappa=case['kappa'], history=case['history'],
                             seed=case['seed'], phi=case['phi'], coefficient=chosen,
                             probe_variance=oracle,
                             fit_diagonal=float(p['d_fit'].double().mean()),
                             fit_cross=float(p['cross_fit'].double().mean()),
                             predicted_fitted=float(case['fitted_inv_n'].double().mean()),
                             predicted_old=float(case['old_inv_n'].double().mean()),
                             current_bures_mse=float(bures(case['current'], target).double().mean()),
                             current_tangent_mse=float(tangent_square(residual, case['kappa']).double().mean()),
                             current_tangent_spatial_bias2=float(tangent_square(residual.mean(1, keepdim=True), case['kappa'])),
                             predicted_current_variance=float((case['current_var']*case['fitted_inv_n']).double().mean())))
        # Unbiased variance across independent repetitions at the SAME pixel,
        # retaining the spatially varying Poisson rotation. Pixelwise centering
        # removes nonlinear selection bias without interpreting it as covariance.
        if repetitions > 1:
            for phi in phis:
                for k in (0., .7, .98):
                    for n in (1, 4, 16, 64):
                        group = [c for c in cases if c['phi'] == phi and c['kappa'] == k and c['history'] == n]
                        group_mean = torch.stack([c['current'] for c in group]).mean(0)
                        central = sum(float(tangent_square(c['current']-group_mean, k).double().mean())
                                      for c in group)/(repetitions-1)
                        pred = sum(float((c['current_var']*c['fitted_inv_n']).double().mean()) for c in group)/repetitions
                        old = sum(float((c['current_var']*c['old_inv_n']).double().mean()) for c in group)/repetitions
                        ensembles.append(dict(split=split, step=step, kappa=k, history=n, phi=phi,
                                              repetitions=repetitions, centered_tangent_variance=central,
                                              predicted_fitted=pred, predicted_old=old))
        ratios = [r['predicted_fitted']/r['probe_variance'] for r in rows if r['step'] == step]
        print(f'{split} step={step:2d} p={chosen:.8f} ratio=[{min(ratios):.4f},{max(ratios):.4f}]', flush=True)
    return measured, rows, ensembles


def summarize(rows):
    result = {}
    for step in STEPS:
        subset = [r for r in rows if r['step'] == step]
        ref = sum(r['probe_variance'] for r in subset)
        ratios = [r['predicted_fitted']/r['probe_variance'] for r in subset]
        result[str(step)] = dict(
            fitted_over_reference=sum(r['predicted_fitted'] for r in subset)/ref,
            old_over_reference=sum(r['predicted_old'] for r in subset)/ref,
            per_case_ratio_min=min(ratios), per_case_ratio_max=max(ratios),
            mean_absolute_relative_error=sum(abs(r-1) for r in ratios)/len(ratios),
            current_predicted_over_tangent_mse=sum(r['predicted_current_variance'] for r in subset)
                /sum(r['current_tangent_mse'] for r in subset))
    return result


def summarize_ensembles(rows):
    result = {}
    first = {(r['kappa'], r['history'], r['phi']): r for r in rows if r['step'] == 1}
    for step in STEPS:
        subset = [r for r in rows if r['step'] == step]
        if not subset:
            continue
        ref = sum(r['centered_tangent_variance'] for r in subset)
        ratios = [r['predicted_fitted']/r['centered_tangent_variance'] for r in subset]
        result[str(step)] = dict(fitted_over_centered_mc=sum(r['predicted_fitted'] for r in subset)/ref,
                                old_over_centered_mc=sum(r['predicted_old'] for r in subset)/ref,
                                per_case_ratio_min=min(ratios), per_case_ratio_max=max(ratios),
                                propagation_only_fitted_ratio=sum(
                                    r['predicted_fitted'] * first[(r['kappa'],r['history'],r['phi'])]['centered_tangent_variance']
                                    /first[(r['kappa'],r['history'],r['phi'])]['predicted_fitted'] for r in subset)/ref,
                                propagation_only_old_ratio=sum(
                                    r['predicted_old'] * first[(r['kappa'],r['history'],r['phi'])]['centered_tangent_variance']
                                    /first[(r['kappa'],r['history'],r['phi'])]['predicted_old'] for r in subset)/ref)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--side', type=int, default=256)
    ap.add_argument('--probes', type=int, default=128)
    ap.add_argument('--train-repeats', type=int, default=4)
    ap.add_argument('--holdout-repeats', type=int, default=8)
    ap.add_argument('--seed', type=int, default=9062026)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', type=Path, default=ROOT/'doc/calibration/bures_plane_correlations.json')
    args = ap.parse_args()
    if args.side < 256 or min(args.probes, args.train_repeats, args.holdout_repeats) < 1:
        ap.error('side must be >=256 (avoid periodic support aliasing); sample counts must be positive')
    config = settings()
    with torch.no_grad():
        fitted, train, train_ensemble = run('train', args, config)
        _, holdout, holdout_ensemble = run('holdout', args, config, fitted)
    paths = ['shaders/lib/lighting/denoiser/atrous_filter.glsl',
             'shaders/lib/lighting/denoiser/variance_prepare.glsl',
             'shaders/lib/lighting/denoiser/atrous_small.glsl',
             'shaders/lib/lighting/denoiser/atrous_large.glsl',
             'tools/calibrate_bures_pass_correlations.py',
             'tools/estimate_maxent_pass_correlations.py']
    result = dict(configuration={**vars(args), 'output': str(args.output)}, shader=config,
                  device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
                  torch_version=torch.__version__, coefficients=fitted,
                  scope='recursive effective overlap of independent errors under frozen Bures weights; homogeneous plane',
                  nonlinear_error='current shares samples with pilot; reported separately, never fitted into overlap',
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  training=summarize(train), holdout=summarize(holdout), rows=train+holdout,
                  independent_mc_holdout=summarize_ensembles(holdout_ensemble),
                  ensembles=train_ensemble+holdout_ensemble)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['holdout'], indent=2))
    print('independent centered MC', json.dumps(result['independent_mc_holdout'], indent=2))
    print('saved', args.output)


if __name__ == '__main__':
    main()
