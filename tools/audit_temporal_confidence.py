"""Reproduce the experimental split-raw temporal resolve on homogeneous planes.

CPU algorithm audit, not a Minecraft capture or a GPU timing benchmark.
python -B tools/audit_temporal_confidence.py --compile
Both modes use the SAME split-current samples in the temporal comparison;
this intentionally isolates temporal policy from the changed spatial support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 906202609
def shader_number(relative, name):
    source = (ROOT / relative).read_text(encoding="utf-8-sig")
    pattern = rf'(?:#define\s+{name}\s+|const\s+float\s+{name}\s*=\s*)([0-9.eE+-]+)'
    return float(re.search(pattern, source)[1])


ALPHA = shader_number("shaders/lib/lighting/denoiser/internal_constants.glsl", "MAXENT_TEMPORAL_FIXED_ALPHA")
DELTA = shader_number("shaders/lib/settings.glsl", "MAXENT_TEMPORAL_CONFIDENCE_FAILURE_PROBABILITY")


def pack_half(x):
    return np.clip(x, -65504, 65504).astype(np.float16).astype(np.float32)


def weights():
    xy = np.array([(x, y) for y in range(-2, 3) for x in range(-2, 3)])
    a = np.exp(-0.5 * np.sum(xy * xy, axis=1))
    pilot = (np.abs(xy).sum(axis=1) % 2) == 1
    wa, wc = a * pilot, a * ~pilot
    return wa / wa.sum(), wc / wc.sum()


WA, WC = weights()
QA, QC = float(WA @ WA), float(WC @ WC)
XY = np.array([(x, y) for y in range(-2, 3) for x in range(-2, 3)])
HALF = (XY[:, 0] < 0) | ((XY[:, 0] == 0) & (XY[:, 1] < 0))
WPA, WPB = WA * HALF, WA * ~HALF
WPA, WPB = WPA / WPA.sum(), WPB / WPB.sum()
QPA, QPB = float(WPA @ WPA), float(WPB @ WPB)


def resolve(h, p, a, c, observation_variance, delta=DELTA, samples=199., bounded=True, checks=None):
    # Fixed direction m=(R,0,0,R), so every squared norm has factor two.
    floor = np.maximum(1e-12, 1e-8 * np.maximum(2 * h * h, 2 * a * a))
    p = np.maximum(p, 0) if bounded else np.maximum(p, floor)
    pa = np.maximum(observation_variance * QA, floor)
    pc = np.maximum(observation_variance * QC, floor)
    d2 = 2 * (a - h) ** 2
    radius2 = (p + pa) / max(delta, 1e-6)
    beta = np.where(d2 > radius2,
        1 - np.sqrt(np.minimum(radius2 / np.maximum(d2, 1e-30), 1)), 0)
    if delta == 0:
        beta = np.zeros_like(beta)
    if checks is not None:
        ca, cb = checks
        ra = (p+np.maximum(observation_variance*QPA, floor))/max(delta, 1e-6)
        rb = (p+np.maximum(observation_variance*QPB, floor))/max(delta, 1e-6)
        corroborated = ((ca-h)*(cb-h) > 0) & (2*(ca-h)**2 > ra) & (2*(cb-h)**2 > rb)
        beta = np.where(corroborated, beta, 0)
    hc = (1 - beta) * h + beta * a
    pclip = ((1 - beta) * np.sqrt(p) + beta * np.sqrt(pa)) ** 2 + beta * (1 - beta) * d2
    gain = np.maximum(ALPHA, pclip / (pclip + pc))
    if bounded:
        cap = np.maximum(ALPHA, 1/(1+np.maximum(samples, 1)))
        gain = np.minimum(gain, cap+(1-cap)*beta)
    out = (1 - gain) * hc + gain * c
    pout = (1 - gain) ** 2 * pclip + gain * gain * pc
    return out, pout, gain, beta


def summarize(x, truth):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), bias=float(x.mean() - truth),
        rmse=float(np.sqrt(np.mean((x - truth) ** 2))),
        p99=float(np.quantile(x, .99)), maximum=float(x.max()))


def stationary(kind, pixels, frames):
    rng = np.random.default_rng(SEED)
    if kind == "rare_100":
        mu, variance = 1.099, .001 * .999 * 99**2
    else:
        mu, variance = 1., 1.
    obs = 2 * variance
    modes = {}
    for mode in ("confidence_oracle", "confidence_plugin_fp16", "confidence_disabled_projection_fp16",
            "confidence_single_pilot_oracle", "confidence_plugin_unbounded_fp16"):
        modes[mode] = [np.full(pixels, mu), np.full(pixels, ALPHA / (2-ALPHA) * obs * QC), 0]
    raw = np.full(pixels, mu)
    rms = np.full(pixels, np.sqrt(variance + mu**2))
    nraw = np.full(pixels, 199.)
    legacy = np.full(pixels, mu)
    legacy_nraw = np.full(pixels, 199.)
    legacy_nfiltered = np.full(pixels, 199. / QC)
    legacy_bures_obs = variance / (4 * mu)
    mean_trace = []
    for frame in range(frames):
        if kind == "rare_100":
            samples = np.where(rng.random((pixels, 25)) < .001, 100., 1.)
        else:
            samples = rng.exponential(size=(pixels, 25))
        a, c = samples @ WA, samples @ WC
        pilot_obs = np.maximum(2 * ((samples * samples) @ WA - a * a), 0) / (1-QA)
        prior = np.maximum(2 * (rms-raw) * (rms+raw), 0) / (1-1/nraw)
        for mode, state in modes.items():
            h, p, count = state
            estimated_obs = np.full(pixels, obs) if mode.endswith("oracle") else np.maximum(prior, pilot_obs)
            h, p, gain, beta = resolve(h, p, a, c, estimated_obs,
                0. if mode == "confidence_disabled_projection_fp16" else DELTA,
                samples=nraw, bounded=mode != "confidence_plugin_unbounded_fp16",
                checks=None if mode in ("confidence_single_pilot_oracle", "confidence_plugin_unbounded_fp16")
                    else (samples @ WPA, samples @ WPB))
            if mode.endswith("fp16"):
                h = pack_half(h)
                p = pack_half(np.sqrt(p)) ** 2
            assert np.all(np.isfinite(h)) and np.all(p >= 0)
            modes[mode] = [h, p, count + int(np.count_nonzero(beta))]
        # Legacy equation with oracle local Bures variance and true group ESS.
        d2 = (np.sqrt(c) - np.sqrt(legacy)) ** 2
        total = legacy_bures_obs * (QC + 1 / legacy_nfiltered) + 1e-10
        a0 = 1 / (legacy_nraw + 1)
        gain = a0 + (1-a0) * .005 * d2 / (total + .005 * d2)
        legacy = (1-gain) * legacy + gain * c
        legacy_nraw = 1 / ((1-gain)**2 / legacy_nraw + gain**2)
        legacy_nfiltered = 1 / ((1-gain)**2 / legacy_nfiltered + gain**2 * QC)
        raw = pack_half((1-ALPHA) * raw + ALPHA * samples[:, 12])
        rms = pack_half(np.sqrt((1-ALPHA) * rms*rms + ALPHA * samples[:, 12]**2))
        nraw = pack_half(1 / ((1-ALPHA)**2 / nraw + ALPHA**2))
        if frame in (0, 9, 99, frames-1):
            mean_trace.append(dict(frame=frame+1, legacy=float(legacy.mean()),
                **{name: float(state[0].mean()) for name, state in modes.items()}))
    result = dict(truth=mu, frames=frames, pixels=pixels,
        legacy_oracle=summarize(legacy, mu), trace=mean_trace)
    for name, (h, p, count) in modes.items():
        result[name] = dict(**summarize(h, mu),
            projection_rate=count / (frames*pixels),
            predicted_trace_variance=float(p.mean()),
            observed_trace_mse=float(2*np.mean((h-mu)**2)))
    return result


def deterministic_sequences():
    result = {}
    for name in ("current_impulse", "pilot_impulse", "bright_history", "step_up", "step_down"):
        h = np.array([100. if name in ("bright_history", "step_down") else 1.])
        p = np.array([1e-8])
        raw, rms = h.copy(), h.copy()
        n = 199.
        trace = []
        for frame in range(121):
            values = np.ones((1, 25)) * (100. if name == "step_up" else 1.)
            if frame == 0 and name == "current_impulse":
                values[0, 12] = 100.
            if frame == 0 and name == "pilot_impulse":
                values[0, 11] = 100.
            a, c = values @ WA, values @ WC
            pa_obs = np.maximum(2*((values*values) @ WA-a*a), 0)/(1-QA)
            prior = np.maximum(2*(rms-raw)*(rms+raw), 0)/(1-1/n)
            h, p, gain, beta = resolve(h, p, a, c, np.maximum(pa_obs, prior), checks=(values @ WPA, values @ WPB))
            if frame in (0, 1, 9, 29, 59, 120):
                trace.append(dict(frame=frame, h=float(h[0]), gain=float(gain[0]),
                    beta=float(beta[0]), trace_variance=float(p[0])))
            raw = (1-ALPHA)*raw + ALPHA*values[:, 12]
            rms = np.sqrt((1-ALPHA)*rms*rms+ALPHA*values[:, 12]**2)
        result[name] = trace
    return result


def identities():
    rng = np.random.default_rng(SEED+1)
    u = rng.normal(size=(20000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    r = rng.exponential(size=len(u))
    x = np.column_stack((r[:, None]*u, r))
    m = x.mean(axis=0)
    trace = float(np.mean(np.sum((x-m)**2, axis=1)))
    identity = float(2*np.mean(r*r)-m@m)
    assert abs(trace-identity) < 1e-12
    assert np.all(WA*WC == 0) and WA[12] == 0 and WC[12] > 0
    # Convex moment interpolation and projected ball radius.
    a, b = x[:10000], x[10000:]
    t = rng.random((10000, 1))
    mixed = a*(1-t)+b*t
    cone_error = float(np.max(np.linalg.norm(mixed[:, :3], axis=1)-mixed[:, 3]))
    assert cone_error < 1e-12
    radius = .3*np.linalg.norm(a-b, axis=1)
    beta = np.maximum(0, 1-radius/np.linalg.norm(a-b, axis=1))
    projected = a*(1-beta[:, None])+b*beta[:, None]
    error = float(np.max(np.abs(np.linalg.norm(projected-b, axis=1)-radius)))
    assert error < 1e-12
    # Exact known-variance Chebyshev threshold audit, independent non-Gaussian
    # innovations. This is separate from the adaptive plug-in pipeline above.
    sample = rng.exponential(size=(100000, 25))
    pilot = sample @ WA
    event = 2*(pilot-1)**2 > (2*QA)/DELTA
    return dict(trace_identity_error=abs(trace-identity), projection_error=error,
        cone_error=cone_error, oracle_exponential_test_rate=float(event.mean()),
        pilot_effective_samples=1/QA, current_effective_samples=1/QC)


def expand(path):
    parts = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r'^\s*#include\s+"([^"]+)"\s*$', line)
        if match:
            name = match[1]
            child = ROOT / "shaders" / name.lstrip("/") if name.startswith("/") else path.parent / name
            parts.append(expand(child))
        else:
            parts.append(line)
    return "\n".join(parts)


def compile_shaders():
    compiler = Path("E:/VulkanSDK/Bin/glslangValidator.exe")
    folder = ROOT / "temp/confidence_validate"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in (0, 1):
        for entry in (1, 50, 51, 56, 58, 61, 63, 65, 66, 71, 72):
            source = expand(ROOT / f"shaders/composite{entry}.csh")
            source = re.sub(r'(#define MAXENT_TEMPORAL_CONFIDENCE_CLAMP) [01]', rf'\g<1> {mode}', source)
            lines = source.splitlines()
            index = next(i for i, s in enumerate(lines) if s.lstrip().startswith("#version"))
            version = lines.pop(index)
            source = version + "\n" + "\n".join(lines) + "\n"
            target = folder / f"mode{mode}_{entry}.glsl"
            target.write_text(source, encoding="utf-8")
            run = subprocess.run([str(compiler), "-V", "-R", "-S", "comp",
                "-DMC_GL_AMD_gpu_shader_half_float=1", "--auto-map-bindings", "--auto-map-locations",
                "-o", str(target.with_suffix(".spv")), str(target)], capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(run.stdout+run.stderr)
            rows.append(dict(mode=mode, entry=entry, sha256=hashlib.sha256(source.encode()).hexdigest()))
    return dict(compiler=str(compiler), backend="SPIR-V syntax validation; AMD half-float define", entries=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--pixels", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/calibration/temporal_confidence.json")
    args = parser.parse_args()
    result = dict(seed=SEED, python=platform.python_version(), numpy=np.__version__,
        scope="CPU flat-plane temporal algorithm audit; disjoint IID current raw groups; no geometry, PT reuse, or real GPU render",
        alpha_floor=ALPHA, nominal_delta=DELTA, identities=identities(),
        rare_100=stationary("rare_100", args.pixels, args.frames),
        exponential=stationary("exponential", args.pixels, args.frames),
        sequences=deterministic_sequences())
    # Default-policy regression contract. Keep failing ablations in the report.
    if ALPHA == .01 and DELTA == .01:
        assert result["rare_100"]["confidence_plugin_fp16"]["rmse"] < .2
        assert abs(result["exponential"]["confidence_plugin_fp16"]["bias"]) < .01
        assert result["sequences"]["current_impulse"][0]["gain"] <= ALPHA+1e-12
        assert result["sequences"]["pilot_impulse"][0]["h"] == 1.
        assert abs(result["sequences"]["step_up"][0]["h"]-100) < .001
        assert abs(result["sequences"]["step_down"][0]["h"]-1) < .001
    if args.compile:
        result["compilation"] = compile_shaders()
    result["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "shaders/lib/lighting/denoiser/temporal_confidence.glsl",
            ROOT / "shaders/lib/settings.glsl", ROOT / "shaders/lib/lighting/denoiser/internal_constants.glsl"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "compilation"}, indent=2))


if __name__ == "__main__":
    main()
