"""Historical observation-field fault/ABI tests plus current GLSL compilation.

Current estimator propagation is tested by audit_estimator_variance.py.

No game state changes, GPU execution or new storage. Run with python -B.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import runpy
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = -2.
MAX = 65504.


def known(s):
    return math.isfinite(s) and 0 <= s <= MAX


def resolve(sigmas, weights):
    pairs = [(s, w) for s, w in zip(sigmas, weights) if known(s) and math.isfinite(w) and w > 0]
    if not pairs:
        return UNKNOWN
    v = sum(s*s*w for s, w in pairs)/sum(w for _, w in pairs)
    return math.sqrt(v) if math.isfinite(v) and 0 <= v <= MAX*MAX else UNKNOWN


def moments_valid(m, rms):
    m = np.asarray(m, dtype=float)
    if not np.isfinite(m).all() or not math.isfinite(rms) or m[3] < 0 or rms < 0:
        return False
    if rms > MAX or np.any(np.abs(m) > MAX):
        return False
    if m[3] == 0:
        return rms == 0 and np.all(m[:3] == 0)
    eps = max(.002*m[3], 2**-23)
    return rms+eps >= m[3] and np.linalg.norm(m[:3]) <= m[3]+eps


def tests():
    invalid = [float('nan'), float('inf'), -float('inf'), -1., UNKNOWN, MAX+1]
    assert not any(known(s) for s in invalid)
    assert resolve(invalid, [1.]*len(invalid)) == UNKNOWN
    assert resolve([UNKNOWN, 3.], [.99, .01]) == 3.
    assert resolve([UNKNOWN, 3.], [1., 0.]) == UNKNOWN
    assert resolve([0., UNKNOWN], [1., 1.]) == 0.
    assert resolve([10000., UNKNOWN], [1., 1.]) == 10000.  # legitimate high V=1e8
    assert resolve([MAX], [1.]) == MAX
    # Light and Kish use all accepted taps, independently of variance validity.
    weights = np.array([.9, .1])
    light = np.array([[0, 0, 1, 2], [0, 0, 3, 4]], dtype=float)
    assert np.allclose(weights @ light, [0, 0, 1.2, 2.2])
    assert resolve([UNKNOWN, UNKNOWN], weights) == UNKNOWN
    # Mixed history / surface-virtual branch weights preserve unknown iff no
    # positively weighted contributor has a known uncertainty.
    for t in [0., .001, .5, 1.]:
        assert resolve([UNKNOWN, UNKNOWN], [1-t, t]) == UNKNOWN
        s = resolve([2., 4.], [1-t, t])
        assert abs(s*s-((1-t)*4+t*16)) < 1e-12
    # Both FP16 history lane orders preserve -2, N, and valid zero variance.
    for s in [UNKNOWN, -1., 0., 1., MAX]:
        for n in [1., 32., 199., MAX]:
            pair = np.array([s, n], dtype=np.float16)
            words = pair.view(np.uint16).astype(np.uint32)
            packed = words[0] | (words[1] << np.uint32(16))
            decoded = np.array([packed & 65535, packed >> np.uint32(16)], dtype=np.uint16).view(np.float16)
            assert np.array_equal(pair, decoded)
    faults = [([0, 0, 0, 0], 1e-7), ([0, 0, 0, 0], 110.),
              ([0, 0, 0, 100], 1.), ([0, 0, 101, 100], 110.),
              ([0, float('nan'), 0, 100], 110.), ([0, 0, 0, 100], float('inf'))]
    assert all(not moments_valid(m, rms) for m, rms in faults)
    assert moments_valid([0, 0, 0, 0], 0.)
    assert moments_valid([0, 0, 100, 100], 100.)
    assert moments_valid([0, 0, 100.05, 100], 100.)  # FP16-scale cone roundoff
    # Finite legitimate states retain their old field interpolation.
    rng = np.random.default_rng(906202620)
    worst = 0.
    for _ in range(10000):
        sigmas = np.exp(rng.uniform(-10, 10, 9))
        a = rng.random(9)
        old = math.sqrt(np.dot(a, sigmas**2)/a.sum())
        worst = max(worst, abs(resolve(sigmas, a)-old)/max(old, 1e-30))
    assert worst < 1e-12
    # No variance-driven freeze when either temporal variance is unknown.
    fallback = max(0., 1/(1+199.))
    assert fallback == .005
    return dict(fault_cases=len(faults), finite_mixture_cases=10000,
                all_unknown_preserves_light=True, half_sentinel_roundtrip=True,
                finite_mixture_max_relative_error=worst,
                no_history_freeze_alpha=fallback, high_valid_variance_preserved=1e8)


def compile_more():
    audit = runpy.run_path(str(ROOT/'tools/audit_temporal_confidence.py'))
    result = audit['compile_shaders']()
    compiler = result['compiler']
    folder = ROOT/'temp/uncertainty_validate'
    folder.mkdir(parents=True, exist_ok=True)
    entries = [(ROOT/f'shaders/composite{i}.csh', 'comp', None)
               for i in (52, 53, 54, 55, 59, 67, 68, 69, 70)]
    entries += [(ROOT/'shaders/post/composite_lighting.glsl', 'frag', v) for v in (24, 25, 37, 38, 50)]
    entries += [(ROOT/f'shaders/ray{i}.rgen', 'rgen', None) for i in range(6)]
    entries += [(ROOT/f'shaders/ray{i}_0.{stage}', stage, None)
                for i in (0, 4, 5) for stage in ('rahit', 'rchit', 'rmiss')]
    for source_path, stage, view in entries:
        source = audit['expand'](source_path)
        lines = source.splitlines()
        i = next(i for i, s in enumerate(lines) if s.lstrip().startswith('#version'))
        version = lines.pop(i)
        target = folder/f'{source_path.stem}_{view}.glsl'
        source = version+'\n'+'\n'.join(lines)+'\n'
        target.write_text(source, encoding='utf-8')
        cmd = [compiler, '-V', '-R', '-S', stage, '-DMC_GL_AMD_gpu_shader_half_float=1',
               '--auto-map-bindings', '--auto-map-locations']
        if stage in ('rgen', 'rahit', 'rchit', 'rmiss'):
            cmd += ['--target-env', 'vulkan1.2']
        if view is not None:
            cmd.append(f'-DDEBUG_VIEW={view}')
        cmd += ['-o', str(target.with_suffix('.spv')), str(target)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout+proc.stderr)
    return dict(variants=len(result['entries'])+len(entries), compiler=compiler,
                scope='SPIR-V compilation only; not GPU execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    result = dict(scope='CPU arithmetic/FP16 model; production GLSL syntax checked only with --compile',
                  tests=tests())
    if args.compile:
        result['compilation'] = compile_more()
    paths = list((ROOT/'shaders/lib/lighting/denoiser').glob('*.glsl'))
    paths += [ROOT/'shaders/lib/math/denoiser_uncertainty.glsl']
    paths += list((ROOT/'shaders/post/denoiser').glob('**/*.glsl'))
    paths += [ROOT/f'shaders/lib/buffers/{name}_buffer.glsl' for name in ('diffuse', 'specular', 'debug')]
    result['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
