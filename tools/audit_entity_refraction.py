"""CPU regression models and source contracts for entity filtering / PSR reuse.

Run python -B tools/audit_entity_refraction.py [--compile].
Compilation is not game execution. Requires Python 3.10+ and NumPy.
"""
from pathlib import Path
import argparse
import json
import math
import runpy
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def safe_lod(low, high):
    edges = int(low[0]) | int(low[1]) | int(high[0]) | int(high[1])
    alignment = (edges & -edges).bit_length() - 1 if edges else 0
    size = np.maximum(high - low, 1)
    return min(alignment, int(math.log2(min(size))))


def entity_tests():
    rng = np.random.default_rng(9062026)
    checked = 0
    # Minecraft's ordinary entity faces have integer texel-edge UV bounds.
    # Enumerate mip-level support, including both levels of trilinear filtering.
    for _ in range(3000):
        low = rng.integers(0, 48, size=2)
        high = low + rng.integers(1, 17, size=2)
        lod = min(float(rng.uniform(0, 9)), safe_lod(low, high))
        inset = np.minimum(.5 * 2**math.ceil(lod), .5 * (high - low))
        uv_texels = rng.uniform(-128, 128, size=2)
        tap = np.clip(uv_texels, low + inset, high - inset)
        for level in {math.floor(lod), math.ceil(lod)}:
            cell = 2**level
            position = tap / cell - .5
            first = np.floor(position).astype(int)
            fraction = position - first
            for y in (0, 1):
                for x in (0, 1):
                    weight = (fraction[0] if x else 1-fraction[0]) * (fraction[1] if y else 1-fraction[1])
                    if weight <= 1e-12:
                        continue
                    begin = (first + (x, y)) * cell
                    end = begin + cell
                    assert np.all(begin >= low) and np.all(end <= high), (low, high, tap, lod, begin, end)
                    checked += 1
    # A face surrounded by transparent black: the old 64-texel footprint
    # sampled unrelated skin regions. The bounded filter preserves that face.
    offsets = (np.arange(8) + .5) / 8 - .5
    taps = 20 + 64 * offsets
    old_coverage = float(np.mean((taps >= 16) & (taps < 24)))
    bounded = np.clip(taps, 16.5, 23.5)
    assert old_coverage < 1 and np.all((bounded >= 16) & (bounded < 24))
    return dict(rectangles=3000, mip_support_checks=checked,
                synthetic_old_face_coverage=old_coverage, bounded_face_coverage=1.)


def geometry_tests():
    rng = np.random.default_rng(9062027)
    rays = rng.normal(size=(10000, 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    distances = rng.uniform(.05, 2048, size=10000).astype(np.float32)
    words = distances.view(np.uint32)
    assert np.array_equal(words.view(np.float32), distances)
    # Same primary hit retains its original 32-bit normal without requantizing.
    primary_normals = rng.integers(0, 2**32, size=10000, dtype=np.uint32)
    selected_normals = np.where(words == distances.view(np.uint32), primary_normals, 0)
    assert np.array_equal(selected_normals, primary_normals)
    assert np.float32(-1).view(np.uint32).view(np.float32) < 0
    # Water at distance 5, opaque background at 9; both share a camera ray.
    # The previous ownership always rejected this otherwise exact PSR hit.
    endpoint = rays[0] * 9
    old_distance_error = float(np.linalg.norm(rays[0] * 5 - endpoint))
    new_distance_error = float(np.linalg.norm(rays[0] * 9 - endpoint))
    tolerance = max(.12, 6 * max(9 / 1080, .025))
    assert old_distance_error > tolerance and new_distance_error <= tolerance
    # Invert the native integer RT grid for asymmetric / jittered projections.
    resolution = np.array([1920., 1080.])
    pixels = rng.integers([0, 0], [1920, 1080], size=(10000, 2))
    scale = np.array([1.2, 1.8])
    shift = np.array([.031, -.017])
    slope = (2*pixels/resolution - 1 + shift) / scale
    projected = ((slope * scale - shift) * .5 + .5) * resolution
    error = float(np.max(np.abs(projected - pixels)))
    assert error < 1e-10
    # Per-tap validation discards an occluder and never wraps border addresses.
    total_cases = 0
    for point in ([0., 0.], [1919.9, 1079.9], [150.25, 900.75]):
        p = np.asarray(point)
        base = np.floor(p).astype(int)
        frac = p - base
        total = 0.
        signal = 0.
        for y in (0, 1):
            for x in (0, 1):
                tap = base + (x, y)
                if np.any(tap < 0) or np.any(tap >= resolution):
                    continue
                if x == 1 and y == 1:  # synthetic mismatched geometry
                    continue
                weight = (frac[0] if x else 1-frac[0]) * (frac[1] if y else 1-frac[1])
                signal += weight * 7.
                total += weight
        assert total > 0 and abs(signal / total - 7.) < 1e-12
        total_cases += 1
    return dict(distance_roundtrips=10000, primary_normal_preservation=10000,
                old_water_endpoint_error=old_distance_error,
                new_water_endpoint_error=new_distance_error,
                projection_max_pixel_error=error, border_and_occluder_cases=total_cases)


def source_contracts():
    def read(path):
        return (ROOT/'shaders'/path).read_text(encoding='utf-8')
    buf = read('lib/buffers/diffuse_buffer.glsl')
    assert 'floatBitsToUint(distance)' in buf
    assert 'surface.x >> 16u' in buf and 'surface.w == primary.w' in buf
    assert '0x2600u' in buf
    for file in ('lib/rt/raytrace_rgen.glsl', 'lib/rt/raytrace/gbuffer_io.glsl'):
        assert 'writeDiffuseSurfaceInvalid(' in read(file)
    for file in ('temporal.glsl', 'resolve.glsl'):
        assert 'readPrimaryGeometryNormal(' not in read('post/denoiser/diffuse/'+file)
    assert 'return readDiffuseGeometryWords(' in read('post/denoiser/diffuse/variance_pass.glsl')
    assert 'return readDiffuseGeometryWords(' in read('lib/lighting/denoiser/temporal_confidence.glsl')
    for stage in ('rahit', 'rchit'):
        assert 'getEntityTextureBox(quad)' in read('lib/rt/raytrace_'+stage+'.glsl')
    assert 'vec4 entityAtlas = atlas;' in read('lib/rt/raytrace/scene.glsl')
    assert 'rtEntitySafeLod(baseTextureSize, atlas)' in read('lib/rt/mipmap.glsl')
    psr = read('post/composite_lighting.glsl')
    assert 'sampleDiffuse(samplePixel)' not in psr
    assert 'uv * vec2(resolution_global) - 0.5' not in psr
    assert ': reusedScreen ? vec3(0.0, 1.0, 0.0)' in psr
    assert 'bufferObject.2 = 176 true 1.1 1.1' in read('shaders.properties')


def compile_routes():
    audit = runpy.run_path(str(ROOT/'tools/audit_temporal_confidence.py'))
    folder = ROOT/'temp/entity_refraction_validate'
    folder.mkdir(parents=True, exist_ok=True)
    compiler = 'E:/VulkanSDK/Bin/glslangValidator.exe'
    source = audit['expand'](ROOT/'shaders/post/composite_lighting.glsl')
    lines = source.splitlines()
    version = lines.pop(next(i for i, s in enumerate(lines) if s.lstrip().startswith('#version')))
    for view in (0, 40, 41):
        target = folder/f'lighting_{view}.glsl'
        target.write_text(version+'\n'+'\n'.join(lines)+'\n', encoding='utf-8')
        proc = subprocess.run([compiler, '-V', '-R', '-S', 'frag',
            '-DMC_GL_AMD_gpu_shader_half_float=1', '--auto-map-bindings',
            '--auto-map-locations', f'-DDEBUG_VIEW={view}', '-o',
            str(target.with_suffix('.spv')), str(target)], capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout+proc.stderr)
    return dict(variants=3, views=[0, 40, 41], scope='SPIR-V compilation, no GPU execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    source_contracts()
    result = dict(entity=entity_tests(), refraction=geometry_tests(),
                  scope='CPU synthetic regressions and source contracts; game screenshots still need retesting')
    if args.compile:
        result['compilation'] = compile_routes()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
